"""
전략 6: 시나리오 레벨 후처리 v2 — model40
================================================================
근거:
  - 배율(Public/CV) 1.156은 train→test 분포 차이에서 기인
  - test 시나리오의 피처 분포를 활용한 비모수 보정으로 배율 축소 가능
  - v4.1A/v4.1B 후처리 실패 교훈: 개별 row 보정은 noise 유발
  - v2 접근: 시나리오 단위 보정 (25행 일괄 스케일링)

접근법:
  A. 시나리오 피처 유사도 기반 보정
     - test 시나리오와 train 시나리오 간 피처 유사도 계산
     - 유사한 train 시나리오의 실제 target 분포로 test 예측 보정
  B. 시나리오 내 순서 보존 스케일링
     - 시나리오 내 예측값의 순서(rank)를 보존하면서
     - 시나리오 평균/std를 유사 train 시나리오의 target 분포로 매칭
  C. 극값 시나리오 증폭
     - 시나리오 피처(order_inflow, congestion 등)가 극단적인 경우
     - 예측값을 더 극단적으로 확장 (과소예측 보정)

실행: python src/run_model40_scenario_pp.py
예상 시간: ~2분 (후처리만, 모델 학습 없음)
※ 체크포인트 불필요, 기존 제출 파일만 사용
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')

TARGET = 'avg_delay_minutes_next_30m'

# ── 시나리오 집계에 사용할 핵심 피처 ──
SC_KEY_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

# ── 극값 판별에 사용할 피처 (축3 분석 결과) ──
EXTREME_INDICATORS = [
    'order_inflow_15m',   # 극값 152 vs 일반 68
    'robot_idle',         # 극값 11 vs 일반 34 (역방향)
    'low_battery_ratio',  # 극값 0.28 vs 일반 0.03
    'congestion_score',   # 극값 19.1 vs 일반 3.1
    'sku_concentration',  # 극값 0.55 vs 일반 0.37
]


def load_and_prepare():
    """train/test 로드 + 시나리오별 집계"""
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    # 시나리오별 집계 피처 (mean만 사용 — 유사도 계산용)
    train_sc = train.groupby('scenario_id')[SC_KEY_COLS].agg(['mean', 'std', 'max']).reset_index()
    train_sc.columns = ['scenario_id'] + [f'{c}_{s}' for c, s in
                                            train_sc.columns[1:]]

    test_sc = test.groupby('scenario_id')[SC_KEY_COLS].agg(['mean', 'std', 'max']).reset_index()
    test_sc.columns = ['scenario_id'] + [f'{c}_{s}' for c, s in
                                           test_sc.columns[1:]]

    # train 시나리오의 타겟 분포
    train_target_sc = train.groupby('scenario_id')[TARGET].agg(['mean', 'std', 'max', 'min', 'median']).reset_index()
    train_target_sc.columns = ['scenario_id', 'target_mean', 'target_std', 'target_max', 'target_min', 'target_median']

    return train, test, train_sc, test_sc, train_target_sc


def method_A_knn_calibration(test, test_sc, train_sc, train_target_sc, pred_col, K=5):
    """
    방법 A: KNN 기반 시나리오 유사도 보정
    - test 시나리오와 가장 유사한 train K개 시나리오의 target 분포로 보정
    - 예측 시나리오 평균을 유사 시나리오 target 평균으로 shift
    """
    # 공통 피처 컬럼
    feat_cols = [c for c in train_sc.columns if c != 'scenario_id']
    feat_cols = [c for c in feat_cols if c in test_sc.columns]

    # 정규화
    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_sc[feat_cols].fillna(0))
    test_feat  = scaler.transform(test_sc[feat_cols].fillna(0))

    # KNN
    knn = NearestNeighbors(n_neighbors=K, metric='euclidean')
    knn.fit(train_feat)
    distances, indices = knn.kneighbors(test_feat)

    # 유사 시나리오의 target 통계
    train_sc_ids = train_sc['scenario_id'].values
    calibrated = test.copy()

    for i, test_sid in enumerate(test_sc['scenario_id'].values):
        # 유사 train 시나리오
        neighbor_sids = train_sc_ids[indices[i]]
        neighbor_stats = train_target_sc[train_target_sc['scenario_id'].isin(neighbor_sids)]

        if len(neighbor_stats) == 0:
            continue

        # 거리 가중 평균
        dists = distances[i]
        weights = 1 / (dists + 1e-8)
        weights /= weights.sum()

        ref_mean = np.average(neighbor_stats['target_mean'].values, weights=weights)
        ref_std  = np.average(neighbor_stats['target_std'].values, weights=weights)

        # 현재 예측의 시나리오 통계
        mask = calibrated['scenario_id'] == test_sid
        pred_vals = calibrated.loc[mask, pred_col].values
        pred_mean = pred_vals.mean()
        pred_std_local = pred_vals.std() + 1e-8

        # 부드러운 보정: 예측 평균을 ref_mean 방향으로 50% shift
        alpha = 0.5  # 보정 강도 (0=무보정, 1=완전대체)
        new_mean = pred_mean * (1 - alpha) + ref_mean * alpha

        # 순서 보존 스케일링: z-score 기반
        z_scores = (pred_vals - pred_mean) / pred_std_local
        new_vals = new_mean + z_scores * pred_std_local  # std는 유지 (mean만 shift)

        calibrated.loc[mask, pred_col] = np.maximum(new_vals, 0)

    return calibrated


def method_B_rank_preserve_scaling(test, test_sc, train_sc, train_target_sc, pred_col, K=5):
    """
    방법 B: 순서 보존 + std 스케일링
    - mean shift (방법 A와 동일) + std도 유사 시나리오에 매칭
    """
    feat_cols = [c for c in train_sc.columns if c != 'scenario_id']
    feat_cols = [c for c in feat_cols if c in test_sc.columns]

    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_sc[feat_cols].fillna(0))
    test_feat  = scaler.transform(test_sc[feat_cols].fillna(0))

    knn = NearestNeighbors(n_neighbors=K, metric='euclidean')
    knn.fit(train_feat)
    distances, indices = knn.kneighbors(test_feat)

    train_sc_ids = train_sc['scenario_id'].values
    calibrated = test.copy()

    for i, test_sid in enumerate(test_sc['scenario_id'].values):
        neighbor_sids = train_sc_ids[indices[i]]
        neighbor_stats = train_target_sc[train_target_sc['scenario_id'].isin(neighbor_sids)]
        if len(neighbor_stats) == 0:
            continue

        dists = distances[i]
        weights = 1 / (dists + 1e-8)
        weights /= weights.sum()

        ref_mean = np.average(neighbor_stats['target_mean'].values, weights=weights)
        ref_std  = np.average(neighbor_stats['target_std'].values, weights=weights)

        mask = calibrated['scenario_id'] == test_sid
        pred_vals = calibrated.loc[mask, pred_col].values
        pred_mean = pred_vals.mean()
        pred_std_local = pred_vals.std() + 1e-8

        # mean + std 둘 다 보정
        alpha = 0.3  # mean shift 강도 (더 보수적)
        beta  = 0.3  # std scaling 강도

        new_mean = pred_mean * (1 - alpha) + ref_mean * alpha
        new_std  = pred_std_local * (1 - beta) + ref_std * beta

        z_scores = (pred_vals - pred_mean) / pred_std_local
        new_vals = new_mean + z_scores * new_std

        calibrated.loc[mask, pred_col] = np.maximum(new_vals, 0)

    return calibrated


def method_C_extreme_amplify(test, pred_col, amplify_factor=1.3, threshold_pct=90):
    """
    방법 C: 극값 시나리오 증폭
    - 시나리오 피처가 극단적(상위 10%)이면 예측값을 증폭
    - 과소예측 보정 (pred/actual = 0.32 → 0.42 목표)
    """
    calibrated = test.copy()

    # 시나리오별 극값 지표 계산
    sc_extreme = test.groupby('scenario_id')[EXTREME_INDICATORS].mean().reset_index()

    # 각 지표의 극값 threshold (상위 percentile)
    extreme_scores = np.zeros(len(sc_extreme))
    for col in EXTREME_INDICATORS:
        if col == 'robot_idle':
            # 역방향: 낮을수록 극값
            threshold = np.percentile(sc_extreme[col], 100 - threshold_pct)
            extreme_scores += (sc_extreme[col] < threshold).astype(float)
        else:
            threshold = np.percentile(sc_extreme[col], threshold_pct)
            extreme_scores += (sc_extreme[col] > threshold).astype(float)

    # 3개 이상 지표에서 극값 → 극값 시나리오
    sc_extreme['extreme_count'] = extreme_scores
    extreme_sids = sc_extreme[sc_extreme['extreme_count'] >= 3]['scenario_id'].values

    print(f'  극값 시나리오: {len(extreme_sids)}개 ({len(extreme_sids)/len(sc_extreme)*100:.1f}%)')

    for sid in extreme_sids:
        mask = calibrated['scenario_id'] == sid
        pred_vals = calibrated.loc[mask, pred_col].values
        sc_mean = pred_vals.mean()

        # 시나리오 평균 이상인 값만 증폭 (저값은 유지)
        amplified = np.where(
            pred_vals > sc_mean,
            sc_mean + (pred_vals - sc_mean) * amplify_factor,
            pred_vals
        )
        calibrated.loc[mask, pred_col] = np.maximum(amplified, 0)

    return calibrated


def main():
    t0 = time.time()
    print('=' * 70)
    print('model40: 시나리오 레벨 후처리 v2')
    print('  기준: blend_m33m34_w80 Public=9.8073')
    print('  방법: A(KNN mean shift), B(rank-preserve scaling), C(극값 증폭)')
    print('=' * 70)

    os.makedirs(SUB_DIR, exist_ok=True)

    # ── 데이터 로드 ──
    print('\n[데이터 로드]')
    train, test, train_sc, test_sc, train_target_sc = load_and_prepare()
    print(f'  train 시나리오: {train_sc.shape[0]}, test 시나리오: {test_sc.shape[0]}')
    print(f'  시나리오 피처: {train_sc.shape[1] - 1}')

    # ── 기준 제출 로드 ──
    base_files = {
        'bw80': 'blend_m33m34_w80.csv',
        'm34':  'model34_6asym20.csv',
        'm33':  'model33_6model_asym.csv',
    }

    for base_key, base_fname in base_files.items():
        base_path = os.path.join(SUB_DIR, base_fname)
        if not os.path.exists(base_path):
            print(f'  ⚠️ {base_fname} 없음 — 스킵')
            continue

        base_sub = pd.read_csv(base_path)
        base_pred = base_sub[TARGET].values
        print(f'\n{"="*70}')
        print(f'기준: {base_fname}')
        print(f'  mean={base_pred.mean():.2f}, std={base_pred.std():.2f}, max={base_pred.max():.2f}')

        # test에 예측값 병합
        test_with_pred = test.copy()
        test_with_pred[TARGET] = base_pred

        # ── 방법 A: KNN mean shift ──
        for K in [3, 5, 10]:
            for alpha_tag, alpha_val in [('a30', 0.3), ('a50', 0.5)]:
                print(f'\n[A] KNN K={K}, α={alpha_val}')

                # alpha를 직접 조절하기 위해 함수 내부 alpha 대신 직접 구현
                feat_cols = [c for c in train_sc.columns if c != 'scenario_id']
                feat_cols = [c for c in feat_cols if c in test_sc.columns]
                scaler = StandardScaler()
                train_feat = scaler.fit_transform(train_sc[feat_cols].fillna(0))
                test_feat  = scaler.transform(test_sc[feat_cols].fillna(0))
                knn = NearestNeighbors(n_neighbors=K, metric='euclidean')
                knn.fit(train_feat)
                distances, indices = knn.kneighbors(test_feat)
                train_sc_ids = train_sc['scenario_id'].values

                cal = test_with_pred.copy()
                for i, test_sid in enumerate(test_sc['scenario_id'].values):
                    neighbor_sids = train_sc_ids[indices[i]]
                    neighbor_stats = train_target_sc[train_target_sc['scenario_id'].isin(neighbor_sids)]
                    if len(neighbor_stats) == 0: continue
                    dists = distances[i]
                    wts = 1 / (dists + 1e-8); wts /= wts.sum()
                    ref_mean = np.average(neighbor_stats['target_mean'].values, weights=wts)
                    mask = cal['scenario_id'] == test_sid
                    pv = cal.loc[mask, TARGET].values
                    pm = pv.mean(); ps = pv.std() + 1e-8
                    nm = pm * (1-alpha_val) + ref_mean * alpha_val
                    cal.loc[mask, TARGET] = np.maximum(nm + (pv - pm), 0)

                cal_pred = cal[TARGET].values
                fname = f'model40_{base_key}_A_K{K}_{alpha_tag}.csv'
                sub = base_sub.copy(); sub[TARGET] = cal_pred
                sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
                print(f'  mean={cal_pred.mean():.2f}, std={cal_pred.std():.2f}, '
                      f'max={cal_pred.max():.2f} → {fname}')

        # ── 방법 B: Rank-preserve + std scaling ──
        for K in [5]:
            for alpha_val, beta_val in [(0.3, 0.3), (0.2, 0.2), (0.3, 0.5)]:
                print(f'\n[B] K={K}, α={alpha_val}, β={beta_val}')
                cal = method_B_rank_preserve_scaling(
                    test_with_pred, test_sc, train_sc, train_target_sc,
                    TARGET, K=K)
                # method_B uses fixed alpha/beta internally — reimplement for variable params
                feat_cols = [c for c in train_sc.columns if c != 'scenario_id']
                feat_cols = [c for c in feat_cols if c in test_sc.columns]
                scaler = StandardScaler()
                train_feat = scaler.fit_transform(train_sc[feat_cols].fillna(0))
                test_feat  = scaler.transform(test_sc[feat_cols].fillna(0))
                knn = NearestNeighbors(n_neighbors=K, metric='euclidean')
                knn.fit(train_feat)
                distances, indices = knn.kneighbors(test_feat)
                train_sc_ids = train_sc['scenario_id'].values

                cal = test_with_pred.copy()
                for i, test_sid in enumerate(test_sc['scenario_id'].values):
                    neighbor_sids = train_sc_ids[indices[i]]
                    neighbor_stats = train_target_sc[train_target_sc['scenario_id'].isin(neighbor_sids)]
                    if len(neighbor_stats) == 0: continue
                    dists = distances[i]
                    wts = 1 / (dists + 1e-8); wts /= wts.sum()
                    ref_mean = np.average(neighbor_stats['target_mean'].values, weights=wts)
                    ref_std  = np.average(neighbor_stats['target_std'].values, weights=wts)
                    mask = cal['scenario_id'] == test_sid
                    pv = cal.loc[mask, TARGET].values
                    pm = pv.mean(); ps = pv.std() + 1e-8
                    nm = pm*(1-alpha_val) + ref_mean*alpha_val
                    ns = ps*(1-beta_val) + ref_std*beta_val
                    z = (pv - pm) / ps
                    cal.loc[mask, TARGET] = np.maximum(nm + z * ns, 0)

                cal_pred = cal[TARGET].values
                tag = f'a{int(alpha_val*100)}b{int(beta_val*100)}'
                fname = f'model40_{base_key}_B_K{K}_{tag}.csv'
                sub = base_sub.copy(); sub[TARGET] = cal_pred
                sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
                print(f'  mean={cal_pred.mean():.2f}, std={cal_pred.std():.2f}, '
                      f'max={cal_pred.max():.2f} → {fname}')

        # ── 방법 C: 극값 증폭 ──
        for amp in [1.2, 1.3, 1.5]:
            print(f'\n[C] 극값 증폭 factor={amp}')
            cal = method_C_extreme_amplify(test_with_pred, TARGET,
                                           amplify_factor=amp, threshold_pct=90)
            cal_pred = cal[TARGET].values
            fname = f'model40_{base_key}_C_amp{int(amp*10)}.csv'
            sub = base_sub.copy(); sub[TARGET] = cal_pred
            sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
            print(f'  mean={cal_pred.mean():.2f}, std={cal_pred.std():.2f}, '
                  f'max={cal_pred.max():.2f} → {fname}')

        # ── 복합: A + C (mean shift + 극값 증폭) ──
        print(f'\n[A+C] KNN K=5 α=0.3 + 극값 증폭 1.3')
        # Step 1: KNN mean shift
        feat_cols = [c for c in train_sc.columns if c != 'scenario_id']
        feat_cols = [c for c in feat_cols if c in test_sc.columns]
        scaler = StandardScaler()
        train_feat = scaler.fit_transform(train_sc[feat_cols].fillna(0))
        test_feat  = scaler.transform(test_sc[feat_cols].fillna(0))
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(train_feat)
        distances, indices = knn.kneighbors(test_feat)
        train_sc_ids = train_sc['scenario_id'].values

        cal = test_with_pred.copy()
        for i, test_sid in enumerate(test_sc['scenario_id'].values):
            neighbor_sids = train_sc_ids[indices[i]]
            neighbor_stats = train_target_sc[train_target_sc['scenario_id'].isin(neighbor_sids)]
            if len(neighbor_stats) == 0: continue
            dists = distances[i]
            wts = 1 / (dists + 1e-8); wts /= wts.sum()
            ref_mean = np.average(neighbor_stats['target_mean'].values, weights=wts)
            mask = cal['scenario_id'] == test_sid
            pv = cal.loc[mask, TARGET].values
            pm = pv.mean()
            nm = pm * 0.7 + ref_mean * 0.3
            cal.loc[mask, TARGET] = np.maximum(nm + (pv - pm), 0)

        # Step 2: 극값 증폭
        cal = method_C_extreme_amplify(cal, TARGET, amplify_factor=1.3, threshold_pct=90)
        cal_pred = cal[TARGET].values
        fname = f'model40_{base_key}_AC.csv'
        sub = base_sub.copy(); sub[TARGET] = cal_pred
        sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f'  mean={cal_pred.mean():.2f}, std={cal_pred.std():.2f}, '
              f'max={cal_pred.max():.2f} → {fname}')

    # ── 종합 ──
    print('\n' + '=' * 70)
    print('생성 완료')
    print('=' * 70)

    # 생성된 파일 목록
    model40_files = sorted([f for f in os.listdir(SUB_DIR) if f.startswith('model40_')])
    print(f'  생성 파일 수: {len(model40_files)}')
    for f in model40_files:
        df = pd.read_csv(os.path.join(SUB_DIR, f))
        pred = df[TARGET].values
        print(f'  {f:<50s} mean={pred.mean():.2f} std={pred.std():.2f} max={pred.max():.2f}')

    elapsed = time.time() - t0
    print(f'\n총 소요 시간: {elapsed:.1f}초')
    print('\n핵심 주의사항:')
    print('  - v4.1A/v4.1B 후처리 실패 이력 있음 (IF/분류기 모두 악화)')
    print('  - 시나리오 단위 보정이 row 단위보다 안정적이지만, 보장 없음')
    print('  - 보수적(α=0.2~0.3)부터 시작, Public 확인 후 강도 조절')
    print('  - std 확대 → 배율 개선 기대, 과도한 std → 노이즈')
    print('※ USER가 제출하여 실제 Public 스코어 확인 필수')


if __name__ == '__main__':
    main()
