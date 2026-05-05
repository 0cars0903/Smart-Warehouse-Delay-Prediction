"""
모델실험28B: 극값 강화 메타 전략 (축3)
=============================================================
model22 base OOF 체크포인트 재사용 — 메타 학습기만 변경.
base learner 재학습 불필요 → 실행시간 ~5분.

핵심 문제:
  - target>=50 (전체 8%): 실제 평균 84.8인데 LGBM 예측 평균 ~16 (19%)
  - log1p 메타 공간에서 target 100 = 4.62, target 5 = 1.79
    → MAE 기준 극값 오차의 gradient가 1/20로 축소
  - 전체 MAE 개선에 극값 기여 낮음 → 모델이 무시

전략 (3단 비교):
  1. Meta-A: model22 재현 (log1p 공간, 기준선)
  2. Meta-B: Raw 공간 MAE + 극값 가중치 (sample_weight)
  3. Meta-C: Dual-space 블렌드 (Meta-A × α + Meta-B × (1-α))

기대:
  - Meta-B: 극값 MAE 대폭 개선 (68.4 → 55~60 목표), 전체 MAE 소폭 악화 가능
  - Meta-C: 블렌드로 전체 MAE 유지하면서 극값 개선 일부 흡수
  - pred_std 확장 → 배율 개선 가능성

핵심 원리:
  raw 공간 MAE에서 target=100 오차 80의 gradient = log1p 공간 gradient의 ~20배
  → 모델이 극값 예측에 훨씬 더 큰 노력을 기울이게 됨

실행: python src/run_exp_model28B_extreme_boost.py
예상 시간: ~5분 (체크포인트 재사용, 메타만 학습)
출력: submissions/model28B_extreme_boost.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import warnings, gc, os, sys, time

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
# model22 체크포인트 직접 참조 (base learner 재학습 불필요)
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model22_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 메타 하이퍼파라미터
# ─────────────────────────────────────────────
# Meta-A: model22 재현 (log1p 공간)
META_A_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# Meta-B: Raw 공간 + 극값 가중치
META_B_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1',  # raw MAE
    'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}


# ─────────────────────────────────────────────
# 체크포인트 로드
# ─────────────────────────────────────────────
def load_ckpt(name):
    oof  = np.load(os.path.join(CKPT_DIR, f'{name}_oof.npy'))
    test = np.load(os.path.join(CKPT_DIR, f'{name}_test.npy'))
    return oof, test


def load_all_base_oof():
    """model22 체크포인트에서 5모델 OOF 로드"""
    names = ['lgbm', 'tw18', 'cb', 'et', 'rf']
    oof_dict, test_dict = {}, {}
    for name in names:
        oof, test = load_ckpt(name)
        oof_dict[name] = oof
        test_dict[name] = test
        print(f'  {name:6s}: oof shape={oof.shape}, test shape={test.shape}')
    return oof_dict, test_dict


# ─────────────────────────────────────────────
# 메타 입력 구성 (model22 동일 형식)
# ─────────────────────────────────────────────
def build_meta_input(oof_dict, test_dict):
    """
    model22 동일: LGBM(log1p), CB(log1p), TW(log1p(clip)), ET(log1p), RF(log1p)
    모든 OOF는 저장 시 각 모델의 학습 공간 그대로 저장됨:
      - LGBM/CB/ET/RF: log1p 공간
      - TW1.8: raw 공간
    """
    meta_train = np.column_stack([
        oof_dict['lgbm'],                              # log1p
        oof_dict['cb'],                                # log1p
        np.log1p(np.maximum(oof_dict['tw18'], 0)),     # raw → log1p
        oof_dict['et'],                                # log1p
        oof_dict['rf'],                                # log1p
    ])

    meta_test = np.column_stack([
        test_dict['lgbm'],
        test_dict['cb'],
        np.log1p(np.maximum(test_dict['tw18'], 0)),
        test_dict['et'],
        test_dict['rf'],
    ])

    return meta_train, meta_test


# ─────────────────────────────────────────────
# Meta-A: model22 재현 (log1p 공간)
# ─────────────────────────────────────────────
def run_meta_A(meta_train, meta_test, y_raw, groups):
    """model22 동일: log1p 공간 MAE 메타"""
    print('\n[Meta-A] log1p 공간 MAE (model22 재현)')
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y_raw))
    test_pred = np.zeros(meta_test.shape[0])

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
        y_tr_log = np.log1p(y_raw.iloc[tr_idx].values)
        y_va_log = np.log1p(y_raw.iloc[va_idx].values)

        m = lgb.LGBMRegressor(**META_A_PARAMS)
        m.fit(X_tr, y_tr_log,
              eval_set=[(X_va, y_va_log)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.expm1(m.predict(X_va))
        test_pred += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_mae = np.abs(oof - y_raw.values).mean()
    print(f'  Meta-A OOF MAE={oof_mae:.4f} | pred_std={oof.std():.2f}')
    return oof, test_pred, oof_mae


# ─────────────────────────────────────────────
# Meta-B: Raw 공간 + 극값 가중치
# ─────────────────────────────────────────────
def run_meta_B(meta_train, meta_test, y_raw, groups):
    """
    Raw 공간 MAE + 극값 upweight

    sample_weight 설계:
      w_i = 1 + sqrt(max(0, y_i - median) / std)
      → target=10 (평범): w ≈ 1.0
      → target=50 (경계): w ≈ 1.0 + sqrt((50-12)/20) ≈ 2.38
      → target=100 (극값): w ≈ 1.0 + sqrt((100-12)/20) ≈ 3.10
      극값에 ~3배 가중치, but 부드러운 sqrt → 급격한 전환 없음
    """
    print('\n[Meta-B] Raw 공간 MAE + 극값 가중치')

    median_y = np.median(y_raw.values)
    std_y = y_raw.std()
    print(f'  target median={median_y:.2f}, std={std_y:.2f}')

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y_raw))
    test_pred = np.zeros(meta_test.shape[0])

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
        y_tr = y_raw.iloc[tr_idx].values
        y_va = y_raw.iloc[va_idx].values

        # 극값 가중치
        weights = 1.0 + np.sqrt(np.maximum(0, y_tr - median_y) / (std_y + 1e-8))

        m = lgb.LGBMRegressor(**META_B_PARAMS)
        m.fit(X_tr, y_tr,
              sample_weight=weights,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_va)
        test_pred += m.predict(meta_test) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_va).mean()
        print(f'  Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_mae = np.abs(oof - y_raw.values).mean()
    print(f'  Meta-B OOF MAE={oof_mae:.4f} | pred_std={oof.std():.2f}')
    return oof, test_pred, oof_mae


# ─────────────────────────────────────────────
# Meta-B2: Raw 공간 Huber loss (극값 gradient 증폭)
# ─────────────────────────────────────────────
def run_meta_B2(meta_train, meta_test, y_raw, groups):
    """
    Raw 공간 Huber loss (delta=10)

    Huber 특성:
      |error| <= 10: quadratic (L2처럼 부드러운 최적화)
      |error| > 10: linear (L1처럼 이상치 강건)

    → target=100 오류 80에서: MAE는 gradient=1, Huber는 gradient=1이지만
      학습 공간이 raw → 극값 오차가 log1p 대비 20배 더 큰 신호
    """
    print('\n[Meta-B2] Raw 공간 Huber (delta=10)')

    huber_params = META_B_PARAMS.copy()
    huber_params['objective'] = 'huber'
    huber_params['huber_delta'] = 10.0  # 중앙값 부근에서 L2, 극값에서 L1

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y_raw))
    test_pred = np.zeros(meta_test.shape[0])

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
        y_tr = y_raw.iloc[tr_idx].values
        y_va = y_raw.iloc[va_idx].values

        m = lgb.LGBMRegressor(**huber_params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_va)
        test_pred += m.predict(meta_test) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_va).mean()
        print(f'  Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_mae = np.abs(oof - y_raw.values).mean()
    print(f'  Meta-B2 OOF MAE={oof_mae:.4f} | pred_std={oof.std():.2f}')
    return oof, test_pred, oof_mae


# ─────────────────────────────────────────────
# Meta-C: Dual-space 블렌드
# ─────────────────────────────────────────────
def find_best_blend(oof_A, oof_B, y_raw, label='C'):
    """OOF에서 최적 블렌드 비율 탐색"""
    best_alpha, best_mae = 0.5, np.inf
    for a in np.arange(0.0, 1.01, 0.01):
        blend = a * oof_A + (1 - a) * oof_B
        mae = np.abs(blend - y_raw.values).mean()
        if mae < best_mae:
            best_mae = mae
            best_alpha = a
    return best_alpha, best_mae


# ─────────────────────────────────────────────
# 타겟 구간별 분석
# ─────────────────────────────────────────────
def segment_analysis(oof, y_raw, label):
    print(f'\n[{label}] 타겟 구간별 MAE')
    bins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 800)]
    results = {}
    for lo, hi in bins:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(oof[mask] - y_raw.values[mask]).mean()
            seg_pred_mean = oof[mask].mean()
            seg_actual_mean = y_raw.values[mask].mean()
            pred_ratio = seg_pred_mean / (seg_actual_mean + 1e-8)
            print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d}  '
                  f'MAE={seg_mae:6.2f}  '
                  f'pred_mean={seg_pred_mean:6.2f}  '
                  f'actual_mean={seg_actual_mean:6.2f}  '
                  f'ratio={pred_ratio:.3f}')
            results[(lo, hi)] = {
                'n': mask.sum(), 'mae': seg_mae,
                'pred_mean': seg_pred_mean, 'actual_mean': seg_actual_mean,
            }
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 60)
    print('모델실험28B: 극값 강화 메타 전략')
    print('기준: Model22 base OOF 재사용 (체크포인트)')
    print('변경: 메타 학습기 3종 비교 (log1p / raw+weight / Huber)')
    print('목표: target>=50 구간 MAE 개선 + 전체 MAE 유지')
    print('=' * 60)

    os.makedirs(SUB_DIR, exist_ok=True)

    # ══════════════════════════════════════════
    # 1. 데이터 및 체크포인트 로드
    # ══════════════════════════════════════════
    print('\n[데이터] 로드 중...')
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    # ★ 핵심: model22 OOF는 build_features() 내부 정렬(scenario_id+ts_idx) 순서로 저장됨
    # ts_idx = groupby('scenario_id').cumcount() 후 sort_values(['scenario_id','ts_idx'])
    # y_raw/groups도 동일하게 정렬해야 OOF와 매칭됨
    train['ts_idx'] = train.groupby('scenario_id').cumcount()
    train = train.sort_values(['scenario_id', 'ts_idx']).reset_index(drop=True)
    print(f'  [정렬] scenario_id + ts_idx 정렬 적용 (model22 OOF 순서 일치)')

    y_raw = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']
    print(f'  train: {len(train)} rows, {train["scenario_id"].nunique()} scenarios')
    print(f'  target: mean={y_raw.mean():.2f}, median={y_raw.median():.2f}, '
          f'std={y_raw.std():.2f}, max={y_raw.max():.2f}')

    # 타겟 분포 요약
    pct_50plus = (y_raw >= 50).mean() * 100
    pct_80plus = (y_raw >= 80).mean() * 100
    print(f'  target>=50: {pct_50plus:.1f}% ({(y_raw>=50).sum()} rows)')
    print(f'  target>=80: {pct_80plus:.1f}% ({(y_raw>=80).sum()} rows)')

    # model22 체크포인트 로드
    print(f'\n[체크포인트] model22 OOF 로드 ({CKPT_DIR})')
    oof_dict, test_dict = load_all_base_oof()

    # 메타 입력 구성
    meta_train, meta_test = build_meta_input(oof_dict, test_dict)
    print(f'  meta_train: {meta_train.shape}, meta_test: {meta_test.shape}')

    # ══════════════════════════════════════════
    # 2. Base Learner OOF 확인
    # ══════════════════════════════════════════
    print('\n' + '─' * 60)
    print('[Base OOF] 개별 모델 MAE')
    print('─' * 60)
    oof_raw_dict = {
        'LGBM': np.expm1(oof_dict['lgbm']),
        'TW':   oof_dict['tw18'],
        'CB':   np.expm1(oof_dict['cb']),
        'ET':   np.expm1(oof_dict['et']),
        'RF':   np.expm1(oof_dict['rf']),
    }
    for name, oof in oof_raw_dict.items():
        mae = np.abs(oof - y_raw.values).mean()
        print(f'  {name:6s} OOF MAE={mae:.4f}')

    # 극값 구간 base learner 예측 확인
    print('\n[극값 진단] target>=50 구간 base learner 예측')
    mask_50 = y_raw.values >= 50
    actual_50_mean = y_raw.values[mask_50].mean()
    for name, oof in oof_raw_dict.items():
        pred_50_mean = oof[mask_50].mean()
        pred_50_std = oof[mask_50].std()
        ratio = pred_50_mean / actual_50_mean
        print(f'  {name:6s}: pred_mean={pred_50_mean:6.2f}  actual_mean={actual_50_mean:6.2f}  '
              f'ratio={ratio:.3f}  pred_std={pred_50_std:.2f}')

    # ══════════════════════════════════════════
    # 3. 메타 학습기 3종 실행
    # ══════════════════════════════════════════
    print('\n' + '═' * 60)
    print('[META] 3종 메타 학습기 비교')
    print('═' * 60)

    # Meta-A: log1p 공간 (model22 재현)
    oof_A, test_A, mae_A = run_meta_A(meta_train, meta_test, y_raw, groups)

    # Meta-B: Raw 공간 + 극값 가중치
    oof_B, test_B, mae_B = run_meta_B(meta_train, meta_test, y_raw, groups)

    # Meta-B2: Raw 공간 Huber loss
    oof_B2, test_B2, mae_B2 = run_meta_B2(meta_train, meta_test, y_raw, groups)

    # ══════════════════════════════════════════
    # 4. 블렌드 최적화
    # ══════════════════════════════════════════
    print('\n' + '═' * 60)
    print('[BLEND] Dual-space 블렌드 최적화')
    print('═' * 60)

    # A+B 블렌드
    alpha_AB, mae_AB = find_best_blend(oof_A, oof_B, y_raw, 'A+B')
    blend_AB_oof = alpha_AB * oof_A + (1 - alpha_AB) * oof_B
    blend_AB_test = alpha_AB * test_A + (1 - alpha_AB) * test_B
    print(f'  Meta-A+B 최적 블렌드: α={alpha_AB:.2f} (A={alpha_AB:.2f}, B={1-alpha_AB:.2f})')
    print(f'  Blend MAE={mae_AB:.4f} | pred_std={blend_AB_oof.std():.2f}')

    # A+B2 블렌드
    alpha_AB2, mae_AB2 = find_best_blend(oof_A, oof_B2, y_raw, 'A+B2')
    blend_AB2_oof = alpha_AB2 * oof_A + (1 - alpha_AB2) * oof_B2
    blend_AB2_test = alpha_AB2 * test_A + (1 - alpha_AB2) * test_B2
    print(f'  Meta-A+B2 최적 블렌드: α={alpha_AB2:.2f} (A={alpha_AB2:.2f}, B2={1-alpha_AB2:.2f})')
    print(f'  Blend MAE={mae_AB2:.4f} | pred_std={blend_AB2_oof.std():.2f}')

    # B+B2 블렌드
    alpha_BB2, mae_BB2 = find_best_blend(oof_B, oof_B2, y_raw, 'B+B2')
    blend_BB2_oof = alpha_BB2 * oof_B + (1 - alpha_BB2) * oof_B2
    blend_BB2_test = alpha_BB2 * test_B + (1 - alpha_BB2) * test_B2
    print(f'  Meta-B+B2 최적 블렌드: α={alpha_BB2:.2f} (B={alpha_BB2:.2f}, B2={1-alpha_BB2:.2f})')
    print(f'  Blend MAE={mae_BB2:.4f} | pred_std={blend_BB2_oof.std():.2f}')

    # 3-way 블렌드 (A + B + B2)
    print(f'\n  [3-way] A + B + B2 블렌드 탐색...')
    best_3way_mae = np.inf
    best_3way_w = (0.33, 0.33, 0.34)
    for a in np.arange(0.0, 1.01, 0.05):
        for b in np.arange(0.0, 1.01 - a, 0.05):
            c = 1.0 - a - b
            if c < 0:
                continue
            blend = a * oof_A + b * oof_B + c * oof_B2
            mae = np.abs(blend - y_raw.values).mean()
            if mae < best_3way_mae:
                best_3way_mae = mae
                best_3way_w = (a, b, c)
    blend_3way_oof = best_3way_w[0]*oof_A + best_3way_w[1]*oof_B + best_3way_w[2]*oof_B2
    blend_3way_test = best_3way_w[0]*test_A + best_3way_w[1]*test_B + best_3way_w[2]*test_B2
    print(f'  3-way 최적: A={best_3way_w[0]:.2f}, B={best_3way_w[1]:.2f}, B2={best_3way_w[2]:.2f}')
    print(f'  3-way MAE={best_3way_mae:.4f} | pred_std={blend_3way_oof.std():.2f}')

    # ══════════════════════════════════════════
    # 5. 타겟 구간별 비교
    # ══════════════════════════════════════════
    print('\n' + '═' * 60)
    print('[구간별] 메타 전략 간 비교')
    print('═' * 60)

    seg_A  = segment_analysis(oof_A, y_raw, 'Meta-A (log1p)')
    seg_B  = segment_analysis(oof_B, y_raw, 'Meta-B (raw+wt)')
    seg_B2 = segment_analysis(oof_B2, y_raw, 'Meta-B2 (huber)')

    # 극값 구간 델타 분석
    print('\n[극값 개선 델타] 기준: Meta-A (model22 재현)')
    bins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 800)]
    for lo, hi in bins:
        key = (lo, hi)
        if key in seg_A and key in seg_B and key in seg_B2:
            delta_B  = seg_B[key]['mae'] - seg_A[key]['mae']
            delta_B2 = seg_B2[key]['mae'] - seg_A[key]['mae']
            marker_B  = '✅' if delta_B < -0.1 else ('⚠️' if delta_B > 0.1 else '—')
            marker_B2 = '✅' if delta_B2 < -0.1 else ('⚠️' if delta_B2 > 0.1 else '—')
            print(f'  [{lo:3d},{hi:3d}): '
                  f'B Δ={delta_B:+6.2f} {marker_B}  '
                  f'B2 Δ={delta_B2:+6.2f} {marker_B2}')

    # ══════════════════════════════════════════
    # 6. 예측 분포 비교
    # ══════════════════════════════════════════
    print('\n' + '═' * 60)
    print('[분포] 예측 분포 비교')
    print('═' * 60)

    for label, oof, test in [
        ('Meta-A', oof_A, test_A),
        ('Meta-B', oof_B, test_B),
        ('Meta-B2', oof_B2, test_B2),
        ('Blend-AB', blend_AB_oof, blend_AB_test),
        ('Blend-AB2', blend_AB2_oof, blend_AB2_test),
        ('3-way', blend_3way_oof, blend_3way_test),
    ]:
        print(f'  {label:12s}: '
              f'OOF mean={oof.mean():.2f} std={oof.std():.2f} max={oof.max():.2f} | '
              f'test mean={test.mean():.2f} std={test.std():.2f} max={test.max():.2f}')

    # ══════════════════════════════════════════
    # 7. 최적 전략 선정 + 제출 파일
    # ══════════════════════════════════════════
    print('\n' + '═' * 60)
    print('[최적 전략 선정]')
    print('═' * 60)

    candidates = {
        'Meta-A':    (mae_A,  oof_A,  test_A),
        'Meta-B':    (mae_B,  oof_B,  test_B),
        'Meta-B2':   (mae_B2, oof_B2, test_B2),
        'Blend-AB':  (mae_AB,  blend_AB_oof,  blend_AB_test),
        'Blend-AB2': (mae_AB2, blend_AB2_oof, blend_AB2_test),
        'Blend-BB2': (mae_BB2, blend_BB2_oof, blend_BB2_test),
        '3-way':     (best_3way_mae, blend_3way_oof, blend_3way_test),
    }

    # 전체 MAE 기준 정렬
    ranked = sorted(candidates.items(), key=lambda x: x[1][0])
    for rank, (name, (mae, oof, test)) in enumerate(ranked, 1):
        test_std = test.std()
        ratio_168 = mae * 1.168
        ratio_163 = mae * 1.163
        marker = '🏆' if rank == 1 else ''
        print(f'  #{rank} {name:12s}: CV={mae:.4f}  test_std={test_std:.2f}  '
              f'Public(×1.168)={ratio_168:.4f}  Public(×1.163)={ratio_163:.4f} {marker}')

    # 최적 전략으로 제출 파일 생성
    best_name = ranked[0][0]
    best_mae, best_oof, best_test = ranked[0][1]
    print(f'\n  → 최적: {best_name} (CV={best_mae:.4f})')

    # 제출 파일 (최적 전략)
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(best_test, 0)
    sub_path = os.path.join(SUB_DIR, 'model28B_extreme_boost.csv')
    sample.to_csv(sub_path, index=False)
    print(f'  제출 파일 (최적): {sub_path}')

    # Meta-B가 최적이 아니더라도 별도 제출 파일 (극값 전략 비교용)
    if best_name != 'Meta-B':
        sample2 = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
        sample2['avg_delay_minutes_next_30m'] = np.maximum(test_B, 0)
        sub_path2 = os.path.join(SUB_DIR, 'model28B_raw_weighted.csv')
        sample2.to_csv(sub_path2, index=False)
        print(f'  제출 파일 (Meta-B): {sub_path2}')

    # Meta-A(기준)와 다르면 A도 별도 저장
    if best_name != 'Meta-A':
        sample3 = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
        sample3['avg_delay_minutes_next_30m'] = np.maximum(test_A, 0)
        sub_path3 = os.path.join(SUB_DIR, 'model28B_baseline_A.csv')
        sample3.to_csv(sub_path3, index=False)
        print(f'  제출 파일 (Meta-A): {sub_path3}')

    # ══════════════════════════════════════════
    # 8. 최종 요약
    # ══════════════════════════════════════════
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 60)
    print(f'모델실험28B 결과 ({elapsed:.1f}분 소요)')
    print('=' * 60)
    print(f'  Meta-A (log1p, 기준) : CV={mae_A:.4f}  pred_std={oof_A.std():.2f}')
    print(f'  Meta-B (raw+weight)  : CV={mae_B:.4f}  pred_std={oof_B.std():.2f}  '
          f'Δ={mae_B - mae_A:+.4f}')
    print(f'  Meta-B2 (raw+huber)  : CV={mae_B2:.4f}  pred_std={oof_B2.std():.2f}  '
          f'Δ={mae_B2 - mae_A:+.4f}')
    print(f'  Blend-AB             : CV={mae_AB:.4f}  α={alpha_AB:.2f}  '
          f'Δ={mae_AB - mae_A:+.4f}')
    print(f'  Blend-AB2            : CV={mae_AB2:.4f}  α={alpha_AB2:.2f}  '
          f'Δ={mae_AB2 - mae_A:+.4f}')
    print(f'  3-way                : CV={best_3way_mae:.4f}  '
          f'A={best_3way_w[0]:.2f}/B={best_3way_w[1]:.2f}/B2={best_3way_w[2]:.2f}  '
          f'Δ={best_3way_mae - mae_A:+.4f}')
    print()
    print(f'  Model22 (기준)       : ~8.51 / Public 9.9385 (배율 ~1.168)')
    print(f'  최적 ({best_name:12s}): CV={best_mae:.4f}')
    print(f'  기대 Public (×1.168) : {best_mae * 1.168:.4f}')
    print(f'  기대 Public (×1.163) : {best_mae * 1.163:.4f}')

    # 극값 개선 요약
    if (50, 80) in seg_A and (50, 80) in seg_B:
        delta_50 = seg_B[(50, 80)]['mae'] - seg_A[(50, 80)]['mae']
        print(f'\n  극값 [50,80) 개선: Meta-B vs A = {delta_50:+.2f}')
    if (80, 800) in seg_A and (80, 800) in seg_B:
        delta_80 = seg_B[(80, 800)]['mae'] - seg_A[(80, 800)]['mae']
        print(f'  극값 [80,800) 개선: Meta-B vs A = {delta_80:+.2f}')

    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
