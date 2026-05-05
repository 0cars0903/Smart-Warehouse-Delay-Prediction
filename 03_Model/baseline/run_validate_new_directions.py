"""
원자 단위 분석 기반 — 3가지 새 방향 순차 검증
==============================================
핵심 발견:
  - 시나리오 간 분산 63.4%, 시나리오 내 36.6%
  - 현재 base 모델은 모든 구간에서 ~15분 상수 예측 (시나리오 레벨 구분 실패)
  - 시나리오 내 피처 diff → 타겟 diff 상관 < 0.05 (거의 랜덤)
  - 시나리오 평균 오라클 MAE: 5.15분 (이론적 하한)

검증 A: 시나리오 집계 피처 broadcast (기존 파이프라인 + 집계 피처)
검증 B: 2단계 분리 모델 (시나리오 레벨 → 잔차)
검증 C: 잔차 타겟 학습 (시나리오 평균을 피처로, 타겟은 잔차)

실행: python src/run_validate_new_directions.py
예상 시간: ~30분 (LGBM 단독 × 3 × 5fold)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings, gc, os, sys, time

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features, get_feature_cols

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
N_SPLITS = 5
RANDOM_STATE = 42

# 기존 최적 LGBM 파라미터
LGBM_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# 시나리오 집계에 사용할 핵심 피처
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]


def load_base_data():
    """기본 FE v1 파이프라인 (lag 1-6, rolling 3/5/10)"""
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(
        train, test, layout,
        lag_lags=[1,2,3,4,5,6],
        rolling_windows=[3,5,10],
    )
    return train, test


def add_scenario_agg_features(df, agg_cols=None):
    """
    시나리오 집계 피처 broadcast.
    test에서도 동일 시나리오 25행이 모두 존재하므로 리크 없음.
    각 행에 시나리오 내 전체 통계를 추가.
    """
    if agg_cols is None:
        agg_cols = SC_AGG_COLS

    df = df.copy()
    for col in agg_cols:
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]
        df[f'sc_{col}_mean'] = grp.transform('mean')
        df[f'sc_{col}_std']  = grp.transform('std').fillna(0)
        df[f'sc_{col}_max']  = grp.transform('max')
        df[f'sc_{col}_min']  = grp.transform('min')
        # 현재값 - 시나리오 평균 (위치 정보)
        df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']
    return df


def run_lgbm_oof(X, y_log, groups, params=None, label='LGBM'):
    """LGBM OOF (log1p 공간) → raw 공간 MAE 반환"""
    if params is None:
        params = LGBM_PARAMS

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X))
    y_raw = np.expm1(y_log)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_log, groups)):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y_log.iloc[tr_idx],
              eval_set=[(X.iloc[va_idx], y_log.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(X.iloc[va_idx])
        mae = np.abs(np.expm1(oof[va_idx]) - y_raw.values[va_idx]).mean()
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_raw = np.expm1(oof)
    total_mae = np.abs(oof_raw - y_raw.values).mean()
    print(f'  [{label}] 전체 OOF MAE={total_mae:.4f}  pred_std={oof_raw.std():.2f}')
    return oof, oof_raw, total_mae


def run_lgbm_oof_raw(X, y_raw, groups, params=None, label='LGBM-raw'):
    """LGBM OOF (raw 공간 직접 학습) → raw 공간 MAE 반환"""
    if params is None:
        params = LGBM_PARAMS.copy()

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_raw, groups)):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y_raw.iloc[tr_idx],
              eval_set=[(X.iloc[va_idx], y_raw.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(X.iloc[va_idx])
        mae = np.abs(oof[va_idx] - y_raw.values[va_idx]).mean()
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    total_mae = np.abs(oof - y_raw.values).mean()
    print(f'  [{label}] 전체 OOF MAE={total_mae:.4f}  pred_std={oof.std():.2f}')
    return oof, total_mae


# ============================================================
# 검증 A: 시나리오 집계 피처 broadcast
# ============================================================
def validate_A(train, test):
    """
    기존 FE v1 + 시나리오 집계 피처(mean/std/max/min/diff) broadcast.
    test에서 시나리오 25행의 피처를 모두 볼 수 있으므로 look-ahead가 아닌 정당한 정보.
    """
    print('\n' + '=' * 60)
    print('검증 A: 시나리오 집계 피처 broadcast')
    print('=' * 60)

    train_a = add_scenario_agg_features(train)
    # test도 동일하게 (실제 제출 시 사용)
    # test_a = add_scenario_agg_features(test)

    feat_cols = [c for c in train_a.columns
                 if c not in {'ID','scenario_id','layout_id','avg_delay_minutes_next_30m'}
                 and train_a[c].dtype != object]
    X = train_a[feat_cols].fillna(0)
    y_log = np.log1p(train_a['avg_delay_minutes_next_30m'])
    groups = train_a['scenario_id']

    sc_feats = [c for c in feat_cols if c.startswith('sc_')]
    print(f'총 피처: {len(feat_cols)} (기존 ~212 + 시나리오 집계 ~{len(sc_feats)})')

    _, oof_raw, mae = run_lgbm_oof(X, y_log, groups, label='A-LGBM')

    # 타겟 구간별 분석
    y_raw = train_a['avg_delay_minutes_next_30m'].values
    print(f'\n[검증 A] 타겟 구간별 MAE:')
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    for lo, hi in bins:
        mask = (y_raw >= lo) & (y_raw < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(oof_raw[mask] - y_raw[mask]).mean()
            seg_pred = oof_raw[mask].mean()
            print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d} MAE={seg_mae:6.2f} pred_mean={seg_pred:6.2f}')

    return mae


# ============================================================
# 검증 B: 2단계 분리 모델
# ============================================================
def validate_B(train, test):
    """
    Stage 1: 시나리오-레벨 모델 (시나리오 집계 피처 → 시나리오 평균 지연)
    Stage 2: 시나리오 내 잔차 모델 (행 단위 피처 + Stage1 예측 → 잔차)
    최종 = Stage1 예측 + Stage2 예측
    """
    print('\n' + '=' * 60)
    print('검증 B: 2단계 분리 모델 (Scenario-Level + Residual)')
    print('=' * 60)

    y_raw = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']

    # ── Stage 1: 시나리오-레벨 모델 ──────────────────────
    print('\n[Stage 1] 시나리오-레벨 모델')

    # 시나리오별 집계 피처 구성
    all_num_cols = [c for c in train.select_dtypes(include=[np.number]).columns
                    if c not in ['ID','avg_delay_minutes_next_30m','scenario_id','ts_idx']]

    sc_data = train.groupby('scenario_id')[all_num_cols].agg(['mean','std','max','min','median'])
    sc_data.columns = ['_'.join(c) for c in sc_data.columns]
    sc_data = sc_data.fillna(0)
    sc_data['sc_target_mean'] = train.groupby('scenario_id')['avg_delay_minutes_next_30m'].mean()

    # 시나리오 고유 ID 리스트 (GroupKFold용)
    sc_ids = sc_data.index.values

    X_sc = sc_data.drop('sc_target_mean', axis=1)
    y_sc = sc_data['sc_target_mean']
    y_sc_log = np.log1p(y_sc)

    print(f'시나리오-레벨 피처: {X_sc.shape[1]}개, 시나리오: {len(X_sc)}개')

    # 시나리오-레벨 LGBM OOF
    sc_params = LGBM_PARAMS.copy()
    sc_params['num_leaves'] = 63  # 시나리오 수 10000이므로 작은 모델
    sc_params['min_child_samples'] = 10
    sc_params['n_estimators'] = 2000

    gkf = GroupKFold(n_splits=N_SPLITS)
    # 시나리오 레벨에서의 GroupKFold는 그냥 KFold와 동일 (각 시나리오가 1행)
    # 하지만 행 단위 GroupKFold와 동일한 fold 분할을 맞춰야 함
    # → 행 단위 GKF에서 validation 시나리오 추출
    row_gkf = GroupKFold(n_splits=N_SPLITS)
    sc_oof_pred = np.zeros(len(sc_ids))

    sc_id_to_idx = {sid: i for i, sid in enumerate(sc_ids)}

    for fold, (tr_idx, va_idx) in enumerate(row_gkf.split(train, y_raw, groups)):
        va_sids = train.iloc[va_idx]['scenario_id'].unique()
        tr_sids = train.iloc[tr_idx]['scenario_id'].unique()

        sc_tr_mask = np.isin(sc_ids, tr_sids)
        sc_va_mask = np.isin(sc_ids, va_sids)

        m = lgb.LGBMRegressor(**sc_params)
        m.fit(X_sc.values[sc_tr_mask], y_sc_log.values[sc_tr_mask],
              eval_set=[(X_sc.values[sc_va_mask], y_sc_log.values[sc_va_mask])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        sc_oof_pred[sc_va_mask] = np.expm1(m.predict(X_sc.values[sc_va_mask]))
        sc_mae = np.abs(sc_oof_pred[sc_va_mask] - y_sc.values[sc_va_mask]).mean()
        print(f'  [S1] Fold {fold+1}  sc_MAE={sc_mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    sc_total_mae = np.abs(sc_oof_pred - y_sc.values).mean()
    sc_corr = np.corrcoef(sc_oof_pred, y_sc.values)[0,1]
    print(f'  [S1] 시나리오 레벨 OOF MAE={sc_total_mae:.4f}  r={sc_corr:.4f}')
    print(f'  [S1] pred range: [{sc_oof_pred.min():.1f}, {sc_oof_pred.max():.1f}]')
    print(f'  [S1] true range: [{y_sc.values.min():.1f}, {y_sc.values.max():.1f}]')

    # Stage 1 예측을 행 단위로 broadcast
    sc_pred_map = pd.Series(sc_oof_pred, index=sc_ids)
    train_B = train.copy()
    train_B['s1_pred'] = train_B['scenario_id'].map(sc_pred_map)

    # Stage 1만으로의 전체 MAE
    s1_broadcast_mae = np.abs(y_raw.values - train_B['s1_pred'].values).mean()
    print(f'  [S1] broadcast MAE={s1_broadcast_mae:.4f}')

    # ── Stage 2: 잔차 모델 ──────────────────────────────
    print('\n[Stage 2] 잔차 모델')

    # 잔차 타겟 = y_raw - s1_pred
    train_B['residual'] = y_raw.values - train_B['s1_pred'].values
    # 시나리오 집계 피처도 추가
    train_B = add_scenario_agg_features(train_B)

    feat_cols_2 = [c for c in train_B.columns
                   if c not in {'ID','scenario_id','layout_id','avg_delay_minutes_next_30m','residual'}
                   and train_B[c].dtype != object]

    X2 = train_B[feat_cols_2].fillna(0)
    y2 = train_B['residual']  # raw 공간 잔차

    print(f'Stage 2 피처: {len(feat_cols_2)}개')

    # 잔차는 음수 가능 → log1p 변환 불가 → raw 공간 학습
    s2_params = LGBM_PARAMS.copy()
    # raw 공간 잔차 학습
    oof_resid, s2_mae = run_lgbm_oof_raw(X2, y2, groups, params=s2_params, label='S2-resid')

    # ── 최종 결합 ──────────────────────────────────────
    final_pred = train_B['s1_pred'].values + oof_resid
    final_mae = np.abs(y_raw.values - final_pred).mean()
    print(f'\n[검증 B] 최종 MAE = {final_mae:.4f}')
    print(f'  Stage 1 MAE: {s1_broadcast_mae:.4f}')
    print(f'  Stage 2 잔차 MAE: {s2_mae:.4f}')
    print(f'  결합 후: {final_mae:.4f}')
    print(f'  pred_std: {final_pred.std():.2f} (실제: {y_raw.std():.2f})')

    # 타겟 구간별
    print(f'\n[검증 B] 타겟 구간별 MAE:')
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    for lo, hi in bins:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(final_pred[mask] - y_raw.values[mask]).mean()
            seg_pred = final_pred[mask].mean()
            print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d} MAE={seg_mae:6.2f} pred_mean={seg_pred:6.2f}')

    return final_mae


# ============================================================
# 검증 C: 잔차 타겟 학습 (시나리오 평균을 피처로)
# ============================================================
def validate_C(train, test):
    """
    기존 파이프라인 그대로 두되:
      1) 시나리오 집계 피처 추가 (검증 A와 동일)
      2) 타겟을 log1p(y) 대신 log1p(y) - log1p(sc_mean)으로 변경
         → 시나리오 레벨 변동 제거, 시나리오 내 잔차만 학습
      3) 예측 시: pred = sc_mean_pred + residual_pred

    핵심: 시나리오 평균은 시나리오 집계 피처에서 계산 (타겟 리크 없음)
    """
    print('\n' + '=' * 60)
    print('검증 C: 잔차 타겟 학습')
    print('=' * 60)

    y_raw = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']

    # 시나리오 집계 피처 추가
    train_c = add_scenario_agg_features(train)

    # 시나리오 평균 (학습 데이터에서는 실제값 사용 가능 — CV에서 검증 fold는 별도)
    sc_mean_bc = train_c.groupby('scenario_id')['avg_delay_minutes_next_30m'].transform('mean')

    # 잔차 타겟: y - scenario_mean
    residual = y_raw - sc_mean_bc

    feat_cols = [c for c in train_c.columns
                 if c not in {'ID','scenario_id','layout_id','avg_delay_minutes_next_30m'}
                 and train_c[c].dtype != object]
    X = train_c[feat_cols].fillna(0)

    print(f'피처: {len(feat_cols)}개')
    print(f'잔차 타겟: mean={residual.mean():.4f}, std={residual.std():.4f}')

    # GroupKFold로 잔차 예측
    # 주의: validation fold에서는 시나리오 평균을 모름!
    # → validation fold의 시나리오 평균은 '시나리오 집계 피처'로부터 별도 예측 필요
    # → 여기서는 간이 검증으로 "시나리오 집계 피처의 sc_*_mean"이 시나리오 레벨의 프록시

    # 방법: validation에서도 시나리오 집계 피처(sc_*_mean 등)는 계산 가능 (피처이므로)
    # 하지만 시나리오 '타겟' 평균은 모름
    # → 잔차 = y - sc_mean_target 에서 sc_mean_target을 모름
    # → 그래서 이 검증에서는 '학습 시나리오 평균'은 알지만 '검증 시나리오 평균'은 모르는 상황

    # 현실적 접근: 검증 fold에서 sc_mean은 예측해야 함
    # → 검증 C를 제대로 하려면 검증 B처럼 2단계가 필요
    # → 여기서는 "시나리오 집계 피처가 포함된 상태에서 직접 y를 예측"하되
    #   log1p(y) - mean(log1p(y_scenario))를 타겟으로 사용하는 간이 방식 테스트

    # ===== 간이 C: 시나리오 집계 피처 + 정상 타겟 (log1p) =====
    # 이건 검증 A와 동일하지만, 추가로 시나리오 레벨 정보 활용도를 측정

    # ===== 진짜 C: fold-aware 잔차 학습 =====
    print('\n[검증 C] Fold-aware 잔차 학습')
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_final = np.zeros(len(train_c))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_raw, groups)):
        # 1. train fold의 시나리오 평균으로 잔차 타겟 생성
        tr_sc_mean = train_c.iloc[tr_idx].groupby('scenario_id')['avg_delay_minutes_next_30m'].transform('mean')
        tr_residual = y_raw.iloc[tr_idx].values - tr_sc_mean.values

        # 2. validation fold: 시나리오 평균을 모르므로
        #    시나리오 집계 피처(sc_*_mean 등)로부터 추정
        #    → 간이: validation 시나리오의 피처 기반으로 sc_mean 예측
        #    → 여기서는 train fold의 (sc_feat → sc_mean_target) 관계를 학습

        # Stage 1 (fold 내): 시나리오 레벨 모델
        tr_sids = train_c.iloc[tr_idx]['scenario_id'].unique()
        va_sids = train_c.iloc[va_idx]['scenario_id'].unique()

        # 시나리오별 집계
        sc_feat_cols = [c for c in feat_cols if c.startswith('sc_')]
        # train 시나리오 집계 (시나리오당 1행 — 첫 행 사용, 이미 broadcast라 동일)
        tr_sc = train_c.iloc[tr_idx].groupby('scenario_id')[sc_feat_cols].first()
        tr_sc['sc_target'] = train_c.iloc[tr_idx].groupby('scenario_id')['avg_delay_minutes_next_30m'].mean()

        va_sc = train_c.iloc[va_idx].groupby('scenario_id')[sc_feat_cols].first()
        va_sc_true = train_c.iloc[va_idx].groupby('scenario_id')['avg_delay_minutes_next_30m'].mean()

        # 시나리오 레벨 LGBM
        sc_m = lgb.LGBMRegressor(
            num_leaves=63, learning_rate=0.05, n_estimators=1000,
            feature_fraction=0.8, bagging_fraction=0.8, min_child_samples=10,
            objective='regression_l1', verbosity=-1, random_state=RANDOM_STATE)
        sc_m.fit(tr_sc.drop('sc_target', axis=1).fillna(0),
                 np.log1p(tr_sc['sc_target']),
                 eval_set=[(va_sc.fillna(0), np.log1p(va_sc_true))],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        va_sc_pred = np.expm1(sc_m.predict(va_sc.fillna(0)))
        sc_mae = np.abs(va_sc_pred - va_sc_true.values).mean()

        # validation 시나리오 평균 예측을 broadcast
        va_sc_pred_map = pd.Series(va_sc_pred, index=va_sids)
        va_sc_broadcast = train_c.iloc[va_idx]['scenario_id'].map(va_sc_pred_map).values

        # 3. Stage 2: 잔차 모델
        resid_m = lgb.LGBMRegressor(**LGBM_PARAMS)
        resid_m.fit(X.iloc[tr_idx], tr_residual,
                    eval_set=[(X.iloc[va_idx],
                               y_raw.iloc[va_idx].values - va_sc_broadcast)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        va_resid_pred = resid_m.predict(X.iloc[va_idx])
        oof_final[va_idx] = va_sc_broadcast + va_resid_pred

        fold_mae = np.abs(oof_final[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [C] Fold {fold+1}  MAE={fold_mae:.4f}  sc_MAE={sc_mae:.4f}  iter={resid_m.best_iteration_}')
        del sc_m, resid_m; gc.collect()

    total_mae = np.abs(oof_final - y_raw.values).mean()
    print(f'\n[검증 C] 최종 OOF MAE={total_mae:.4f}  pred_std={oof_final.std():.2f}')

    # 타겟 구간별
    print(f'\n[검증 C] 타겟 구간별 MAE:')
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    for lo, hi in bins:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(oof_final[mask] - y_raw.values[mask]).mean()
            seg_pred = oof_final[mask].mean()
            print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d} MAE={seg_mae:6.2f} pred_mean={seg_pred:6.2f}')

    return total_mae


# ============================================================
# 기준선: 기존 FE v1 LGBM 단독 (시나리오 집계 없이)
# ============================================================
def validate_baseline(train, test):
    """기존 FE v1 + LGBM 단독 (비교 기준)"""
    print('\n' + '=' * 60)
    print('기준선: FE v1 + LGBM 단독 (시나리오 집계 없음)')
    print('=' * 60)

    feat_cols = get_feature_cols(train)
    X = train[feat_cols].fillna(0)
    y_log = np.log1p(train['avg_delay_minutes_next_30m'])
    groups = train['scenario_id']

    print(f'피처: {len(feat_cols)}개')
    _, oof_raw, mae = run_lgbm_oof(X, y_log, groups, label='Baseline')

    # 타겟 구간별
    y_raw = train['avg_delay_minutes_next_30m'].values
    print(f'\n[기준선] 타겟 구간별 MAE:')
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    for lo, hi in bins:
        mask = (y_raw >= lo) & (y_raw < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(oof_raw[mask] - y_raw[mask]).mean()
            seg_pred = oof_raw[mask].mean()
            print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d} MAE={seg_mae:6.2f} pred_mean={seg_pred:6.2f}')

    return mae


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    print('데이터 로드 및 FE v1...')
    train, test = load_base_data()

    # 0. 기준선
    mae_base = validate_baseline(train, test)

    # A. 시나리오 집계 피처 broadcast
    mae_A = validate_A(train, test)

    # B. 2단계 분리 모델
    mae_B = validate_B(train, test)

    # C. 잔차 타겟 학습
    mae_C = validate_C(train, test)

    # 종합
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 60)
    print(f'종합 결과 ({elapsed:.1f}분 소요)')
    print('=' * 60)
    print(f'  기준선 (FE v1 LGBM 단독):  {mae_base:.4f}')
    print(f'  검증 A (시나리오 집계 FE):  {mae_A:.4f}  (Δ{mae_A - mae_base:+.4f})')
    print(f'  검증 B (2단계 분리):        {mae_B:.4f}  (Δ{mae_B - mae_base:+.4f})')
    print(f'  검증 C (잔차 타겟):         {mae_C:.4f}  (Δ{mae_C - mae_base:+.4f})')
    print(f'\n  현재 최고 (RF5 스태킹):    8.7911 (참고)')
    print(f'  시나리오 평균 오라클:       5.1526 (이론적 하한)')
    print('=' * 60)


if __name__ == '__main__':
    main()
