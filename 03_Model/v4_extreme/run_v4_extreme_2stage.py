"""
v4.0: 2-Stage 극값 전략 + Tier 3 비율 + 시나리오 집계 확장
=============================================================
핵심 문제: 8% 데이터(target≥50)가 전체 MAE의 45% 차지.
           base learner가 [80,800) 구간에서 실제의 32%만 예측.
           log1p 타겟 압축이 근본 원인으로 의심.

전략:
  [Stage 1] 시나리오 분류기 → extreme_prob 피처 생성
            - 시나리오 레벨 집계 피처 → P(mean_target ≥ 40) 예측
            - train/test 모두에 broadcast → base learner 입력에 포함

  [Stage 2] 7 Base Learner (기존 5 + raw-target 2)
            - log1p 5종: LGBM, CB, TW1.8, ET, RF (model30 동일)
            - raw 2종: LGBM-raw, CB-raw (log1p 압축 해소 → 극값 보존)
            - 모든 모델이 전체 데이터에서 학습 (hard routing 아님)
            - raw 모델의 OOF → log1p 변환하여 메타 입력 통일

  [Stage 3] 7모델 메타 스태킹 (LGBM meta)
            - 메타가 extreme_prob 높은 구간에서 raw 모델 가중 학습
            - log1p 공간에서 학습, expm1로 최종 예측

신규 피처:
  - Tier 3 비율 피처 5종 (극값 시나리오 특성 기반 복합 지표)
  - 시나리오 교차 집계 5종 (top 구분자 상호작용)
  - extreme_prob (시나리오 분류기 출력)

기준: model30 CV 8.4838 / Public 9.8279 / 배율 1.1584
목표: [80,800) MAE 20%↓ → 전체 MAE Δ-0.467 (시뮬레이션)

실행: python src/run_v4_extreme_2stage.py
예상 시간: ~50분 (분류기 + 7모델 × 5fold)
출력: submissions/v4_extreme_2stage.csv
체크포인트: docs/v4_ckpt/
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import warnings, gc, os, sys, time

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features, get_feature_cols

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'v4_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

EXTREME_THRESHOLD = 40  # 시나리오 mean_target ≥ 40 → 극값

# ─────────────────────────────────────────────
# 파라미터 (model30 기반)
# ─────────────────────────────────────────────

# log1p 모델: model30 Optuna 파라미터
LGBM_PARAMS = {
    'num_leaves': 129, 'learning_rate': 0.01021,
    'feature_fraction': 0.465, 'bagging_fraction': 0.947,
    'min_child_samples': 30, 'reg_alpha': 1.468, 'reg_lambda': 0.396,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

CB_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.01144,
    'depth': 9, 'l2_leaf_reg': 1.561,
    'random_strength': 1.359, 'bagging_temperature': 0.285,
    'loss_function': 'MAE', 'random_seed': RANDOM_STATE,
    'verbose': 0, 'early_stopping_rounds': 50,
}

TW18_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3.0,
    'loss_function': 'Tweedie:variance_power=1.8',
    'random_seed': RANDOM_STATE, 'verbose': 0,
    'early_stopping_rounds': 50,
}

ET_PARAMS = {
    'n_estimators': 500, 'max_depth': 20,
    'min_samples_leaf': 5, 'max_features': 0.7,
    'random_state': RANDOM_STATE, 'n_jobs': -1,
}

RF_PARAMS = {
    'n_estimators': 500, 'max_depth': 20,
    'min_samples_leaf': 5, 'max_features': 0.7,
    'random_state': RANDOM_STATE, 'n_jobs': -1,
}

# ★ raw-target 모델 파라미터
# log1p 대신 raw MAE로 학습 → 극값 압축 해소
# 정규화 더 강화 (raw 스케일이 크므로)
LGBM_RAW_PARAMS = {
    'num_leaves': 100, 'learning_rate': 0.01,
    'feature_fraction': 0.45, 'bagging_fraction': 0.90,
    'min_child_samples': 40, 'reg_alpha': 2.0, 'reg_lambda': 1.0,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

CB_RAW_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.01,
    'depth': 8, 'l2_leaf_reg': 3.0,
    'random_strength': 2.0, 'bagging_temperature': 0.5,
    'loss_function': 'MAE', 'random_seed': RANDOM_STATE,
    'verbose': 0, 'early_stopping_rounds': 50,
}

META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# 시나리오 집계 대상 피처 (18종)
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]


# ─────────────────────────────────────────────
# 체크포인트
# ─────────────────────────────────────────────
def save_ckpt(name, oof, test_pred):
    np.save(os.path.join(CKPT_DIR, f'{name}_oof.npy'), oof)
    np.save(os.path.join(CKPT_DIR, f'{name}_test.npy'), test_pred)

def load_ckpt(name):
    return (np.load(os.path.join(CKPT_DIR, f'{name}_oof.npy')),
            np.load(os.path.join(CKPT_DIR, f'{name}_test.npy')))

def ckpt_exists(name):
    return (os.path.exists(os.path.join(CKPT_DIR, f'{name}_oof.npy')) and
            os.path.exists(os.path.join(CKPT_DIR, f'{name}_test.npy')))


# ─────────────────────────────────────────────
# [FE] 시나리오 집계 피처 (11통계 — model22 동일)
# ─────────────────────────────────────────────
def add_scenario_agg_features(df):
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]
        df[f'sc_{col}_mean'] = grp.transform('mean')
        df[f'sc_{col}_std']  = grp.transform('std').fillna(0)
        df[f'sc_{col}_max']  = grp.transform('max')
        df[f'sc_{col}_min']  = grp.transform('min')
        df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']
        df[f'sc_{col}_median'] = grp.transform('median')
        df[f'sc_{col}_p10'] = grp.transform(lambda x: x.quantile(0.10))
        df[f'sc_{col}_p90'] = grp.transform(lambda x: x.quantile(0.90))
        df[f'sc_{col}_skew'] = grp.transform(lambda x: x.skew()).fillna(0)
        df[f'sc_{col}_kurtosis'] = grp.transform(lambda x: x.kurtosis()).fillna(0)
        cv_series = df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)
        df[f'sc_{col}_cv'] = cv_series.fillna(0)
    return df


# ─────────────────────────────────────────────
# [FE] Tier 1 비율 피처 (model28A — 5종)
# ─────────────────────────────────────────────
def add_layout_ratio_features_tier1(df):
    def safe_div(a, b, fill=0):
        return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = safe_div(
            df['sc_order_inflow_15m_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = safe_div(
            df['sc_congestion_score_mean'], df['intersection_count'])
    if 'sc_low_battery_ratio_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_stress'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df.get('sc_charge_queue_length_mean', 0),
            df['charger_count'])
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = safe_div(
            df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']
    return df


# ─────────────────────────────────────────────
# [FE] Tier 2 비율 피처 (model29A — 7종)
# ─────────────────────────────────────────────
def add_layout_ratio_features_tier2(df):
    def safe_div(a, b, fill=0):
        return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

    if all(c in df.columns for c in ['sc_congestion_score_mean', 'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'],
            df['robot_total'] ** 2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(
            df['robot_total'], df['floor_area_sqm'] / 100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(
            df['pack_station_count'], df['floor_area_sqm'] / 1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(
            df['sc_robot_charging_mean'], df['charger_count'])
    if 'sc_battery_mean_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_battery_per_robot'] = safe_div(
            df['sc_battery_mean_mean'] * df['robot_total'], df['robot_total'])
        if 'sc_robot_utilization_mean' in df.columns and 'charger_count' in df.columns:
            df['ratio_battery_per_robot'] = safe_div(
                df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'],
                df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(
            df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(
            df['sc_robot_idle_mean'], df['robot_total'])
    return df


# ─────────────────────────────────────────────
# ★ [FE] Tier 3 비율 피처 (v4 신규 — 극값 구분 복합 지표 5종)
# axis3 분석에서 극값 시나리오의 top 구분자를 조합
# ─────────────────────────────────────────────
def add_layout_ratio_features_tier3(df):
    """
    극값 시나리오 특성 기반 복합 비율 피처:
    - 극값 시나리오: order_inflow↑, robot_idle↓, low_battery↑, congestion↑
    - 이 조합을 layout capacity로 정규화하여 극값 감지 능력 부여
    """
    def safe_div(a, b, fill=0):
        return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

    # 13. 종합 스트레스 지수: (수요 × 혼잡 × 배터리위기) / (로봇수 × 충전기)
    #     극값 시나리오의 3대 신호를 하나로 결합
    cols_needed = ['sc_order_inflow_15m_mean', 'sc_congestion_score_mean',
                   'sc_low_battery_ratio_mean', 'robot_total', 'charger_count']
    if all(c in df.columns for c in cols_needed):
        df['ratio_total_stress'] = safe_div(
            df['sc_order_inflow_15m_mean'] *
            df['sc_congestion_score_mean'] *
            (df['sc_low_battery_ratio_mean'] + 0.01),  # +0.01: 0일 때도 값 유지
            df['robot_total'] * df['charger_count'])

    # 14. 유휴 부족 위험: (1 - idle_fraction) × 수요 / 면적
    #     로봇이 쉬지 못하면서 수요까지 높은 상황 = 극값 전조
    cols_needed = ['sc_robot_idle_mean', 'robot_total',
                   'sc_order_inflow_15m_mean', 'floor_area_sqm']
    if all(c in df.columns for c in cols_needed):
        idle_frac = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
        df['ratio_no_idle_demand'] = safe_div(
            (1 - idle_frac) * df['sc_order_inflow_15m_mean'],
            df['floor_area_sqm'] / 100)

    # 15. 배터리 위기 심도: low_battery × charge_queue / 충전기
    #     배터리 부족 + 충전 대기열 동시 발생 = 회복 불가 상태
    cols_needed = ['sc_low_battery_ratio_mean', 'sc_charge_queue_length_mean',
                   'charger_count']
    if all(c in df.columns for c in cols_needed):
        df['ratio_battery_crisis'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df['sc_charge_queue_length_mean'],
            df['charger_count'])

    # 16. SKU 집중도 × 혼잡: 특정 SKU에 주문 집중 + 혼잡 = 병목 극대화
    cols_needed = ['sc_sku_concentration_mean', 'sc_congestion_score_mean',
                   'intersection_count']
    if all(c in df.columns for c in cols_needed):
        df['ratio_sku_congestion'] = safe_div(
            df['sc_sku_concentration_mean'] * df['sc_congestion_score_mean'],
            df['intersection_count'])

    # 17. 처리량 갭: 수요 / (패킹활용률 × 패킹스테이션) — 처리 능력 대비 초과 수요
    cols_needed = ['sc_order_inflow_15m_mean', 'sc_pack_utilization_mean',
                   'pack_station_count']
    if all(c in df.columns for c in cols_needed):
        df['ratio_throughput_gap'] = safe_div(
            df['sc_order_inflow_15m_mean'],
            (df['sc_pack_utilization_mean'] + 0.01) * df['pack_station_count'])

    return df


# ─────────────────────────────────────────────
# ★ [FE] 시나리오 교차 집계 피처 (v4 신규 — top 구분자 상호작용)
# ─────────────────────────────────────────────
def add_scenario_cross_features(df):
    """
    극값 구분력이 높은 피처 쌍의 시나리오 레벨 상호작용:
    - axis3에서 확인된 극값 vs 일반 구분자 조합
    - 시나리오 내 range(max-min)과 IQR도 추가
    """
    # Top 구분자 쌍 상호작용 (시나리오 레벨)
    cross_pairs = [
        ('order_inflow_15m', 'congestion_score'),     # 수요 × 혼잡
        ('order_inflow_15m', 'low_battery_ratio'),    # 수요 × 배터리위기
        ('congestion_score', 'low_battery_ratio'),    # 혼잡 × 배터리위기
        ('robot_utilization', 'charge_queue_length'), # 가동률 × 충전대기
        ('sku_concentration', 'max_zone_density'),    # SKU집중 × 밀집도
    ]

    for col_a, col_b in cross_pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        # 시나리오 내 상호작용 평균
        interaction = df[col_a] * df[col_b]
        grp = interaction.groupby(df['scenario_id'])
        fname = f'sc_cross_{col_a[:6]}_{col_b[:6]}_mean'
        df[fname] = grp.transform('mean')

    # 핵심 피처의 range(max-min)과 IQR 추가 — 시나리오 내 변동성
    volatility_cols = ['order_inflow_15m', 'congestion_score',
                       'low_battery_ratio', 'robot_utilization']
    for col in volatility_cols:
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]
        df[f'sc_{col}_range'] = grp.transform('max') - grp.transform('min')
        df[f'sc_{col}_iqr'] = (grp.transform(lambda x: x.quantile(0.75)) -
                                grp.transform(lambda x: x.quantile(0.25)))

    return df


# ─────────────────────────────────────────────
# ★ [Stage 1] 시나리오 분류기: P(extreme) 예측
# ─────────────────────────────────────────────
def build_scenario_classifier(train, test, y_raw):
    """
    시나리오 레벨에서 극값 확률을 예측하는 LightGBM 분류기.

    train에서:
    - 시나리오별 mean target 계산 → 이진 라벨 (≥ threshold)
    - 시나리오 레벨 피처(sc_*_mean 등)로 분류기 학습
    - OOF 방식으로 train에 극값 확률 생성

    test에서:
    - 동일한 시나리오 레벨 피처로 극값 확률 예측

    Returns: train extreme_prob, test extreme_prob
    """
    print('\n' + '─' * 60)
    print('[Stage 1] 시나리오 극값 분류기')
    print('─' * 60)

    # 시나리오 레벨 집계
    sc_mean_target = train.groupby('scenario_id')[
        'avg_delay_minutes_next_30m'].mean()
    sc_label = (sc_mean_target >= EXTREME_THRESHOLD).astype(int)

    n_extreme = sc_label.sum()
    n_total = len(sc_label)
    print(f'  극값 시나리오: {n_extreme}/{n_total} ({100*n_extreme/n_total:.1f}%)')
    print(f'  임계값: mean_target ≥ {EXTREME_THRESHOLD}')

    # 시나리오 레벨 피처 추출 (sc_*_mean 계열)
    sc_feat_cols = [c for c in train.columns
                    if c.startswith('sc_') and c.endswith('_mean')]
    # layout 피처도 추가 (시나리오 내 동일값이므로 first)
    layout_cols = [c for c in train.columns
                   if c in ['robot_total', 'charger_count', 'floor_area_sqm',
                            'intersection_count', 'pack_station_count',
                            'aisle_width_avg', 'zone_count']]
    # ratio 피처도 추가
    ratio_cols = [c for c in train.columns if c.startswith('ratio_')]

    all_sc_cols = sc_feat_cols + layout_cols + ratio_cols

    # 시나리오 레벨로 집약
    train_sc = train.groupby('scenario_id')[all_sc_cols].mean()
    test_sc  = test.groupby('scenario_id')[all_sc_cols].mean()

    print(f'  분류기 피처 수: {len(all_sc_cols)}')

    # OOF 방식으로 train 확률 생성
    clf_params = {
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.7, 'bagging_fraction': 0.8,
        'min_child_samples': 20, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
        'objective': 'binary', 'metric': 'binary_logloss',
        'n_estimators': 500, 'bagging_freq': 1,
        'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
        'is_unbalance': True,  # 극값이 소수이므로
    }

    scenarios = train_sc.index.values
    X_sc = train_sc.fillna(0).values
    y_sc = sc_label.loc[train_sc.index].values
    X_te_sc = test_sc.fillna(0).values

    # 5-fold OOF for scenario-level classifier
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = np.zeros(len(scenarios))
    test_prob = np.zeros(len(test_sc))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_sc, y_sc)):
        m = lgb.LGBMClassifier(**clf_params)
        m.fit(X_sc[tr_idx], y_sc[tr_idx],
              eval_set=[(X_sc[va_idx], y_sc[va_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        oof_prob[va_idx] = m.predict_proba(X_sc[va_idx])[:, 1]
        test_prob += m.predict_proba(X_te_sc)[:, 1] / N_SPLITS
        del m; gc.collect()

    # 분류 성능 평가
    from sklearn.metrics import roc_auc_score, f1_score
    auc = roc_auc_score(y_sc, oof_prob)
    pred_label = (oof_prob >= 0.5).astype(int)
    f1 = f1_score(y_sc, pred_label)
    print(f'  분류기 AUC: {auc:.4f}, F1: {f1:.4f}')
    print(f'  OOF prob: mean={oof_prob.mean():.4f}, extreme mean={oof_prob[y_sc==1].mean():.4f}, '
          f'normal mean={oof_prob[y_sc==0].mean():.4f}')

    # 시나리오 레벨 → 행 레벨 broadcast
    # train
    sc_prob_map_train = dict(zip(scenarios, oof_prob))
    train_extreme_prob = train['scenario_id'].map(sc_prob_map_train).values

    # test
    test_scenarios = test_sc.index.values
    sc_prob_map_test = dict(zip(test_scenarios, test_prob))
    test_extreme_prob = test['scenario_id'].map(sc_prob_map_test).values

    print(f'  train extreme_prob: mean={train_extreme_prob.mean():.4f}, '
          f'std={train_extreme_prob.std():.4f}')
    print(f'  test  extreme_prob: mean={test_extreme_prob.mean():.4f}, '
          f'std={test_extreme_prob.std():.4f}')

    # shift 확인
    shift = abs(train_extreme_prob.mean() - test_extreme_prob.mean()) / (train_extreme_prob.std() + 1e-8)
    print(f'  extreme_prob shift: {shift:.3f}σ {"✅" if shift < 0.4 else "⚠️"}')

    return train_extreme_prob, test_extreme_prob


# ─────────────────────────────────────────────
# 데이터 로드 + 전체 FE 파이프라인
# ─────────────────────────────────────────────
def load_data():
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    train, test = build_features(train, test, layout,
                                  lag_lags=[1,2,3,4,5,6],
                                  rolling_windows=[3,5,10])

    # 시나리오 집계 (11통계)
    train = add_scenario_agg_features(train)
    test  = add_scenario_agg_features(test)

    # Tier 1 비율 (5종)
    train = add_layout_ratio_features_tier1(train)
    test  = add_layout_ratio_features_tier1(test)

    # Tier 2 비율 (7종)
    train = add_layout_ratio_features_tier2(train)
    test  = add_layout_ratio_features_tier2(test)

    # ★ Tier 3 비율 (v4 신규 5종)
    train = add_layout_ratio_features_tier3(train)
    test  = add_layout_ratio_features_tier3(test)

    # ★ 시나리오 교차 집계 (v4 신규)
    train = add_scenario_cross_features(train)
    test  = add_scenario_cross_features(test)

    ratio_cols = [c for c in train.columns if c.startswith('ratio_')]
    cross_cols = [c for c in train.columns if c.startswith('sc_cross_')]
    range_cols = [c for c in train.columns if c.endswith('_range') or c.endswith('_iqr')]

    return train, test, ratio_cols, cross_cols, range_cols


def get_feat_cols(train):
    return [c for c in train.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and train[c].dtype != object]


# ─────────────────────────────────────────────
# [Stage 2A] Base Learner — log1p target (5종, model30 동일)
# ─────────────────────────────────────────────
def train_lgbm_oof(X_train, X_test, y_log, groups, feat_cols):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0); X_te_np = X_test[feat_cols].fillna(0)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr_np.iloc[tr_idx], y_log.iloc[tr_idx],
              eval_set=[(X_tr_np.iloc[va_idx], y_log.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(X_tr_np.iloc[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.iloc[va_idx].values)).mean()
        print(f'  [LGBM] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred

def train_tw18_oof(X_train, X_test, y_raw, groups, feat_cols):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values; X_te_np = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_raw, groups)):
        train_pool = cb.Pool(X_tr_np[tr_idx], y_raw.values[tr_idx])
        val_pool   = cb.Pool(X_tr_np[va_idx], y_raw.values[va_idx])
        m = cb.CatBoostRegressor(**TW18_PARAMS)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.values[va_idx]).mean()
        print(f'  [TW1.8] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred

def train_cb_oof(X_train, X_test, y_log, groups, feat_cols):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values; X_te_np = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        train_pool = cb.Pool(X_tr_np[tr_idx], y_log.values[tr_idx])
        val_pool   = cb.Pool(X_tr_np[va_idx], y_log.values[va_idx])
        m = cb.CatBoostRegressor(**CB_PARAMS)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [CB] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred

def train_et_oof(X_train, X_test, y_log, groups, feat_cols):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values; X_te_np = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = ExtraTreesRegressor(**ET_PARAMS)
        m.fit(X_tr_np[tr_idx], y_log.values[tr_idx])
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [ET] Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    return oof, test_pred

def train_rf_oof(X_train, X_test, y_log, groups, feat_cols):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values; X_te_np = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = RandomForestRegressor(**RF_PARAMS)
        m.fit(X_tr_np[tr_idx], y_log.values[tr_idx])
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [RF] Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    return oof, test_pred


# ─────────────────────────────────────────────
# ★ [Stage 2B] Base Learner — raw target (2종, v4 신규)
#   log1p 압축 없이 원본 타겟으로 학습
#   극값 구간(target≥50)에서 예측 범위 확장 기대
# ─────────────────────────────────────────────
def train_lgbm_raw_oof(X_train, X_test, y_raw, groups, feat_cols):
    """LGBM trained on RAW target (no log1p) — preserves extreme values"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0); X_te_np = X_test[feat_cols].fillna(0)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_raw, groups)):
        m = lgb.LGBMRegressor(**LGBM_RAW_PARAMS)
        m.fit(X_tr_np.iloc[tr_idx], y_raw.iloc[tr_idx],
              eval_set=[(X_tr_np.iloc[va_idx], y_raw.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(X_tr_np.iloc[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [LGBM-raw] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred

def train_cb_raw_oof(X_train, X_test, y_raw, groups, feat_cols):
    """CatBoost trained on RAW target — preserves extreme values"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values; X_te_np = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_raw, groups)):
        train_pool = cb.Pool(X_tr_np[tr_idx], y_raw.values[tr_idx])
        val_pool   = cb.Pool(X_tr_np[va_idx], y_raw.values[va_idx])
        m = cb.CatBoostRegressor(**CB_RAW_PARAMS)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.values[va_idx]).mean()
        print(f'  [CB-raw] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred


# ─────────────────────────────────────────────
# [Stage 3] 7모델 메타 스태킹
# ─────────────────────────────────────────────
def run_meta_lgbm(meta_train, meta_test, y_raw, groups, label='LGBM-meta'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_meta = np.zeros(len(y_raw)); test_meta = np.zeros(meta_test.shape[0])
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(meta_train[tr_idx], np.log1p(y_raw.iloc[tr_idx].values),
              eval_set=[(meta_train[va_idx], np.log1p(y_raw.iloc[va_idx].values))],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[va_idx] = np.expm1(m.predict(meta_train[va_idx]))
        test_meta += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    oof_mae = np.abs(oof_meta - y_raw.values).mean()
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | pred_std={oof_meta.std():.2f}')
    return oof_meta, test_meta, oof_mae


def main():
    t0 = time.time()
    print('=' * 60)
    print('v4.0: 2-Stage 극값 전략')
    print('기준: model30 CV 8.4838 / Public 9.8279 (배율 1.1584)')
    print('전략: 시나리오 분류 + raw-target 모델 + Tier 3 + 교차 집계')
    print('=' * 60)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    # ── 데이터 로드 + FE ──
    train, test, ratio_cols, cross_cols, range_cols = load_data()
    feat_cols_before_prob = get_feat_cols(train)
    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']

    print(f'\n피처 수 (분류기 전): {len(feat_cols_before_prob)}')
    print(f'  비율 피처: {len(ratio_cols)}종 (T1: 5 + T2: 7 + T3: 5 = 17)')
    print(f'  교차 집계: {len(cross_cols)}종')
    print(f'  range/IQR: {len(range_cols)}종')

    # 새 피처 shift 분석
    new_cols = [c for c in ratio_cols if 'total_stress' in c or 'no_idle' in c or
                'crisis' in c or 'sku_cong' in c or 'throughput' in c]
    new_cols += cross_cols + range_cols
    if new_cols:
        print(f'\n[v4 신규 피처 shift 분석]')
        for col in new_cols:
            if col in train.columns:
                tr_m = train[col].mean(); te_m = test[col].mean()
                tr_s = train[col].std()
                shift = abs(tr_m - te_m) / (tr_s + 1e-8)
                marker = '✅' if shift < 0.4 else '⚠️'
                print(f'  {col:50s}: shift={shift:.3f}σ {marker}')

    # ── Stage 1: 시나리오 분류기 ──
    train_eprob, test_eprob = build_scenario_classifier(train, test, y_raw)
    train['extreme_prob'] = train_eprob
    test['extreme_prob']  = test_eprob

    feat_cols = get_feat_cols(train)  # extreme_prob 포함
    print(f'\n피처 수 (분류기 후): {len(feat_cols)}')

    # ── Stage 2A: log1p Base Learner (5종) ──
    print('\n' + '─' * 60)
    print('[Stage 2A] log1p Base Learner OOF 생성 (5종)')
    print('─' * 60)

    if ckpt_exists('lgbm'):
        print('\n[LGBM] ckpt 로드'); oof_lg, test_lg = load_ckpt('lgbm')
    else:
        print('\n[LGBM] 학습...')
        oof_lg, test_lg = train_lgbm_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('lgbm', oof_lg, test_lg)
    print(f'  LGBM OOF MAE={np.abs(np.expm1(oof_lg) - y_raw.values).mean():.4f}')

    if ckpt_exists('tw18'):
        print('\n[TW1.8] ckpt 로드'); oof_tw, test_tw = load_ckpt('tw18')
    else:
        print('\n[TW1.8] 학습...')
        oof_tw, test_tw = train_tw18_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt('tw18', oof_tw, test_tw)
    print(f'  TW1.8 OOF MAE={np.abs(oof_tw - y_raw.values).mean():.4f}')

    if ckpt_exists('cb'):
        print('\n[CB] ckpt 로드'); oof_cb, test_cb = load_ckpt('cb')
    else:
        print('\n[CB] 학습...')
        oof_cb, test_cb = train_cb_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('cb', oof_cb, test_cb)
    print(f'  CB OOF MAE={np.abs(np.expm1(oof_cb) - y_raw.values).mean():.4f}')

    if ckpt_exists('et'):
        print('\n[ET] ckpt 로드'); oof_et, test_et = load_ckpt('et')
    else:
        print('\n[ET] 학습...')
        oof_et, test_et = train_et_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('et', oof_et, test_et)
    print(f'  ET OOF MAE={np.abs(np.expm1(oof_et) - y_raw.values).mean():.4f}')

    if ckpt_exists('rf'):
        print('\n[RF] ckpt 로드'); oof_rf, test_rf = load_ckpt('rf')
    else:
        print('\n[RF] 학습...')
        oof_rf, test_rf = train_rf_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('rf', oof_rf, test_rf)
    print(f'  RF OOF MAE={np.abs(np.expm1(oof_rf) - y_raw.values).mean():.4f}')

    # ── Stage 2B: raw-target Base Learner (2종) ──
    print('\n' + '─' * 60)
    print('[Stage 2B] raw-target Base Learner OOF 생성 (2종)')
    print('이 모델들은 log1p 압축 없이 원본 타겟으로 학습 → 극값 예측 강화')
    print('─' * 60)

    if ckpt_exists('lgbm_raw'):
        print('\n[LGBM-raw] ckpt 로드'); oof_lg_raw, test_lg_raw = load_ckpt('lgbm_raw')
    else:
        print('\n[LGBM-raw] 학습...')
        oof_lg_raw, test_lg_raw = train_lgbm_raw_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt('lgbm_raw', oof_lg_raw, test_lg_raw)
    print(f'  LGBM-raw OOF MAE={np.abs(oof_lg_raw - y_raw.values).mean():.4f}')

    if ckpt_exists('cb_raw'):
        print('\n[CB-raw] ckpt 로드'); oof_cb_raw, test_cb_raw = load_ckpt('cb_raw')
    else:
        print('\n[CB-raw] 학습...')
        oof_cb_raw, test_cb_raw = train_cb_raw_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt('cb_raw', oof_cb_raw, test_cb_raw)
    print(f'  CB-raw OOF MAE={np.abs(oof_cb_raw - y_raw.values).mean():.4f}')

    # ── 극값 구간 log1p vs raw 비교 ──
    print('\n' + '─' * 60)
    print('[비교] 극값 구간 log1p vs raw 모델')
    print('─' * 60)
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    print(f'  {"구간":>10s}  {"LGBM(log1p)":>12s}  {"LGBM(raw)":>12s}  {"CB(log1p)":>12s}  {"CB(raw)":>12s}  {"개선":>6s}')
    for lo, hi in bins:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            lg_log = np.abs(np.expm1(oof_lg[mask]) - y_raw.values[mask]).mean()
            lg_raw = np.abs(oof_lg_raw[mask] - y_raw.values[mask]).mean()
            cb_log = np.abs(np.expm1(oof_cb[mask]) - y_raw.values[mask]).mean()
            cb_raw = np.abs(oof_cb_raw[mask] - y_raw.values[mask]).mean()
            # raw가 극값에서 더 나은지?
            improved = '✅' if (lg_raw < lg_log and hi > 50) else ''
            print(f'  [{lo:3d},{hi:3d})  {lg_log:12.2f}  {lg_raw:12.2f}  {cb_log:12.2f}  {cb_raw:12.2f}  {improved}')

    # raw 모델의 극값 예측 비율
    for name, oof_pred in [('LGBM-log1p', np.expm1(oof_lg)), ('LGBM-raw', oof_lg_raw),
                            ('CB-log1p', np.expm1(oof_cb)), ('CB-raw', oof_cb_raw)]:
        mask = y_raw.values >= 80
        pred_ratio = oof_pred[mask].mean() / y_raw.values[mask].mean()
        print(f'  {name:12s} [80+] pred/actual={pred_ratio:.3f}')

    # ── 상관관계 분석 (7모델) ──
    print('\n' + '─' * 60)
    print('[다양성] 7모델 OOF 상관관계')
    print('─' * 60)
    oof_raw_dict = {
        'LGBM': np.expm1(oof_lg), 'TW': oof_tw, 'CB': np.expm1(oof_cb),
        'ET': np.expm1(oof_et), 'RF': np.expm1(oof_rf),
        'LG-r': oof_lg_raw, 'CB-r': oof_cb_raw,
    }
    names = list(oof_raw_dict.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = np.corrcoef(oof_raw_dict[names[i]], oof_raw_dict[names[j]])[0,1]
            marker = ' ★' if c < 0.95 else ''
            print(f'  {names[i]:4s}-{names[j]:4s}: {c:.4f}{marker}')

    # ── 가중 앙상블 (7모델) ──
    arrs = [oof_raw_dict[n] for n in names]
    def loss_n(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        return np.mean(np.abs(sum(w[i]*arrs[i] for i in range(len(arrs))) - y_raw.values))
    best_loss, best_w = np.inf, np.ones(len(arrs))/len(arrs)
    for _ in range(500):
        w0 = np.random.dirichlet(np.ones(len(arrs)))
        res = minimize(loss_n, w0, method='Nelder-Mead')
        if res.fun < best_loss:
            best_loss = res.fun; best_w = np.abs(res.x) / np.abs(res.x).sum()
    print(f'\n  7모델 가중 앙상블 CV MAE: {best_loss:.4f}')
    for i, n in enumerate(names):
        print(f'    {n}={best_w[i]:.3f}', end='')
    print()

    # ── Stage 3: 7모델 메타 스태킹 ──
    print('\n' + '─' * 60)
    print('[Stage 3] 7모델 LGBM 메타 학습기')
    print('─' * 60)

    # 메타 입력 구성: 모든 OOF를 log1p 공간으로 통일
    test_tw_clipped = np.maximum(test_tw, 0)
    meta_train = np.column_stack([
        oof_lg,                                     # log1p space
        oof_cb,                                     # log1p space
        np.log1p(np.maximum(oof_tw, 0)),           # raw → log1p
        oof_et,                                     # log1p space
        oof_rf,                                     # log1p space
        np.log1p(np.maximum(oof_lg_raw, 0)),       # raw → log1p (★ 신규)
        np.log1p(np.maximum(oof_cb_raw, 0)),       # raw → log1p (★ 신규)
    ])
    meta_test = np.column_stack([
        test_lg,                                    # log1p space
        test_cb,                                    # log1p space
        np.log1p(test_tw_clipped),                 # raw → log1p
        test_et,                                    # log1p space
        test_rf,                                    # log1p space
        np.log1p(np.maximum(test_lg_raw, 0)),      # raw → log1p (★ 신규)
        np.log1p(np.maximum(test_cb_raw, 0)),      # raw → log1p (★ 신규)
    ])

    print(f'  메타 입력: {meta_train.shape[1]}종 (5 log1p + 2 raw-log1p)')

    oof_meta, test_meta, mae_meta = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    # 5모델 비교를 위해 5모델 메타도 실행
    print('\n[비교] 5모델 메타 (model30 동일 구조)')
    meta_train_5 = np.column_stack([
        oof_lg, oof_cb, np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf
    ])
    meta_test_5 = np.column_stack([
        test_lg, test_cb, np.log1p(test_tw_clipped), test_et, test_rf
    ])
    oof_meta_5, test_meta_5, mae_meta_5 = run_meta_lgbm(
        meta_train_5, meta_test_5, y_raw, groups, label='5model-meta')

    print(f'\n  7모델 메타 MAE: {mae_meta:.4f}')
    print(f'  5모델 메타 MAE: {mae_meta_5:.4f}')
    print(f'  raw 모델 기여 : {mae_meta_5 - mae_meta:+.4f}')

    # 제출 파일 (7모델 기준)
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    sub_path = os.path.join(SUB_DIR, 'v4_extreme_2stage.csv')
    sample.to_csv(sub_path, index=False)
    print(f'\n제출 파일: {sub_path}')

    # 5모델도 별도 저장 (비교용)
    sample_5 = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample_5['avg_delay_minutes_next_30m'] = np.maximum(test_meta_5, 0)
    sub_path_5 = os.path.join(SUB_DIR, 'v4_5model_newfe.csv')
    sample_5.to_csv(sub_path_5, index=False)
    print(f'비교용 5모델 제출 파일: {sub_path_5}')

    # ── 분석 ──
    print('\n' + '─' * 60)
    print('[분석] 타겟 구간별 MAE (7모델 vs model30)')
    print('─' * 60)
    print(f'  {"구간":>10s}  {"v4(7모델)":>10s}  {"v4(5모델)":>10s}  {"model30참고":>12s}')
    # model30 구간별 MAE 참고값 (출력에서 읽음)
    m30_ref = {(0,5): 2.73, (5,10): 2.84, (10,20): 7.30, (20,30): 9.18,
               (30,50): 8.06, (50,80): 27.78, (80,800): 92.89}
    for lo, hi in bins:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            seg_7 = np.abs(oof_meta[mask] - y_raw.values[mask]).mean()
            seg_5 = np.abs(oof_meta_5[mask] - y_raw.values[mask]).mean()
            ref = m30_ref.get((lo, hi), 0)
            delta = seg_7 - ref if ref > 0 else 0
            marker = '✅' if delta < 0 else ('⚠️' if delta > 0 else '')
            print(f'  [{lo:3d},{hi:3d})  {seg_7:10.2f}  {seg_5:10.2f}  {ref:12.2f}  {delta:+.2f} {marker}')

    print('\n' + '─' * 60)
    print('[분석] 예측 분포')
    print('─' * 60)
    print(f'  OOF:  mean={oof_meta.mean():.2f}, std={oof_meta.std():.2f}, max={oof_meta.max():.2f}')
    print(f'  test: mean={test_meta.mean():.2f}, std={test_meta.std():.2f}, max={test_meta.max():.2f}')
    print(f'  (model30: test mean=19.47, std=15.83, max=101.65)')

    # 최종 요약
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 60)
    print(f'v4.0 결과 ({elapsed:.1f}분 소요)')
    print('=' * 60)
    print(f'  피처 수       : {len(feat_cols)} (model30: 422)')
    print(f'  7모델 메타    : {mae_meta:.4f}  pred_std={oof_meta.std():.2f}')
    print(f'  5모델 메타    : {mae_meta_5:.4f}  (FE 효과만)')
    print(f'  test pred     : mean={test_meta.mean():.2f}, std={test_meta.std():.2f}')
    print(f'  model30 기준  : CV 8.4838 / Public 9.8279 (배율 1.1584)')
    print(f'  v4 변화       : {mae_meta - 8.4838:+.4f} (7모델 vs model30)')
    print(f'  기대 Public (×1.158): {mae_meta * 1.158:.4f}')
    print(f'  기대 Public (×1.170): {mae_meta * 1.170:.4f}')

    # 핵심 질문: raw 모델이 극값에서 실제로 개선했는가?
    mask_ext = y_raw.values >= 50
    mae_ext_7 = np.abs(oof_meta[mask_ext] - y_raw.values[mask_ext]).mean()
    mae_ext_5 = np.abs(oof_meta_5[mask_ext] - y_raw.values[mask_ext]).mean()
    print(f'\n  [극값≥50] 7모델={mae_ext_7:.2f}, 5모델={mae_ext_5:.2f}, Δ={mae_ext_7-mae_ext_5:+.2f}')
    if mae_ext_7 < mae_ext_5:
        print(f'  ✅ raw 모델이 극값 구간 개선에 기여!')
    else:
        print(f'  ⚠️ raw 모델의 극값 기여 미미 — 배율 확인 후 제출 판단')

    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
