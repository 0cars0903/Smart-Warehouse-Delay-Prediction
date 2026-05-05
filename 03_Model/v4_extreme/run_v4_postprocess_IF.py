"""
v4.1A: Isolation Forest 기반 극값 탐지 + 2D 보정 테이블 후처리
================================================================
전략:
  1. model30 파이프라인 재현 (422피처, 5모델 스태킹 → LGBM-meta)
     - OOF 예측 + test 예측 생성
  2. 시나리오 레벨 피처로 IsolationForest 학습 → anomaly_score 산출
     - train/test 모두 동일 피처로 score 계산 (리크 없음)
     - anomaly_score ∈ [-1, 1]: 높을수록 이상(극값 가능성 높음)
  3. OOF에서 2D 보정 테이블 구축
     - (prediction_bin, anomaly_score_bin) → correction_factor
     - correction_factor = mean(actual / predicted) per cell
  4. test 예측에 보정 테이블 적용
     - 안전장치: correction_factor [0.8, 3.0] 클리핑
     - anomaly_score 낮은 구간은 보정 최소화 (정상 예측 보호)

기준: model30 CV 8.4838 / Public 9.8279 / 배율 1.1584
목표: [80,800) MAE 개선 → 전체 MAE 하락

실행: python src/run_v4_postprocess_IF.py
예상 시간: ~35분 (5모델 스태킹 + IF + 보정)
출력: submissions/v4_postprocess_IF.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, IsolationForest
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
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model30_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# model30 파라미터 (동일)
# ─────────────────────────────────────────────
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

META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# 시나리오 집계 대상 피처 (model22: 11통계)
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

# Isolation Forest에 사용할 시나리오 레벨 피처
# 극값 시나리오 top 구분자 (axis3 분석 결과)
IF_SCENARIO_COLS = [
    'sc_order_inflow_15m_mean', 'sc_order_inflow_15m_std', 'sc_order_inflow_15m_max',
    'sc_robot_idle_mean', 'sc_robot_idle_std',
    'sc_low_battery_ratio_mean', 'sc_low_battery_ratio_max',
    'sc_sku_concentration_mean', 'sc_sku_concentration_std',
    'sc_congestion_score_mean', 'sc_congestion_score_max',
    'sc_robot_utilization_mean', 'sc_robot_utilization_std',
    'sc_charge_queue_length_mean', 'sc_charge_queue_length_max',
    'sc_max_zone_density_mean', 'sc_max_zone_density_max',
    'sc_battery_mean_mean', 'sc_battery_std_mean',
]


# ─────────────────────────────────────────────
# 체크포인트
# ─────────────────────────────────────────────
def save_ckpt(name, oof, test_pred):
    os.makedirs(CKPT_DIR, exist_ok=True)
    np.save(os.path.join(CKPT_DIR, f'{name}_oof.npy'), oof)
    np.save(os.path.join(CKPT_DIR, f'{name}_test.npy'), test_pred)

def load_ckpt(name):
    return (np.load(os.path.join(CKPT_DIR, f'{name}_oof.npy')),
            np.load(os.path.join(CKPT_DIR, f'{name}_test.npy')))

def ckpt_exists(name):
    return (os.path.exists(os.path.join(CKPT_DIR, f'{name}_oof.npy')) and
            os.path.exists(os.path.join(CKPT_DIR, f'{name}_test.npy')))


# ─────────────────────────────────────────────
# 시나리오 집계 + 비율 피처 (model30 동일)
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
            df['sc_battery_mean_mean'] * df['robot_total'],
            df['robot_total'])
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


def load_data():
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(train, test, layout,
                                  lag_lags=[1,2,3,4,5,6],
                                  rolling_windows=[3,5,10])
    train = add_scenario_agg_features(train)
    test  = add_scenario_agg_features(test)
    train = add_layout_ratio_features_tier1(train)
    test  = add_layout_ratio_features_tier1(test)
    train = add_layout_ratio_features_tier2(train)
    test  = add_layout_ratio_features_tier2(test)
    return train, test


def get_feat_cols(train):
    return [c for c in train.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and train[c].dtype != object]


# ─────────────────────────────────────────────
# Base Learner 학습 (model30 동일)
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


# ─────────────────────────────────────────────
# [핵심] Isolation Forest 기반 이상치 점수 산출
# ─────────────────────────────────────────────
def compute_isolation_forest_scores(train, test):
    """
    시나리오 레벨 피처로 IsolationForest 학습 → anomaly_score 산출.

    핵심 아이디어:
    - 극값 시나리오는 피처 공간에서 '고립'되기 쉬움
      (order_inflow 152 vs 일반 68, low_battery 0.28 vs 0.03 등)
    - IF는 unsupervised → train/test 모두 동일 기준 적용 가능
    - score_samples()가 반환하는 anomaly score가 높을수록 정상,
      낮을수록 이상치 → 부호 반전하여 '극값 가능성' 지표로 활용

    Returns:
        train_scores, test_scores: anomaly_score (높을수록 극값 가능성 높음)
    """
    print('\n[Isolation Forest] 시나리오 레벨 이상치 점수 산출')

    # 사용 가능한 피처만 선택
    available_cols = [c for c in IF_SCENARIO_COLS if c in train.columns]
    print(f'  IF 입력 피처: {len(available_cols)}종')

    # 시나리오 레벨로 집계 (행 단위 → 시나리오 단위)
    # 같은 시나리오의 모든 행은 sc_ 피처가 동일하므로 첫 행만 사용
    train_sc = train.groupby('scenario_id')[available_cols].first().reset_index()
    test_sc  = test.groupby('scenario_id')[available_cols].first().reset_index()

    print(f'  train 시나리오: {len(train_sc)}, test 시나리오: {len(test_sc)}')

    # train + test 전체로 IF 학습 (unsupervised이므로 리크 없음)
    all_sc = pd.concat([train_sc[available_cols], test_sc[available_cols]], axis=0)
    all_sc = all_sc.fillna(0)

    # IsolationForest 학습
    # contamination: 극값 비율 추정 (~8-13%)
    # n_estimators: 트리 수 (충분히 많이)
    iforest = IsolationForest(
        n_estimators=300,
        contamination=0.10,  # 극값 시나리오 비율 ~10%
        max_samples='auto',
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iforest.fit(all_sc.values)

    # anomaly score 산출
    # score_samples: 낮을수록 이상치 → 부호 반전
    train_sc_scores = -iforest.score_samples(train_sc[available_cols].fillna(0).values)
    test_sc_scores  = -iforest.score_samples(test_sc[available_cols].fillna(0).values)

    # 시나리오 레벨 → 행 레벨로 broadcast
    train_sc_map = dict(zip(train_sc['scenario_id'], train_sc_scores))
    test_sc_map  = dict(zip(test_sc['scenario_id'], test_sc_scores))

    train_scores = train['scenario_id'].map(train_sc_map).values
    test_scores  = test['scenario_id'].map(test_sc_map).values

    # score 통계
    print(f'  train anomaly_score: mean={train_scores.mean():.4f}, '
          f'std={train_scores.std():.4f}, min={train_scores.min():.4f}, max={train_scores.max():.4f}')
    print(f'  test  anomaly_score: mean={test_scores.mean():.4f}, '
          f'std={test_scores.std():.4f}, min={test_scores.min():.4f}, max={test_scores.max():.4f}')

    # 극값과의 상관 분석 (train만)
    if 'avg_delay_minutes_next_30m' in train.columns:
        y = train['avg_delay_minutes_next_30m'].values
        is_extreme = (y >= 50).astype(float)
        corr_target = np.corrcoef(train_scores, y)[0,1]
        corr_extreme = np.corrcoef(train_scores, is_extreme)[0,1]
        print(f'  anomaly_score vs target: corr={corr_target:.4f}')
        print(f'  anomaly_score vs extreme(≥50): corr={corr_extreme:.4f}')

        # 극값 시나리오에서의 score 분포
        extreme_mask = y >= 50
        normal_mask = y < 50
        print(f'  정상 구간 score: mean={train_scores[normal_mask].mean():.4f}')
        print(f'  극값 구간 score: mean={train_scores[extreme_mask].mean():.4f}')

    return train_scores, test_scores


# ─────────────────────────────────────────────
# [핵심] 2D 보정 테이블 구축 + 적용
# ─────────────────────────────────────────────
def build_2d_calibration_table(oof_pred, y_actual, anomaly_scores,
                                pred_bins=None, score_bins=None):
    """
    OOF 예측에서 2D 보정 테이블 구축.

    축1: prediction_bin (예측값 구간)
    축2: anomaly_score_bin (이상치 점수 구간)
    셀값: correction_factor = mean(actual / predicted)

    핵심 논리:
    - 정상 구간(low score): correction ≈ 1.0 (보정 불필요)
    - 극값 구간(high score) + 높은 예측: correction > 1.0 (상향 보정)
    - 극값 구간(high score) + 낮은 예측: correction ≈ 1.0 (트리가 잘 맞춘 영역)
    """
    print('\n[2D 보정 테이블] 구축')

    # 기본 bin 설정
    if pred_bins is None:
        pred_bins = [0, 5, 10, 15, 20, 30, 50, 80, 200]
    if score_bins is None:
        # anomaly score percentile 기반 bin
        score_pcts = np.percentile(anomaly_scores, [0, 25, 50, 75, 90, 95, 100])
        score_bins = np.unique(score_pcts)

    pred_bin_idx = np.digitize(oof_pred, pred_bins) - 1
    pred_bin_idx = np.clip(pred_bin_idx, 0, len(pred_bins) - 2)

    score_bin_idx = np.digitize(anomaly_scores, score_bins) - 1
    score_bin_idx = np.clip(score_bin_idx, 0, len(score_bins) - 2)

    # 보정 테이블 구축
    table = {}
    print(f'  pred_bins: {[f"{b:.0f}" for b in pred_bins]}')
    print(f'  score_bins: {[f"{b:.3f}" for b in score_bins]}')
    print()

    header = f'  {"pred_bin":>12s} | {"score_bin":>12s} | {"n":>6s} | {"mean_pred":>10s} | {"mean_actual":>10s} | {"correction":>10s} | {"clipped":>10s}'
    print(header)
    print('  ' + '-' * len(header))

    for pi in range(len(pred_bins) - 1):
        for si in range(len(score_bins) - 1):
            mask = (pred_bin_idx == pi) & (score_bin_idx == si)
            n = mask.sum()
            if n < 5:  # 최소 샘플 수
                table[(pi, si)] = 1.0  # 데이터 부족 → 보정 안함
                continue

            mean_pred = oof_pred[mask].mean()
            mean_actual = y_actual[mask].mean()

            # correction_factor: actual / predicted (predicted가 0에 가까우면 보호)
            if mean_pred > 1.0:
                raw_correction = mean_actual / mean_pred
            else:
                raw_correction = 1.0

            # 안전 클리핑
            clipped = np.clip(raw_correction, 0.8, 3.0)
            table[(pi, si)] = clipped

            # score 하위 50% (정상 구간)은 추가 보호: correction → 1.0에 가깝게
            median_score_bin = (len(score_bins) - 1) // 2
            if si < median_score_bin:
                # 정상 구간은 보정 폭 축소 (1.0 방향으로 shrink)
                shrink_factor = 0.3  # 정상 구간은 보정의 30%만 적용
                table[(pi, si)] = 1.0 + (clipped - 1.0) * shrink_factor

            flag = '⬆' if clipped > 1.05 else ('⬇' if clipped < 0.95 else '  ')
            print(f'  [{pred_bins[pi]:3.0f},{pred_bins[pi+1]:3.0f}) | '
                  f'[{score_bins[si]:.3f},{score_bins[si+1]:.3f}) | '
                  f'{n:6d} | {mean_pred:10.2f} | {mean_actual:10.2f} | '
                  f'{raw_correction:10.4f} | {table[(pi,si)]:10.4f} {flag}')

    return table, pred_bins, score_bins


def apply_2d_calibration(predictions, anomaly_scores, table, pred_bins, score_bins):
    """
    2D 보정 테이블을 예측에 적용.
    """
    pred_bin_idx = np.digitize(predictions, pred_bins) - 1
    pred_bin_idx = np.clip(pred_bin_idx, 0, len(pred_bins) - 2)

    score_bin_idx = np.digitize(anomaly_scores, score_bins) - 1
    score_bin_idx = np.clip(score_bin_idx, 0, len(score_bins) - 2)

    corrected = predictions.copy()
    for i in range(len(predictions)):
        key = (pred_bin_idx[i], score_bin_idx[i])
        factor = table.get(key, 1.0)
        corrected[i] = predictions[i] * factor

    return np.maximum(corrected, 0)


# ─────────────────────────────────────────────
# 구간별 분석 유틸
# ─────────────────────────────────────────────
def segment_analysis(pred, actual, label=''):
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    total_mae = np.abs(pred - actual).mean()
    total_weighted = 0
    print(f'\n[구간 분석] {label} (전체 MAE={total_mae:.4f})')
    for lo, hi in bins:
        mask = (actual >= lo) & (actual < hi)
        if mask.sum() == 0:
            continue
        seg_mae = np.abs(pred[mask] - actual[mask]).mean()
        contribution = seg_mae * mask.sum() / len(actual)
        pct = mask.sum() / len(actual) * 100
        pred_ratio = pred[mask].mean() / (actual[mask].mean() + 1e-8)
        print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d} ({pct:5.1f}%) '
              f'MAE={seg_mae:7.2f}  contrib={contribution:5.3f}  '
              f'pred/actual={pred_ratio:.3f}')
        total_weighted += contribution
    print(f'  합계 가중 MAE: {total_weighted:.4f}')
    return total_mae


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 70)
    print('v4.1A: Isolation Forest 극값 탐지 + 2D 보정 테이블 후처리')
    print('기준: model30 CV 8.4838 / Public 9.8279 / 배율 1.1584')
    print('=' * 70)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    # ── Step 1: 데이터 로드 + model30 재현 ──
    print('\n[Step 1] 데이터 로드 + model30 파이프라인 재현')
    train, test = load_data()
    feat_cols = get_feat_cols(train)
    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'  피처 수: {len(feat_cols)}')

    # ── Step 2: 5모델 Base Learner OOF ──
    print('\n' + '─' * 70)
    print('[Step 2] Base Learner OOF 생성 (model30 동일)')
    print('─' * 70)

    # LGBM
    if ckpt_exists('lgbm'):
        print('\n[LGBM] 체크포인트 로드'); oof_lg, test_lg = load_ckpt('lgbm')
    else:
        print('\n[LGBM] 학습 시작...')
        oof_lg, test_lg = train_lgbm_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('lgbm', oof_lg, test_lg)
    print(f'  LGBM OOF MAE={np.abs(np.expm1(oof_lg) - y_raw.values).mean():.4f}')

    # TW1.8
    if ckpt_exists('tw18'):
        print('\n[TW1.8] 체크포인트 로드'); oof_tw, test_tw = load_ckpt('tw18')
    else:
        print('\n[TW1.8] 학습 시작...')
        oof_tw, test_tw = train_tw18_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt('tw18', oof_tw, test_tw)
    print(f'  TW1.8 OOF MAE={np.abs(oof_tw - y_raw.values).mean():.4f}')

    # CB
    if ckpt_exists('cb'):
        print('\n[CB] 체크포인트 로드'); oof_cb, test_cb = load_ckpt('cb')
    else:
        print('\n[CB] 학습 시작...')
        oof_cb, test_cb = train_cb_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('cb', oof_cb, test_cb)
    print(f'  CB OOF MAE={np.abs(np.expm1(oof_cb) - y_raw.values).mean():.4f}')

    # ET
    if ckpt_exists('et'):
        print('\n[ET] 체크포인트 로드'); oof_et, test_et = load_ckpt('et')
    else:
        print('\n[ET] 학습 시작...')
        oof_et, test_et = train_et_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('et', oof_et, test_et)
    print(f'  ET OOF MAE={np.abs(np.expm1(oof_et) - y_raw.values).mean():.4f}')

    # RF
    if ckpt_exists('rf'):
        print('\n[RF] 체크포인트 로드'); oof_rf, test_rf = load_ckpt('rf')
    else:
        print('\n[RF] 학습 시작...')
        oof_rf, test_rf = train_rf_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('rf', oof_rf, test_rf)
    print(f'  RF OOF MAE={np.abs(np.expm1(oof_rf) - y_raw.values).mean():.4f}')

    # ── Step 3: 메타 스태킹 ──
    print('\n' + '─' * 70)
    print('[Step 3] 5모델 LGBM 메타 스태킹')
    print('─' * 70)

    test_tw_clipped = np.maximum(test_tw, 0)
    meta_train = np.column_stack([oof_lg, oof_cb,
                                   np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf])
    meta_test  = np.column_stack([test_lg, test_cb,
                                   np.log1p(test_tw_clipped), test_et, test_rf])

    oof_meta, test_meta, mae_baseline = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    # 기준선 분석
    segment_analysis(oof_meta, y_raw.values, label='model30 기준선 (후처리 전)')

    # ── Step 4: Isolation Forest 이상치 점수 ──
    print('\n' + '─' * 70)
    print('[Step 4] Isolation Forest 이상치 점수 산출')
    print('─' * 70)

    train_if_scores, test_if_scores = compute_isolation_forest_scores(train, test)

    # ── Step 5: 2D 보정 테이블 구축 (OOF 기반) ──
    print('\n' + '─' * 70)
    print('[Step 5] 2D 보정 테이블 구축')
    print('─' * 70)

    # 예측값 bin: 극값 영역을 세분화
    pred_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]

    # anomaly score bin: percentile 기반
    score_pcts = [0, 30, 50, 70, 85, 92, 97, 100]
    score_bins = np.percentile(train_if_scores, score_pcts)
    score_bins = np.unique(np.round(score_bins, 6))  # 중복 제거

    table, p_bins, s_bins = build_2d_calibration_table(
        oof_meta, y_raw.values, train_if_scores,
        pred_bins=pred_bins, score_bins=score_bins
    )

    # ── Step 6: OOF 보정 적용 + 평가 ──
    print('\n' + '─' * 70)
    print('[Step 6] OOF 보정 적용 + 평가')
    print('─' * 70)

    oof_corrected = apply_2d_calibration(oof_meta, train_if_scores, table, p_bins, s_bins)
    mae_corrected = np.abs(oof_corrected - y_raw.values).mean()

    print(f'\n  기준선 MAE: {mae_baseline:.4f}')
    print(f'  보정 후 MAE: {mae_corrected:.4f}')
    print(f'  변화:       {mae_corrected - mae_baseline:+.4f}')

    segment_analysis(oof_corrected, y_raw.values, label='IF 2D 보정 후')

    # ── Step 7: 보정 강도 탐색 ──
    # 보정 테이블의 correction factor에 α (0~1) 블렌딩
    print('\n' + '─' * 70)
    print('[Step 7] 보정 강도 (α) 최적화')
    print('─' * 70)

    best_alpha, best_mae_alpha = 1.0, mae_corrected
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blended = oof_meta * (1 - alpha) + oof_corrected * alpha
        blended = np.maximum(blended, 0)
        mae_a = np.abs(blended - y_raw.values).mean()
        marker = ' ✅ best' if mae_a < best_mae_alpha else ''
        print(f'  α={alpha:.1f}: MAE={mae_a:.4f}{marker}')
        if mae_a < best_mae_alpha:
            best_mae_alpha = mae_a
            best_alpha = alpha

    print(f'\n  최적 α={best_alpha:.1f}, MAE={best_mae_alpha:.4f} (기준 {mae_baseline:.4f})')

    # 최적 α 적용
    if best_alpha > 0:
        oof_final = oof_meta * (1 - best_alpha) + oof_corrected * best_alpha
    else:
        oof_final = oof_meta.copy()
    oof_final = np.maximum(oof_final, 0)

    segment_analysis(oof_final, y_raw.values, label=f'최적 α={best_alpha:.1f} 적용')

    # ── Step 8: test 예측 보정 + 제출 ──
    print('\n' + '─' * 70)
    print('[Step 8] test 예측 보정 + 제출')
    print('─' * 70)

    test_corrected = apply_2d_calibration(test_meta, test_if_scores, table, p_bins, s_bins)

    if best_alpha > 0 and best_alpha < 1.0:
        test_final = test_meta * (1 - best_alpha) + test_corrected * best_alpha
    elif best_alpha >= 1.0:
        test_final = test_corrected
    else:
        test_final = test_meta.copy()
    test_final = np.maximum(test_final, 0)

    print(f'  기준 test: mean={test_meta.mean():.2f}, std={test_meta.std():.2f}, max={test_meta.max():.2f}')
    print(f'  보정 test: mean={test_final.mean():.2f}, std={test_final.std():.2f}, max={test_final.max():.2f}')

    # 제출 파일 생성
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    # 원본 model30 (보정 없음)
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    sub_base = os.path.join(SUB_DIR, 'v4_IF_baseline.csv')
    sample.to_csv(sub_base, index=False)

    # IF 보정 적용
    sample['avg_delay_minutes_next_30m'] = test_final
    sub_if = os.path.join(SUB_DIR, 'v4_postprocess_IF.csv')
    sample.to_csv(sub_if, index=False)

    print(f'\n  기준선 제출: {sub_base}')
    print(f'  IF 보정 제출: {sub_if}')

    # ── Step 9: train-test 분포 비교 ──
    print('\n' + '─' * 70)
    print('[Step 9] train-test 분포 비교')
    print('─' * 70)

    print(f'  IF score shift: train={train_if_scores.mean():.4f}, test={test_if_scores.mean():.4f}')
    print(f'  IF score shift (σ): {abs(train_if_scores.mean() - test_if_scores.mean()) / (train_if_scores.std() + 1e-8):.4f}')

    # 예측 분포
    for name, pred in [('기준 (model30)', test_meta), ('IF 보정', test_final)]:
        print(f'\n  [{name}]')
        print(f'    mean={pred.mean():.2f}, std={pred.std():.2f}, '
              f'max={pred.max():.2f}, min={pred.min():.2f}')
        for pct in [50, 75, 90, 95, 99]:
            print(f'    p{pct}={np.percentile(pred, pct):.2f}', end='')
        print()

    # ── 최종 요약 ──
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 70)
    print(f'v4.1A 결과 ({elapsed:.1f}분 소요)')
    print('=' * 70)
    print(f'  model30 기준선 CV:   {mae_baseline:.4f}')
    print(f'  IF 2D 보정 CV:       {mae_corrected:.4f} (Δ={mae_corrected - mae_baseline:+.4f})')
    print(f'  최적 α={best_alpha:.1f} CV:      {best_mae_alpha:.4f} (Δ={best_mae_alpha - mae_baseline:+.4f})')
    print(f'  test pred std:       {test_final.std():.2f} (기준 {test_meta.std():.2f})')

    ratio_est = best_mae_alpha * 1.158 if best_mae_alpha < mae_baseline else mae_baseline * 1.158
    print(f'  기대 Public (×1.158): {ratio_est:.4f}')

    if best_mae_alpha < mae_baseline:
        print(f'\n  ✅ IF 후처리 유효! CV Δ={best_mae_alpha - mae_baseline:+.4f}')
        print(f'  → v4_postprocess_IF.csv 제출 추천')
    else:
        print(f'\n  ⚠️ IF 후처리 무효 — 기준선 유지')
        print(f'  → 보정이 OOF 기준 개선 없음. 배율 변화 가능성은 있으나 리스크')

    print(f'\n{"=" * 70}')


if __name__ == '__main__':
    main()
