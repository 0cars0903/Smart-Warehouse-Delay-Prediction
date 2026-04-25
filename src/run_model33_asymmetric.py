"""
model33: Asymmetric Loss 기반 base learner 추가
================================================================
Notion 분석 문서 핵심 근거:
  - MAE = 조건부 중앙값 회귀 → 극단값 체계적 under-prediction
  - 잔차 정규화/후처리 = zero-sum (model28B, v4.1A/BC 모두 실패)
  - 해결책: "손실함수 자체의 재설계" (Koenker 2005, Newey & Powell 1987)

전략:
  model31 파이프라인 (429피처, 5모델 스태킹) 유지 +
  2종의 신규 base learner 추가:

  [1] Asymmetric MAE (LGBM custom objective)
      - under-prediction 페널티 α > 1.0 (위쪽 오차에 더 큰 가중)
      - 극값 구간에서 예측을 위로 밀어올림
      - α=1.5: under-prediction에 1.5배 페널티

  [2] Expectile Regression (LGBM custom objective)
      - τ=0.7: 상위 30% tail에 집중
      - MSE 기반이지만 비대칭 → 극값 쪽으로 예측 편향
      - Tweedie와 다른 오차 패턴 → 다양성 기대

  메타 스태킹: 7모델 (기존 5 + Asym + Expectile) → LGBM-meta

핵심 가설:
  - 기존 5모델은 모두 symmetric loss → 극값 under-prediction 공유
  - asymmetric loss 모델은 이질적 오차 패턴 → 메타에서 극값 구간 보정
  - model27(CNN+LSTM)과 달리 동일 피처/트리 → test 분포 붕괴 위험 낮음

기준: model31 CV 8.4786 / Public 9.8255 / 배율 1.1589
목표: 극값 구간 개선 + 배율 유지

실행: python src/run_model33_asymmetric.py
예상 시간: ~40분 (7모델 × 5fold)
출력: submissions/model33_asymmetric.csv
체크포인트: docs/model33_ckpt/
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import warnings, gc, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features, get_feature_cols

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_31  = os.path.join(_BASE, '..', 'docs', 'model31_ckpt')
CKPT_33  = os.path.join(_BASE, '..', 'docs', 'model33_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# ── Asymmetric Loss 하이퍼파라미터 ──
ASYM_ALPHA = 1.5   # under-prediction 페널티 배율 (>1 → 위쪽 오차에 가중)
EXPECTILE_TAU = 0.7  # expectile 파라미터 (>0.5 → 상위 tail 강조)

# ── model31 파라미터 ──
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

# Asymmetric LGBM: 더 강한 정규화 (신규 모델이므로 과적합 방지)
ASYM_LGBM_PARAMS = {
    'num_leaves': 127, 'learning_rate': 0.015,
    'feature_fraction': 0.50, 'bagging_fraction': 0.90,
    'min_child_samples': 35, 'reg_alpha': 2.0, 'reg_lambda': 1.0,
    'n_estimators': 3000, 'bagging_freq': 1,
    'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
}

# Expectile LGBM
EXPECTILE_LGBM_PARAMS = {
    'num_leaves': 127, 'learning_rate': 0.015,
    'feature_fraction': 0.50, 'bagging_fraction': 0.90,
    'min_child_samples': 35, 'reg_alpha': 2.0, 'reg_lambda': 1.0,
    'n_estimators': 3000, 'bagging_freq': 1,
    'random_state': RANDOM_STATE + 1, 'verbosity': -1, 'n_jobs': -1,
}

SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]


# ─────────────────────────────────────────────
# Custom Loss Functions (LightGBM)
# ─────────────────────────────────────────────
def asymmetric_mae_objective(y_pred, dtrain):
    """
    Asymmetric MAE: under-prediction에 α배 더 큰 페널티.

    residual = y_true - y_pred
    loss = α * |residual| if residual > 0  (under-prediction)
           1 * |residual| if residual ≤ 0  (over-prediction)

    gradient: -α * sign(residual) if residual > 0
               1 * sign(residual) if residual ≤ 0
    → -α if y_pred < y_true (under)
    →  1 if y_pred > y_true (over)

    hessian: constant (MAE의 2차 도함수는 0이지만, LGBM은 hessian > 0 필요)
    """
    y_true = dtrain.get_label()
    residual = y_true - y_pred  # positive = under-prediction

    alpha = ASYM_ALPHA
    grad = np.where(residual > 0, -alpha, 1.0)  # under → -α, over → +1
    hess = np.ones_like(y_pred)  # constant hessian
    return grad, hess


def asymmetric_mae_metric(y_pred, dtrain):
    """평가 지표: 일반 MAE (expm1 공간)"""
    y_true = dtrain.get_label()
    mae = np.abs(np.expm1(y_pred) - np.expm1(y_true)).mean()
    return 'asym_mae', mae, False


def expectile_objective(y_pred, dtrain):
    """
    Expectile Regression (Newey & Powell, 1987):

    loss = τ * (y - ŷ)²     if y ≥ ŷ  (under-prediction)
         = (1-τ) * (y - ŷ)²  if y < ŷ  (over-prediction)

    τ > 0.5 → 상위 tail에 집중 → 예측이 위로 이동
    τ = 0.5 → 일반 MSE (mean regression)
    τ = 0.7 → 상위 30% tail 강조

    gradient: -2τ(y-ŷ) if y ≥ ŷ, -2(1-τ)(y-ŷ) if y < ŷ
    hessian:   2τ      if y ≥ ŷ,  2(1-τ)       if y < ŷ
    """
    y_true = dtrain.get_label()
    residual = y_true - y_pred
    tau = EXPECTILE_TAU

    weight = np.where(residual >= 0, tau, 1 - tau)
    grad = -2 * weight * residual
    hess = 2 * weight
    return grad, hess


def expectile_metric(y_pred, dtrain):
    """평가 지표: 일반 MAE (expm1 공간)"""
    y_true = dtrain.get_label()
    mae = np.abs(np.expm1(y_pred) - np.expm1(y_true)).mean()
    return 'expectile_mae', mae, False


# ─────────────────────────────────────────────
# 체크포인트
# ─────────────────────────────────────────────
def save_ckpt(ckpt_dir, name, oof, test_pred):
    os.makedirs(ckpt_dir, exist_ok=True)
    np.save(os.path.join(ckpt_dir, f'{name}_oof.npy'), oof)
    np.save(os.path.join(ckpt_dir, f'{name}_test.npy'), test_pred)

def load_ckpt(ckpt_dir, name):
    return (np.load(os.path.join(ckpt_dir, f'{name}_oof.npy')),
            np.load(os.path.join(ckpt_dir, f'{name}_test.npy')))

def ckpt_exists(ckpt_dir, name):
    return (os.path.exists(os.path.join(ckpt_dir, f'{name}_oof.npy')) and
            os.path.exists(os.path.join(ckpt_dir, f'{name}_test.npy')))


# ─────────────────────────────────────────────
# FE (model31 동일)
# ─────────────────────────────────────────────
def add_scenario_agg_features(df):
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns: continue
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
        df[f'sc_{col}_cv'] = (df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)
    return df

def safe_div(a, b, fill=0):
    return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

def add_ratio_tier1(df):
    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = safe_div(df['sc_order_inflow_15m_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = safe_div(df['sc_congestion_score_mean'], df['intersection_count'])
    if 'sc_low_battery_ratio_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_stress'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df.get('sc_charge_queue_length_mean', 0), df['charger_count'])
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = safe_div(df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']
    return df

def add_ratio_tier2(df):
    if all(c in df.columns for c in ['sc_congestion_score_mean', 'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'], df['robot_total'] ** 2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm'] / 100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(df['pack_station_count'], df['floor_area_sqm'] / 1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if 'sc_battery_mean_mean' in df.columns:
        if 'sc_robot_utilization_mean' in df.columns and 'charger_count' in df.columns:
            df['ratio_battery_per_robot'] = safe_div(
                df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'], df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    return df

def add_ratio_tier3_selected(df):
    cols = ['sc_order_inflow_15m_mean', 'sc_congestion_score_mean',
            'sc_low_battery_ratio_mean', 'robot_total', 'charger_count']
    if all(c in df.columns for c in cols):
        df['ratio_total_stress'] = safe_div(
            df['sc_order_inflow_15m_mean'] * df['sc_congestion_score_mean'] *
            (df['sc_low_battery_ratio_mean'] + 0.01), df['robot_total'] * df['charger_count'])
    cols2 = ['sc_sku_concentration_mean', 'sc_congestion_score_mean', 'intersection_count']
    if all(c in df.columns for c in cols2):
        df['ratio_sku_congestion'] = safe_div(
            df['sc_sku_concentration_mean'] * df['sc_congestion_score_mean'], df['intersection_count'])
    cols3 = ['sc_robot_idle_mean', 'robot_total', 'sc_order_inflow_15m_mean', 'floor_area_sqm']
    if all(c in df.columns for c in cols3):
        idle_frac = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
        df['ratio_no_idle_demand'] = safe_div(
            (1 - idle_frac) * df['sc_order_inflow_15m_mean'], df['floor_area_sqm'] / 100)
    cols4 = ['sc_low_battery_ratio_mean', 'sc_charge_queue_length_mean', 'charger_count']
    if all(c in df.columns for c in cols4):
        df['ratio_battery_crisis'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df['sc_charge_queue_length_mean'], df['charger_count'])
    return df

def add_cross_selected(df):
    safe_pairs = [
        ('congestion_score', 'low_battery_ratio'),
        ('sku_concentration', 'max_zone_density'),
        ('robot_utilization', 'charge_queue_length'),
    ]
    for col_a, col_b in safe_pairs:
        if col_a not in df.columns or col_b not in df.columns: continue
        interaction = df[col_a] * df[col_b]
        grp = interaction.groupby(df['scenario_id'])
        df[f'sc_cross_{col_a[:6]}_{col_b[:6]}_mean'] = grp.transform('mean')
    return df

def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(train, test, layout, lag_lags=[1,2,3,4,5,6], rolling_windows=[3,5,10])
    for fn in [add_scenario_agg_features, add_ratio_tier1, add_ratio_tier2,
               add_ratio_tier3_selected, add_cross_selected]:
        train = fn(train); test = fn(test)
    return train, test

def get_feat_cols(df):
    return [c for c in df.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and df[c].dtype != object]


# ─────────────────────────────────────────────
# Base Learner 학습 (기존 5종 — model31 체크포인트 재사용)
# ─────────────────────────────────────────────
def train_lgbm_oof(X_train, X_test, y_log, groups, feat_cols):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr = X_train[feat_cols].fillna(0); X_te = X_test[feat_cols].fillna(0)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr.iloc[tr_idx], y_log.iloc[tr_idx],
              eval_set=[(X_tr.iloc[va_idx], y_log.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.iloc[va_idx].values)).mean()
        print(f'  [LGBM] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred

def train_tw18_oof(X_train, X_test, y_raw, groups, feat_cols):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr = X_train[feat_cols].fillna(0).values; X_te = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_raw, groups)):
        m = cb.CatBoostRegressor(**TW18_PARAMS)
        m.fit(cb.Pool(X_tr[tr_idx], y_raw.values[tr_idx]),
              eval_set=cb.Pool(X_tr[va_idx], y_raw.values[va_idx]), use_best_model=True)
        oof[va_idx] = m.predict(X_tr[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.values[va_idx]).mean()
        print(f'  [TW1.8] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred

def train_cb_oof(X_train, X_test, y_log, groups, feat_cols):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr = X_train[feat_cols].fillna(0).values; X_te = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = cb.CatBoostRegressor(**CB_PARAMS)
        m.fit(cb.Pool(X_tr[tr_idx], y_log.values[tr_idx]),
              eval_set=cb.Pool(X_tr[va_idx], y_log.values[va_idx]), use_best_model=True)
        oof[va_idx] = m.predict(X_tr[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [CB] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred

def train_et_oof(X_train, X_test, y_log, groups, feat_cols):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr = X_train[feat_cols].fillna(0).values; X_te = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = ExtraTreesRegressor(**ET_PARAMS)
        m.fit(X_tr[tr_idx], y_log.values[tr_idx])
        oof[va_idx] = m.predict(X_tr[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [ET] Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    return oof, test_pred

def train_rf_oof(X_train, X_test, y_log, groups, feat_cols):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr = X_train[feat_cols].fillna(0).values; X_te = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = RandomForestRegressor(**RF_PARAMS)
        m.fit(X_tr[tr_idx], y_log.values[tr_idx])
        oof[va_idx] = m.predict(X_tr[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [RF] Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    return oof, test_pred


# ─────────────────────────────────────────────
# ★ 신규 Base Learner: Asymmetric MAE
# ─────────────────────────────────────────────
def train_asymmetric_oof(X_train, X_test, y_log, groups, feat_cols):
    """
    Asymmetric MAE LGBM: under-prediction에 α배 페널티.
    log1p 공간에서 학습 → OOF도 log1p 공간.
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr = X_train[feat_cols].fillna(0); X_te = X_test[feat_cols].fillna(0)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        dtrain = lgb.Dataset(X_tr.iloc[tr_idx], label=y_log.iloc[tr_idx].values)
        dval   = lgb.Dataset(X_tr.iloc[va_idx], label=y_log.iloc[va_idx].values, reference=dtrain)

        params = {k: v for k, v in ASYM_LGBM_PARAMS.items()
                  if k not in ['n_estimators']}
        params['objective'] = asymmetric_mae_objective  # LightGBM 4.x: callable in params

        bst = lgb.train(
            params, dtrain,
            num_boost_round=ASYM_LGBM_PARAMS['n_estimators'],
            valid_sets=[dval],
            feval=asymmetric_mae_metric,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )

        oof[va_idx] = bst.predict(X_tr.iloc[va_idx])
        test_pred += bst.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.iloc[va_idx].values)).mean()
        print(f'  [Asym-MAE(α={ASYM_ALPHA})] Fold {fold+1}  MAE={mae:.4f}  iter={bst.best_iteration}')
        del bst; gc.collect()
    return oof, test_pred


# ─────────────────────────────────────────────
# ★ 신규 Base Learner: Expectile Regression
# ─────────────────────────────────────────────
def train_expectile_oof(X_train, X_test, y_log, groups, feat_cols):
    """
    Expectile Regression LGBM: τ>0.5로 상위 tail 강조.
    log1p 공간에서 학습 → OOF도 log1p 공간.
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr = X_train[feat_cols].fillna(0); X_te = X_test[feat_cols].fillna(0)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        dtrain = lgb.Dataset(X_tr.iloc[tr_idx], label=y_log.iloc[tr_idx].values)
        dval   = lgb.Dataset(X_tr.iloc[va_idx], label=y_log.iloc[va_idx].values, reference=dtrain)

        params = {k: v for k, v in EXPECTILE_LGBM_PARAMS.items()
                  if k not in ['n_estimators']}
        params['objective'] = expectile_objective  # LightGBM 4.x: callable in params

        bst = lgb.train(
            params, dtrain,
            num_boost_round=EXPECTILE_LGBM_PARAMS['n_estimators'],
            valid_sets=[dval],
            feval=expectile_metric,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )

        oof[va_idx] = bst.predict(X_tr.iloc[va_idx])
        test_pred += bst.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.iloc[va_idx].values)).mean()
        print(f'  [Expectile(τ={EXPECTILE_TAU})] Fold {fold+1}  MAE={mae:.4f}  iter={bst.best_iteration}')
        del bst; gc.collect()
    return oof, test_pred


# ─────────────────────────────────────────────
# 메타 학습기
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


def segment_analysis(pred, actual, label=''):
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    total = np.abs(pred - actual).mean()
    print(f'\n[구간 분석] {label} (전체 MAE={total:.4f})')
    for lo, hi in bins:
        mask = (actual >= lo) & (actual < hi)
        if mask.sum() == 0: continue
        seg = np.abs(pred[mask] - actual[mask]).mean()
        pct = mask.sum() / len(actual) * 100
        pr = pred[mask].mean() / (actual[mask].mean() + 1e-8)
        print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d} ({pct:5.1f}%) MAE={seg:7.2f} pred/actual={pr:.3f}')
    return total


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 70)
    print('model33: Asymmetric Loss + Expectile 기반 base learner 추가')
    print('기준: model31 CV 8.4786 / Public 9.8255 / 배율 1.1589')
    print('이론: MAE = 조건부 중앙값 → 극값 under-prediction 구조적 원인')
    print('      → 비대칭 손실로 극값 쪽 예측 상향')
    print('=' * 70)

    os.makedirs(CKPT_33, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    # ── 데이터 로드 ──
    print('\n[데이터] model31 피처 파이프라인 로드')
    train, test = load_data()
    feat_cols = get_feat_cols(train)
    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'  피처: {len(feat_cols)}')

    # ── 기존 5모델 (model31 체크포인트 재사용) ──
    print('\n' + '─' * 70)
    print('[Layer 1A] 기존 5모델 OOF (model31 체크포인트)')
    print('─' * 70)

    oof_dict = {}
    test_dict = {}

    for name, fn, y, ckpt_src in [
        ('lgbm', train_lgbm_oof, y_log, CKPT_31),
        ('cb',   train_cb_oof,   y_log, CKPT_31),
        ('et',   train_et_oof,   y_log, CKPT_31),
        ('rf',   train_rf_oof,   y_log, CKPT_31),
    ]:
        if ckpt_exists(ckpt_src, name):
            print(f'\n[{name.upper()}] 체크포인트 로드')
            oof_dict[name], test_dict[name] = load_ckpt(ckpt_src, name)
        else:
            print(f'\n[{name.upper()}] 학습 시작...')
            oof_dict[name], test_dict[name] = fn(train, test, y, groups, feat_cols)
            save_ckpt(CKPT_33, name, oof_dict[name], test_dict[name])
        if name == 'tw18':
            mae = np.abs(oof_dict[name] - y_raw.values).mean()
        else:
            mae = np.abs(np.expm1(oof_dict[name]) - y_raw.values).mean()
        print(f'  {name.upper()} OOF MAE: {mae:.4f}')

    # TW1.8 별도
    if ckpt_exists(CKPT_31, 'tw18'):
        print(f'\n[TW1.8] 체크포인트 로드')
        oof_dict['tw18'], test_dict['tw18'] = load_ckpt(CKPT_31, 'tw18')
    else:
        print(f'\n[TW1.8] 학습 시작...')
        oof_dict['tw18'], test_dict['tw18'] = train_tw18_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt(CKPT_33, 'tw18', oof_dict['tw18'], test_dict['tw18'])
    print(f'  TW1.8 OOF MAE: {np.abs(oof_dict["tw18"] - y_raw.values).mean():.4f}')

    # ── 신규 2모델 ──
    print('\n' + '─' * 70)
    print('[Layer 1B] 신규 Asymmetric Loss 모델 (2종)')
    print('─' * 70)

    # Asymmetric MAE
    if ckpt_exists(CKPT_33, 'asym'):
        print(f'\n[Asym-MAE] 체크포인트 로드')
        oof_dict['asym'], test_dict['asym'] = load_ckpt(CKPT_33, 'asym')
    else:
        print(f'\n[Asym-MAE(α={ASYM_ALPHA})] 학습 시작...')
        oof_dict['asym'], test_dict['asym'] = train_asymmetric_oof(train, test, y_log, groups, feat_cols)
        save_ckpt(CKPT_33, 'asym', oof_dict['asym'], test_dict['asym'])
    asym_mae = np.abs(np.expm1(oof_dict['asym']) - y_raw.values).mean()
    print(f'  Asym-MAE OOF MAE: {asym_mae:.4f}')

    # Expectile
    if ckpt_exists(CKPT_33, 'expectile'):
        print(f'\n[Expectile] 체크포인트 로드')
        oof_dict['expectile'], test_dict['expectile'] = load_ckpt(CKPT_33, 'expectile')
    else:
        print(f'\n[Expectile(τ={EXPECTILE_TAU})] 학습 시작...')
        oof_dict['expectile'], test_dict['expectile'] = train_expectile_oof(train, test, y_log, groups, feat_cols)
        save_ckpt(CKPT_33, 'expectile', oof_dict['expectile'], test_dict['expectile'])
    exp_mae = np.abs(np.expm1(oof_dict['expectile']) - y_raw.values).mean()
    print(f'  Expectile OOF MAE: {exp_mae:.4f}')

    # ── 다양성 분석 ──
    print('\n' + '─' * 70)
    print('[다양성] 7모델 OOF 상관관계')
    print('─' * 70)

    oof_raw = {
        'LGBM': np.expm1(oof_dict['lgbm']),
        'TW':   oof_dict['tw18'],
        'CB':   np.expm1(oof_dict['cb']),
        'ET':   np.expm1(oof_dict['et']),
        'RF':   np.expm1(oof_dict['rf']),
        'Asym': np.expm1(oof_dict['asym']),
        'Exp':  np.expm1(oof_dict['expectile']),
    }
    names = list(oof_raw.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = np.corrcoef(oof_raw[names[i]], oof_raw[names[j]])[0,1]
            marker = '✅' if c < 0.95 else ('⚠️' if c < 0.98 else '❌')
            print(f'  {names[i]:4s}-{names[j]:4s}: {c:.4f} {marker}')

    # 극값 구간별 pred/actual 비교 (핵심!)
    print('\n' + '─' * 70)
    print('[극값 분석] 모델별 [80,800) 예측/실제 비율')
    print('─' * 70)
    extreme_mask = y_raw.values >= 80
    if extreme_mask.sum() > 0:
        for name in names:
            pred_ext = oof_raw[name][extreme_mask]
            actual_ext = y_raw.values[extreme_mask]
            ratio = pred_ext.mean() / actual_ext.mean()
            mae_ext = np.abs(pred_ext - actual_ext).mean()
            print(f'  {name:5s}: pred/actual={ratio:.3f}, MAE={mae_ext:.2f}')

    # ── 5모델 메타 기준선 재현 ──
    print('\n' + '─' * 70)
    print('[Layer 2A] 5모델 메타 (model31 재현)')
    print('─' * 70)

    meta5_train = np.column_stack([
        oof_dict['lgbm'], oof_dict['cb'],
        np.log1p(np.maximum(oof_dict['tw18'], 0)),
        oof_dict['et'], oof_dict['rf']
    ])
    meta5_test = np.column_stack([
        test_dict['lgbm'], test_dict['cb'],
        np.log1p(np.maximum(test_dict['tw18'], 0)),
        test_dict['et'], test_dict['rf']
    ])
    oof5, test5, mae5 = run_meta_lgbm(meta5_train, meta5_test, y_raw, groups, '5모델-meta')
    segment_analysis(oof5, y_raw.values, '5모델 기준선 (model31 재현)')

    # ── 7모델 메타 ──
    print('\n' + '─' * 70)
    print('[Layer 2B] 7모델 메타 (5 + Asym + Expectile)')
    print('─' * 70)

    meta7_train = np.column_stack([
        oof_dict['lgbm'], oof_dict['cb'],
        np.log1p(np.maximum(oof_dict['tw18'], 0)),
        oof_dict['et'], oof_dict['rf'],
        oof_dict['asym'], oof_dict['expectile']
    ])
    meta7_test = np.column_stack([
        test_dict['lgbm'], test_dict['cb'],
        np.log1p(np.maximum(test_dict['tw18'], 0)),
        test_dict['et'], test_dict['rf'],
        test_dict['asym'], test_dict['expectile']
    ])
    oof7, test7, mae7 = run_meta_lgbm(meta7_train, meta7_test, y_raw, groups, '7모델-meta')
    segment_analysis(oof7, y_raw.values, '7모델 (5+Asym+Expectile)')

    # ── 6모델 변형들 (ablation) ──
    print('\n' + '─' * 70)
    print('[Layer 2C] 6모델 변형 (Ablation)')
    print('─' * 70)

    # 5+Asym
    meta6a_train = np.column_stack([
        oof_dict['lgbm'], oof_dict['cb'],
        np.log1p(np.maximum(oof_dict['tw18'], 0)),
        oof_dict['et'], oof_dict['rf'],
        oof_dict['asym']
    ])
    meta6a_test = np.column_stack([
        test_dict['lgbm'], test_dict['cb'],
        np.log1p(np.maximum(test_dict['tw18'], 0)),
        test_dict['et'], test_dict['rf'],
        test_dict['asym']
    ])
    oof6a, test6a, mae6a = run_meta_lgbm(meta6a_train, meta6a_test, y_raw, groups, '6모델(+Asym)')

    # 5+Expectile
    meta6b_train = np.column_stack([
        oof_dict['lgbm'], oof_dict['cb'],
        np.log1p(np.maximum(oof_dict['tw18'], 0)),
        oof_dict['et'], oof_dict['rf'],
        oof_dict['expectile']
    ])
    meta6b_test = np.column_stack([
        test_dict['lgbm'], test_dict['cb'],
        np.log1p(np.maximum(test_dict['tw18'], 0)),
        test_dict['et'], test_dict['rf'],
        test_dict['expectile']
    ])
    oof6b, test6b, mae6b = run_meta_lgbm(meta6b_train, meta6b_test, y_raw, groups, '6모델(+Exp)')

    # ── 제출 파일 생성 ──
    print('\n' + '─' * 70)
    print('[제출 파일]')
    print('─' * 70)

    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    configs = [
        ('model33_7model', test7, mae7),
        ('model33_6model_asym', test6a, mae6a),
        ('model33_6model_expectile', test6b, mae6b),
    ]

    for name, pred, cv in configs:
        pred_clipped = np.maximum(pred, 0)
        sample['avg_delay_minutes_next_30m'] = pred_clipped
        fpath = os.path.join(SUB_DIR, f'{name}.csv')
        sample.to_csv(fpath, index=False)
        print(f'  {name}: CV={cv:.4f}, test_std={pred_clipped.std():.2f}, '
              f'test_max={pred_clipped.max():.2f}')

    # ── 최종 비교 ──
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 70)
    print(f'model33 결과 ({elapsed:.1f}분 소요)')
    print('=' * 70)
    print(f'  Asymmetric MAE(α={ASYM_ALPHA}) OOF MAE: {asym_mae:.4f}')
    print(f'  Expectile(τ={EXPECTILE_TAU})    OOF MAE: {exp_mae:.4f}')
    print()
    print(f'  5모델 기준 (model31)   : CV={mae5:.4f}, test_std={np.maximum(test5,0).std():.2f}')
    print(f'  6모델 (+Asym only)     : CV={mae6a:.4f} (Δ={mae6a-mae5:+.4f})')
    print(f'  6모델 (+Expectile only): CV={mae6b:.4f} (Δ={mae6b-mae5:+.4f})')
    print(f'  7모델 (전체)           : CV={mae7:.4f} (Δ={mae7-mae5:+.4f})')
    print()

    # 최적 구성 판정
    results = [('5모델 (model31)', mae5, test5),
               ('6모델 (+Asym)', mae6a, test6a),
               ('6모델 (+Exp)', mae6b, test6b),
               ('7모델 (전체)', mae7, test7)]
    results.sort(key=lambda x: x[1])
    best_name, best_cv, best_test = results[0]

    print(f'  최적: {best_name} (CV={best_cv:.4f})')
    print(f'  기대 Public (×1.159): {best_cv * 1.159:.4f}')

    # test 분포 비교
    print('\n[test 분포 비교]')
    for name, _, pred in results:
        pc = np.maximum(pred, 0)
        print(f'  {name:25s}: mean={pc.mean():.2f}, std={pc.std():.2f}, max={pc.max():.2f}')

    # 판정
    if best_cv < mae5 - 0.001:
        print(f'\n  ✅ 비대칭 손실 모델 유효! CV Δ={best_cv - mae5:+.4f}')
        if np.maximum(best_test, 0).std() >= 15.5:
            print(f'  ✅ 분포 유지 → 제출 강력 추천')
        else:
            print(f'  ⚠️ 분포 압축 주의 → 제출하되 배율 확인')
    elif best_cv < mae5:
        print(f'\n  △ 미미한 개선 (Δ={best_cv - mae5:+.4f}). 제출 검토 가치 있음')
    else:
        print(f'\n  ⚠️ 개선 없음. 5모델(model31)이 여전히 최적')

    print(f'\n{"=" * 70}')


if __name__ == '__main__':
    main()
