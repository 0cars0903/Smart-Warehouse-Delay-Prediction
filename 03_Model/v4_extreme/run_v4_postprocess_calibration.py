"""
v4.1: 극값 후처리 + 피처 선별
=============================================================
v4.0 교훈: 트리 모델의 외삽 한계 확정.
           base learner 수준에서 극값 해결 불가.
           → "모델은 건드리지 않고, 출력만 보정"하는 후처리 전략.

전략:
  [A] 극값 후처리 (Calibration)
      - model30 OOF에서 예측값 vs 실제값 관계 분석
      - 예측값 구간별 보정 계수(scale factor) 계산
      - test 예측에 동일 보정 적용
      - 방식들: Isotonic, 구간별 선형 보정, extreme_prob 가중 스케일링

  [B] 피처 선별 (shift-safe Tier 3만)
      - v4 Tier 3 중 shift < 0.3σ만 사용 (range/IQR 제외)
      - model30 구조(5모델 스태킹) + 선별 피처 = model31

  [C] A+B 결합 (model31 + 후처리)

기준: model30 CV 8.4838 / Public 9.8279 / 배율 1.1584
핵심 수치: [80,800) MAE 92.89, 전체 MAE의 27.6% 차지

실행: python src/run_v4_postprocess_calibration.py
예상 시간: ~40분 (피처 선별 모델 학습) + 후처리 분석 (~2분)
출력: submissions/v4_postprocess_*.csv, submissions/model31_*.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
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
CKPT_30  = os.path.join(_BASE, '..', 'docs', 'model30_ckpt')
CKPT_31  = os.path.join(_BASE, '..', 'docs', 'model31_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# model30 파라미터 (Optuna 최적)
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
def save_ckpt(ckpt_dir, name, oof, test_pred):
    np.save(os.path.join(ckpt_dir, f'{name}_oof.npy'), oof)
    np.save(os.path.join(ckpt_dir, f'{name}_test.npy'), test_pred)

def load_ckpt(ckpt_dir, name):
    return (np.load(os.path.join(ckpt_dir, f'{name}_oof.npy')),
            np.load(os.path.join(ckpt_dir, f'{name}_test.npy')))

def ckpt_exists(ckpt_dir, name):
    return (os.path.exists(os.path.join(ckpt_dir, f'{name}_oof.npy')) and
            os.path.exists(os.path.join(ckpt_dir, f'{name}_test.npy')))


# ─────────────────────────────────────────────
# FE 함수들 (model30 동일)
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


def add_layout_ratio_features_tier2(df):
    def safe_div(a, b, fill=0):
        return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)
    if all(c in df.columns for c in ['sc_congestion_score_mean', 'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'], df['robot_total'] ** 2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm'] / 100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(df['pack_station_count'], df['floor_area_sqm'] / 1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if 'sc_battery_mean_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_battery_per_robot'] = safe_div(
            df['sc_battery_mean_mean'] * df['robot_total'], df['robot_total'])
        if 'sc_robot_utilization_mean' in df.columns and 'charger_count' in df.columns:
            df['ratio_battery_per_robot'] = safe_div(
                df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'], df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    return df


# ★ Tier 3 선별: shift < 0.3σ만 유지 (range/IQR 제외)
def add_layout_ratio_features_tier3_selected(df):
    """
    v4에서 shift 양호했던 Tier 3 비율 피처만 선별:
    - ratio_total_stress: 0.224σ ✅
    - ratio_sku_congestion: 0.007σ ✅ (최저)
    - ratio_no_idle_demand: 0.293σ ✅
    - ratio_battery_crisis: 0.294σ ✅
    제외: ratio_throughput_gap (0.316σ — 경계선, 안전하게 제외)
    """
    def safe_div(a, b, fill=0):
        return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

    # 종합 스트레스 지수 (shift 0.224σ ✅)
    cols_needed = ['sc_order_inflow_15m_mean', 'sc_congestion_score_mean',
                   'sc_low_battery_ratio_mean', 'robot_total', 'charger_count']
    if all(c in df.columns for c in cols_needed):
        df['ratio_total_stress'] = safe_div(
            df['sc_order_inflow_15m_mean'] *
            df['sc_congestion_score_mean'] *
            (df['sc_low_battery_ratio_mean'] + 0.01),
            df['robot_total'] * df['charger_count'])

    # SKU집중도 × 혼잡 (shift 0.007σ ✅ — 최저!)
    cols_needed = ['sc_sku_concentration_mean', 'sc_congestion_score_mean',
                   'intersection_count']
    if all(c in df.columns for c in cols_needed):
        df['ratio_sku_congestion'] = safe_div(
            df['sc_sku_concentration_mean'] * df['sc_congestion_score_mean'],
            df['intersection_count'])

    # 유휴 부족 위험 (shift 0.293σ ✅)
    cols_needed = ['sc_robot_idle_mean', 'robot_total',
                   'sc_order_inflow_15m_mean', 'floor_area_sqm']
    if all(c in df.columns for c in cols_needed):
        idle_frac = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
        df['ratio_no_idle_demand'] = safe_div(
            (1 - idle_frac) * df['sc_order_inflow_15m_mean'],
            df['floor_area_sqm'] / 100)

    # 배터리 위기 심도 (shift 0.294σ ✅)
    cols_needed = ['sc_low_battery_ratio_mean', 'sc_charge_queue_length_mean',
                   'charger_count']
    if all(c in df.columns for c in cols_needed):
        df['ratio_battery_crisis'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df['sc_charge_queue_length_mean'],
            df['charger_count'])

    return df


# ★ 교차 집계: shift 양호한 것만 선별
def add_scenario_cross_features_selected(df):
    """
    v4에서 shift 양호한 교차 집계만 선별 (range/IQR 완전 제외):
    - sc_cross_conges_low_ba_mean: 0.186σ ✅
    - sc_cross_sku_co_max_zo_mean: 0.104σ ✅
    - sc_cross_robot__charge_mean: 0.211σ ✅
    제외: order_inflow 관련 교차 (shift > 0.28σ)
    """
    safe_pairs = [
        ('congestion_score', 'low_battery_ratio'),     # 0.186σ
        ('sku_concentration', 'max_zone_density'),     # 0.104σ
        ('robot_utilization', 'charge_queue_length'),  # 0.211σ
    ]

    for col_a, col_b in safe_pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        interaction = df[col_a] * df[col_b]
        grp = interaction.groupby(df['scenario_id'])
        fname = f'sc_cross_{col_a[:6]}_{col_b[:6]}_mean'
        df[fname] = grp.transform('mean')

    return df


def load_data_model30():
    """model30 동일 피처 (422종)"""
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(train, test, layout, lag_lags=[1,2,3,4,5,6], rolling_windows=[3,5,10])
    train = add_scenario_agg_features(train)
    test  = add_scenario_agg_features(test)
    train = add_layout_ratio_features_tier1(train)
    test  = add_layout_ratio_features_tier1(test)
    train = add_layout_ratio_features_tier2(train)
    test  = add_layout_ratio_features_tier2(test)
    return train, test


def load_data_model31():
    """model30 + Tier 3 선별 + 교차 선별 (range/IQR 제외)"""
    train, test = load_data_model30()
    train = add_layout_ratio_features_tier3_selected(train)
    test  = add_layout_ratio_features_tier3_selected(test)
    train = add_scenario_cross_features_selected(train)
    test  = add_scenario_cross_features_selected(test)
    return train, test


def get_feat_cols(train):
    return [c for c in train.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and train[c].dtype != object]


# ─────────────────────────────────────────────
# Base Learner 학습 함수 (model30 동일)
# ─────────────────────────────────────────────
def train_lgbm_oof(X_train, X_test, y_log, groups, feat_cols, ckpt_dir):
    if ckpt_exists(ckpt_dir, 'lgbm'):
        print('  [LGBM] ckpt 로드'); return load_ckpt(ckpt_dir, 'lgbm')
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
    save_ckpt(ckpt_dir, 'lgbm', oof, test_pred)
    return oof, test_pred

def train_tw18_oof(X_train, X_test, y_raw, groups, feat_cols, ckpt_dir):
    if ckpt_exists(ckpt_dir, 'tw18'):
        print('  [TW1.8] ckpt 로드'); return load_ckpt(ckpt_dir, 'tw18')
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values; X_te_np = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_raw, groups)):
        m = cb.CatBoostRegressor(**TW18_PARAMS)
        m.fit(cb.Pool(X_tr_np[tr_idx], y_raw.values[tr_idx]),
              eval_set=cb.Pool(X_tr_np[va_idx], y_raw.values[va_idx]), use_best_model=True)
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.values[va_idx]).mean()
        print(f'  [TW1.8] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    save_ckpt(ckpt_dir, 'tw18', oof, test_pred)
    return oof, test_pred

def train_cb_oof(X_train, X_test, y_log, groups, feat_cols, ckpt_dir):
    if ckpt_exists(ckpt_dir, 'cb'):
        print('  [CB] ckpt 로드'); return load_ckpt(ckpt_dir, 'cb')
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values; X_te_np = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = cb.CatBoostRegressor(**CB_PARAMS)
        m.fit(cb.Pool(X_tr_np[tr_idx], y_log.values[tr_idx]),
              eval_set=cb.Pool(X_tr_np[va_idx], y_log.values[va_idx]), use_best_model=True)
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [CB] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    save_ckpt(ckpt_dir, 'cb', oof, test_pred)
    return oof, test_pred

def train_et_oof(X_train, X_test, y_log, groups, feat_cols, ckpt_dir):
    if ckpt_exists(ckpt_dir, 'et'):
        print('  [ET] ckpt 로드'); return load_ckpt(ckpt_dir, 'et')
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
    save_ckpt(ckpt_dir, 'et', oof, test_pred)
    return oof, test_pred

def train_rf_oof(X_train, X_test, y_log, groups, feat_cols, ckpt_dir):
    if ckpt_exists(ckpt_dir, 'rf'):
        print('  [RF] ckpt 로드'); return load_ckpt(ckpt_dir, 'rf')
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
    save_ckpt(ckpt_dir, 'rf', oof, test_pred)
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


def build_stacking(train, test, y_raw, y_log, groups, feat_cols, ckpt_dir, label=''):
    """5모델 스태킹 파이프라인 (model30 동일 구조)"""
    print(f'\n[{label}] 5모델 Base Learner 학습...')
    oof_lg, test_lg = train_lgbm_oof(train, test, y_log, groups, feat_cols, ckpt_dir)
    oof_tw, test_tw = train_tw18_oof(train, test, y_raw, groups, feat_cols, ckpt_dir)
    oof_cb, test_cb = train_cb_oof(train, test, y_log, groups, feat_cols, ckpt_dir)
    oof_et, test_et = train_et_oof(train, test, y_log, groups, feat_cols, ckpt_dir)
    oof_rf, test_rf = train_rf_oof(train, test, y_log, groups, feat_cols, ckpt_dir)

    # OOF MAE 보고
    for name, oof, is_raw in [('LGBM', oof_lg, False), ('TW', oof_tw, True),
                               ('CB', oof_cb, False), ('ET', oof_et, False), ('RF', oof_rf, False)]:
        if is_raw:
            mae = np.abs(oof - y_raw.values).mean()
        else:
            mae = np.abs(np.expm1(oof) - y_raw.values).mean()
        print(f'  {name} OOF MAE: {mae:.4f}')

    # 메타 입력
    meta_train = np.column_stack([
        oof_lg, oof_cb, np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf
    ])
    meta_test = np.column_stack([
        test_lg, test_cb, np.log1p(np.maximum(test_tw, 0)), test_et, test_rf
    ])

    # 메타 학습
    print(f'\n[{label}] 메타 학습기...')
    oof_meta, test_meta, mae_meta = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    return oof_meta, test_meta, mae_meta


# ─────────────────────────────────────────────
# ★ [A] 극값 후처리 함수들
# ─────────────────────────────────────────────
def analyze_calibration(oof_pred, y_actual):
    """OOF에서 예측값 구간별 보정 필요성 분석"""
    print('\n' + '─' * 60)
    print('[A] 극값 후처리: OOF 보정 분석')
    print('─' * 60)

    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    print(f'  {"예측구간":>10s}  {"n":>6s}  {"pred_mean":>10s}  {"actual_mean":>12s}  {"ratio":>6s}  {"MAE":>6s}')
    for lo, hi in bins:
        mask = (oof_pred >= lo) & (oof_pred < hi)
        if mask.sum() > 0:
            p_mean = oof_pred[mask].mean()
            a_mean = y_actual[mask].mean()
            ratio = p_mean / (a_mean + 1e-8)
            mae = np.abs(oof_pred[mask] - y_actual[mask]).mean()
            print(f'  [{lo:3d},{hi:3d})  {mask.sum():6d}  {p_mean:10.2f}  {a_mean:12.2f}  {ratio:6.3f}  {mae:6.2f}')


def postprocess_isotonic(oof_pred, y_actual, test_pred):
    """
    Isotonic Regression 보정:
    OOF에서 pred→actual 단조 증가 함수 학습, test에 적용
    """
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(oof_pred, y_actual)
    oof_calibrated = ir.predict(oof_pred)
    test_calibrated = ir.predict(test_pred)
    return oof_calibrated, test_calibrated


def postprocess_percentile_stretch(oof_pred, y_actual, test_pred, alpha=0.5):
    """
    Percentile-based stretch:
    예측값 상위 percentile을 실제 분포에 맞게 stretch
    alpha: 보정 강도 (0=없음, 1=완전 보정)
    """
    # 예측값의 percentile별 보정 계수 학습
    percentiles = np.arange(0, 101, 5)
    pred_pctls = np.percentile(oof_pred, percentiles)
    actual_pctls = np.percentile(y_actual, percentiles)

    # 상위 percentile에서 실제 분포가 더 넓으므로 stretch
    stretch_factors = actual_pctls / (pred_pctls + 1e-8)
    stretch_factors = np.clip(stretch_factors, 0.5, 5.0)  # 안전 범위

    # 각 예측값에 대해 해당 percentile의 stretch factor 적용
    from scipy.interpolate import interp1d
    stretch_fn = interp1d(pred_pctls, stretch_factors,
                          bounds_error=False, fill_value=(stretch_factors[0], stretch_factors[-1]))

    test_factors = stretch_fn(test_pred)
    test_calibrated = test_pred * (1 + alpha * (test_factors - 1))

    oof_factors = stretch_fn(oof_pred)
    oof_calibrated = oof_pred * (1 + alpha * (oof_factors - 1))

    return oof_calibrated, test_calibrated


def postprocess_extreme_scale(oof_pred, y_actual, test_pred, threshold=30, max_scale=2.0):
    """
    극값 구간 선택적 스케일링:
    예측값 > threshold인 경우에만 보정 계수 적용
    OOF에서 학습한 구간별 actual/pred 비율 사용
    """
    # 구간별 보정 계수 학습 (예측값 기준 구간)
    scale_bins = [(30, 40), (40, 50), (50, 70), (70, 100), (100, 500)]
    scales = {}
    for lo, hi in scale_bins:
        mask = (oof_pred >= lo) & (oof_pred < hi)
        if mask.sum() >= 50:  # 최소 50개 이상
            ratio = y_actual[mask].mean() / (oof_pred[mask].mean() + 1e-8)
            scales[(lo, hi)] = min(ratio, max_scale)
        else:
            scales[(lo, hi)] = 1.0

    print(f'\n  극값 스케일링 보정 계수:')
    for (lo, hi), s in scales.items():
        print(f'    [{lo},{hi}): scale={s:.3f}')

    # test에 적용
    test_calibrated = test_pred.copy()
    oof_calibrated = oof_pred.copy()
    for (lo, hi), s in scales.items():
        mask_test = (test_pred >= lo) & (test_pred < hi)
        test_calibrated[mask_test] = test_pred[mask_test] * s

        mask_oof = (oof_pred >= lo) & (oof_pred < hi)
        oof_calibrated[mask_oof] = oof_pred[mask_oof] * s

    return oof_calibrated, test_calibrated


def evaluate_postprocess(name, oof_calib, y_actual, test_calib, oof_orig, test_orig):
    """후처리 결과 평가"""
    mae_orig = np.abs(oof_orig - y_actual).mean()
    mae_calib = np.abs(oof_calib - y_actual).mean()

    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    print(f'\n  [{name}] OOF MAE: {mae_orig:.4f} → {mae_calib:.4f} (Δ{mae_calib-mae_orig:+.4f})')
    print(f'  test: mean={test_calib.mean():.2f}, std={test_calib.std():.2f}, max={test_calib.max():.2f}')

    # 구간별 변화
    improvements = []
    for lo, hi in bins:
        mask = (y_actual >= lo) & (y_actual < hi)
        if mask.sum() > 0:
            seg_orig = np.abs(oof_orig[mask] - y_actual[mask]).mean()
            seg_calib = np.abs(oof_calib[mask] - y_actual[mask]).mean()
            delta = seg_calib - seg_orig
            marker = '✅' if delta < 0 else '⚠️'
            improvements.append((lo, hi, seg_orig, seg_calib, delta, marker))
            print(f'    [{lo:3d},{hi:3d}): {seg_orig:.2f} → {seg_calib:.2f} ({delta:+.2f}) {marker}')

    return mae_calib


def main():
    t0 = time.time()
    print('=' * 60)
    print('v4.1: 극값 후처리 + 피처 선별')
    print('기준: model30 CV 8.4838 / Public 9.8279')
    print('=' * 60)

    os.makedirs(CKPT_31, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    # ════════════════════════════════════════════
    # Part 1: model30 OOF 로드 + 후처리 분석
    # ════════════════════════════════════════════
    print('\n' + '═' * 60)
    print('[Part 1] model30 기반 후처리')
    print('═' * 60)

    # model30 피처 로드 (OOF 재구축용)
    train_30, test_30 = load_data_model30()
    y_raw = train_30['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train_30['scenario_id']
    feat_cols_30 = get_feat_cols(train_30)
    print(f'model30 피처: {len(feat_cols_30)}')

    # model30 체크포인트에서 OOF 로드 + 메타 재실행
    if all(ckpt_exists(CKPT_30, n) for n in ['lgbm', 'tw18', 'cb', 'et', 'rf']):
        print('\n[model30 체크포인트 로드]')
        oof_lg, test_lg = load_ckpt(CKPT_30, 'lgbm')
        oof_tw, test_tw = load_ckpt(CKPT_30, 'tw18')
        oof_cb, test_cb = load_ckpt(CKPT_30, 'cb')
        oof_et, test_et = load_ckpt(CKPT_30, 'et')
        oof_rf, test_rf = load_ckpt(CKPT_30, 'rf')

        meta_train_30 = np.column_stack([
            oof_lg, oof_cb, np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf
        ])
        meta_test_30 = np.column_stack([
            test_lg, test_cb, np.log1p(np.maximum(test_tw, 0)), test_et, test_rf
        ])

        oof_meta_30, test_meta_30, mae_30 = run_meta_lgbm(
            meta_train_30, meta_test_30, y_raw, groups, label='model30-meta')

        print(f'\n  model30 재현 CV: {mae_30:.4f}')
    else:
        print('\n⚠️ model30 체크포인트 없음 — 5모델 재학습...')
        oof_meta_30, test_meta_30, mae_30 = build_stacking(
            train_30, test_30, y_raw, y_log, groups, feat_cols_30, CKPT_30, 'model30')

    # 보정 분석
    analyze_calibration(oof_meta_30, y_raw.values)

    # 후처리 A1: Isotonic Regression
    print('\n' + '─' * 60)
    print('[A1] Isotonic Regression 보정')
    print('─' * 60)
    oof_iso, test_iso = postprocess_isotonic(oof_meta_30, y_raw.values, test_meta_30)
    mae_iso = evaluate_postprocess('Isotonic', oof_iso, y_raw.values, test_iso,
                                    oof_meta_30, test_meta_30)

    # 후처리 A2: Percentile stretch
    print('\n' + '─' * 60)
    print('[A2] Percentile Stretch 보정')
    print('─' * 60)
    best_alpha_mae = np.inf; best_alpha = 0
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        oof_ps, _ = postprocess_percentile_stretch(oof_meta_30, y_raw.values, test_meta_30, alpha)
        mae_ps = np.abs(oof_ps - y_raw.values).mean()
        marker = ' ← best' if mae_ps < best_alpha_mae else ''
        print(f'  alpha={alpha:.1f}: OOF MAE={mae_ps:.4f}{marker}')
        if mae_ps < best_alpha_mae:
            best_alpha_mae = mae_ps; best_alpha = alpha

    oof_ps, test_ps = postprocess_percentile_stretch(
        oof_meta_30, y_raw.values, test_meta_30, best_alpha)
    mae_ps = evaluate_postprocess(f'PctlStretch(α={best_alpha})', oof_ps, y_raw.values,
                                   test_ps, oof_meta_30, test_meta_30)

    # 후처리 A3: Extreme Scale
    print('\n' + '─' * 60)
    print('[A3] Extreme Scale 보정 (예측≥30 구간)')
    print('─' * 60)
    oof_es, test_es = postprocess_extreme_scale(oof_meta_30, y_raw.values, test_meta_30)
    mae_es = evaluate_postprocess('ExtremeScale', oof_es, y_raw.values,
                                   test_es, oof_meta_30, test_meta_30)

    # 후처리 결과 비교
    print('\n' + '─' * 60)
    print('[후처리 결과 비교]')
    print('─' * 60)
    results_a = {
        'model30 원본': (mae_30, oof_meta_30, test_meta_30),
        'A1 Isotonic': (mae_iso, oof_iso, test_iso),
        f'A2 PctlStretch(α={best_alpha})': (mae_ps, oof_ps, test_ps),
        'A3 ExtremeScale': (mae_es, oof_es, test_es),
    }
    for name, (mae, oof, test) in results_a.items():
        print(f'  {name:30s}: CV={mae:.4f}  test_std={test.std():.2f}  test_max={test.max():.2f}')

    # 최적 후처리 제출 파일 저장
    best_name = min(results_a, key=lambda k: results_a[k][0])
    best_mae, _, best_test = results_a[best_name]
    if best_name != 'model30 원본':
        sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
        sample['avg_delay_minutes_next_30m'] = np.maximum(best_test, 0)
        sub_a = os.path.join(SUB_DIR, 'v4_postprocess_best.csv')
        sample.to_csv(sub_a, index=False)
        print(f'\n  최적 후처리 제출 파일: {sub_a} ({best_name})')

    # 모든 후처리 제출 파일 저장 (비교용)
    for name_key, fname in [('A1 Isotonic', 'v4_postprocess_isotonic.csv'),
                             (f'A2 PctlStretch(α={best_alpha})', 'v4_postprocess_stretch.csv'),
                             ('A3 ExtremeScale', 'v4_postprocess_extreme_scale.csv')]:
        if name_key in results_a:
            _, _, t_pred = results_a[name_key]
            sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
            sample['avg_delay_minutes_next_30m'] = np.maximum(t_pred, 0)
            sample.to_csv(os.path.join(SUB_DIR, fname), index=False)

    # ════════════════════════════════════════════
    # Part 2: model31 (피처 선별) + 후처리
    # ════════════════════════════════════════════
    print('\n' + '═' * 60)
    print('[Part 2] model31: model30 + shift-safe 피처 선별')
    print('═' * 60)

    train_31, test_31 = load_data_model31()
    feat_cols_31 = get_feat_cols(train_31)

    new_feats = sorted(set(feat_cols_31) - set(feat_cols_30))
    print(f'model31 피처: {len(feat_cols_31)} (model30: {len(feat_cols_30)}, 추가: {len(new_feats)})')
    for nf in new_feats:
        tr_m = train_31[nf].mean(); te_m = test_31[nf].mean()
        tr_s = train_31[nf].std()
        shift = abs(tr_m - te_m) / (tr_s + 1e-8)
        print(f'  {nf:45s}: shift={shift:.3f}σ ✅')

    oof_meta_31, test_meta_31, mae_31 = build_stacking(
        train_31, test_31, y_raw, y_log, groups, feat_cols_31, CKPT_31, 'model31')

    print(f'\n  model31 CV: {mae_31:.4f} (model30: {mae_30:.4f}, Δ={mae_31-mae_30:+.4f})')
    print(f'  model31 test: std={test_meta_31.std():.2f}, max={test_meta_31.max():.2f}')

    # model31 제출 파일
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta_31, 0)
    sub_31 = os.path.join(SUB_DIR, 'model31_selected_fe.csv')
    sample.to_csv(sub_31, index=False)
    print(f'  model31 제출 파일: {sub_31}')

    # model31 + 최적 후처리 결합
    if best_name != 'model30 원본':
        print(f'\n  model31 + 후처리 ({best_name.split(" ")[0]}) 적용...')
        if 'Isotonic' in best_name:
            oof_31_cal, test_31_cal = postprocess_isotonic(oof_meta_31, y_raw.values, test_meta_31)
        elif 'PctlStretch' in best_name:
            oof_31_cal, test_31_cal = postprocess_percentile_stretch(
                oof_meta_31, y_raw.values, test_meta_31, best_alpha)
        else:
            oof_31_cal, test_31_cal = postprocess_extreme_scale(oof_meta_31, y_raw.values, test_meta_31)

        mae_31_cal = np.abs(oof_31_cal - y_raw.values).mean()
        print(f'  model31+후처리 CV: {mae_31_cal:.4f}')

        sample_31c = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
        sample_31c['avg_delay_minutes_next_30m'] = np.maximum(test_31_cal, 0)
        sub_31c = os.path.join(SUB_DIR, 'model31_postprocess.csv')
        sample_31c.to_csv(sub_31c, index=False)

    # ════════════════════════════════════════════
    # 구간별 상세 비교
    # ════════════════════════════════════════════
    print('\n' + '═' * 60)
    print('[최종 비교] 구간별 MAE')
    print('═' * 60)
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    print(f'  {"구간":>10s}  {"model30":>8s}  {"model31":>8s}  {"30+PP":>8s}')
    for lo, hi in bins:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            m30 = np.abs(oof_meta_30[mask] - y_raw.values[mask]).mean()
            m31 = np.abs(oof_meta_31[mask] - y_raw.values[mask]).mean()
            if best_name != 'model30 원본':
                _, best_oof, _ = results_a[best_name]
                m30pp = np.abs(best_oof[mask] - y_raw.values[mask]).mean()
            else:
                m30pp = m30
            print(f'  [{lo:3d},{hi:3d})  {m30:8.2f}  {m31:8.2f}  {m30pp:8.2f}')

    # 최종 요약
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 60)
    print(f'v4.1 결과 ({elapsed:.1f}분 소요)')
    print('=' * 60)
    print(f'  model30 기준     : CV={mae_30:.4f}, test_std={test_meta_30.std():.2f}')
    print(f'  model31 (FE선별) : CV={mae_31:.4f}, test_std={test_meta_31.std():.2f}')
    print(f'  최적 후처리      : {best_name}, CV={best_mae:.4f}')
    print(f'  기대 Public (×1.158):')
    print(f'    model30       : {mae_30 * 1.158:.4f}')
    print(f'    model31       : {mae_31 * 1.158:.4f}')
    print(f'    최적 후처리   : {best_mae * 1.158:.4f}')

    print(f'\n제출 파일 목록:')
    print(f'  1. v4_postprocess_isotonic.csv')
    print(f'  2. v4_postprocess_stretch.csv')
    print(f'  3. v4_postprocess_extreme_scale.csv')
    print(f'  4. model31_selected_fe.csv')
    if best_name != 'model30 원본':
        print(f'  5. model31_postprocess.csv')
    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
