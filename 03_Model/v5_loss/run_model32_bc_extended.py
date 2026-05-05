"""
model32: model31 기반 확장 — B+C 후처리 + 추가 shift-safe FE
================================================================
3가지 출력:
  [A] model31 + B+C 분류기 후처리 → submissions/model31_BC.csv
  [B] model32: model31 + 추가 shift-safe FE → submissions/model32_extended_fe.csv
  [C] model32 + B+C 후처리 → submissions/model32_BC.csv

model31 기준: CV 8.4786 / Public 9.8255 / 배율 1.1589

B+C 후처리 설계 (IF 실패 교훈 반영):
  - IF(unsupervised): 극값 정렬 부정확 → 9.8458 ❌
  - B+C(supervised): 라벨 사용, AUC 0.897 → 극값 탐지 정확도 ↑
  - 정상 구간 보호 강화: prob < 0.20 → correction=1.0 (보정 안함)
  - 극값 구간만 선택적 보정: prob ≥ 0.50 and pred ≥ 20

추가 FE 설계 (shift-safe 원칙 유지):
  - Layout 정규화 비율: 기존 피처를 warehouse capacity로 정규화
  - 시나리오 내 분위수 차이: p90-p10 (이미 있는 통계 활용)
  - 안전 교차 확장: 기존 safe ratio 간 상호작용

실행: python src/run_model32_bc_extended.py
예상 시간: ~35분 (model31 ckpt 있으면 ~8분)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
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
CKPT_31  = os.path.join(_BASE, '..', 'docs', 'model31_ckpt')
CKPT_32  = os.path.join(_BASE, '..', 'docs', 'model32_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

EXTREME_THRESHOLD = 40

# ── model30/31 파라미터 (동일) ──
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
CLF_PARAMS = {
    'objective': 'binary', 'metric': 'auc',
    'num_leaves': 63, 'learning_rate': 0.03,
    'feature_fraction': 0.7, 'bagging_fraction': 0.8,
    'min_child_samples': 30, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'n_estimators': 1000, 'bagging_freq': 1,
    'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
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
# FE: 시나리오 집계 (11통계) + 비율 Tier 1/2/3
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
    if 'sc_battery_mean_mean' in df.columns and 'robot_total' in df.columns:
        if 'sc_robot_utilization_mean' in df.columns and 'charger_count' in df.columns:
            df['ratio_battery_per_robot'] = safe_div(
                df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'], df['charger_count'])
        else:
            df['ratio_battery_per_robot'] = safe_div(
                df['sc_battery_mean_mean'] * df['robot_total'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    return df

def add_ratio_tier3_selected(df):
    """model31 shift-safe Tier 3 (4종)"""
    cols = ['sc_order_inflow_15m_mean', 'sc_congestion_score_mean',
            'sc_low_battery_ratio_mean', 'robot_total', 'charger_count']
    if all(c in df.columns for c in cols):
        df['ratio_total_stress'] = safe_div(
            df['sc_order_inflow_15m_mean'] * df['sc_congestion_score_mean'] *
            (df['sc_low_battery_ratio_mean'] + 0.01),
            df['robot_total'] * df['charger_count'])
    cols = ['sc_sku_concentration_mean', 'sc_congestion_score_mean', 'intersection_count']
    if all(c in df.columns for c in cols):
        df['ratio_sku_congestion'] = safe_div(
            df['sc_sku_concentration_mean'] * df['sc_congestion_score_mean'], df['intersection_count'])
    cols = ['sc_robot_idle_mean', 'robot_total', 'sc_order_inflow_15m_mean', 'floor_area_sqm']
    if all(c in df.columns for c in cols):
        idle_frac = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
        df['ratio_no_idle_demand'] = safe_div(
            (1 - idle_frac) * df['sc_order_inflow_15m_mean'], df['floor_area_sqm'] / 100)
    cols = ['sc_low_battery_ratio_mean', 'sc_charge_queue_length_mean', 'charger_count']
    if all(c in df.columns for c in cols):
        df['ratio_battery_crisis'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df['sc_charge_queue_length_mean'], df['charger_count'])
    return df

def add_cross_selected(df):
    """model31 shift-safe 교차 집계 (3종)"""
    safe_pairs = [
        ('congestion_score', 'low_battery_ratio'),
        ('sku_concentration', 'max_zone_density'),
        ('robot_utilization', 'charge_queue_length'),
    ]
    for col_a, col_b in safe_pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        interaction = df[col_a] * df[col_b]
        grp = interaction.groupby(df['scenario_id'])
        df[f'sc_cross_{col_a[:6]}_{col_b[:6]}_mean'] = grp.transform('mean')
    return df


# ─────────────────────────────────────────────
# ★ model32 추가 FE: shift-safe 확장
# ─────────────────────────────────────────────
def add_model32_extended_features(df):
    """
    model32 추가 shift-safe 피처:

    1. 시나리오 IQR (p90-p10): 이미 있는 통계 조합 → shift 매우 낮을 것
       - 높은 IQR = 시나리오 내 변동성 큼 → 극값 가능성
    2. Layout 정규화 비율: 기존 safe 비율을 layout capacity로 2차 정규화
    3. 안전 교차 확장: 기존 safe ratio 간 상호작용
    4. 비율 피처 log 변환: 극값 피처 분포를 트리에 더 효과적으로 전달
    """
    new_cols = []

    # [1] 시나리오 IQR (p90 - p10) — 핵심 구분자 대상
    iqr_targets = [
        'order_inflow_15m', 'congestion_score', 'low_battery_ratio',
        'robot_utilization', 'sku_concentration',
    ]
    for col in iqr_targets:
        p90_col = f'sc_{col}_p90'
        p10_col = f'sc_{col}_p10'
        if p90_col in df.columns and p10_col in df.columns:
            fname = f'sc_{col}_iqr_p90p10'
            df[fname] = df[p90_col] - df[p10_col]
            new_cols.append(fname)

    # [2] Layout 정규화 2차 비율
    # robot_total 대비 극값 신호 정규화 (layout 규모 영향 제거)
    if 'ratio_total_stress' in df.columns and 'robot_total' in df.columns:
        fname = 'ratio_stress_per_robot'
        df[fname] = safe_div(df['ratio_total_stress'], df['robot_total'] / 10)
        new_cols.append(fname)

    if 'ratio_battery_crisis' in df.columns and 'robot_total' in df.columns:
        fname = 'ratio_crisis_per_robot'
        df[fname] = safe_div(df['ratio_battery_crisis'], df['robot_total'] / 10)
        new_cols.append(fname)

    # [3] 안전 교차: 기존 ratio 간 상호작용
    # demand_per_robot × congestion_per_intersection: 부하 복합 지표
    if 'ratio_demand_per_robot' in df.columns and 'ratio_congestion_per_intersection' in df.columns:
        fname = 'ratio_demand_x_congestion'
        df[fname] = df['ratio_demand_per_robot'] * df['ratio_congestion_per_intersection']
        new_cols.append(fname)

    # battery_stress × charge_competition: 충전 복합 위기
    if 'ratio_battery_stress' in df.columns and 'ratio_charge_competition' in df.columns:
        fname = 'ratio_battery_x_charge'
        df[fname] = df['ratio_battery_stress'] * df['ratio_charge_competition']
        new_cols.append(fname)

    # total_stress × idle_fraction_inv: 스트레스 상태에서 유휴 여유도
    if 'ratio_total_stress' in df.columns and 'ratio_idle_fraction' in df.columns:
        fname = 'ratio_stress_x_nonidle'
        df[fname] = df['ratio_total_stress'] * (1 - df['ratio_idle_fraction'].clip(0, 1))
        new_cols.append(fname)

    # [4] 극값 변별 복합 지표
    # congestion_max / congestion_mean: 피크 대비 평균 비율
    if 'sc_congestion_score_max' in df.columns and 'sc_congestion_score_mean' in df.columns:
        fname = 'sc_congestion_peak_ratio'
        df[fname] = safe_div(df['sc_congestion_score_max'], df['sc_congestion_score_mean'])
        new_cols.append(fname)

    # order_inflow max / mean: 수요 피크 대비 평균
    if 'sc_order_inflow_15m_max' in df.columns and 'sc_order_inflow_15m_mean' in df.columns:
        fname = 'sc_inflow_peak_ratio'
        df[fname] = safe_div(df['sc_order_inflow_15m_max'], df['sc_order_inflow_15m_mean'])
        new_cols.append(fname)

    # low_battery_ratio cv (변동계수): 배터리 불안정도
    if 'sc_low_battery_ratio_cv' in df.columns:
        # 이미 있으므로 pass
        pass

    print(f'  [model32 추가 FE] {len(new_cols)}종: {new_cols}')
    return df, new_cols


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
def load_data_model31():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(train, test, layout, lag_lags=[1,2,3,4,5,6], rolling_windows=[3,5,10])
    for fn in [add_scenario_agg_features, add_ratio_tier1, add_ratio_tier2,
               add_ratio_tier3_selected, add_cross_selected]:
        train = fn(train); test = fn(test)
    return train, test

def load_data_model32():
    train, test = load_data_model31()
    train, new_cols_tr = add_model32_extended_features(train)
    test, new_cols_te = add_model32_extended_features(test)
    return train, test, new_cols_tr

def get_feat_cols(df):
    return [c for c in df.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and df[c].dtype != object]


# ─────────────────────────────────────────────
# Base Learner 학습
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


def build_base_oof(train, test, y_log, y_raw, groups, feat_cols, ckpt_dir, label=''):
    """5모델 Base Learner OOF 생성 (체크포인트 지원)"""
    results = {}
    for name, fn, y in [
        ('lgbm', train_lgbm_oof, y_log),
        ('cb', train_cb_oof, y_log),
        ('et', train_et_oof, y_log),
        ('rf', train_rf_oof, y_log),
    ]:
        if ckpt_exists(ckpt_dir, name):
            print(f'\n[{name.upper()}] {label} 체크포인트 로드')
            results[name] = load_ckpt(ckpt_dir, name)
        else:
            print(f'\n[{name.upper()}] {label} 학습 시작...')
            oof, tpred = fn(train, test, y, groups, feat_cols)
            save_ckpt(ckpt_dir, name, oof, tpred)
            results[name] = (oof, tpred)

    # TW1.8은 raw target
    if ckpt_exists(ckpt_dir, 'tw18'):
        print(f'\n[TW1.8] {label} 체크포인트 로드')
        results['tw18'] = load_ckpt(ckpt_dir, 'tw18')
    else:
        print(f'\n[TW1.8] {label} 학습 시작...')
        oof, tpred = train_tw18_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt(ckpt_dir, 'tw18', oof, tpred)
        results['tw18'] = (oof, tpred)

    # MAE 출력
    for name in ['lgbm', 'tw18', 'cb', 'et', 'rf']:
        oof, _ = results[name]
        if name == 'tw18':
            mae = np.abs(oof - y_raw.values).mean()
        else:
            mae = np.abs(np.expm1(oof) - y_raw.values).mean()
        print(f'  {name.upper()} OOF MAE: {mae:.4f}')

    return results


def build_meta_inputs(results):
    """5모델 OOF → 메타 입력"""
    oof_lg, test_lg = results['lgbm']
    oof_cb, test_cb = results['cb']
    oof_tw, test_tw = results['tw18']
    oof_et, test_et = results['et']
    oof_rf, test_rf = results['rf']

    meta_train = np.column_stack([
        oof_lg, oof_cb, np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf])
    meta_test = np.column_stack([
        test_lg, test_cb, np.log1p(np.maximum(test_tw, 0)), test_et, test_rf])
    return meta_train, meta_test


# ─────────────────────────────────────────────
# B+C: 시나리오 분류기 + 2D 보정 (IF 교훈 반영)
# ─────────────────────────────────────────────
def compute_extreme_prob(train, test, feat_cols):
    """supervised 분류기로 extreme_prob 산출"""
    print('\n[시나리오 분류기] extreme_prob 산출')
    y_raw = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']

    sc_mean = train.groupby('scenario_id')['avg_delay_minutes_next_30m'].mean()
    sc_label = (sc_mean >= EXTREME_THRESHOLD).astype(int)
    row_label = train['scenario_id'].map(sc_label).values
    print(f'  극값 시나리오: {sc_label.sum()}/{len(sc_label)} ({sc_label.sum()/len(sc_label)*100:.1f}%)')

    clf_feat = [c for c in feat_cols if c.startswith('sc_') or c.startswith('ratio_')]
    print(f'  분류기 피처: {len(clf_feat)}종')

    X_tr = train[clf_feat].fillna(0); X_te = test[clf_feat].fillna(0)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_prob = np.zeros(len(train)); test_prob = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, row_label, groups)):
        m = lgb.LGBMClassifier(**CLF_PARAMS)
        m.fit(X_tr.iloc[tr_idx], row_label[tr_idx],
              eval_set=[(X_tr.iloc[va_idx], row_label[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_prob[va_idx] = m.predict_proba(X_tr.iloc[va_idx])[:, 1]
        test_prob += m.predict_proba(X_te)[:, 1] / N_SPLITS
        auc = roc_auc_score(row_label[va_idx], oof_prob[va_idx])
        print(f'  [CLF] Fold {fold+1}  AUC={auc:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_auc = roc_auc_score(row_label, oof_prob)
    print(f'  OOF AUC={oof_auc:.4f}')
    print(f'  train prob: mean={oof_prob.mean():.4f}, test prob: mean={test_prob.mean():.4f}')
    return oof_prob, test_prob


def build_bc_calibration(oof_pred, y_actual, extreme_prob):
    """
    B+C 2D 보정 테이블 — IF 실패 교훈 반영:
    - 정상 구간(prob < 0.20) 완전 보호 (correction=1.0)
    - 극값 구간(prob ≥ 0.50 and pred ≥ 20)만 선택적 보정
    - correction factor 범위 [0.90, 2.5] (보수적)
    """
    print('\n[B+C 보정 테이블] 구축 (정상 구간 보호 강화)')

    pred_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
    prob_bins = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.01]

    pred_bi = np.clip(np.digitize(oof_pred, pred_bins) - 1, 0, len(pred_bins) - 2)
    prob_bi = np.clip(np.digitize(extreme_prob, prob_bins) - 1, 0, len(prob_bins) - 2)

    table = {}
    print(f'  pred_bins: {pred_bins}')
    print(f'  prob_bins: {prob_bins}')
    print(f'\n  {"pred":>12s} | {"prob":>12s} | {"n":>5s} | {"pred_m":>7s} | {"actual_m":>8s} | {"correction":>10s}')
    print('  ' + '-' * 70)

    for pi in range(len(pred_bins) - 1):
        for pbi in range(len(prob_bins) - 1):
            mask = (pred_bi == pi) & (prob_bi == pbi)
            n = mask.sum()

            # 정상 구간: prob < 0.20 → 절대 보정 안함
            if prob_bins[pbi + 1] <= 0.20:
                table[(pi, pbi)] = 1.0
                if n >= 5:
                    mp = oof_pred[mask].mean(); ma = y_actual[mask].mean()
                    print(f'  [{pred_bins[pi]:3.0f},{pred_bins[pi+1]:3.0f}) | '
                          f'[{prob_bins[pbi]:.2f},{prob_bins[pbi+1]:.2f}) | '
                          f'{n:5d} | {mp:7.2f} | {ma:8.2f} | {"1.0000 (보호)":>10s}')
                continue

            if n < 10:
                table[(pi, pbi)] = 1.0
                continue

            mp = oof_pred[mask].mean()
            ma = y_actual[mask].mean()

            if mp > 1.0:
                raw = ma / mp
            else:
                raw = 1.0

            # 보수적 클리핑: [0.90, 2.5]
            clipped = np.clip(raw, 0.90, 2.5)

            # 중간 구간(0.20~0.50): 보정 50%만 적용
            if prob_bins[pbi + 1] <= 0.50:
                clipped = 1.0 + (clipped - 1.0) * 0.5

            table[(pi, pbi)] = clipped
            flag = '⬆' if clipped > 1.05 else ('⬇' if clipped < 0.95 else '  ')
            print(f'  [{pred_bins[pi]:3.0f},{pred_bins[pi+1]:3.0f}) | '
                  f'[{prob_bins[pbi]:.2f},{prob_bins[pbi+1]:.2f}) | '
                  f'{n:5d} | {mp:7.2f} | {ma:8.2f} | {clipped:10.4f} {flag}')

    return table, pred_bins, prob_bins


def apply_bc_calibration(predictions, extreme_prob, table, pred_bins, prob_bins):
    pred_bi = np.clip(np.digitize(predictions, pred_bins) - 1, 0, len(pred_bins) - 2)
    prob_bi = np.clip(np.digitize(extreme_prob, prob_bins) - 1, 0, len(prob_bins) - 2)
    corrected = predictions.copy()
    for i in range(len(predictions)):
        corrected[i] *= table.get((pred_bi[i], prob_bi[i]), 1.0)
    return np.maximum(corrected, 0)


# ─────────────────────────────────────────────
# 분석 유틸
# ─────────────────────────────────────────────
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


def shift_analysis(train, test, cols, label=''):
    print(f'\n[shift 분석] {label}')
    for col in cols:
        if col not in train.columns or col not in test.columns:
            continue
        tr_m, te_m, tr_s = train[col].mean(), test[col].mean(), train[col].std()
        shift = abs(tr_m - te_m) / (tr_s + 1e-8)
        marker = '✅' if shift < 0.3 else ('⚠️' if shift < 0.4 else '❌')
        print(f'  {col:45s}: shift={shift:.3f}σ {marker}')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 70)
    print('model32: model31 기반 확장 — B+C 후처리 + 추가 shift-safe FE')
    print('기준: model31 CV 8.4786 / Public 9.8255 / 배율 1.1589')
    print('=' * 70)

    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(CKPT_31, exist_ok=True)
    os.makedirs(CKPT_32, exist_ok=True)

    # ════════════════════════════════════════════
    # [Part A] model31 재현 + B+C 후처리
    # ════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('[Part A] model31 재현 + B+C 후처리')
    print('═' * 70)

    train31, test31 = load_data_model31()
    feat_cols_31 = get_feat_cols(train31)
    y_raw = train31['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train31['scenario_id']
    print(f'  model31 피처: {len(feat_cols_31)}')

    # Base Learner OOF
    results31 = build_base_oof(train31, test31, y_log, y_raw, groups,
                                feat_cols_31, CKPT_31, label='model31')

    # 메타 스태킹
    meta_tr31, meta_te31 = build_meta_inputs(results31)
    oof31, test31_pred, mae31 = run_meta_lgbm(meta_tr31, meta_te31, y_raw, groups, 'model31-meta')
    segment_analysis(oof31, y_raw.values, 'model31 기준선')

    # B+C 분류기
    prob_tr31, prob_te31 = compute_extreme_prob(train31, test31, feat_cols_31)

    # 2D 보정 테이블
    table31, pb31, prb31 = build_bc_calibration(oof31, y_raw.values, prob_tr31)

    # OOF 보정
    oof31_bc = apply_bc_calibration(oof31, prob_tr31, table31, pb31, prb31)
    mae31_bc = np.abs(oof31_bc - y_raw.values).mean()
    print(f'\n  model31 기준 MAE: {mae31:.4f}')
    print(f'  model31+BC MAE:  {mae31_bc:.4f} (Δ={mae31_bc - mae31:+.4f})')
    segment_analysis(oof31_bc, y_raw.values, 'model31 + B+C 보정')

    # α 블렌딩 탐색
    print('\n[α 블렌딩 탐색]')
    best_a31, best_mae31 = 0.0, mae31
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        bl = oof31 * (1 - alpha) + oof31_bc * alpha
        m = np.abs(np.maximum(bl, 0) - y_raw.values).mean()
        marker = ' ✅' if m < best_mae31 else ''
        print(f'  α={alpha:.1f}: MAE={m:.4f}{marker}')
        if m < best_mae31:
            best_mae31 = m; best_a31 = alpha

    # test 보정 적용
    test31_bc = apply_bc_calibration(test31_pred, prob_te31, table31, pb31, prb31)
    if best_a31 > 0:
        test31_final = test31_pred * (1 - best_a31) + test31_bc * best_a31
    else:
        test31_final = test31_pred.copy()
    test31_final = np.maximum(test31_final, 0)

    # 제출 파일
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = test31_final
    sub31_bc = os.path.join(SUB_DIR, 'model31_BC.csv')
    sample.to_csv(sub31_bc, index=False)
    print(f'\n  model31+BC 제출: {sub31_bc}')
    print(f'  test: mean={test31_final.mean():.2f}, std={test31_final.std():.2f}, max={test31_final.max():.2f}')

    # ════════════════════════════════════════════
    # [Part B] model32: 추가 shift-safe FE
    # ════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('[Part B] model32: model31 + 추가 shift-safe FE')
    print('═' * 70)

    train32, test32, new_fe_cols = load_data_model32()
    feat_cols_32 = get_feat_cols(train32)
    print(f'  model32 피처: {len(feat_cols_32)} (model31: {len(feat_cols_31)}, 추가: {len(feat_cols_32)-len(feat_cols_31)})')

    # 추가 피처 shift 분석
    shift_analysis(train32, test32, new_fe_cols, 'model32 추가 피처')

    # 위험 피처 필터링 (shift ≥ 0.35σ 제거)
    safe_new = []
    for col in new_fe_cols:
        if col not in train32.columns or col not in test32.columns:
            continue
        tr_m, te_m, tr_s = train32[col].mean(), test32[col].mean(), train32[col].std()
        shift = abs(tr_m - te_m) / (tr_s + 1e-8)
        if shift < 0.35:
            safe_new.append(col)
        else:
            print(f'  ⚠️ 제거: {col} (shift={shift:.3f}σ)')
            train32.drop(columns=[col], inplace=True)
            test32.drop(columns=[col], inplace=True)

    feat_cols_32 = get_feat_cols(train32)
    print(f'  필터 후 model32 피처: {len(feat_cols_32)} (안전 추가: {len(safe_new)}종)')

    if len(safe_new) == 0:
        print('  ⚠️ 안전한 추가 피처 없음 — model31 유지')
        # model31 결과 그대로 사용
        oof32, test32_pred, mae32 = oof31, test31_pred, mae31
    else:
        # Base Learner 학습
        results32 = build_base_oof(train32, test32, y_log, y_raw, groups,
                                    feat_cols_32, CKPT_32, label='model32')
        meta_tr32, meta_te32 = build_meta_inputs(results32)
        oof32, test32_pred, mae32 = run_meta_lgbm(meta_tr32, meta_te32, y_raw, groups, 'model32-meta')
        segment_analysis(oof32, y_raw.values, 'model32')

        # 상관 분석
        print(f'\n  model31-model32 OOF 상관: {np.corrcoef(oof31, oof32)[0,1]:.6f}')

        # 제출 파일
        sample['avg_delay_minutes_next_30m'] = np.maximum(test32_pred, 0)
        sub32 = os.path.join(SUB_DIR, 'model32_extended_fe.csv')
        sample.to_csv(sub32, index=False)
        print(f'\n  model32 제출: {sub32}')
        print(f'  test: mean={test32_pred.mean():.2f}, std={np.maximum(test32_pred, 0).std():.2f}, max={test32_pred.max():.2f}')

    # ════════════════════════════════════════════
    # [Part C] model32 + B+C 후처리
    # ════════════════════════════════════════════
    if len(safe_new) > 0 and mae32 <= mae31 + 0.01:
        print('\n' + '═' * 70)
        print('[Part C] model32 + B+C 후처리')
        print('═' * 70)

        # model32 피처 기반 분류기
        prob_tr32, prob_te32 = compute_extreme_prob(train32, test32, feat_cols_32)
        table32, pb32, prb32 = build_bc_calibration(oof32, y_raw.values, prob_tr32)

        oof32_bc = apply_bc_calibration(oof32, prob_tr32, table32, pb32, prb32)
        mae32_bc = np.abs(oof32_bc - y_raw.values).mean()

        print(f'\n  model32 기준 MAE: {mae32:.4f}')
        print(f'  model32+BC MAE:  {mae32_bc:.4f} (Δ={mae32_bc - mae32:+.4f})')

        # α 탐색
        best_a32, best_mae32 = 0.0, mae32
        for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            bl = oof32 * (1 - alpha) + oof32_bc * alpha
            m = np.abs(np.maximum(bl, 0) - y_raw.values).mean()
            if m < best_mae32:
                best_mae32 = m; best_a32 = alpha

        test32_bc = apply_bc_calibration(test32_pred, prob_te32, table32, pb32, prb32)
        if best_a32 > 0:
            test32_final = test32_pred * (1 - best_a32) + test32_bc * best_a32
        else:
            test32_final = test32_pred.copy()
        test32_final = np.maximum(test32_final, 0)

        sample['avg_delay_minutes_next_30m'] = test32_final
        sub32_bc = os.path.join(SUB_DIR, 'model32_BC.csv')
        sample.to_csv(sub32_bc, index=False)
        print(f'\n  model32+BC 제출: {sub32_bc}')

    # ════════════════════════════════════════════
    # [Part D] 크로스 블렌드 (model31 × model32)
    # ════════════════════════════════════════════
    if len(safe_new) > 0:
        print('\n' + '═' * 70)
        print('[Part D] model31 × model32 블렌드')
        print('═' * 70)

        best_w, best_blend_mae = 0.0, mae31
        for w32 in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            bl = oof31 * (1 - w32) + oof32 * w32
            m = np.abs(np.maximum(bl, 0) - y_raw.values).mean()
            marker = ' ✅' if m < best_blend_mae else ''
            print(f'  w32={w32:.1f}: MAE={m:.4f}{marker}')
            if m < best_blend_mae:
                best_blend_mae = m; best_w = w32

        if best_w > 0:
            test_blend = test31_pred * (1 - best_w) + test32_pred * best_w
            test_blend = np.maximum(test_blend, 0)
            sample['avg_delay_minutes_next_30m'] = test_blend
            sub_blend = os.path.join(SUB_DIR, 'blend_m31m32.csv')
            sample.to_csv(sub_blend, index=False)
            print(f'\n  블렌드(w32={best_w:.1f}) 제출: {sub_blend}')
            print(f'  test: mean={test_blend.mean():.2f}, std={test_blend.std():.2f}')

    # ════════════════════════════════════════════
    # 최종 요약
    # ════════════════════════════════════════════
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 70)
    print(f'최종 결과 ({elapsed:.1f}분 소요)')
    print('=' * 70)
    print(f'  model31 기준       : CV={mae31:.4f}, test_std={test31_pred.std():.2f}')
    print(f'  model31+BC(α={best_a31:.1f})  : CV={best_mae31:.4f} (Δ={best_mae31-mae31:+.4f})')
    if len(safe_new) > 0:
        print(f'  model32 FE 확장    : CV={mae32:.4f} (Δ={mae32-mae31:+.4f}), 피처 {len(feat_cols_32)}')
        print(f'  test: mean={np.maximum(test32_pred,0).mean():.2f}, std={np.maximum(test32_pred,0).std():.2f}')

    print(f'\n  기대 Public (×1.159):')
    print(f'    model31       : {mae31 * 1.159:.4f}')
    print(f'    model31+BC    : {best_mae31 * 1.159:.4f}')
    if len(safe_new) > 0:
        print(f'    model32       : {mae32 * 1.159:.4f}')

    # 제출 추천
    print('\n[제출 추천]')
    candidates = [('model31 (기준)', mae31)]
    candidates.append(('model31+BC', best_mae31))
    if len(safe_new) > 0:
        candidates.append(('model32', mae32))
    candidates.sort(key=lambda x: x[1])
    for rank, (name, cv) in enumerate(candidates, 1):
        print(f'  {rank}. {name}: CV={cv:.4f}')

    print(f'\n{"=" * 70}')


if __name__ == '__main__':
    main()
