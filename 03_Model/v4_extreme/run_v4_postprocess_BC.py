"""
v4.1B: 시나리오 분류기 extreme_prob + 2D 보정 테이블 (B+C) 후처리
================================================================
전략:
  1. model30 파이프라인 재현 (422피처, 5모델 스태킹 → LGBM-meta)
     - OOF 예측 + test 예측 생성
  2. LightGBM 이진 분류기로 extreme_prob 산출
     - 시나리오 레벨: mean_target ≥ 40 → 극값 시나리오 (AUC ~0.897)
     - GroupKFold OOF로 train extreme_prob (리크 방지)
     - 전체 학습 모델로 test extreme_prob
  3. OOF에서 2D 보정 테이블 구축
     - (prediction_bin, extreme_prob_bin) → correction_factor
     - correction_factor = mean(actual / predicted) per cell
  4. test 예측에 보정 테이블 적용
     - 안전장치: correction_factor [0.8, 3.0] 클리핑
     - extreme_prob 낮은 구간은 보정 최소화

model30 대비 차이점:
  - IF (unsupervised) vs 분류기 (supervised) → 분류기가 라벨 정보 활용
  - extreme_prob는 이미 v4에서 AUC 0.897 달성 → 검증된 신호

기준: model30 CV 8.4838 / Public 9.8279 / 배율 1.1584
목표: [80,800) MAE 개선 → 전체 MAE 하락

실행: python src/run_v4_postprocess_BC.py
예상 시간: ~35분 (5모델 스태킹 + 분류기 + 보정)
출력: submissions/v4_postprocess_BC.csv
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
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model30_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

EXTREME_THRESHOLD = 40  # 시나리오 mean_target ≥ 40 → 극값

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

# 분류기 파라미터 (v4에서 검증됨)
CLF_PARAMS = {
    'objective': 'binary', 'metric': 'auc',
    'num_leaves': 63, 'learning_rate': 0.03,
    'feature_fraction': 0.7, 'bagging_fraction': 0.8,
    'min_child_samples': 30, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'n_estimators': 1000, 'bagging_freq': 1,
    'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
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
# [핵심] 시나리오 분류기 → extreme_prob
# ─────────────────────────────────────────────
def compute_extreme_prob(train, test, feat_cols):
    """
    LightGBM 이진 분류기로 극값 시나리오 확률 산출.

    라벨: 시나리오 mean_target ≥ EXTREME_THRESHOLD (40) → 1
    입력: 시나리오 집계 피처 (sc_* 피처)
    OOF: GroupKFold로 train extreme_prob 산출 (리크 방지)
    test: 전체 train으로 학습 → test 예측

    Returns:
        train_prob, test_prob: extreme_prob ∈ [0, 1]
    """
    print('\n[시나리오 분류기] extreme_prob 산출')

    y_raw = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']

    # 시나리오 레벨 라벨 생성
    sc_mean = train.groupby('scenario_id')['avg_delay_minutes_next_30m'].mean()
    sc_label = (sc_mean >= EXTREME_THRESHOLD).astype(int)
    # 행 레벨로 broadcast
    row_label = train['scenario_id'].map(sc_label).values

    n_extreme = sc_label.sum()
    n_total = len(sc_label)
    print(f'  극값 시나리오: {n_extreme}/{n_total} ({n_extreme/n_total*100:.1f}%)')

    # 분류기 입력 피처: sc_ 피처 중심
    clf_feat_cols = [c for c in feat_cols
                     if c.startswith('sc_') or c.startswith('ratio_')]
    print(f'  분류기 피처: {len(clf_feat_cols)}종')

    X_tr = train[clf_feat_cols].fillna(0)
    X_te = test[clf_feat_cols].fillna(0)

    # OOF extreme_prob (GroupKFold → 리크 방지)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_prob = np.zeros(len(train))
    test_prob = np.zeros(len(test))

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

    # 전체 AUC
    oof_auc = roc_auc_score(row_label, oof_prob)
    # F1 at threshold=0.5
    oof_f1 = f1_score(row_label, (oof_prob >= 0.5).astype(int))
    print(f'  OOF AUC={oof_auc:.4f}, F1(0.5)={oof_f1:.4f}')

    # extreme_prob 분포
    print(f'  train prob: mean={oof_prob.mean():.4f}, std={oof_prob.std():.4f}, '
          f'max={oof_prob.max():.4f}')
    print(f'  test  prob: mean={test_prob.mean():.4f}, std={test_prob.std():.4f}, '
          f'max={test_prob.max():.4f}')

    # 극값 구간별 prob 확인
    bins_check = [(0,5), (5,20), (20,50), (50,80), (80,800)]
    print(f'\n  타겟 구간별 extreme_prob:')
    for lo, hi in bins_check:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            print(f'    [{lo:3d},{hi:3d}): mean_prob={oof_prob[mask].mean():.4f}, '
                  f'n={mask.sum()}')

    return oof_prob, test_prob


# ─────────────────────────────────────────────
# [핵심] 2D 보정 테이블 구축 + 적용
# ─────────────────────────────────────────────
def build_2d_calibration_table(oof_pred, y_actual, extreme_prob,
                                pred_bins=None, prob_bins=None):
    """
    OOF에서 2D 보정 테이블 구축.

    축1: prediction_bin (예측값 구간)
    축2: extreme_prob_bin (극값 확률 구간)
    셀값: correction_factor = mean(actual / predicted)
    """
    print('\n[2D 보정 테이블] 구축')

    if pred_bins is None:
        pred_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
    if prob_bins is None:
        # extreme_prob에 맞는 bin: 대부분 0에 가깝고 극값만 높음
        prob_bins = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.01]

    pred_bin_idx = np.digitize(oof_pred, pred_bins) - 1
    pred_bin_idx = np.clip(pred_bin_idx, 0, len(pred_bins) - 2)

    prob_bin_idx = np.digitize(extreme_prob, prob_bins) - 1
    prob_bin_idx = np.clip(prob_bin_idx, 0, len(prob_bins) - 2)

    table = {}
    print(f'  pred_bins: {[f"{b:.0f}" for b in pred_bins]}')
    print(f'  prob_bins: {[f"{b:.2f}" for b in prob_bins]}')
    print()

    header = f'  {"pred_bin":>12s} | {"prob_bin":>12s} | {"n":>6s} | {"mean_pred":>10s} | {"mean_actual":>10s} | {"raw_corr":>10s} | {"final_corr":>10s}'
    print(header)
    print('  ' + '-' * len(header))

    for pi in range(len(pred_bins) - 1):
        for pbi in range(len(prob_bins) - 1):
            mask = (pred_bin_idx == pi) & (prob_bin_idx == pbi)
            n = mask.sum()
            if n < 5:
                table[(pi, pbi)] = 1.0
                continue

            mean_pred = oof_pred[mask].mean()
            mean_actual = y_actual[mask].mean()

            if mean_pred > 1.0:
                raw_correction = mean_actual / mean_pred
            else:
                raw_correction = 1.0

            # 안전 클리핑
            clipped = np.clip(raw_correction, 0.8, 3.0)
            table[(pi, pbi)] = clipped

            # extreme_prob < 0.20 (정상 구간)은 보정 축소
            if prob_bins[pbi + 1] <= 0.20:
                shrink_factor = 0.2  # 정상 구간은 보정의 20%만 적용
                table[(pi, pbi)] = 1.0 + (clipped - 1.0) * shrink_factor

            flag = '⬆' if table[(pi, pbi)] > 1.05 else ('⬇' if table[(pi, pbi)] < 0.95 else '  ')
            print(f'  [{pred_bins[pi]:3.0f},{pred_bins[pi+1]:3.0f}) | '
                  f'[{prob_bins[pbi]:.2f},{prob_bins[pbi+1]:.2f}) | '
                  f'{n:6d} | {mean_pred:10.2f} | {mean_actual:10.2f} | '
                  f'{raw_correction:10.4f} | {table[(pi,pbi)]:10.4f} {flag}')

    return table, pred_bins, prob_bins


def apply_2d_calibration(predictions, extreme_prob, table, pred_bins, prob_bins):
    """2D 보정 테이블을 예측에 적용."""
    pred_bin_idx = np.digitize(predictions, pred_bins) - 1
    pred_bin_idx = np.clip(pred_bin_idx, 0, len(pred_bins) - 2)

    prob_bin_idx = np.digitize(extreme_prob, prob_bins) - 1
    prob_bin_idx = np.clip(prob_bin_idx, 0, len(prob_bins) - 2)

    corrected = predictions.copy()
    for i in range(len(predictions)):
        key = (pred_bin_idx[i], prob_bin_idx[i])
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
    print('v4.1B: 시나리오 분류기 extreme_prob + 2D 보정 테이블 (B+C)')
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

    if ckpt_exists('lgbm'):
        print('\n[LGBM] 체크포인트 로드'); oof_lg, test_lg = load_ckpt('lgbm')
    else:
        print('\n[LGBM] 학습 시작...')
        oof_lg, test_lg = train_lgbm_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('lgbm', oof_lg, test_lg)
    print(f'  LGBM OOF MAE={np.abs(np.expm1(oof_lg) - y_raw.values).mean():.4f}')

    if ckpt_exists('tw18'):
        print('\n[TW1.8] 체크포인트 로드'); oof_tw, test_tw = load_ckpt('tw18')
    else:
        print('\n[TW1.8] 학습 시작...')
        oof_tw, test_tw = train_tw18_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt('tw18', oof_tw, test_tw)
    print(f'  TW1.8 OOF MAE={np.abs(oof_tw - y_raw.values).mean():.4f}')

    if ckpt_exists('cb'):
        print('\n[CB] 체크포인트 로드'); oof_cb, test_cb = load_ckpt('cb')
    else:
        print('\n[CB] 학습 시작...')
        oof_cb, test_cb = train_cb_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('cb', oof_cb, test_cb)
    print(f'  CB OOF MAE={np.abs(np.expm1(oof_cb) - y_raw.values).mean():.4f}')

    if ckpt_exists('et'):
        print('\n[ET] 체크포인트 로드'); oof_et, test_et = load_ckpt('et')
    else:
        print('\n[ET] 학습 시작...')
        oof_et, test_et = train_et_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('et', oof_et, test_et)
    print(f'  ET OOF MAE={np.abs(np.expm1(oof_et) - y_raw.values).mean():.4f}')

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

    segment_analysis(oof_meta, y_raw.values, label='model30 기준선 (후처리 전)')

    # ── Step 4: 시나리오 분류기 extreme_prob ──
    print('\n' + '─' * 70)
    print('[Step 4] 시나리오 분류기 → extreme_prob')
    print('─' * 70)

    train_prob, test_prob = compute_extreme_prob(train, test, feat_cols)

    # ── Step 5: 2D 보정 테이블 구축 ──
    print('\n' + '─' * 70)
    print('[Step 5] 2D 보정 테이블 구축 (B+C)')
    print('─' * 70)

    # 예측값 bin
    pred_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]

    # extreme_prob bin: 0에 밀집, 높은 쪽 세분화
    prob_bins = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.01]

    table, p_bins, pb_bins = build_2d_calibration_table(
        oof_meta, y_raw.values, train_prob,
        pred_bins=pred_bins, prob_bins=prob_bins
    )

    # ── Step 6: OOF 보정 적용 + 평가 ──
    print('\n' + '─' * 70)
    print('[Step 6] OOF 보정 적용 + 평가')
    print('─' * 70)

    oof_corrected = apply_2d_calibration(oof_meta, train_prob, table, p_bins, pb_bins)
    mae_corrected = np.abs(oof_corrected - y_raw.values).mean()

    print(f'\n  기준선 MAE: {mae_baseline:.4f}')
    print(f'  B+C 보정 MAE: {mae_corrected:.4f}')
    print(f'  변화:        {mae_corrected - mae_baseline:+.4f}')

    segment_analysis(oof_corrected, y_raw.values, label='B+C 2D 보정 후')

    # ── Step 7: 보정 강도 (α) 최적화 ──
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

    test_corrected = apply_2d_calibration(test_meta, test_prob, table, p_bins, pb_bins)

    if best_alpha > 0 and best_alpha < 1.0:
        test_final = test_meta * (1 - best_alpha) + test_corrected * best_alpha
    elif best_alpha >= 1.0:
        test_final = test_corrected
    else:
        test_final = test_meta.copy()
    test_final = np.maximum(test_final, 0)

    print(f'  기준 test: mean={test_meta.mean():.2f}, std={test_meta.std():.2f}, max={test_meta.max():.2f}')
    print(f'  B+C  test: mean={test_final.mean():.2f}, std={test_final.std():.2f}, max={test_final.max():.2f}')

    # 제출 파일 생성
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    # 원본 model30 (보정 없음)
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    sub_base = os.path.join(SUB_DIR, 'v4_BC_baseline.csv')
    sample.to_csv(sub_base, index=False)

    # B+C 보정 적용
    sample['avg_delay_minutes_next_30m'] = test_final
    sub_bc = os.path.join(SUB_DIR, 'v4_postprocess_BC.csv')
    sample.to_csv(sub_bc, index=False)

    print(f'\n  기준선 제출: {sub_base}')
    print(f'  B+C 보정 제출: {sub_bc}')

    # ── Step 9: IF vs B+C 비교용 통계 ──
    print('\n' + '─' * 70)
    print('[Step 9] extreme_prob 분포 + 예측 통계')
    print('─' * 70)

    print(f'  extreme_prob shift: '
          f'train={train_prob.mean():.4f}, test={test_prob.mean():.4f}, '
          f'diff={abs(train_prob.mean()-test_prob.mean()):.4f}')

    for name, pred in [('기준 (model30)', test_meta), ('B+C 보정', test_final)]:
        print(f'\n  [{name}]')
        print(f'    mean={pred.mean():.2f}, std={pred.std():.2f}, '
              f'max={pred.max():.2f}, min={pred.min():.2f}')
        for pct in [50, 75, 90, 95, 99]:
            print(f'    p{pct}={np.percentile(pred, pct):.2f}', end='')
        print()

    # ── 최종 요약 ──
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 70)
    print(f'v4.1B 결과 ({elapsed:.1f}분 소요)')
    print('=' * 70)
    print(f'  시나리오 분류기 OOF AUC: {roc_auc_score(train["scenario_id"].map(train.groupby("scenario_id")["avg_delay_minutes_next_30m"].mean() >= EXTREME_THRESHOLD).values, train_prob):.4f}')
    print(f'  model30 기준선 CV:   {mae_baseline:.4f}')
    print(f'  B+C 보정 CV:         {mae_corrected:.4f} (Δ={mae_corrected - mae_baseline:+.4f})')
    print(f'  최적 α={best_alpha:.1f} CV:      {best_mae_alpha:.4f} (Δ={best_mae_alpha - mae_baseline:+.4f})')
    print(f'  test pred std:       {test_final.std():.2f} (기준 {test_meta.std():.2f})')

    ratio_est = best_mae_alpha * 1.158 if best_mae_alpha < mae_baseline else mae_baseline * 1.158
    print(f'  기대 Public (×1.158): {ratio_est:.4f}')

    if best_mae_alpha < mae_baseline:
        print(f'\n  ✅ B+C 후처리 유효! CV Δ={best_mae_alpha - mae_baseline:+.4f}')
        print(f'  → v4_postprocess_BC.csv 제출 추천')
    else:
        print(f'\n  ⚠️ B+C 후처리 무효 — 기준선 유지')
        print(f'  → 보정이 OOF 기준 개선 없음. 하지만 배율 변화 가능성은 있으므로 제출 검토 가능')

    print(f'\n{"=" * 70}')


if __name__ == '__main__':
    main()
