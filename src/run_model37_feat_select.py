"""
전략 3: Aggressive Feature Selection — model37
================================================================
근거:
  - 현재 429피처 중 다수가 노이즈 → 과적합 유발, 배율 악화
  - model29A 교훈: 약간의 노이즈가 정규화 역할을 할 수 있지만,
    불필요한 피처 제거가 더 근본적인 정규화
  - LGBM feature importance 기반 하위 피처 제거 → CV 유지/소폭 악화 + 배율 개선 기대
  - 3단계 실험: Top 80% / Top 60% / Top 50% 피처로 6모델 스태킹

접근법:
  1. model31 LGBM 체크포인트의 fold별 feature importance 평균
  2. importance 하위 피처 제거 (3단계 컷오프)
  3. 각 컷오프에서 6모델 full 재학습 + meta stacking
  4. CV + pred_std + 배율 추정 비교 → 최적 컷오프 선정

모델 구성 (6모델, model34-B 동일):
  LGBM (log1p), CB (log1p), ET (log1p), RF (log1p),
  TW1.5 (raw), Asym2.0 (log1p)

실행: python src/run_model37_feat_select.py
예상 시간: ~20분 (3컷오프 × 6모델 = 18회 학습, 체크포인트 일부 재사용 불가)
※ USER 로컬 실행 필수
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
import warnings, gc, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model37_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

TARGET = 'avg_delay_minutes_next_30m'

# ── 하이퍼파라미터 (model30 = 29A피처 + 29B Optuna 결합) ──
BEST_LGBM_PARAMS = {
    'num_leaves': 129, 'learning_rate': 0.01021,
    'feature_fraction': 0.465, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 1.468, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

CB_PARAMS = {
    'depth': 9, 'learning_rate': 0.01144,
    'iterations': 3000, 'l2_leaf_reg': 1.561,
    'random_strength': 1.359, 'bagging_temperature': 0.29,
    'random_seed': RANDOM_STATE,
    'loss_function': 'MAE', 'verbose': 0, 'thread_count': -1,
}

TW_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'tweedie', 'tweedie_variance_power': 1.5,
    'n_estimators': 3000, 'bagging_freq': 1,
    'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
}

ASYM_ALPHA = 2.0
ASYM_LGBM_PARAMS = {
    'num_leaves': 127, 'learning_rate': 0.015,
    'feature_fraction': 0.50, 'bagging_fraction': 0.90,
    'min_child_samples': 35, 'reg_alpha': 2.0, 'reg_lambda': 1.0,
    'n_estimators': 3000, 'bagging_freq': 1,
    'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
}

META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}


# ── FE (model31 동일) ──
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

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
    if 'sc_battery_mean_mean' in df.columns and 'sc_robot_utilization_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_per_robot'] = safe_div(
            df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'], df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    return df


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(train, test, layout, lag_lags=[1,2,3,4,5,6], rolling_windows=[3,5,10])
    for fn in [add_scenario_agg_features, add_ratio_tier1, add_ratio_tier2]:
        train = fn(train); test = fn(test)
    return train, test


def get_feat_cols(df):
    return [c for c in df.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and df[c].dtype != object]


# ── Custom Loss ──
def asymmetric_mae_objective(y_pred, dtrain):
    y_true = dtrain.get_label()
    residual = y_true - y_pred
    grad = np.where(residual > 0, -ASYM_ALPHA, 1.0)
    hess = np.ones_like(y_pred)
    return grad, hess

def asymmetric_mae_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    mae = np.abs(np.expm1(y_pred) - np.expm1(y_true)).mean()
    return 'asym_mae', mae, False


# ── 체크포인트 ──
def save_ckpt(d, name, oof, test_pred):
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f'{name}_oof.npy'), oof)
    np.save(os.path.join(d, f'{name}_test.npy'), test_pred)


# ── Phase 0: Feature Importance 수집 ──
def collect_feature_importance(X_train, y_log, groups, feat_cols):
    """LGBM 5-fold 학습 후 평균 feature importance 반환"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    imp_sum = np.zeros(len(feat_cols))

    print('\n[Phase 0] Feature Importance 수집 (LGBM 5-fold)')
    X = X_train[feat_cols].fillna(0)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_log, groups)):
        m = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
        m.fit(X.iloc[tr_idx], y_log.iloc[tr_idx].values,
              eval_set=[(X.iloc[va_idx], y_log.iloc[va_idx].values)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        imp_sum += m.feature_importances_
        mae = np.abs(np.expm1(m.predict(X.iloc[va_idx])) - np.expm1(y_log.iloc[va_idx].values)).mean()
        print(f'  Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    imp_mean = imp_sum / N_SPLITS
    imp_df = pd.DataFrame({'feature': feat_cols, 'importance': imp_mean})
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # 중요도 분포 출력
    total_imp = imp_df['importance'].sum()
    cum_pct = imp_df['importance'].cumsum() / total_imp * 100
    imp_df['cum_pct'] = cum_pct

    print(f'\n  총 피처: {len(feat_cols)}, 총 importance: {total_imp:.0f}')
    for pct in [50, 60, 70, 80, 90, 95]:
        n = (cum_pct <= pct).sum() + 1
        print(f'  Top {pct}% importance: {min(n, len(feat_cols))} 피처')

    # 하위 20개 피처 출력
    print(f'\n  하위 20개 피처 (importance):')
    for _, row in imp_df.tail(20).iterrows():
        print(f'    {row["feature"]:<50s} imp={row["importance"]:.1f} ({row["importance"]/total_imp*100:.2f}%)')

    return imp_df


# ── 6모델 학습 함수 ──
def train_6models(X_train, X_test, y_raw, y_log, groups, feat_cols, tag=''):
    """6모델 OOF + test 예측"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    X_tr = X_train[feat_cols].fillna(0)
    X_te = X_test[feat_cols].fillna(0)

    oof_dict = {}
    test_dict = {}
    n_feat = len(feat_cols)

    # ── 1. LGBM (log1p) ──
    print(f'\n  [LGBM] {n_feat}피처')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
        m.fit(X_tr.iloc[tr_idx], y_log.iloc[tr_idx].values,
              eval_set=[(X_tr.iloc[va_idx], y_log.iloc[va_idx].values)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - y_raw.iloc[va_idx].values).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    oof_dict['lgbm'] = oof; test_dict['lgbm'] = test_pred
    lgbm_mae = np.abs(np.expm1(oof) - y_raw.values).mean()
    print(f'    LGBM OOF MAE={lgbm_mae:.4f}')

    # ── 2. CatBoost (log1p) ──
    print(f'\n  [CB] {n_feat}피처')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        pool_tr = cb.Pool(X_tr.iloc[tr_idx], label=y_log.iloc[tr_idx].values)
        pool_va = cb.Pool(X_tr.iloc[va_idx], label=y_log.iloc[va_idx].values)
        m = cb.CatBoostRegressor(**CB_PARAMS)
        m.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=50, verbose=0)
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - y_raw.iloc[va_idx].values).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m, pool_tr, pool_va; gc.collect()
    oof_dict['cb'] = oof; test_dict['cb'] = test_pred
    cb_mae = np.abs(np.expm1(oof) - y_raw.values).mean()
    print(f'    CB OOF MAE={cb_mae:.4f}')

    # ── 3. Tweedie 1.5 (raw space) ──
    print(f'\n  [TW1.5] {n_feat}피처 (raw space)')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_raw, groups)):
        m = lgb.LGBMRegressor(**TW_PARAMS)
        m.fit(X_tr.iloc[tr_idx], y_raw.iloc[tr_idx].values,
              eval_set=[(X_tr.iloc[va_idx], y_raw.iloc[va_idx].values)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    oof_dict['tw15'] = oof; test_dict['tw15'] = test_pred
    tw_mae = np.abs(oof - y_raw.values).mean()
    print(f'    TW1.5 OOF MAE={tw_mae:.4f}')

    # ── 4. ExtraTrees (log1p) ──
    print(f'\n  [ET] {n_feat}피처')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=5,
                                random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_tr.iloc[tr_idx], y_log.iloc[tr_idx].values)
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - y_raw.iloc[va_idx].values).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    oof_dict['et'] = oof; test_dict['et'] = test_pred
    et_mae = np.abs(np.expm1(oof) - y_raw.values).mean()
    print(f'    ET OOF MAE={et_mae:.4f}')

    # ── 5. RandomForest (log1p) ──
    print(f'\n  [RF] {n_feat}피처')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=5,
                                  random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_tr.iloc[tr_idx], y_log.iloc[tr_idx].values)
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - y_raw.iloc[va_idx].values).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    oof_dict['rf'] = oof; test_dict['rf'] = test_pred
    rf_mae = np.abs(np.expm1(oof) - y_raw.values).mean()
    print(f'    RF OOF MAE={rf_mae:.4f}')

    # ── 6. Asymmetric MAE (log1p) ──
    print(f'\n  [Asym(α={ASYM_ALPHA})] {n_feat}피처')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        dtrain = lgb.Dataset(X_tr.iloc[tr_idx], label=y_log.iloc[tr_idx].values)
        dval   = lgb.Dataset(X_tr.iloc[va_idx], label=y_log.iloc[va_idx].values, reference=dtrain)
        params = {k: v for k, v in ASYM_LGBM_PARAMS.items() if k not in ['n_estimators']}
        params['objective'] = asymmetric_mae_objective
        bst = lgb.train(
            params, dtrain,
            num_boost_round=ASYM_LGBM_PARAMS['n_estimators'],
            valid_sets=[dval], feval=asymmetric_mae_metric,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va_idx] = bst.predict(X_tr.iloc[va_idx])
        test_pred += bst.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.iloc[va_idx].values)).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={bst.best_iteration}')
        del bst; gc.collect()
    oof_dict['asym20'] = oof; test_dict['asym20'] = test_pred
    asym_mae = np.abs(np.expm1(oof) - y_raw.values).mean()
    print(f'    Asym2.0 OOF MAE={asym_mae:.4f}')

    return oof_dict, test_dict


# ── 메타 학습기 ──
def run_meta(meta_train, meta_test, y_raw, groups, label='meta'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y_raw)); test_pred = np.zeros(meta_test.shape[0])
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(meta_train[tr_idx], np.log1p(y_raw.iloc[tr_idx].values),
              eval_set=[(meta_train[va_idx], np.log1p(y_raw.iloc[va_idx].values))],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = np.expm1(m.predict(meta_train[va_idx]))
        test_pred += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    oof_mae = np.abs(oof - y_raw.values).mean()
    test_std = np.maximum(test_pred, 0).std()
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | test_std={test_std:.2f}')
    return oof, test_pred, oof_mae


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


# ── MAIN ──
def main():
    t0 = time.time()
    print('=' * 70)
    print('model37: Aggressive Feature Selection (6모델 스태킹)')
    print('  기준: model34-B blend CV=8.4803 / Public=9.8073')
    print('  목표: 피처 축소로 과적합 억제 → 배율 개선')
    print('  3단계: Top 80% / Top 60% / Top 50%')
    print('=' * 70)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    # ── 데이터 로드 ──
    print('\n[데이터 로드] build_features + scenario agg + ratios')
    train, test = load_data()
    all_feat_cols = get_feat_cols(train)
    y_raw = train[TARGET]
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'  train: {len(train)}, test: {len(test)}, 전체 피처: {len(all_feat_cols)}')

    # ── Phase 0: Feature Importance 수집 ──
    imp_df = collect_feature_importance(train, y_log, groups, all_feat_cols)

    # importance 저장 (분석용)
    imp_path = os.path.join(CKPT_DIR, 'feature_importance.csv')
    imp_df.to_csv(imp_path, index=False)
    print(f'\n  Feature importance 저장: {imp_path}')

    # ── 3단계 컷오프 실험 ──
    cutoffs = [
        ('top80', 0.80),
        ('top60', 0.60),
        ('top50', 0.50),
    ]

    total_imp = imp_df['importance'].sum()
    results = []

    ref_df = pd.read_csv(os.path.join(SUB_DIR, 'blend_m33m34_w80.csv'))  # 제출 템플릿

    for tag, pct in cutoffs:
        cum = imp_df['importance'].cumsum() / total_imp
        n_keep = (cum <= pct).sum() + 1  # pct까지의 피처 수
        n_keep = min(n_keep, len(all_feat_cols))
        selected_cols = imp_df['feature'].iloc[:n_keep].tolist()

        print('\n' + '=' * 70)
        print(f'[{tag}] 피처 {n_keep}/{len(all_feat_cols)} 선택 '
              f'(importance {pct:.0%} 커버)')
        print('=' * 70)

        # 선택된 피처 저장
        sel_path = os.path.join(CKPT_DIR, f'selected_features_{tag}.txt')
        with open(sel_path, 'w') as f:
            f.write('\n'.join(selected_cols))

        # ── 6모델 학습 ──
        oof_dict, test_dict = train_6models(
            train, test, y_raw, y_log, groups, selected_cols, tag=tag)

        # OOF 체크포인트 저장
        tag_ckpt = os.path.join(CKPT_DIR, tag)
        for name in oof_dict:
            save_ckpt(tag_ckpt, name, oof_dict[name], test_dict[name])

        # ── 상관 분석 ──
        print(f'\n[{tag}] 모델 간 상관')
        keys = list(oof_dict.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                ki, kj = keys[i], keys[j]
                if ki == 'tw15':
                    a = oof_dict[ki]
                else:
                    a = np.expm1(oof_dict[ki])
                if kj == 'tw15':
                    b = oof_dict[kj]
                else:
                    b = np.expm1(oof_dict[kj])
                corr = np.corrcoef(a, b)[0, 1]
                if corr < 0.95:
                    print(f'  {ki}-{kj}: {corr:.4f} ✅')
                else:
                    print(f'  {ki}-{kj}: {corr:.4f}')

        # ── 메타 스태킹 ──
        print(f'\n[{tag}] 메타 스태킹')

        # meta 입력: log1p 모델은 그대로, TW1.5는 log1p 변환
        meta_oof_list = []
        meta_test_list = []
        for name in ['lgbm', 'cb', 'et', 'rf', 'asym20']:
            meta_oof_list.append(oof_dict[name])
            meta_test_list.append(test_dict[name])
        # TW1.5: raw → log1p
        meta_oof_list.append(np.log1p(np.maximum(oof_dict['tw15'], 0)))
        meta_test_list.append(np.log1p(np.maximum(test_dict['tw15'], 0)))

        meta_train_arr = np.column_stack(meta_oof_list)
        meta_test_arr  = np.column_stack(meta_test_list)

        oof_final, test_final, meta_mae = run_meta(
            meta_train_arr, meta_test_arr, y_raw, groups, label=f'{tag}-meta')

        test_final = np.maximum(test_final, 0)

        # 예측 통계
        pred_std = test_final.std()
        pred_mean = test_final.mean()
        pred_max = test_final.max()
        print(f'\n  [{tag}] test: mean={pred_mean:.2f}, std={pred_std:.2f}, max={pred_max:.2f}')
        print(f'  [{tag}] 배율 추정: ×1.156={meta_mae*1.156:.4f}, ×1.160={meta_mae*1.160:.4f}')

        # 구간 분석
        segment_analysis(oof_final, y_raw.values, label=f'{tag} meta')

        # 제출 파일 생성
        sub = ref_df.copy()
        sub[TARGET] = test_final
        fname = f'model37_{tag}.csv'
        sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f'  → {fname} 저장')

        results.append({
            'tag': tag,
            'n_feat': n_keep,
            'meta_mae': meta_mae,
            'pred_std': pred_std,
            'pred_max': pred_max,
            'fname': fname,
        })

    # ── 종합 비교 ──
    print('\n' + '=' * 70)
    print('종합 비교')
    print('=' * 70)
    print(f'  {"Config":<12s}  {"피처":>5s}  {"Meta MAE":>10s}  {"pred_std":>10s}  '
          f'{"pred_max":>10s}  {"예상 Public (×1.156)":>20s}')
    print(f'  {"-"*12}  {"-"*5}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*20}')

    # 기준선 (model34-B)
    print(f'  {"baseline":<12s}  {"429":>5s}  {"8.4803":>10s}  {"16.15":>10s}  '
          f'{"~100":>10s}  {"9.8073 (actual)":>20s}')

    for r in results:
        est_pub = r['meta_mae'] * 1.156
        print(f'  {r["tag"]:<12s}  {r["n_feat"]:>5d}  {r["meta_mae"]:>10.4f}  '
              f'{r["pred_std"]:>10.2f}  {r["pred_max"]:>10.2f}  {est_pub:>20.4f}')

    # ── 최적 컷오프 추천 ──
    print('\n핵심 해석:')
    print('  - CV 소폭 악화(<0.03)이면서 pred_std 유지/확대 → 배율 개선 기대')
    print('  - CV 대폭 악화(>0.05) → 정보 손실 과다, 해당 컷오프 부적합')
    print('  - 배율 1.156 이하 달성 시 Public 개선 가능')

    elapsed = time.time() - t0
    print(f'\n총 소요 시간: {elapsed/60:.1f}분')
    print('※ USER가 제출하여 실제 Public 스코어 확인 필요')


if __name__ == '__main__':
    main()
