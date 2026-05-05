"""
model34: Loss Ablation 기반 최적 스태킹
================================================================
§3 Loss ablation 핵심 발견:
  - Tweedie(1.5): [80+] MAE 86.28 (기존 TW1.8의 91.68 대비 Δ-5.4)
  - Asym(α=2.0): [80+] MAE 87.99, pred/actual 0.349 (기준 0.272 대비 +28%)
  - 각 loss는 서로 다른 구간에서 최적 → 메타가 구간별 선택 가능

모델 구성 (7모델):
  [유지] LGBM(MAE+log1p), CB(MAE+log1p), ET, RF — model31 체크포인트
  [교체] TW1.8 → TW1.5 (극값 챔피언)
  [교체] Asym(α=1.5) → Asym(α=2.0) (극값 2위 + 더 좋은 다양성)
  [신규] LGBM-Tweedie1.5 (log1p 공간 Tweedie — CatBoost TW1.5와 다른 오차)

기준: model33 6model+Asym CV 8.4756 / Public 9.8223
목표: 극값 [80+] MAE ↓ + 전체 CV ↓ + 배율 유지

실행: python src/run_model34_loss_opt.py
예상 시간: ~15분 (model31 ckpt 4종 재사용, 신규 3종 학습)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
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
CKPT_31  = os.path.join(_BASE, '..', 'docs', 'model31_ckpt')
CKPT_34  = os.path.join(_BASE, '..', 'docs', 'model34_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# ── 파라미터 ──
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
# ★ TW1.5로 변경 (기존 TW1.8에서)
TW15_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3.0,
    'loss_function': 'Tweedie:variance_power=1.5',
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

# ★ Asymmetric α=2.0 (model33의 α=1.5에서 상향)
ASYM_ALPHA = 2.0
ASYM_LGBM_PARAMS = {
    'num_leaves': 127, 'learning_rate': 0.015,
    'feature_fraction': 0.50, 'bagging_fraction': 0.90,
    'min_child_samples': 35, 'reg_alpha': 2.0, 'reg_lambda': 1.0,
    'n_estimators': 3000, 'bagging_freq': 1,
    'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
}

# ★ 신규: LGBM Tweedie 1.5 (log1p 공간 — CatBoost TW1.5와 다른 구현)
LGBM_TW15_PARAMS = {
    'num_leaves': 129, 'learning_rate': 0.01021,
    'feature_fraction': 0.465, 'bagging_fraction': 0.947,
    'min_child_samples': 30, 'reg_alpha': 1.468, 'reg_lambda': 0.396,
    'objective': 'tweedie', 'tweedie_variance_power': 1.5,
    'n_estimators': 3000, 'bagging_freq': 1,
    'random_state': RANDOM_STATE + 2, 'verbosity': -1, 'n_jobs': -1,
}

SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]


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

def load_ckpt(d, name):
    return (np.load(os.path.join(d, f'{name}_oof.npy')),
            np.load(os.path.join(d, f'{name}_test.npy')))

def ckpt_exists(d, name):
    return (os.path.exists(os.path.join(d, f'{name}_oof.npy')) and
            os.path.exists(os.path.join(d, f'{name}_test.npy')))


# ── FE (model31 동일) ──
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


# ── Base Learner 학습 함수 ──
def train_tw15_oof(X_train, X_test, y_raw, groups, feat_cols):
    """CatBoost Tweedie 1.5 (raw space)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr = X_train[feat_cols].fillna(0).values; X_te = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_raw, groups)):
        m = cb.CatBoostRegressor(**TW15_PARAMS)
        m.fit(cb.Pool(X_tr[tr_idx], y_raw.values[tr_idx]),
              eval_set=cb.Pool(X_tr[va_idx], y_raw.values[va_idx]), use_best_model=True)
        oof[va_idx] = m.predict(X_tr[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.values[va_idx]).mean()
        print(f'  [TW1.5] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred

def train_asym20_oof(X_train, X_test, y_log, groups, feat_cols):
    """Asymmetric MAE α=2.0 (log1p space)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr = X_train[feat_cols].fillna(0); X_te = X_test[feat_cols].fillna(0)
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
        print(f'  [Asym(α={ASYM_ALPHA})] Fold {fold+1}  MAE={mae:.4f}  iter={bst.best_iteration}')
        del bst; gc.collect()
    return oof, test_pred

def train_lgbm_tw15_oof(X_train, X_test, y_raw, groups, feat_cols):
    """LGBM Tweedie 1.5 (raw space — CatBoost와 다른 구현)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr = X_train[feat_cols].fillna(0); X_te = X_test[feat_cols].fillna(0)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_raw, groups)):
        m = lgb.LGBMRegressor(**LGBM_TW15_PARAMS)
        m.fit(X_tr.iloc[tr_idx], y_raw.iloc[tr_idx].values,
              eval_set=[(X_tr.iloc[va_idx], y_raw.iloc[va_idx].values)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.values[va_idx]).mean()
        print(f'  [LGBM-TW1.5] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred


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
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | pred_std={oof.std():.2f}')
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
    print('model34: Loss Ablation 기반 최적 스태킹')
    print('  변경: TW1.8→TW1.5, Asym α1.5→α2.0, +LGBM-TW1.5')
    print('  기준: model33 6model+Asym CV=8.4756 / Public=9.8223')
    print('=' * 70)

    os.makedirs(CKPT_34, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    print('\n[데이터 로드]')
    train, test = load_data()
    feat_cols = get_feat_cols(train)
    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'  피처: {len(feat_cols)}')

    oof_dict = {}
    test_dict = {}

    # ── Layer 1A: model31 체크포인트 재사용 (4종) ──
    print('\n' + '─' * 70)
    print('[Layer 1A] model31 체크포인트 재사용 (LGBM, CB, ET, RF)')
    print('─' * 70)

    for name in ['lgbm', 'cb', 'et', 'rf']:
        if ckpt_exists(CKPT_31, name):
            print(f'  [{name.upper()}] 체크포인트 로드')
            oof_dict[name], test_dict[name] = load_ckpt(CKPT_31, name)
        else:
            print(f'  ⚠️ [{name.upper()}] 체크포인트 없음 — 학습 필요')
            # 필요 시 학습 코드 추가 가능
            continue
        if name == 'tw15':
            mae = np.abs(oof_dict[name] - y_raw.values).mean()
        else:
            mae = np.abs(np.expm1(oof_dict[name]) - y_raw.values).mean()
        print(f'    OOF MAE: {mae:.4f}')

    # ── Layer 1B: 신규 3종 ──
    print('\n' + '─' * 70)
    print('[Layer 1B] 신규 모델 (TW1.5, Asym α=2.0, LGBM-TW1.5)')
    print('─' * 70)

    # CatBoost Tweedie 1.5
    if ckpt_exists(CKPT_34, 'tw15'):
        print('\n  [TW1.5] 체크포인트 로드')
        oof_dict['tw15'], test_dict['tw15'] = load_ckpt(CKPT_34, 'tw15')
    else:
        print(f'\n  [TW1.5] 학습 시작...')
        oof_dict['tw15'], test_dict['tw15'] = train_tw15_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt(CKPT_34, 'tw15', oof_dict['tw15'], test_dict['tw15'])
    print(f'    TW1.5 OOF MAE: {np.abs(oof_dict["tw15"] - y_raw.values).mean():.4f}')

    # Asymmetric α=2.0
    if ckpt_exists(CKPT_34, 'asym20'):
        print('\n  [Asym α=2.0] 체크포인트 로드')
        oof_dict['asym20'], test_dict['asym20'] = load_ckpt(CKPT_34, 'asym20')
    else:
        print(f'\n  [Asym α=2.0] 학습 시작...')
        oof_dict['asym20'], test_dict['asym20'] = train_asym20_oof(train, test, y_log, groups, feat_cols)
        save_ckpt(CKPT_34, 'asym20', oof_dict['asym20'], test_dict['asym20'])
    print(f'    Asym2.0 OOF MAE: {np.abs(np.expm1(oof_dict["asym20"]) - y_raw.values).mean():.4f}')

    # LGBM Tweedie 1.5
    if ckpt_exists(CKPT_34, 'lgbm_tw15'):
        print('\n  [LGBM-TW1.5] 체크포인트 로드')
        oof_dict['lgbm_tw15'], test_dict['lgbm_tw15'] = load_ckpt(CKPT_34, 'lgbm_tw15')
    else:
        print(f'\n  [LGBM-TW1.5] 학습 시작...')
        oof_dict['lgbm_tw15'], test_dict['lgbm_tw15'] = train_lgbm_tw15_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt(CKPT_34, 'lgbm_tw15', oof_dict['lgbm_tw15'], test_dict['lgbm_tw15'])
    print(f'    LGBM-TW1.5 OOF MAE: {np.abs(oof_dict["lgbm_tw15"] - y_raw.values).mean():.4f}')

    # ── 다양성 분석 ──
    print('\n' + '─' * 70)
    print('[다양성] 7모델 OOF 상관관계')
    print('─' * 70)

    oof_raw = {
        'LGBM': np.expm1(oof_dict['lgbm']),
        'CB':   np.expm1(oof_dict['cb']),
        'TW1.5': oof_dict['tw15'],
        'ET':   np.expm1(oof_dict['et']),
        'RF':   np.expm1(oof_dict['rf']),
        'Asym2.0': np.expm1(oof_dict['asym20']),
        'LTW1.5': oof_dict['lgbm_tw15'],
    }
    names = list(oof_raw.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = np.corrcoef(oof_raw[names[i]], oof_raw[names[j]])[0,1]
            marker = '✅' if c < 0.95 else ('⚠️' if c < 0.98 else '❌')
            print(f'  {names[i]:8s}-{names[j]:8s}: {c:.4f} {marker}')

    # 극값 분석
    print('\n[극값 분석] 모델별 [80,800) pred/actual')
    extreme_mask = y_raw.values >= 80
    if extreme_mask.sum() > 0:
        for name in names:
            pred_ext = oof_raw[name][extreme_mask]
            actual_ext = y_raw.values[extreme_mask]
            ratio = pred_ext.mean() / actual_ext.mean()
            mae_ext = np.abs(pred_ext - actual_ext).mean()
            print(f'  {name:8s}: pred/actual={ratio:.3f}, MAE={mae_ext:.2f}')

    # ── 메타 스태킹: 여러 구성 테스트 ──
    print('\n' + '─' * 70)
    print('[Layer 2] 메타 스태킹 (여러 구성)')
    print('─' * 70)

    # Config A: 5모델 기준 (LGBM, CB, TW1.5, ET, RF) — TW1.8→1.5 교체
    meta_A_train = np.column_stack([
        oof_dict['lgbm'], oof_dict['cb'],
        np.log1p(np.maximum(oof_dict['tw15'], 0)),
        oof_dict['et'], oof_dict['rf']
    ])
    meta_A_test = np.column_stack([
        test_dict['lgbm'], test_dict['cb'],
        np.log1p(np.maximum(test_dict['tw15'], 0)),
        test_dict['et'], test_dict['rf']
    ])
    print('\n[A] 5모델 (TW1.5 교체)')
    oof_A, test_A, mae_A = run_meta(meta_A_train, meta_A_test, y_raw, groups, '5model-TW15')
    segment_analysis(oof_A, y_raw.values, '5model-TW15')

    # Config B: 6모델 (A + Asym2.0)
    meta_B_train = np.column_stack([meta_A_train, oof_dict['asym20']])
    meta_B_test  = np.column_stack([meta_A_test,  test_dict['asym20']])
    print('\n[B] 6모델 (+Asym2.0)')
    oof_B, test_B, mae_B = run_meta(meta_B_train, meta_B_test, y_raw, groups, '6model+Asym2.0')
    segment_analysis(oof_B, y_raw.values, '6model+Asym2.0')

    # Config C: 7모델 (B + LGBM-TW1.5)
    meta_C_train = np.column_stack([meta_B_train, np.log1p(np.maximum(oof_dict['lgbm_tw15'], 0))])
    meta_C_test  = np.column_stack([meta_B_test,  np.log1p(np.maximum(test_dict['lgbm_tw15'], 0))])
    print('\n[C] 7모델 (+LGBM-TW1.5)')
    oof_C, test_C, mae_C = run_meta(meta_C_train, meta_C_test, y_raw, groups, '7model-full')
    segment_analysis(oof_C, y_raw.values, '7model-full')

    # Config D: 6모델 (A + LGBM-TW1.5, Asym 없이)
    meta_D_train = np.column_stack([meta_A_train, np.log1p(np.maximum(oof_dict['lgbm_tw15'], 0))])
    meta_D_test  = np.column_stack([meta_A_test,  np.log1p(np.maximum(test_dict['lgbm_tw15'], 0))])
    print('\n[D] 6모델 (+LGBM-TW1.5 only)')
    oof_D, test_D, mae_D = run_meta(meta_D_train, meta_D_test, y_raw, groups, '6model+LTW15')
    segment_analysis(oof_D, y_raw.values, '6model+LTW15')

    # ── 제출 파일 ──
    print('\n' + '─' * 70)
    print('[제출 파일]')
    print('─' * 70)

    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    configs = [
        ('model34_5tw15', test_A, mae_A),
        ('model34_6asym20', test_B, mae_B),
        ('model34_7full', test_C, mae_C),
        ('model34_6ltw15', test_D, mae_D),
    ]
    for name, pred, cv in configs:
        pc = np.maximum(pred, 0)
        sample['avg_delay_minutes_next_30m'] = pc
        fpath = os.path.join(SUB_DIR, f'{name}.csv')
        sample.to_csv(fpath, index=False)
        print(f'  {name}: CV={cv:.4f}, test_std={pc.std():.2f}, test_max={pc.max():.2f}')

    # ── 최종 비교 ──
    elapsed = (time.time() - t0) / 60
    print(f'\n{"=" * 70}')
    print(f'model34 결과 ({elapsed:.1f}분 소요)')
    print('=' * 70)
    print(f'  [A] 5model-TW15          : CV={mae_A:.4f} (Δ={mae_A-8.4786:+.4f} vs model31)')
    print(f'  [B] 6model+Asym2.0       : CV={mae_B:.4f} (Δ={mae_B-8.4786:+.4f})')
    print(f'  [C] 7model-full          : CV={mae_C:.4f} (Δ={mae_C-8.4786:+.4f})')
    print(f'  [D] 6model+LTW15         : CV={mae_D:.4f} (Δ={mae_D-8.4786:+.4f})')
    print()

    results = [('A:5model-TW15', mae_A), ('B:6model+Asym2.0', mae_B),
               ('C:7model-full', mae_C), ('D:6model+LTW15', mae_D)]
    results.sort(key=lambda x: x[1])
    best = results[0]
    print(f'  최적: {best[0]} (CV={best[1]:.4f})')
    print(f'  기대 Public (×1.159): {best[1] * 1.159:.4f}')
    print(f'  model33 대비: Δ={best[1] - 8.4756:+.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
