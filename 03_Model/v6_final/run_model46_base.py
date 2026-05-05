"""
run_model46_base.py
===================
model46a/b/c 공통 베이스 — 파라미터, 학습 함수, 메타 함수 공유
  model46a : SC_AGG 확장   (run_model46a_sc_expand.py)
  model46b : KEY_COLS 확장 (run_model46b_key_expand.py)
  model46c : Layout 교호작용 (run_model46c_layout_cross.py)

기준: model34 Config B  CV=8.4803 / Public=9.8078
      model45c q95(7모델) CV=8.4684 / Public=9.7931 ← 현 최고
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
import warnings, gc, os, sys

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

# ── 공통 경로 ──
_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
DOCS_DIR = os.path.join(_BASE, '..', 'docs')

N_SPLITS     = 5
RANDOM_STATE = 42

# ── 하이퍼파라미터 (model34 Optuna 결과) ──
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
ASYM_ALPHA = 2.0
ASYM_LGBM_PARAMS = {
    'num_leaves': 127, 'learning_rate': 0.015,
    'feature_fraction': 0.50, 'bagging_fraction': 0.90,
    'min_child_samples': 35, 'reg_alpha': 2.0, 'reg_lambda': 1.0,
    'n_estimators': 3000, 'bagging_freq': 1,
    'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
}
META_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# ── model34 기준 SC_AGG 18개 ──
SC_AGG_BASE = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]


# ── Custom Loss ──
def asymmetric_mae_objective(y_pred, dtrain):
    y_true   = dtrain.get_label()
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


# ── SC_AGG ──
def add_scenario_agg(df, agg_cols):
    """시나리오 집계 피처 (11통계)"""
    df = df.copy()
    for col in agg_cols:
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]
        df[f'sc_{col}_mean']     = grp.transform('mean')
        df[f'sc_{col}_std']      = grp.transform('std').fillna(0)
        df[f'sc_{col}_max']      = grp.transform('max')
        df[f'sc_{col}_min']      = grp.transform('min')
        df[f'sc_{col}_diff']     = df[col] - df[f'sc_{col}_mean']
        df[f'sc_{col}_median']   = grp.transform('median')
        df[f'sc_{col}_p10']      = grp.transform(lambda x: x.quantile(0.10))
        df[f'sc_{col}_p90']      = grp.transform(lambda x: x.quantile(0.90))
        df[f'sc_{col}_skew']     = grp.transform(lambda x: x.skew()).fillna(0)
        df[f'sc_{col}_kurtosis'] = grp.transform(lambda x: x.kurtosis()).fillna(0)
        df[f'sc_{col}_cv']       = (df[f'sc_{col}_std'] /
                                     (df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)
    return df

def safe_div(a, b, fill=0.0):
    return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

def add_ratio_tier1(df):
    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = safe_div(
            df['sc_order_inflow_15m_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = safe_div(
            df['sc_congestion_score_mean'], df['intersection_count'])
    if all(c in df.columns for c in ['sc_low_battery_ratio_mean',
                                      'sc_charge_queue_length_mean', 'charger_count']):
        df['ratio_battery_stress'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df['sc_charge_queue_length_mean'],
            df['charger_count'])
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = safe_div(
            df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']
    return df

def add_ratio_tier2(df):
    if all(c in df.columns for c in ['sc_congestion_score_mean',
                                      'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'],
            df['robot_total'] ** 2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm'] / 100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(
            df['pack_station_count'], df['floor_area_sqm'] / 1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(
            df['sc_robot_charging_mean'], df['charger_count'])
    if all(c in df.columns for c in ['sc_battery_mean_mean',
                                      'sc_robot_utilization_mean', 'charger_count']):
        df['ratio_battery_per_robot'] = safe_div(
            df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'],
            df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(
            df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    return df


# ── 기본 FE 파이프라인 (model34 동일) ──
def load_base_fe(sc_agg_cols=None, lag_key_cols=None):
    """
    Parameters
    ----------
    sc_agg_cols   : SC_AGG에 사용할 컬럼 목록 (None이면 SC_AGG_BASE 18개)
    lag_key_cols  : lag/rolling에 사용할 KEY_COLS (None이면 feature_engineering 기본 8개)
    """
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    # feature_engineering의 KEY_COLS를 원하는 것으로 교체
    if lag_key_cols is not None:
        import feature_engineering as fe_module
        _orig = fe_module.KEY_COLS[:]
        fe_module.KEY_COLS = lag_key_cols
        train, test = build_features(train, test, layout,
                                     lag_lags=[1, 2, 3, 4, 5, 6],
                                     rolling_windows=[3, 5, 10],
                                     verbose=True)
        fe_module.KEY_COLS = _orig   # 복원
    else:
        train, test = build_features(train, test, layout,
                                     lag_lags=[1, 2, 3, 4, 5, 6],
                                     rolling_windows=[3, 5, 10],
                                     verbose=True)

    _agg = sc_agg_cols if sc_agg_cols is not None else SC_AGG_BASE

    print(f'  SC_AGG 컬럼 수: {len(_agg)}  → sc 피처: {len(_agg) * 11}')
    for fn in [lambda df: add_scenario_agg(df, _agg),
               add_ratio_tier1, add_ratio_tier2]:
        train = fn(train); test = fn(test)

    return train, test

def get_feat_cols(df):
    return [c for c in df.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and df[c].dtype != object]


# ── Base Learner 학습 함수 ──
def train_lgbm(X_tr, X_te, y_log, groups, feat_cols, ckpt_dir, name='lgbm'):
    if ckpt_exists(ckpt_dir, name):
        print(f'  [{name.upper()}] 체크포인트 로드'); return load_ckpt(ckpt_dir, name)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); preds = np.zeros(len(X_te))
    Xtr = X_tr[feat_cols].fillna(0); Xte = X_te[feat_cols].fillna(0)
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, y_log, groups)):
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(Xtr.iloc[ti], y_log.iloc[ti],
              eval_set=[(Xtr.iloc[vi], y_log.iloc[vi])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[vi] = m.predict(Xtr.iloc[vi])
        preds   += m.predict(Xte) / N_SPLITS
        mae = np.abs(np.expm1(oof[vi]) - np.expm1(y_log.iloc[vi].values)).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    save_ckpt(ckpt_dir, name, oof, preds); return oof, preds

def train_cb(X_tr, X_te, y_log, groups, feat_cols, ckpt_dir, name='cb'):
    if ckpt_exists(ckpt_dir, name):
        print(f'  [{name.upper()}] 체크포인트 로드'); return load_ckpt(ckpt_dir, name)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); preds = np.zeros(len(X_te))
    Xtr = X_tr[feat_cols].fillna(0).values; Xte = X_te[feat_cols].fillna(0).values
    y_log_arr = y_log.values
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, y_log_arr, groups)):
        m = cb.CatBoostRegressor(**CB_PARAMS)
        m.fit(cb.Pool(Xtr[ti], y_log_arr[ti]),
              eval_set=cb.Pool(Xtr[vi], y_log_arr[vi]), use_best_model=True)
        oof[vi] = m.predict(Xtr[vi])
        preds   += m.predict(Xte) / N_SPLITS
        mae = np.abs(np.expm1(oof[vi]) - np.expm1(y_log_arr[vi])).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    save_ckpt(ckpt_dir, name, oof, preds); return oof, preds

def train_tw15(X_tr, X_te, y_raw, groups, feat_cols, ckpt_dir, name='tw15'):
    if ckpt_exists(ckpt_dir, name):
        print(f'  [{name.upper()}] 체크포인트 로드'); return load_ckpt(ckpt_dir, name)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); preds = np.zeros(len(X_te))
    Xtr = X_tr[feat_cols].fillna(0).values; Xte = X_te[feat_cols].fillna(0).values
    y_arr = y_raw.values
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, y_arr, groups)):
        m = cb.CatBoostRegressor(**TW15_PARAMS)
        m.fit(cb.Pool(Xtr[ti], y_arr[ti]),
              eval_set=cb.Pool(Xtr[vi], y_arr[vi]), use_best_model=True)
        oof[vi] = m.predict(Xtr[vi])
        preds   += m.predict(Xte) / N_SPLITS
        mae = np.abs(oof[vi] - y_arr[vi]).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    save_ckpt(ckpt_dir, name, oof, preds); return oof, preds

def train_et(X_tr, X_te, y_log, groups, feat_cols, ckpt_dir, name='et'):
    if ckpt_exists(ckpt_dir, name):
        print(f'  [{name.upper()}] 체크포인트 로드'); return load_ckpt(ckpt_dir, name)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); preds = np.zeros(len(X_te))
    Xtr = X_tr[feat_cols].fillna(0).values; Xte = X_te[feat_cols].fillna(0).values
    y_arr = y_log.values
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, y_arr, groups)):
        m = ExtraTreesRegressor(**ET_PARAMS)
        m.fit(Xtr[ti], y_arr[ti])
        oof[vi] = m.predict(Xtr[vi])
        preds   += m.predict(Xte) / N_SPLITS
        mae = np.abs(np.expm1(oof[vi]) - np.expm1(y_arr[vi])).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    save_ckpt(ckpt_dir, name, oof, preds); return oof, preds

def train_rf(X_tr, X_te, y_log, groups, feat_cols, ckpt_dir, name='rf'):
    if ckpt_exists(ckpt_dir, name):
        print(f'  [{name.upper()}] 체크포인트 로드'); return load_ckpt(ckpt_dir, name)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); preds = np.zeros(len(X_te))
    Xtr = X_tr[feat_cols].fillna(0).values; Xte = X_te[feat_cols].fillna(0).values
    y_arr = y_log.values
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, y_arr, groups)):
        m = RandomForestRegressor(**RF_PARAMS)
        m.fit(Xtr[ti], y_arr[ti])
        oof[vi] = m.predict(Xtr[vi])
        preds   += m.predict(Xte) / N_SPLITS
        mae = np.abs(np.expm1(oof[vi]) - np.expm1(y_arr[vi])).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    save_ckpt(ckpt_dir, name, oof, preds); return oof, preds

def train_asym20(X_tr, X_te, y_log, groups, feat_cols, ckpt_dir, name='asym20'):
    if ckpt_exists(ckpt_dir, name):
        print(f'  [{name.upper()}] 체크포인트 로드'); return load_ckpt(ckpt_dir, name)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); preds = np.zeros(len(X_te))
    Xtr = X_tr[feat_cols].fillna(0); Xte = X_te[feat_cols].fillna(0)
    y_arr = y_log.values
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, y_arr, groups)):
        dtrain = lgb.Dataset(Xtr.iloc[ti], label=y_arr[ti])
        dval   = lgb.Dataset(Xtr.iloc[vi], label=y_arr[vi], reference=dtrain)
        params = {k: v for k, v in ASYM_LGBM_PARAMS.items() if k != 'n_estimators'}
        params['objective'] = asymmetric_mae_objective
        bst = lgb.train(params, dtrain,
                        num_boost_round=ASYM_LGBM_PARAMS['n_estimators'],
                        valid_sets=[dval], feval=asymmetric_mae_metric,
                        callbacks=[lgb.early_stopping(50, verbose=False),
                                   lgb.log_evaluation(0)])
        oof[vi] = bst.predict(Xtr.iloc[vi])
        preds   += bst.predict(Xte) / N_SPLITS
        mae = np.abs(np.expm1(oof[vi]) - np.expm1(y_arr[vi])).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={bst.best_iteration}')
        del bst; gc.collect()
    save_ckpt(ckpt_dir, name, oof, preds); return oof, preds


# ── 메타 스태킹 ──
def run_meta(oof_dict, test_dict, y_raw, groups, label='meta'):
    names   = list(oof_dict.keys())
    Xm_tr   = np.column_stack([oof_dict[n] for n in names])
    Xm_te   = np.column_stack([test_dict[n] for n in names])
    y_log   = np.log1p(y_raw.values)
    oof_meta = np.zeros(len(y_raw)); preds = []
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (ti, vi) in enumerate(gkf.split(Xm_tr, y_log, groups)):
        m = lgb.LGBMRegressor(**META_PARAMS)
        m.fit(Xm_tr[ti], y_log[ti],
              eval_set=[(Xm_tr[vi], y_log[vi])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[vi] = np.expm1(m.predict(Xm_tr[vi]))
        preds.append(np.expm1(m.predict(Xm_te)))
        mae = np.abs(oof_meta[vi] - y_raw.values[vi]).mean()
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    test_meta = np.mean(preds, axis=0)
    cv = np.abs(oof_meta - y_raw.values).mean()
    print(f'\n  [{label}] OOF CV={cv:.4f} | pred_std={test_meta.std():.2f} '
          f'| test_mean={test_meta.mean():.2f}')
    return cv, oof_meta, test_meta


# ── 구간 분석 ──
def segment_report(pred, actual, label=''):
    cv = np.abs(pred - actual).mean()
    print(f'\n[구간 분석] {label} (전체 MAE={cv:.4f})')
    for lo, hi in [(0,5),(5,20),(20,50),(50,80),(80,800)]:
        mask = (actual >= lo) & (actual < hi)
        if not mask.any(): continue
        mae = np.abs(pred[mask] - actual[mask]).mean()
        pr  = pred[mask].mean() / (actual[mask].mean() + 1e-8)
        print(f'  [{lo:3d},{hi:3d}) n={mask.sum():6d} '
              f'MAE={mae:7.2f}  pred/actual={pr:.3f}')
    return cv


# ── 다양성 분석 ──
def diversity_report(oof_dict):
    """OOF 간 상관 출력 (raw space 통일)"""
    raw = {}
    for k, v in oof_dict.items():
        raw[k] = np.expm1(v) if (v.max() < 15 and v.min() >= -1) else v
    names = list(raw.keys())
    print('\n[다양성] OOF 상관')
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = np.corrcoef(raw[names[i]], raw[names[j]])[0,1]
            mark = '✅' if c < 0.95 else ('⚠️' if c < 0.98 else '❌')
            print(f'  {names[i]:8s}-{names[j]:8s}: {c:.4f} {mark}')
