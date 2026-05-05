"""
model42: LGBM 피처 중요도 기반 경량화
================================================================
동기:
  - model41(458 feat) CV 8.4851 / Public 9.8449 → 배율 1.1602 (model31보다 악화)
  - 궤적 피처 29종이 순노이즈로 작용 → 전체 피처 중 하위권 예상
  - 피처 수 축소 = 암묵적 정규화 → 배율 개선 가능성 (model29A 교훈)

전략:
  Phase 1 (빠름, ~10분): 2-fold quick LGBM으로 gain 기반 중요도 산출
  Phase 2 (전체, ~240분): 상위 K개 피처로 6모델 스태킹
  K 탐색: top-300 / top-250 / top-200 중 가장 높은 pred_std 선택

기준:
  - CV MAE: model31 8.4786 (model41 8.4851)
  - Public:  blend_w80 9.8073 (model31 9.8255)
  - pred_std: model31 15.89

경고: model37(v5)에서 top-80(163 feat) → CV 8.8061 참패 이력 있음
     상위 피처 비율이 44%(200/458)로 훨씬 보수적이나 주의 필요
     Phase 1 결과에서 pred_std 경고 기준 적용

실행: python src/run_model42_feat_select.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings, gc, os, sys, time, pickle

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model42_ckpt')
N_SPLITS  = 5
RANDOM_STATE = 42

# ── 피처 선택 K값 (Phase 2에서 사용) ──
# 여러 값을 비교하려면 TOPK_LIST = [300, 250, 200]으로 변경
TOPK_LIST = [250]  # 기본값: 250 (보수적 단일 실험)

# ── 하이퍼파라미터 (model41 동일) ──
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
ET_PARAMS = {'n_estimators': 500, 'max_depth': 20, 'min_samples_leaf': 5,
             'max_features': 0.7, 'random_state': RANDOM_STATE, 'n_jobs': -1}
RF_PARAMS = {'n_estimators': 500, 'max_depth': 20, 'min_samples_leaf': 5,
             'max_features': 0.7, 'random_state': RANDOM_STATE, 'n_jobs': -1}
ASYM_ALPHA = 2.0
ASYM_PARAMS = {
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

SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]
TRAJ_COLS  = ['robot_utilization','order_inflow_15m','congestion_score',
              'low_battery_ratio','battery_mean','charge_queue_length',
              'robot_idle','max_zone_density']
PEAK_COLS  = ['order_inflow_15m','congestion_score','low_battery_ratio',
              'charge_queue_length','max_zone_density']
MONO_COLS  = ['robot_utilization','congestion_score','order_inflow_15m']


# ═══════════════════════════════════════════════════════
# Feature Engineering (model41과 동일)
# ═══════════════════════════════════════════════════════

def _safe_div(a, b, fill=0):
    return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

def add_scenario_agg_features(df):
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns: continue
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
        df[f'sc_{col}_cv']       = (
            df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)
    return df

def add_ratio_features(df):
    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = _safe_div(df['sc_order_inflow_15m_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = _safe_div(df['sc_congestion_score_mean'], df['intersection_count'])
    if all(c in df.columns for c in ['sc_low_battery_ratio_mean','sc_charge_queue_length_mean','charger_count']):
        df['ratio_battery_stress'] = _safe_div(
            df['sc_low_battery_ratio_mean'] * df['sc_charge_queue_length_mean'], df['charger_count'])
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = _safe_div(df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']
    if all(c in df.columns for c in ['sc_congestion_score_mean','sc_order_inflow_15m_mean','robot_total']):
        df['ratio_cross_stress'] = _safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'], df['robot_total']**2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = _safe_div(df['robot_total'], df['floor_area_sqm'] / 100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = _safe_div(df['pack_station_count'], df['floor_area_sqm'] / 1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = _safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if all(c in df.columns for c in ['sc_battery_mean_mean','sc_robot_utilization_mean','charger_count']):
        df['ratio_battery_per_robot'] = _safe_div(
            df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'], df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = _safe_div(df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = _safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    return df

def add_shift_safe_fe(df):
    if 'robot_utilization' in df.columns and 'order_inflow_15m' in df.columns:
        df['feat_util_x_order']    = df['robot_utilization'] * df['order_inflow_15m']
    if 'low_battery_ratio' in df.columns and 'congestion_score' in df.columns:
        df['feat_batt_x_cong']     = df['low_battery_ratio'] * df['congestion_score']
    if 'charge_queue_length' in df.columns and 'charger_count' in df.columns:
        df['feat_queue_per_charger'] = _safe_div(df['charge_queue_length'].fillna(0), df['charger_count'])
    if 'robot_idle' in df.columns and 'robot_utilization' in df.columns:
        df['feat_idle_util_ratio'] = _safe_div(df['robot_idle'].fillna(0), df['robot_utilization'].fillna(0)+1e-8)
    if 'order_inflow_15m' in df.columns and 'congestion_score' in df.columns:
        df['feat_order_cong']      = df['order_inflow_15m'].fillna(0) * df['congestion_score'].fillna(0)
    if 'battery_mean' in df.columns and 'low_battery_ratio' in df.columns:
        df['feat_batt_risk']       = (100-df['battery_mean'].fillna(100)) * df['low_battery_ratio'].fillna(0)
    if 'max_zone_density' in df.columns and 'order_inflow_15m' in df.columns:
        df['feat_density_order']   = df['max_zone_density'].fillna(0) * df['order_inflow_15m'].fillna(0)
    return df

def add_trajectory_features(df):
    """궤적 형상 피처 (model41과 동일, 29종)"""
    df = df.copy()
    if 'ts_idx' not in df.columns:
        df['ts_idx'] = df.groupby('scenario_id').cumcount()
    ts_arr = np.arange(25, dtype=np.float64)
    for col in TRAJ_COLS:
        if col not in df.columns: continue
        slope_map = (df.groupby('scenario_id')[col]
                     .apply(lambda x: np.polyfit(ts_arr[:len(x)], x.fillna(x.mean()).values, 1)[0]
                            if len(x)>1 else 0.0).fillna(0))
        df[f'sc_{col}_slope'] = df['scenario_id'].map(slope_map)
    for col in TRAJ_COLS:
        if col not in df.columns: continue
        f5 = df[df['ts_idx']<5].groupby('scenario_id')[col].mean()
        l5 = df[df['ts_idx']>=20].groupby('scenario_id')[col].mean()
        fl = (l5/(f5.abs()+1e-8)).fillna(1.0).replace([np.inf,-np.inf],1.0)
        df[f'sc_{col}_fl_ratio'] = df['scenario_id'].map(fl)
    for col in PEAK_COLS:
        if col not in df.columns: continue
        peak_map = (df.groupby('scenario_id')
                    .apply(lambda g: g.loc[g[col].fillna(-np.inf).idxmax(),'ts_idx']/24.0
                           if col in g.columns else 0.5).fillna(0.5))
        df[f'sc_{col}_peak_pos'] = df['scenario_id'].map(peak_map)
    for col in PEAK_COLS:
        if col not in df.columns: continue
        sm = f'sc_{col}_mean'; ss = f'sc_{col}_std'
        if sm not in df.columns or ss not in df.columns: continue
        above_map = ((df[col].fillna(0) > df[sm]+0.5*df[ss]).astype(int)
                     .groupby(df['scenario_id']).sum())
        df[f'sc_{col}_above_cnt'] = df['scenario_id'].map(above_map).fillna(0)
    for col in MONO_COLS:
        if col not in df.columns: continue
        def _mono(x):
            v = x.fillna(x.mean()).values
            return float((np.diff(v)>0).sum())/len(np.diff(v)) if len(v)>1 else 0.5
        mono_map = df.groupby('scenario_id')[col].apply(_mono).fillna(0.5)
        df[f'sc_{col}_mono'] = df['scenario_id'].map(mono_map)
    return df

def load_and_prepare_data():
    t0 = time.time()
    print('데이터 로드 중...')
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    print('기본 FE (lag+rolling+ts+layout)...')
    train, test = build_features(train, test, layout,
                                 lag_lags=[1,2,3,4,5,6],
                                 rolling_windows=[3,5,10])
    print('sc_agg 피처 추가...')
    train = add_scenario_agg_features(train)
    test  = add_scenario_agg_features(test)
    print('비율 피처 추가...')
    train = add_ratio_features(train)
    test  = add_ratio_features(test)
    print('shift-safe 피처 추가...')
    train = add_shift_safe_fe(train)
    test  = add_shift_safe_fe(test)
    print('궤적 피처 추가...')
    train = add_trajectory_features(train)
    test  = add_trajectory_features(test)
    elapsed = time.time() - t0
    print(f'FE 완료: {train.shape}, {test.shape}, {elapsed:.1f}s')
    return train, test

def get_feat_cols(df):
    return [c for c in df.columns
            if c not in {'ID','scenario_id','layout_id','avg_delay_minutes_next_30m'}
            and df[c].dtype != object]

def classify_feat(col):
    """피처를 카테고리로 분류"""
    if any(col.endswith(s) for s in ['_slope','_fl_ratio','_peak_pos','_above_cnt','_mono']):
        return 'trajectory'
    if col.startswith('sc_'):
        return 'sc_agg'
    if col.startswith('ratio_'):
        return 'ratio'
    if col.startswith('feat_'):
        return 'shift_safe'
    if 'lag' in col:
        return 'lag'
    if 'roll' in col:
        return 'rolling'
    return 'base'


# ═══════════════════════════════════════════════════════
# Phase 1: Quick Importance 산출 (2-fold LGBM)
# ═══════════════════════════════════════════════════════

def compute_quick_importance(X_tr, y_log, groups, feat_cols):
    """2-fold quick LGBM으로 gain 기반 피처 중요도 산출 (~10분)"""
    print('\n[Phase 1] 피처 중요도 산출 중 (2-fold quick LGBM)...')
    quick_params = {**LGBM_PARAMS, 'n_estimators': 1000, 'learning_rate': 0.02}
    gkf = GroupKFold(n_splits=2)
    Xt = X_tr[feat_cols].fillna(0)
    importance_sum = np.zeros(len(feat_cols))

    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt, y_log, groups)):
        m = lgb.LGBMRegressor(**quick_params)
        m.fit(Xt.iloc[tr_i], y_log.iloc[tr_i],
              eval_set=[(Xt.iloc[va_i], y_log.iloc[va_i])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        importance_sum += m.feature_importances_
        mae_va = np.abs(np.expm1(m.predict(Xt.iloc[va_i]))
                        - np.expm1(y_log.iloc[va_i].values)).mean()
        print(f'  Quick Fold {fold+1}  val MAE={mae_va:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    importance_df = pd.DataFrame({
        'feature': feat_cols,
        'importance': importance_sum / 2,
        'category': [classify_feat(c) for c in feat_cols]
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    importance_df['rank'] = importance_df.index + 1
    importance_df['cum_pct'] = importance_df['importance'].cumsum() / importance_df['importance'].sum() * 100

    # 카테고리별 통계 출력
    print('\n[Phase 1] 카테고리별 피처 수 & 평균 중요도:')
    for cat, grp in importance_df.groupby('category'):
        print(f'  {cat:12s}: {len(grp):3d}개  avg_imp={grp["importance"].mean():.1f}  '
              f'top10 포함 {(grp["rank"]<=10).sum()}개')

    print(f'\n[Phase 1] 상위 K에서 카테고리 구성:')
    for k in [100, 200, 250, 300]:
        top_k = importance_df[importance_df['rank'] <= k]
        cats = top_k['category'].value_counts().to_dict()
        cum = importance_df[importance_df['rank'] <= k]['cum_pct'].max()
        print(f'  top-{k:3d}: {dict(cats)}  → 누적 중요도 {cum:.1f}%')

    # 저중요도 피처 분포
    zero_imp = (importance_df['importance'] == 0).sum()
    print(f'\n  중요도=0 피처: {zero_imp}개 (전체 {len(feat_cols)}개 중)')

    # 저장
    os.makedirs(CKPT_DIR, exist_ok=True)
    importance_df.to_csv(os.path.join(CKPT_DIR, 'feature_importance.csv'), index=False)
    print(f'  → 저장: {CKPT_DIR}/feature_importance.csv')

    return importance_df


# ═══════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════

def _ckpt_path(name, topk):
    return os.path.join(CKPT_DIR, f'top{topk}_{name}')

def save_ckpt(name, topk, oof, test_pred):
    os.makedirs(CKPT_DIR, exist_ok=True)
    np.save(f'{_ckpt_path(name, topk)}_oof.npy',  oof)
    np.save(f'{_ckpt_path(name, topk)}_test.npy', test_pred)

def load_ckpt(name, topk):
    return (np.load(f'{_ckpt_path(name, topk)}_oof.npy'),
            np.load(f'{_ckpt_path(name, topk)}_test.npy'))

def ckpt_exists(name, topk):
    return (os.path.exists(f'{_ckpt_path(name, topk)}_oof.npy') and
            os.path.exists(f'{_ckpt_path(name, topk)}_test.npy'))


# ═══════════════════════════════════════════════════════
# Asymmetric Loss
# ═══════════════════════════════════════════════════════

def asym_obj(y_pred, dtrain):
    y_true = dtrain.get_label()
    residual = y_true - y_pred
    grad = np.where(residual > 0, -ASYM_ALPHA, 1.0)
    return grad, np.ones_like(y_pred)

def asym_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    mae = np.abs(np.expm1(y_pred) - np.expm1(y_true)).mean()
    return 'asym_mae', mae, False


# ═══════════════════════════════════════════════════════
# Base Learner 학습
# ═══════════════════════════════════════════════════════

def train_lgbm_oof(X_tr, X_te, y_log, groups, feat_cols, topk, name='lgbm'):
    if ckpt_exists(name, topk):
        print(f'  [{name}] 체크포인트 로드')
        return load_ckpt(name, topk)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt = X_tr[feat_cols].fillna(0); Xte = X_te[feat_cols].fillna(0)
    fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt, y_log, groups)):
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(Xt.iloc[tr_i], y_log.iloc[tr_i],
              eval_set=[(Xt.iloc[va_i], y_log.iloc[va_i])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_i]   = np.expm1(m.predict(Xt.iloc[va_i]))
        test_pred  += np.expm1(m.predict(Xte)) / N_SPLITS
        mae = np.abs(oof[va_i] - np.expm1(y_log.iloc[va_i].values)).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    print(f'  [{name}] OOF MAE = {np.mean(fold_maes):.4f}')
    save_ckpt(name, topk, oof, test_pred)
    return oof, test_pred

def train_cb_oof(X_tr, X_te, y_log, groups, feat_cols, topk, name='cb'):
    if ckpt_exists(name, topk):
        print(f'  [{name}] 체크포인트 로드')
        return load_ckpt(name, topk)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt = X_tr[feat_cols].fillna(0).values; Xte = X_te[feat_cols].fillna(0).values
    y_arr = y_log.values; fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt, y_arr, groups)):
        m = cb.CatBoostRegressor(**CB_PARAMS)
        m.fit(cb.Pool(Xt[tr_i], y_arr[tr_i]),
              eval_set=cb.Pool(Xt[va_i], y_arr[va_i]), use_best_model=True)
        oof[va_i]  = np.expm1(m.predict(Xt[va_i]))
        test_pred += np.expm1(m.predict(Xte)) / N_SPLITS
        mae = np.abs(oof[va_i] - np.expm1(y_arr[va_i])).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    print(f'  [{name}] OOF MAE = {np.mean(fold_maes):.4f}')
    save_ckpt(name, topk, oof, test_pred)
    return oof, test_pred

def train_tw15_oof(X_tr, X_te, y_raw, groups, feat_cols, topk, name='tw15'):
    if ckpt_exists(name, topk):
        print(f'  [{name}] 체크포인트 로드')
        return load_ckpt(name, topk)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt = X_tr[feat_cols].fillna(0).values; Xte = X_te[feat_cols].fillna(0).values
    y_arr = y_raw.values; fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt, y_arr, groups)):
        m = cb.CatBoostRegressor(**TW15_PARAMS)
        m.fit(cb.Pool(Xt[tr_i], y_arr[tr_i]),
              eval_set=cb.Pool(Xt[va_i], y_arr[va_i]), use_best_model=True)
        oof[va_i]  = m.predict(Xt[va_i])
        test_pred += m.predict(Xte) / N_SPLITS
        mae = np.abs(oof[va_i] - y_arr[va_i]).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    print(f'  [{name}] OOF MAE = {np.mean(fold_maes):.4f}')
    save_ckpt(name, topk, oof, test_pred)
    return oof, test_pred

def train_tree_oof(X_tr, X_te, y_raw, groups, feat_cols, topk, params, name='et'):
    if ckpt_exists(name, topk):
        print(f'  [{name}] 체크포인트 로드')
        return load_ckpt(name, topk)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Cls = ExtraTreesRegressor if name == 'et' else RandomForestRegressor
    Xt = X_tr[feat_cols].fillna(0); Xte = X_te[feat_cols].fillna(0)
    y_arr = y_raw.values; fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt, y_arr, groups)):
        m = Cls(**params)
        m.fit(Xt.iloc[tr_i], y_arr[tr_i])
        oof[va_i]  = m.predict(Xt.iloc[va_i])
        test_pred += m.predict(Xte) / N_SPLITS
        mae = np.abs(oof[va_i] - y_arr[va_i]).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    print(f'  [{name}] OOF MAE = {np.mean(fold_maes):.4f}')
    save_ckpt(name, topk, oof, test_pred)
    return oof, test_pred

def train_asym_oof(X_tr, X_te, y_log, groups, feat_cols, topk, name='asym20'):
    if ckpt_exists(name, topk):
        print(f'  [{name}] 체크포인트 로드')
        return load_ckpt(name, topk)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt = X_tr[feat_cols].fillna(0); Xte = X_te[feat_cols].fillna(0)
    fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt, y_log, groups)):
        dtr = lgb.Dataset(Xt.iloc[tr_i], y_log.iloc[tr_i])
        dva = lgb.Dataset(Xt.iloc[va_i], y_log.iloc[va_i], reference=dtr)
        raw_p = {k: v for k, v in ASYM_PARAMS.items()
                 if k not in ('n_estimators','random_state')}
        raw_p['seed'] = RANDOM_STATE
        raw_p['objective'] = asym_obj   # lgb 4.x: fobj 제거, params에 전달
        m = lgb.train(raw_p, dtr, num_boost_round=3000,
                      valid_sets=[dva],
                      feval=asym_metric,
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        oof[va_i]  = np.expm1(m.predict(Xt.iloc[va_i]))
        test_pred += np.expm1(m.predict(Xte)) / N_SPLITS
        mae = np.abs(oof[va_i] - np.expm1(y_log.iloc[va_i].values)).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration}')
        del m; gc.collect()
    print(f'  [{name}] OOF MAE = {np.mean(fold_maes):.4f}')
    save_ckpt(name, topk, oof, test_pred)
    return oof, test_pred


# ═══════════════════════════════════════════════════════
# Phase 2: Top-K 스태킹 실험
# ═══════════════════════════════════════════════════════

def run_topk_experiment(topk, importance_df, train, test, feat_cols_all):
    print(f'\n{"="*60}')
    print(f'Phase 2: top-{topk} 피처 실험')
    print(f'{"="*60}')

    # 피처 선택
    sel_feats = importance_df[importance_df['rank'] <= topk]['feature'].tolist()
    sel_feats = [c for c in sel_feats if c in feat_cols_all]
    print(f'선택된 피처: {len(sel_feats)}개 / 전체 {len(feat_cols_all)}개')
    cats = pd.Series([classify_feat(c) for c in sel_feats]).value_counts().to_dict()
    print(f'카테고리 구성: {cats}')

    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id'].values

    print(f'\n[1/6] LGBM MAE + log1p')
    lgbm_oof, lgbm_test = train_lgbm_oof(train, test, y_log, groups, sel_feats, topk)

    print(f'\n[2/6] CatBoost MAE + log1p')
    cb_oof, cb_test = train_cb_oof(train, test, y_log, groups, sel_feats, topk)

    print(f'\n[3/6] CatBoost Tweedie 1.5')
    tw_oof, tw_test = train_tw15_oof(train, test, y_raw, groups, sel_feats, topk)

    print(f'\n[4/6] ExtraTrees')
    et_oof, et_test = train_tree_oof(train, test, y_raw, groups, sel_feats, topk, ET_PARAMS, 'et')

    print(f'\n[5/6] RandomForest')
    rf_oof, rf_test = train_tree_oof(train, test, y_raw, groups, sel_feats, topk, RF_PARAMS, 'rf')

    print(f'\n[6/6] Asymmetric MAE α=2.0')
    asym_oof, asym_test = train_asym_oof(train, test, y_log, groups, sel_feats, topk)

    # 상관 행렬
    print(f'\n{"="*60}')
    print(f'Meta Stacking (top-{topk})')
    y_raw_arr = y_raw.values
    oofs = [lgbm_oof, cb_oof, tw_oof, et_oof, rf_oof, asym_oof]
    names = ['lgbm','cb','tw15','et','rf','asym20']
    corr = pd.DataFrame(np.corrcoef(oofs), index=names, columns=names)
    print('상관 행렬:')
    for n in names:
        vals = '  '.join([f'{corr.loc[n,m]:.4f}' for m in names])
        print(f'  {n:8s}: {vals}')

    # 메타 LGBM
    gkf = GroupKFold(n_splits=N_SPLITS)
    meta_X = np.column_stack(oofs)
    meta_oof = np.zeros(len(train))
    meta_test_preds = []
    fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(meta_X, y_raw_arr, groups)):
        Xtr_m = meta_X[tr_i]; Xva_m = meta_X[va_i]
        m = lgb.LGBMRegressor(**META_PARAMS)
        m.fit(Xtr_m, y_raw_arr[tr_i],
              eval_set=[(Xva_m, y_raw_arr[va_i])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        meta_oof[va_i] = m.predict(Xva_m)
        meta_test_preds.append(m.predict(np.column_stack(
            [lgbm_test, cb_test, tw_test, et_test, rf_test, asym_test])))
        mae = np.abs(meta_oof[va_i] - y_raw_arr[va_i]).mean()
        fold_maes.append(mae)
        print(f'  [meta] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    cv_mae   = np.mean(fold_maes)
    meta_test = np.mean(meta_test_preds, axis=0)
    pred_std  = np.std(meta_test)
    pred_mean = np.mean(meta_test)
    pred_max  = np.max(meta_test)

    print(f'\n[model42-top{topk}]')
    print(f'  CV MAE    : {cv_mae:.4f}  (기준 model31: 8.4786 / model41: 8.4851)')
    print(f'  pred_std  : {pred_std:.2f}  (기준 model31: 15.89 / model41: 15.73)')
    print(f'  pred_mean : {pred_mean:.2f}')
    print(f'  pred_max  : {pred_max:.2f}')
    print(f'  피처 수   : {len(sel_feats)}  (model41: 458)')
    folds_str = ' / '.join([f'{m:.4f}' for m in fold_maes])
    print(f'  Fold MAE  : {folds_str}')

    # ⚠️ 경고: pred_std 급락 시 제출 주의
    baseline_std = 15.89
    if pred_std < 14.0:
        print(f'  ⚠️⚠️ pred_std={pred_std:.2f} < 14.0 → 배율 악화 위험 높음, 제출 비권장')
        submit = False
    elif pred_std < 14.8:
        print(f'  ⚠️  pred_std={pred_std:.2f} < 14.8 → 주의 필요. 기준 대비 {pred_std-baseline_std:+.2f}')
        submit = True
    else:
        print(f'  ✅ pred_std={pred_std:.2f} ≥ 14.8 → 배율 정상 범위, 제출 권장')
        submit = True

    # 극값 분석
    bins = [0, 5, 20, 50, 80, 800]
    print(f'\n[극값 분석] 구간별 OOF MAE:')
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (y_raw_arr >= lo) & (y_raw_arr < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(meta_oof[mask] - y_raw_arr[mask]).mean()
            pr = meta_oof[mask].mean() / (y_raw_arr[mask].mean() + 1e-8)
            print(f'  [{lo:3d},{hi:4d}): n={mask.sum():5d}  MAE={seg_mae:.2f}  pred/actual={pr:.3f}')

    # 제출 파일 생성
    os.makedirs(SUB_DIR, exist_ok=True)
    sub_name = f'model42_top{topk}_cv{cv_mae:.4f}.csv'
    sub_path = os.path.join(SUB_DIR, sub_name)
    sub = pd.DataFrame({'ID': test['ID'], 'avg_delay_minutes_next_30m': meta_test})
    sub.to_csv(sub_path, index=False)
    print(f'\n  제출 파일: {sub_name}')
    if not submit:
        print(f'  ⚠️  pred_std 기준 미달 — 제출 전 pred_std 재확인 필요')

    return {'topk': topk, 'cv_mae': cv_mae, 'pred_std': pred_std,
            'pred_mean': pred_mean, 'fold_maes': fold_maes}


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    t_total = time.time()
    os.makedirs(SUB_DIR, exist_ok=True)

    train, test = load_and_prepare_data()
    feat_cols_all = get_feat_cols(train)
    print(f'\n전체 피처 수: {len(feat_cols_all)}')

    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id'].values

    # ── Phase 1: 피처 중요도 산출 ──
    imp_path = os.path.join(CKPT_DIR, 'feature_importance.csv')
    if os.path.exists(imp_path):
        print(f'\n[Phase 1] 기존 importance 파일 로드: {imp_path}')
        importance_df = pd.read_csv(imp_path)
    else:
        importance_df = compute_quick_importance(train, y_log, groups, feat_cols_all)

    # ── Phase 2: Top-K 실험 ──
    results = []
    for topk in TOPK_LIST:
        r = run_topk_experiment(topk, importance_df, train, test, feat_cols_all)
        results.append(r)

    # ── 최종 비교 ──
    print(f'\n{"="*60}')
    print(f'최종 비교')
    print(f'{"="*60}')
    print(f'  {"모델":20s}  {"CV":8s}  {"pred_std":10s}')
    print(f'  {"model31 (baseline)":20s}  {"8.4786":8s}  {"15.89":10s}')
    print(f'  {"model41 (458 feat)":20s}  {"8.4851":8s}  {"15.73":10s}')
    for r in results:
        status = '✅' if r['pred_std'] >= 14.8 else '⚠️'
        label = f'model42-top{r["topk"]}'
        print(f'  {label:20s}  {r["cv_mae"]:.4f}    {r["pred_std"]:.2f}  {status}')

    elapsed_total = (time.time() - t_total) / 60
    print(f'\n총 소요 시간: {elapsed_total:.1f}분')
