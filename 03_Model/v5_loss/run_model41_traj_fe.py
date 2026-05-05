"""
model41: 궤적 형상(Trajectory Shape) 피처 통합
================================================================
v6 핵심 전략 — 미탐색 시간 축 동적 피처 추가

현재 sc_agg의 공백:
  mean/std/max/min/p10/p90/skew/kurtosis/cv → 정적 분포 통계만 커버
  ❌ slope(추세), fl_ratio(성장방향), peak_pos(극값 위치),
     above_count(이벤트빈도), mono(단조성) → 전부 미탐색

v6 피처 (29종 신규):
  A. slope × 8       — linear trend over ts 0-24 (r=-0.387 확인됨)
  B. fl_ratio × 8    — last5_mean / first5_mean (성장 방향)
  C. peak_pos × 5    — argmax normalized (극값 발생 시점)
  D. above_count × 5 — threshold crossing (이벤트 빈도)
  E. mono × 3        — 단조증가 비율 (방향 일관성)

base: model31 (429 feat) → model41 (458 feat, +29 궤적 피처)
모델: 5모델 스태킹 (LGBM+CB+TW1.5+ET+RF → LGBM-meta)
     + Asym2.0 = 6모델 (model33 구성과 동일 + 새 피처)

기준: blend_m33m34_w80 Public 9.8073 / 배율 1.1564
목표: CV < 8.47 OR pred_std ≥ 15.5 → 즉시 제출

실행: python src/run_model41_traj_fe.py
예상 시간: ~30분 (전체 재학습, 체크포인트 없음)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
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
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model41_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# ── 파라미터 (model30/31 기준, 피처만 변경) ──
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
ET_PARAMS  = {'n_estimators': 500, 'max_depth': 20, 'min_samples_leaf': 5,
               'max_features': 0.7, 'random_state': RANDOM_STATE, 'n_jobs': -1}
RF_PARAMS  = {'n_estimators': 500, 'max_depth': 20, 'min_samples_leaf': 5,
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

# ── 시나리오 집계 대상 컬럼 ──
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

# ── 궤적 피처 대상 컬럼 ──
TRAJ_COLS = [
    'robot_utilization',   # slope r=-0.387 확인
    'order_inflow_15m',    # 극값 시나리오 최강 구분자
    'congestion_score',    # 극값 시나리오 2위
    'low_battery_ratio',   # 극값 시나리오 3위
    'battery_mean',        # 핵심 상관 피처
    'charge_queue_length', # 배터리 위기와 연동
    'robot_idle',          # 역방향 신호
    'max_zone_density',    # 공간 압박
]
# Category C/D/E 대상 (극값 직접 관련 5종)
PEAK_COLS  = ['order_inflow_15m', 'congestion_score',
              'low_battery_ratio', 'charge_queue_length', 'max_zone_density']
MONO_COLS  = ['robot_utilization', 'congestion_score', 'order_inflow_15m']


# ═══════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════

def add_scenario_agg_features(df):
    """기존 sc_agg (11통계 × 18컬럼 = 198 피처)"""
    df = df.copy()
    for col in SC_AGG_COLS:
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
        df[f'sc_{col}_cv']       = (
            df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)
        ).fillna(0)
    return df


def add_trajectory_features(df):
    """
    v6 신규: 궤적 형상 피처 (29종)
    ───────────────────────────────
    A. slope      × 8  — 선형 추세 기울기
    B. fl_ratio   × 8  — last5 / first5 평균 비율
    C. peak_pos   × 5  — argmax 정규화 위치 (0=초반, 1=후반)
    D. above_cnt  × 5  — 시나리오 내 mean+0.5σ 초과 횟수
    E. mono       × 3  — 단조증가 비율

    모두 시나리오 전체 25행 기준 → broadcast (leakage 없음)
    """
    df = df.copy()

    # 미리 ts_idx 확인 (없으면 생성)
    if 'ts_idx' not in df.columns:
        df['ts_idx'] = df.groupby('scenario_id').cumcount()

    ts_arr = np.arange(25, dtype=np.float64)  # [0..24]

    # ── A: Slope (선형 추세) ──────────────────────────────
    print('  [traj] A. slope 계산 중...')
    for col in TRAJ_COLS:
        if col not in df.columns:
            continue
        # 시나리오별 polyfit(1차) 기울기
        slope_map = (
            df.groupby('scenario_id')[col]
            .apply(lambda x: np.polyfit(ts_arr[:len(x)], x.fillna(x.mean()).values, 1)[0]
                   if len(x) > 1 else 0.0)
            .fillna(0)
        )
        df[f'sc_{col}_slope'] = df['scenario_id'].map(slope_map)

    # ── B: First5 / Last5 비율 ────────────────────────────
    print('  [traj] B. fl_ratio 계산 중...')
    for col in TRAJ_COLS:
        if col not in df.columns:
            continue
        # first5: ts_idx 0~4, last5: ts_idx 20~24
        first5 = (df[df['ts_idx'] < 5]
                  .groupby('scenario_id')[col].mean()
                  .rename('first5'))
        last5  = (df[df['ts_idx'] >= 20]
                  .groupby('scenario_id')[col].mean()
                  .rename('last5'))
        fl = (last5 / (first5.abs() + 1e-8)).fillna(1.0).replace([np.inf, -np.inf], 1.0)
        df[f'sc_{col}_fl_ratio'] = df['scenario_id'].map(fl)

    # ── C: Peak Position ──────────────────────────────────
    print('  [traj] C. peak_pos 계산 중...')
    for col in PEAK_COLS:
        if col not in df.columns:
            continue
        # 각 시나리오에서 max가 발생한 ts_idx (0~24 → /24로 정규화)
        peak_map = (
            df.groupby('scenario_id')
            .apply(lambda g: g.loc[g[col].fillna(-np.inf).idxmax(), 'ts_idx'] / 24.0
                   if col in g.columns else 0.5)
            .fillna(0.5)
        )
        df[f'sc_{col}_peak_pos'] = df['scenario_id'].map(peak_map)

    # ── D: Above Threshold Count ──────────────────────────
    print('  [traj] D. above_count 계산 중...')
    for col in PEAK_COLS:
        if col not in df.columns:
            continue
        sc_mean_col = f'sc_{col}_mean'
        sc_std_col  = f'sc_{col}_std'
        if sc_mean_col not in df.columns or sc_std_col not in df.columns:
            continue
        threshold = df[sc_mean_col] + 0.5 * df[sc_std_col]
        above_flag = (df[col].fillna(0) > threshold).astype(int)
        above_map = above_flag.groupby(df['scenario_id']).sum()
        df[f'sc_{col}_above_cnt'] = df['scenario_id'].map(above_map).fillna(0)

    # ── E: Monotonicity Score ─────────────────────────────
    print('  [traj] E. mono 계산 중...')
    for col in MONO_COLS:
        if col not in df.columns:
            continue
        def _mono(x):
            vals = x.fillna(x.mean()).values
            if len(vals) < 2:
                return 0.5
            diffs = np.diff(vals)
            return float((diffs > 0).sum()) / len(diffs)

        mono_map = df.groupby('scenario_id')[col].apply(_mono).fillna(0.5)
        df[f'sc_{col}_mono'] = df['scenario_id'].map(mono_map)

    # 통계 출력
    new_traj_cols = [c for c in df.columns
                     if any(c.endswith(s) for s in
                            ['_slope', '_fl_ratio', '_peak_pos', '_above_cnt', '_mono'])]
    print(f'  [traj] 생성된 궤적 피처: {len(new_traj_cols)}종')
    return df


def safe_div(a, b, fill=0):
    return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)


def add_ratio_features(df):
    """비율 피처 (model29A Tier 1+2, 총 12종)"""
    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = safe_div(df['sc_order_inflow_15m_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = safe_div(df['sc_congestion_score_mean'], df['intersection_count'])
    if all(c in df.columns for c in ['sc_low_battery_ratio_mean', 'sc_charge_queue_length_mean', 'charger_count']):
        df['ratio_battery_stress'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df['sc_charge_queue_length_mean'], df['charger_count'])
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = safe_div(df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']
    if all(c in df.columns for c in ['sc_congestion_score_mean', 'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'], df['robot_total'] ** 2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm'] / 100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(df['pack_station_count'], df['floor_area_sqm'] / 1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if all(c in df.columns for c in ['sc_battery_mean_mean', 'sc_robot_utilization_mean', 'charger_count']):
        df['ratio_battery_per_robot'] = safe_div(
            df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'], df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    return df


def add_shift_safe_fe(df):
    """model31 shift-safe 7종 (cross-ratio 내 원본 피처)"""
    for df_ in [df]:
        if 'robot_utilization' in df_.columns and 'order_inflow_15m' in df_.columns:
            df_['feat_util_x_order'] = df_['robot_utilization'] * df_['order_inflow_15m']
        if 'low_battery_ratio' in df_.columns and 'congestion_score' in df_.columns:
            df_['feat_batt_x_cong'] = df_['low_battery_ratio'] * df_['congestion_score']
        if 'charge_queue_length' in df_.columns and 'charger_count' in df_.columns:
            df_['feat_queue_per_charger'] = safe_div(
                df_['charge_queue_length'].fillna(0), df_['charger_count'])
        if 'robot_idle' in df_.columns and 'robot_utilization' in df_.columns:
            df_['feat_idle_util_ratio'] = safe_div(
                df_['robot_idle'].fillna(0), df_['robot_utilization'].fillna(0) + 1e-8)
        if 'order_inflow_15m' in df_.columns and 'congestion_score' in df_.columns:
            df_['feat_order_cong'] = df_['order_inflow_15m'].fillna(0) * df_['congestion_score'].fillna(0)
        if 'battery_mean' in df_.columns and 'low_battery_ratio' in df_.columns:
            df_['feat_batt_risk'] = (100 - df_['battery_mean'].fillna(100)) * df_['low_battery_ratio'].fillna(0)
        if 'max_zone_density' in df_.columns and 'order_inflow_15m' in df_.columns:
            df_['feat_density_order'] = df_['max_zone_density'].fillna(0) * df_['order_inflow_15m'].fillna(0)
    return df


def load_and_prepare_data():
    """데이터 로드 + 전체 FE 파이프라인"""
    t0 = time.time()
    print('데이터 로드 중...')
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    # 기본 FE (lag/rolling/ts/layout)
    print('기본 FE (lag+rolling+ts+layout)...')
    train, test = build_features(train, test, layout,
                                 lag_lags=[1,2,3,4,5,6],
                                 rolling_windows=[3,5,10])

    # 시나리오 집계 (기존 11통계)
    print('sc_agg 피처 추가...')
    train = add_scenario_agg_features(train)
    test  = add_scenario_agg_features(test)

    # 비율 피처 (12종)
    print('비율 피처 추가...')
    train = add_ratio_features(train)
    test  = add_ratio_features(test)

    # shift-safe cross 피처 (7종)
    print('shift-safe 피처 추가...')
    train = add_shift_safe_fe(train)
    test  = add_shift_safe_fe(test)

    # ★ v6 신규: 궤적 형상 피처 (29종)
    print('[v6] 궤적 형상 피처 추가 (train)...')
    train = add_trajectory_features(train)
    print('[v6] 궤적 형상 피처 추가 (test)...')
    test  = add_trajectory_features(test)

    elapsed = time.time() - t0
    print(f'FE 완료: train {train.shape}, test {test.shape}, {elapsed:.1f}s')
    return train, test


def get_feat_cols(df):
    return [c for c in df.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and df[c].dtype != object]


# ═══════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════
# Asymmetric Loss
# ═══════════════════════════════════════════════════════

def asym_obj(y_pred, dtrain):
    y_true = dtrain.get_label()
    residual = y_true - y_pred
    grad = np.where(residual > 0, -ASYM_ALPHA, 1.0)
    hess = np.ones_like(y_pred)
    return grad, hess

def asym_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    mae = np.abs(np.expm1(y_pred) - np.expm1(y_true)).mean()
    return 'asym_mae', mae, False


# ═══════════════════════════════════════════════════════
# Base Learner 학습
# ═══════════════════════════════════════════════════════

def train_lgbm_oof(X_tr, X_te, y_log, groups, feat_cols, name='lgbm'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt = X_tr[feat_cols].fillna(0); Xte = X_te[feat_cols].fillna(0)
    fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt, y_log, groups)):
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(Xt.iloc[tr_i], y_log.iloc[tr_i],
              eval_set=[(Xt.iloc[va_i], y_log.iloc[va_i])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_i] = np.expm1(m.predict(Xt.iloc[va_i]))
        test_pred += np.expm1(m.predict(Xte)) / N_SPLITS
        raw_oof = m.predict(Xt.iloc[va_i])
        y_raw_va = np.expm1(y_log.iloc[va_i].values)
        mae = np.abs(oof[va_i] - y_raw_va).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    print(f'  [{name}] OOF MAE = {np.mean(fold_maes):.4f}')
    return oof, test_pred


def train_cb_oof(X_tr, X_te, y_log, groups, feat_cols, name='cb'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt = X_tr[feat_cols].fillna(0).values; Xte = X_te[feat_cols].fillna(0).values
    y_log_arr = y_log.values
    fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt, y_log_arr, groups)):
        m = cb.CatBoostRegressor(**CB_PARAMS)
        m.fit(cb.Pool(Xt[tr_i], y_log_arr[tr_i]),
              eval_set=cb.Pool(Xt[va_i], y_log_arr[va_i]),
              use_best_model=True)
        oof[va_i] = np.expm1(m.predict(Xt[va_i]))
        test_pred += np.expm1(m.predict(Xte)) / N_SPLITS
        y_raw_va = np.expm1(y_log_arr[va_i])
        mae = np.abs(oof[va_i] - y_raw_va).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    print(f'  [{name}] OOF MAE = {np.mean(fold_maes):.4f}')
    return oof, test_pred


def train_tw15_oof(X_tr, X_te, y_raw, groups, feat_cols, name='tw15'):
    """CatBoost Tweedie 1.5 (raw space)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt = X_tr[feat_cols].fillna(0).values; Xte = X_te[feat_cols].fillna(0).values
    y_arr = y_raw.values
    fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt, y_arr, groups)):
        m = cb.CatBoostRegressor(**TW15_PARAMS)
        m.fit(cb.Pool(Xt[tr_i], y_arr[tr_i]),
              eval_set=cb.Pool(Xt[va_i], y_arr[va_i]),
              use_best_model=True)
        oof[va_i] = m.predict(Xt[va_i])
        test_pred += m.predict(Xte) / N_SPLITS
        mae = np.abs(oof[va_i] - y_arr[va_i]).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    print(f'  [{name}] OOF MAE = {np.mean(fold_maes):.4f}')
    return oof, test_pred


def train_tree_oof(X_tr, X_te, y_raw, groups, feat_cols, params, name):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt = X_tr[feat_cols].fillna(0).values; Xte = X_te[feat_cols].fillna(0).values
    y_arr = y_raw.values
    ModelClass = ExtraTreesRegressor if name == 'et' else RandomForestRegressor
    fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt, y_arr, groups)):
        m = ModelClass(**params)
        m.fit(Xt[tr_i], y_arr[tr_i])
        oof[va_i] = m.predict(Xt[va_i])
        test_pred += m.predict(Xte) / N_SPLITS
        mae = np.abs(oof[va_i] - y_arr[va_i]).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    print(f'  [{name}] OOF MAE = {np.mean(fold_maes):.4f}')
    return oof, test_pred


def train_asym_oof(X_tr, X_te, y_log, groups, feat_cols, name='asym20'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt = X_tr[feat_cols].fillna(0); Xte = X_te[feat_cols].fillna(0)
    y_log_arr = y_log
    fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt, y_log_arr, groups)):
        dtrain = lgb.Dataset(Xt.iloc[tr_i], label=y_log_arr.iloc[tr_i].values)
        dval   = lgb.Dataset(Xt.iloc[va_i], label=y_log_arr.iloc[va_i].values,
                             reference=dtrain)
        params = {k: v for k, v in ASYM_PARAMS.items() if k != 'n_estimators'}
        params['objective'] = asym_obj
        bst = lgb.train(
            params, dtrain, num_boost_round=ASYM_PARAMS['n_estimators'],
            valid_sets=[dval],
            feval=asym_metric,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        oof[va_i] = np.expm1(bst.predict(Xt.iloc[va_i]))
        test_pred += np.expm1(bst.predict(Xte)) / N_SPLITS
        y_raw_va = np.expm1(y_log_arr.iloc[va_i].values)
        mae = np.abs(oof[va_i] - y_raw_va).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}  iter={bst.best_iteration}')
        del bst; gc.collect()
    print(f'  [{name}] OOF MAE = {np.mean(fold_maes):.4f}')
    return oof, test_pred


# ═══════════════════════════════════════════════════════
# Meta Stacking
# ═══════════════════════════════════════════════════════

def run_meta_stacking(oofs: dict, tests: dict, y_raw, groups):
    """LGBM-meta 5-fold"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    names = list(oofs.keys())
    meta_tr = np.column_stack([oofs[n] for n in names])
    meta_te = np.column_stack([tests[n] for n in names])

    # OOF 상관 출력
    print('\n[meta] OOF 상관행렬:')
    corr = np.corrcoef(meta_tr.T)
    for i, ni in enumerate(names):
        row = '  ' + ni + ': ' + ' '.join(f'{corr[i,j]:.4f}' for j in range(len(names)))
        print(row)

    meta_oof = np.zeros(len(y_raw)); meta_test = np.zeros(len(meta_te))
    fold_maes = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(meta_tr, y_raw.values, groups)):
        m = lgb.LGBMRegressor(**META_PARAMS)
        m.fit(meta_tr[tr_i], y_raw.values[tr_i],
              eval_set=[(meta_tr[va_i], y_raw.values[va_i])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        meta_oof[va_i] = m.predict(meta_tr[va_i])
        meta_test += m.predict(meta_te) / N_SPLITS
        mae = np.abs(meta_oof[va_i] - y_raw.values[va_i]).mean()
        fold_maes.append(mae)
        print(f'  [meta] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    meta_test = meta_test.clip(min=0)
    print(f'\n[meta] CV MAE = {np.mean(fold_maes):.4f}  '
          f'(folds: {" / ".join(f"{m:.4f}" for m in fold_maes)})')
    print(f'[meta] pred_std  = {meta_test.std():.2f}')
    print(f'[meta] pred_mean = {meta_test.mean():.2f}')
    print(f'[meta] pred_max  = {meta_test.max():.2f}')
    return meta_oof, meta_test, fold_maes


# ═══════════════════════════════════════════════════════
# 극값 구간 분석
# ═══════════════════════════════════════════════════════

def analyze_extreme(oof, y_raw):
    bins = [(0,5), (5,20), (20,50), (50,80), (80,800)]
    print('\n[극값 분석] 구간별 OOF MAE:')
    for lo, hi in bins:
        mask = (y_raw >= lo) & (y_raw < hi)
        if mask.sum() == 0: continue
        mae = np.abs(oof[mask] - y_raw[mask]).mean()
        ratio = oof[mask].mean() / (y_raw[mask].mean() + 1e-8)
        n = mask.sum()
        print(f'  [{lo:>3},{hi:>4}): n={n:>5}  MAE={mae:.2f}  pred/actual={ratio:.3f}')


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    t_start = time.time()
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── 데이터 준비 ──
    train, test = load_and_prepare_data()
    feat_cols = get_feat_cols(train)
    print(f'\n총 피처 수: {len(feat_cols)}  (model31 기준 +29 궤적 피처 예상)')

    TARGET  = 'avg_delay_minutes_next_30m'
    y_raw   = train[TARGET]
    y_log   = np.log1p(y_raw)
    groups  = train['scenario_id']

    # 궤적 피처만 별도 출력
    traj_feats = [c for c in feat_cols
                  if any(c.endswith(s) for s in
                         ['_slope', '_fl_ratio', '_peak_pos', '_above_cnt', '_mono'])]
    print(f'궤적 피처 목록 ({len(traj_feats)}종): {traj_feats[:5]}...')

    # ── Base Learners ──
    oofs = {}; tests_pred = {}

    # 1. LGBM (MAE + log1p)
    print('\n[1/6] LGBM MAE + log1p')
    if ckpt_exists('lgbm'):
        oofs['lgbm'], tests_pred['lgbm'] = load_ckpt('lgbm')
        print('  → 체크포인트 로드')
    else:
        oofs['lgbm'], tests_pred['lgbm'] = train_lgbm_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('lgbm', oofs['lgbm'], tests_pred['lgbm'])

    # 2. CatBoost (MAE + log1p)
    print('\n[2/6] CatBoost MAE + log1p')
    if ckpt_exists('cb'):
        oofs['cb'], tests_pred['cb'] = load_ckpt('cb')
        print('  → 체크포인트 로드')
    else:
        oofs['cb'], tests_pred['cb'] = train_cb_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('cb', oofs['cb'], tests_pred['cb'])

    # 3. CatBoost Tweedie 1.5 (raw space)
    print('\n[3/6] CatBoost Tweedie 1.5 (raw)')
    if ckpt_exists('tw15'):
        oofs['tw15'], tests_pred['tw15'] = load_ckpt('tw15')
        print('  → 체크포인트 로드')
    else:
        oofs['tw15'], tests_pred['tw15'] = train_tw15_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt('tw15', oofs['tw15'], tests_pred['tw15'])

    # 4. ExtraTrees
    print('\n[4/6] ExtraTrees')
    if ckpt_exists('et'):
        oofs['et'], tests_pred['et'] = load_ckpt('et')
        print('  → 체크포인트 로드')
    else:
        oofs['et'], tests_pred['et'] = train_tree_oof(train, test, y_raw, groups, feat_cols, ET_PARAMS, 'et')
        save_ckpt('et', oofs['et'], tests_pred['et'])

    # 5. RandomForest
    print('\n[5/6] RandomForest')
    if ckpt_exists('rf'):
        oofs['rf'], tests_pred['rf'] = load_ckpt('rf')
        print('  → 체크포인트 로드')
    else:
        oofs['rf'], tests_pred['rf'] = train_tree_oof(train, test, y_raw, groups, feat_cols, RF_PARAMS, 'rf')
        save_ckpt('rf', oofs['rf'], tests_pred['rf'])

    # 6. Asymmetric MAE α=2.0
    print('\n[6/6] Asymmetric MAE α=2.0')
    if ckpt_exists('asym20'):
        oofs['asym20'], tests_pred['asym20'] = load_ckpt('asym20')
        print('  → 체크포인트 로드')
    else:
        oofs['asym20'], tests_pred['asym20'] = train_asym_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('asym20', oofs['asym20'], tests_pred['asym20'])

    # ── Meta Stacking ──
    print('\n' + '='*60)
    print('Meta Stacking (LGBM-meta, 6모델)')
    meta_oof, meta_test, fold_maes = run_meta_stacking(oofs, tests_pred, y_raw, groups)
    cv_mae = np.mean(fold_maes)

    # ── 극값 분석 ──
    analyze_extreme(meta_oof, y_raw.values)

    # ── 제출 파일 저장 ──
    sub = pd.read_csv(os.path.join(DATA_DIR, '..', 'data', 'sample_submission.csv')
                      if os.path.exists(os.path.join(DATA_DIR, 'sample_submission.csv'))
                      else os.path.join(DATA_DIR, 'sample_submission.csv'))
    sub_path = os.path.join(DATA_DIR, '..', 'submissions')
    os.makedirs(sub_path, exist_ok=True)

    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = meta_test
    fname = f'model41_traj_fe_cv{cv_mae:.4f}.csv'
    fpath = os.path.join(SUB_DIR, fname)
    sample.to_csv(fpath, index=False)

    total_time = (time.time() - t_start) / 60
    print(f'\n{"="*60}')
    print(f'[model41] 완료!')
    print(f'  CV MAE    : {cv_mae:.4f}  (기준 model31: 8.4786)')
    print(f'  pred_std  : {meta_test.std():.2f}  (기준 model31: 15.89)')
    print(f'  pred_mean : {meta_test.mean():.2f}')
    print(f'  피처 수   : {len(feat_cols)}  (model31: 429)')
    print(f'  소요 시간 : {total_time:.1f}분')
    print(f'  제출 파일 : {fname}')
    print(f'\n[판단 기준]')
    if cv_mae < 8.4786:
        print(f'  ✅ CV 개선 (Δ{cv_mae - 8.4786:+.4f}) → 즉시 제출 권장')
    elif meta_test.std() >= 15.5:
        print(f'  ⚠️  CV 악화이나 pred_std={meta_test.std():.2f} ≥ 15.5 → model29A 패턴, 제출 시도')
    else:
        print(f'  ❌ CV 악화 + pred_std={meta_test.std():.2f} < 15.5 → Phase 1B (피처 절반) 시도')


if __name__ == '__main__':
    main()
