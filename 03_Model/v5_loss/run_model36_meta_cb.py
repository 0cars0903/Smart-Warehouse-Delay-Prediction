"""
전략 2: CatBoost / XGBoost 메타 러너 실험 (model36)
================================================================
근거:
  - 현재 메타 러너 LGBM 고정 → CB/XGB 교체로 극값 처리 차이 활용
  - OOF 체크포인트(model31+model34) 재사용 → base learner 재학습 불필요
  - 메타 러너 3종 비교 + 블렌드 → 최적 조합 탐색

모델 구성:
  [Layer 1] model31 ckpt: LGBM, CB, ET, RF (log1p space)
             model34 ckpt: TW1.5 (raw), Asym2.0 (log1p)
  [Layer 2] Meta-A: LGBM (기준 재현)
             Meta-B: CatBoost
             Meta-C: XGBoost
             Meta-D: 블렌드 탐색

실행: python src/run_model36_meta_cb.py
예상 시간: ~5분 (데이터 로드 + 메타 3종)
※ USER 로컬 실행 권장 (체크포인트 정합성)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
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

TARGET = 'avg_delay_minutes_next_30m'

# ── 메타 파라미터 ──
META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

META_CB_PARAMS = {
    'depth': 6,
    'learning_rate': 0.05,
    'iterations': 500,
    'l2_leaf_reg': 3.0,
    'random_seed': RANDOM_STATE,
    'loss_function': 'MAE',
    'verbose': 0,
    'thread_count': -1,
}

META_XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:absoluteerror',
    'random_state': RANDOM_STATE,
    'verbosity': 0,
    'n_jobs': -1,
}


# ── 체크포인트 유틸 ──
def load_ckpt(d, name):
    return (np.load(os.path.join(d, f'{name}_oof.npy')),
            np.load(os.path.join(d, f'{name}_test.npy')))

def ckpt_exists(d, name):
    return (os.path.exists(os.path.join(d, f'{name}_oof.npy')) and
            os.path.exists(os.path.join(d, f'{name}_test.npy')))


# ── 메타 학습기: LGBM ──
def run_meta_lgbm(meta_train, meta_test, y_raw, groups, label='LGBM-meta'):
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


# ── 메타 학습기: CatBoost ──
def run_meta_cb(meta_train, meta_test, y_raw, groups, label='CB-meta'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y_raw)); test_pred = np.zeros(meta_test.shape[0])
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        train_pool = cb.Pool(meta_train[tr_idx], label=np.log1p(y_raw.iloc[tr_idx].values))
        val_pool   = cb.Pool(meta_train[va_idx], label=np.log1p(y_raw.iloc[va_idx].values))
        m = cb.CatBoostRegressor(**META_CB_PARAMS)
        m.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)
        oof[va_idx] = np.expm1(m.predict(meta_train[va_idx]))
        test_pred += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m, train_pool, val_pool; gc.collect()
    oof_mae = np.abs(oof - y_raw.values).mean()
    test_std = np.maximum(test_pred, 0).std()
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | test_std={test_std:.2f}')
    return oof, test_pred, oof_mae


# ── 메타 학습기: XGBoost ──
def run_meta_xgb(meta_train, meta_test, y_raw, groups, label='XGB-meta'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y_raw)); test_pred = np.zeros(meta_test.shape[0])
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        m = xgb.XGBRegressor(**META_XGB_PARAMS)
        m.fit(meta_train[tr_idx], np.log1p(y_raw.iloc[tr_idx].values),
              eval_set=[(meta_train[va_idx], np.log1p(y_raw.iloc[va_idx].values))],
              verbose=False)
        oof[va_idx] = np.expm1(m.predict(meta_train[va_idx]))
        test_pred += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.iloc[va_idx].values).mean()
        best_iter = getattr(m, 'best_iteration', META_XGB_PARAMS['n_estimators'])
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={best_iter}')
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


def main():
    t0 = time.time()
    print('=' * 70)
    print('model36: CatBoost / XGBoost 메타 러너 실험')
    print('  기준: model34-B LGBM-meta CV=8.4803 / Public=9.8078')
    print('  목표: 메타 러너 교체로 CV/배율 개선')
    print('=' * 70)

    os.makedirs(SUB_DIR, exist_ok=True)

    # ── 데이터 로드 (FE 파이프라인 포함) ──
    print('\n[데이터 로드] build_features + scenario agg + ratios')
    train, test = load_data()
    y_raw = train[TARGET]
    groups = train['scenario_id']
    print(f'  train: {len(train)}, test: {len(test)}')

    # ── OOF 체크포인트 로드 ──
    print('\n' + '─' * 70)
    print('[Layer 1] OOF 체크포인트 로드')
    print('─' * 70)

    oof_dict = {}
    test_dict = {}

    # model31 체크포인트: LGBM, CB, ET, RF (log1p space)
    for name in ['lgbm', 'cb', 'et', 'rf']:
        if ckpt_exists(CKPT_31, name):
            oof_dict[name], test_dict[name] = load_ckpt(CKPT_31, name)
            mae = np.abs(np.expm1(oof_dict[name]) - y_raw.values).mean()
            print(f'  [{name.upper():>6s}] ckpt → OOF MAE={mae:.4f}')
        else:
            print(f'  ⚠️ [{name.upper()}] 체크포인트 없음!'); return

    # model34 체크포인트: TW1.5 (raw), Asym2.0 (log1p)
    if ckpt_exists(CKPT_34, 'tw15'):
        oof_dict['tw15'], test_dict['tw15'] = load_ckpt(CKPT_34, 'tw15')
        mae = np.abs(oof_dict['tw15'] - y_raw.values).mean()
        print(f'  [  TW15] ckpt → OOF MAE={mae:.4f}')
    else:
        print(f'  ⚠️ [TW15] 체크포인트 없음!'); return

    if ckpt_exists(CKPT_34, 'asym20'):
        oof_dict['asym20'], test_dict['asym20'] = load_ckpt(CKPT_34, 'asym20')
        mae = np.abs(np.expm1(oof_dict['asym20']) - y_raw.values).mean()
        print(f'  [ASY20] ckpt → OOF MAE={mae:.4f}')
    else:
        print(f'  ⚠️ [ASYM20] 체크포인트 없음!'); return

    # ── 체크포인트 정합성 검증 ──
    lgbm_mae = np.abs(np.expm1(oof_dict['lgbm']) - y_raw.values).mean()
    if lgbm_mae > 10.0:
        print(f'\n  ⚠️ LGBM OOF MAE={lgbm_mae:.2f} 비정상! 체크포인트 불일치 의심')
        print(f'     기대값: ~8.5. build_features 후 행 순서와 체크포인트 확인 필요')
        print(f'     중단합니다.')
        return
    print(f'\n  ✅ 체크포인트 정합성 확인 (LGBM MAE={lgbm_mae:.4f})')

    # ── 메타 입력 구성 (6모델: model34-B 동일) ──
    print('\n' + '─' * 70)
    print('[메타 입력 구성] 6모델 (LGBM, CB, TW1.5, ET, RF, Asym2.0)')
    print('─' * 70)

    meta_names = ['lgbm', 'cb', 'tw15', 'et', 'rf', 'asym20']
    meta_oof_list = []
    meta_test_list = []

    for name in meta_names:
        if name == 'tw15':
            meta_oof_list.append(np.log1p(np.maximum(oof_dict[name], 0)))
            meta_test_list.append(np.log1p(np.maximum(test_dict[name], 0)))
        else:
            meta_oof_list.append(oof_dict[name])
            meta_test_list.append(test_dict[name])

    meta_train = np.column_stack(meta_oof_list)
    meta_test  = np.column_stack(meta_test_list)
    print(f'  메타 입력: {meta_train.shape[1]}개 ({", ".join(meta_names)})')

    # ══════════════════════════════════════════════════════════════
    # Meta-A: LGBM (기준, model34-B 재현)
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('[Meta-A] LGBM 메타 (model34-B 재현)')
    print('=' * 70)
    oof_lgbm, test_lgbm, mae_lgbm = run_meta_lgbm(
        meta_train, meta_test, y_raw, groups)
    segment_analysis(np.maximum(oof_lgbm, 0), y_raw.values, 'LGBM-meta')

    # ══════════════════════════════════════════════════════════════
    # Meta-B: CatBoost
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('[Meta-B] CatBoost 메타')
    print('=' * 70)
    oof_cb, test_cb, mae_cb = run_meta_cb(
        meta_train, meta_test, y_raw, groups)
    segment_analysis(np.maximum(oof_cb, 0), y_raw.values, 'CB-meta')

    # ══════════════════════════════════════════════════════════════
    # Meta-C: XGBoost
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('[Meta-C] XGBoost 메타')
    print('=' * 70)
    oof_xgb, test_xgb, mae_xgb = run_meta_xgb(
        meta_train, meta_test, y_raw, groups)
    segment_analysis(np.maximum(oof_xgb, 0), y_raw.values, 'XGB-meta')

    # ── 메타 간 상관 ──
    print('\n' + '─' * 70)
    print('[메타 간 상관 분석]')
    print('─' * 70)
    corr_lc = np.corrcoef(oof_lgbm, oof_cb)[0, 1]
    corr_lx = np.corrcoef(oof_lgbm, oof_xgb)[0, 1]
    corr_cx = np.corrcoef(oof_cb, oof_xgb)[0, 1]
    print(f'  LGBM-CB:  {corr_lc:.4f}')
    print(f'  LGBM-XGB: {corr_lx:.4f}')
    print(f'  CB-XGB:   {corr_cx:.4f}')

    # ══════════════════════════════════════════════════════════════
    # Meta-D: 블렌드 탐색
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('[Meta-D] 메타 블렌드 탐색')
    print('=' * 70)

    sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    # 단독 제출
    for name_str, test_arr, mae_val in [
        ('model36_meta_lgbm', test_lgbm, mae_lgbm),
        ('model36_meta_cb',   test_cb,   mae_cb),
        ('model36_meta_xgb',  test_xgb,  mae_xgb),
    ]:
        out = sub.copy()
        out[TARGET] = np.maximum(test_arr, 0)
        out.to_csv(os.path.join(SUB_DIR, f'{name_str}.csv'), index=False)
        t_std = np.maximum(test_arr, 0).std()
        t_mean = np.maximum(test_arr, 0).mean()
        print(f'  {name_str}: CV={mae_val:.4f}, mean={t_mean:.2f}, std={t_std:.2f}')

    # LGBM × CB 블렌드
    best_blend_mae = 999
    best_blend_file = ''

    print(f'\n  [LGBM × CB 블렌드]')
    print(f'  {"LGBM%":>6s}  {"CV MAE":>8s}  {"test_std":>8s}  {"test_max":>8s}  파일')
    for wl in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        wc = 1 - wl
        b_oof = wl * oof_lgbm + wc * oof_cb
        b_test = np.maximum(wl * test_lgbm + wc * test_cb, 0)
        b_mae = np.abs(b_oof - y_raw.values).mean()
        fname = f'model36_blend_lc_{int(wl*100)}.csv'
        out = sub.copy(); out[TARGET] = b_test
        out.to_csv(os.path.join(SUB_DIR, fname), index=False)
        flag = ' ✅' if b_mae < best_blend_mae else ''
        if b_mae < best_blend_mae: best_blend_mae = b_mae; best_blend_file = fname
        print(f'  {wl:>6.0%}  {b_mae:>8.4f}  {b_test.std():>8.2f}  {b_test.max():>8.2f}  {fname}{flag}')

    # LGBM × XGB 블렌드
    print(f'\n  [LGBM × XGB 블렌드]')
    for wl in [0.5, 0.6, 0.7, 0.8]:
        wx = 1 - wl
        b_oof = wl * oof_lgbm + wx * oof_xgb
        b_test = np.maximum(wl * test_lgbm + wx * test_xgb, 0)
        b_mae = np.abs(b_oof - y_raw.values).mean()
        fname = f'model36_blend_lx_{int(wl*100)}.csv'
        out = sub.copy(); out[TARGET] = b_test
        out.to_csv(os.path.join(SUB_DIR, fname), index=False)
        flag = ' ✅' if b_mae < best_blend_mae else ''
        if b_mae < best_blend_mae: best_blend_mae = b_mae; best_blend_file = fname
        print(f'  {wl:>6.0%}  {b_mae:>8.4f}  {b_test.std():>8.2f}  {b_test.max():>8.2f}  {fname}{flag}')

    # 3종 블렌드
    print(f'\n  [3종 메타 블렌드 (L+C+X)]')
    for (wl, wc, wx) in [(0.5,0.3,0.2), (0.6,0.2,0.2), (0.4,0.4,0.2),
                          (0.7,0.2,0.1), (0.5,0.25,0.25)]:
        b_oof = wl*oof_lgbm + wc*oof_cb + wx*oof_xgb
        b_test = np.maximum(wl*test_lgbm + wc*test_cb + wx*test_xgb, 0)
        b_mae = np.abs(b_oof - y_raw.values).mean()
        fname = f'model36_blend3_{int(wl*100)}_{int(wc*100)}_{int(wx*100)}.csv'
        out = sub.copy(); out[TARGET] = b_test
        out.to_csv(os.path.join(SUB_DIR, fname), index=False)
        flag = ' ✅' if b_mae < best_blend_mae else ''
        if b_mae < best_blend_mae: best_blend_mae = b_mae; best_blend_file = fname
        print(f'  L={wl:.0%} C={wc:.0%} X={wx:.0%}  {b_mae:>8.4f}  {b_test.std():>8.2f}  '
              f'{b_test.max():>8.2f}  {fname}{flag}')

    # ══════════════════════════════════════════════════════════════
    # 종합
    # ══════════════════════════════════════════════════════════════
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 70)
    print(f'[종합 비교] ({elapsed:.1f}분 소요)')
    print('=' * 70)
    print(f'  기준 (model34-B): CV=8.4803 / Public=9.8078')
    print(f'  LGBM-meta (재현): CV={mae_lgbm:.4f}  (Δ{mae_lgbm - 8.4803:+.4f})')
    print(f'  CB-meta:          CV={mae_cb:.4f}  (Δ{mae_cb - 8.4803:+.4f})')
    print(f'  XGB-meta:         CV={mae_xgb:.4f}  (Δ{mae_xgb - 8.4803:+.4f})')
    print(f'  Best 블렌드:      CV={best_blend_mae:.4f}  → {best_blend_file}')
    print(f'  LGBM-CB 상관:     {corr_lc:.4f}')

    print(f'\n추천 제출:')
    print(f'  1순위: {best_blend_file} (CV {best_blend_mae:.4f})')
    print(f'  2순위: model36_meta_cb.csv (CB 단독, CV {mae_cb:.4f})')
    print(f'  ※ CV만으로 판단 불가 — Public 제출 후 배율 확인 필수')
    print(f'  ※ model29A 패턴 가능: CV↓여도 Public↑ (배율 개선)')


if __name__ == '__main__':
    main()
