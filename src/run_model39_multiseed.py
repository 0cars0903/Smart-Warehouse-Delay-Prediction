"""
전략 5: Multi-seed 앙상블 — model39
================================================================
근거:
  - 단일 seed(42) 학습은 랜덤 초기화에 의한 분산 존재
  - 동일 구조 + 다른 seed → 예측값 평균 = 분산 감소
  - 특히 ET/RF (bagging 기반)와 Asym (custom loss) 모델에서 seed 효과 클 수 있음
  - 배율 변동 없이 CV/Public 동시 개선 기대

접근법:
  1. seed 3종 (42, 123, 7777) × 6모델 → 18회 학습
  2. 각 seed에서 독립 OOF + test 예측
  3. seed별 meta stacking → 3개 최종 예측
  4. 3-seed 평균 → 최종 제출
  5. 개별 seed 제출도 생성 (비교용)

모델 구성 (seed별 6모델, model34-B 동일):
  LGBM (log1p), CB (log1p), ET (log1p), RF (log1p),
  TW1.5 (raw), Asym2.0 (log1p)

실행: python src/run_model39_multiseed.py
예상 시간: ~30분 (3seeds × 6모델 = 18회 학습 + 3회 meta)
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
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model39_ckpt')
N_SPLITS = 5

TARGET = 'avg_delay_minutes_next_30m'

SEEDS = [42, 123, 7777]

# ── 파라미터 템플릿 (seed는 동적으로 교체) ──
def lgbm_params(seed):
    return {
        'num_leaves': 129, 'learning_rate': 0.01021,
        'feature_fraction': 0.465, 'bagging_fraction': 0.9049,
        'min_child_samples': 26, 'reg_alpha': 1.468, 'reg_lambda': 0.3630,
        'objective': 'regression_l1', 'n_estimators': 3000,
        'bagging_freq': 1, 'random_state': seed,
        'verbosity': -1, 'n_jobs': -1,
    }

def cb_params(seed):
    return {
        'depth': 9, 'learning_rate': 0.01144,
        'iterations': 3000, 'l2_leaf_reg': 1.561,
        'random_strength': 1.359, 'bagging_temperature': 0.29,
        'random_seed': seed,
        'loss_function': 'MAE', 'verbose': 0, 'thread_count': -1,
    }

def tw_params(seed):
    return {
        'num_leaves': 181, 'learning_rate': 0.020616,
        'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
        'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
        'objective': 'tweedie', 'tweedie_variance_power': 1.5,
        'n_estimators': 3000, 'bagging_freq': 1,
        'random_state': seed, 'verbosity': -1, 'n_jobs': -1,
    }

ASYM_ALPHA = 2.0
def asym_lgbm_params(seed):
    return {
        'num_leaves': 127, 'learning_rate': 0.015,
        'feature_fraction': 0.50, 'bagging_fraction': 0.90,
        'min_child_samples': 35, 'reg_alpha': 2.0, 'reg_lambda': 1.0,
        'n_estimators': 3000, 'bagging_freq': 1,
        'random_state': seed, 'verbosity': -1, 'n_jobs': -1,
    }

def meta_params(seed):
    return {
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'objective': 'regression_l1', 'n_estimators': 500,
        'bagging_freq': 1, 'random_state': seed,
        'verbosity': -1, 'n_jobs': -1,
    }


# ── FE ──
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


# ── 6모델 학습 (특정 seed) ──
def train_6models_seed(X_train, X_test, y_raw, y_log, groups, feat_cols, seed):
    gkf = GroupKFold(n_splits=N_SPLITS)
    X_tr = X_train[feat_cols].fillna(0)
    X_te = X_test[feat_cols].fillna(0)

    oof_dict = {}
    test_dict = {}

    # ── 1. LGBM ──
    print(f'\n  [LGBM] seed={seed}')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    params = lgbm_params(seed)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr.iloc[tr_idx], y_log.iloc[tr_idx].values,
              eval_set=[(X_tr.iloc[va_idx], y_log.iloc[va_idx].values)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - y_raw.iloc[va_idx].values).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    oof_dict['lgbm'] = oof; test_dict['lgbm'] = test_pred
    print(f'    LGBM OOF MAE={np.abs(np.expm1(oof) - y_raw.values).mean():.4f}')

    # ── 2. CatBoost ──
    print(f'\n  [CB] seed={seed}')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    params = cb_params(seed)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        pool_tr = cb.Pool(X_tr.iloc[tr_idx], label=y_log.iloc[tr_idx].values)
        pool_va = cb.Pool(X_tr.iloc[va_idx], label=y_log.iloc[va_idx].values)
        m = cb.CatBoostRegressor(**params)
        m.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=50, verbose=0)
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - y_raw.iloc[va_idx].values).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m, pool_tr, pool_va; gc.collect()
    oof_dict['cb'] = oof; test_dict['cb'] = test_pred
    print(f'    CB OOF MAE={np.abs(np.expm1(oof) - y_raw.values).mean():.4f}')

    # ── 3. TW1.5 (raw) ──
    print(f'\n  [TW1.5] seed={seed}')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    params = tw_params(seed)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_raw, groups)):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr.iloc[tr_idx], y_raw.iloc[tr_idx].values,
              eval_set=[(X_tr.iloc[va_idx], y_raw.iloc[va_idx].values)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    oof_dict['tw15'] = oof; test_dict['tw15'] = test_pred
    print(f'    TW1.5 OOF MAE={np.abs(oof - y_raw.values).mean():.4f}')

    # ── 4. ET ──
    print(f'\n  [ET] seed={seed}')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_leaf=5,
                                random_state=seed, n_jobs=-1)
        m.fit(X_tr.iloc[tr_idx], y_log.iloc[tr_idx].values)
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - y_raw.iloc[va_idx].values).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    oof_dict['et'] = oof; test_dict['et'] = test_pred
    print(f'    ET OOF MAE={np.abs(np.expm1(oof) - y_raw.values).mean():.4f}')

    # ── 5. RF ──
    print(f'\n  [RF] seed={seed}')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=5,
                                  random_state=seed, n_jobs=-1)
        m.fit(X_tr.iloc[tr_idx], y_log.iloc[tr_idx].values)
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        test_pred += m.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - y_raw.iloc[va_idx].values).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    oof_dict['rf'] = oof; test_dict['rf'] = test_pred
    print(f'    RF OOF MAE={np.abs(np.expm1(oof) - y_raw.values).mean():.4f}')

    # ── 6. Asym2.0 ──
    print(f'\n  [Asym(α={ASYM_ALPHA})] seed={seed}')
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    a_params = asym_lgbm_params(seed)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        dtrain = lgb.Dataset(X_tr.iloc[tr_idx], label=y_log.iloc[tr_idx].values)
        dval   = lgb.Dataset(X_tr.iloc[va_idx], label=y_log.iloc[va_idx].values, reference=dtrain)
        params = {k: v for k, v in a_params.items() if k not in ['n_estimators']}
        params['objective'] = asymmetric_mae_objective
        bst = lgb.train(
            params, dtrain,
            num_boost_round=a_params['n_estimators'],
            valid_sets=[dval], feval=asymmetric_mae_metric,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va_idx] = bst.predict(X_tr.iloc[va_idx])
        test_pred += bst.predict(X_te) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.iloc[va_idx].values)).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={bst.best_iteration}')
        del bst; gc.collect()
    oof_dict['asym20'] = oof; test_dict['asym20'] = test_pred
    print(f'    Asym2.0 OOF MAE={np.abs(np.expm1(oof) - y_raw.values).mean():.4f}')

    return oof_dict, test_dict


# ── 메타 학습기 ──
def run_meta(meta_train, meta_test, y_raw, groups, seed, label='meta'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y_raw)); test_pred = np.zeros(meta_test.shape[0])
    params = meta_params(seed)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        m = lgb.LGBMRegressor(**params)
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
    print('model39: Multi-seed 앙상블 (3 seeds × 6모델 → seed-average)')
    print(f'  Seeds: {SEEDS}')
    print('  기준: blend_m33m34_w80 Public=9.8073')
    print('  목표: seed 분산 감소 → 안정적 개선')
    print('=' * 70)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    # ── 데이터 로드 ──
    print('\n[데이터 로드]')
    train, test = load_data()
    feat_cols = get_feat_cols(train)
    y_raw = train[TARGET]
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'  피처: {len(feat_cols)}, train: {len(train)}, test: {len(test)}')

    ref_df = pd.read_csv(os.path.join(SUB_DIR, 'blend_m33m34_w80.csv'))

    all_test_finals = []
    all_oof_finals = []
    seed_results = []

    for si, seed in enumerate(SEEDS):
        print('\n' + '=' * 70)
        print(f'[Seed {si+1}/{len(SEEDS)}] seed={seed}')
        print('=' * 70)

        # ── 6모델 학습 ──
        oof_dict, test_dict = train_6models_seed(
            train, test, y_raw, y_log, groups, feat_cols, seed)

        # 체크포인트 저장
        seed_ckpt = os.path.join(CKPT_DIR, f'seed_{seed}')
        for name in oof_dict:
            save_ckpt(seed_ckpt, name, oof_dict[name], test_dict[name])

        # ── 메타 스태킹 ──
        print(f'\n[Seed {seed}] 메타 스태킹')
        meta_oof_list = []
        meta_test_list = []
        for name in ['lgbm', 'cb', 'et', 'rf', 'asym20']:
            meta_oof_list.append(oof_dict[name])
            meta_test_list.append(test_dict[name])
        meta_oof_list.append(np.log1p(np.maximum(oof_dict['tw15'], 0)))
        meta_test_list.append(np.log1p(np.maximum(test_dict['tw15'], 0)))

        meta_train_arr = np.column_stack(meta_oof_list)
        meta_test_arr  = np.column_stack(meta_test_list)

        oof_final, test_final, meta_mae = run_meta(
            meta_train_arr, meta_test_arr, y_raw, groups, seed,
            label=f'seed{seed}-meta')

        test_final = np.maximum(test_final, 0)
        oof_final_clipped = np.maximum(oof_final, 0)

        pred_std = test_final.std()
        print(f'\n  [Seed {seed}] Meta MAE={meta_mae:.4f}, test_std={pred_std:.2f}, '
              f'test_max={test_final.max():.2f}')

        # 개별 seed 제출
        sub = ref_df.copy()
        sub[TARGET] = test_final
        fname = f'model39_seed{seed}.csv'
        sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f'  → {fname} 저장')

        all_test_finals.append(test_final)
        all_oof_finals.append(oof_final_clipped)
        seed_results.append({
            'seed': seed,
            'meta_mae': meta_mae,
            'pred_std': pred_std,
            'pred_max': test_final.max(),
        })

        del oof_dict, test_dict; gc.collect()

    # ── Multi-seed 평균 ──
    print('\n' + '=' * 70)
    print('Multi-seed 평균')
    print('=' * 70)

    avg_test = np.mean(all_test_finals, axis=0)
    avg_test = np.maximum(avg_test, 0)
    avg_oof  = np.mean(all_oof_finals, axis=0)

    avg_oof_mae = np.abs(avg_oof - y_raw.values).mean()
    avg_std = avg_test.std()
    avg_max = avg_test.max()

    print(f'  OOF MAE (avg): {avg_oof_mae:.4f}')
    print(f'  test: mean={avg_test.mean():.2f}, std={avg_std:.2f}, max={avg_max:.2f}')
    print(f'  배율 추정: ×1.156={avg_oof_mae*1.156:.4f}')

    segment_analysis(avg_oof, y_raw.values, label='multi-seed avg')

    # 평균 제출
    sub = ref_df.copy()
    sub[TARGET] = avg_test
    fname = 'model39_multiseed_avg.csv'
    sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
    print(f'\n  → {fname} 저장')

    # ── Seed 간 예측 편차 분석 ──
    print('\n[Seed 간 예측 편차]')
    test_stack = np.column_stack(all_test_finals)
    per_row_std = test_stack.std(axis=1)
    print(f'  Per-row std: mean={per_row_std.mean():.4f}, max={per_row_std.max():.4f}, '
          f'p90={np.percentile(per_row_std, 90):.4f}')
    print(f'  Seed 간 상관:')
    for i in range(len(SEEDS)):
        for j in range(i+1, len(SEEDS)):
            corr = np.corrcoef(all_test_finals[i], all_test_finals[j])[0, 1]
            print(f'    seed{SEEDS[i]}-seed{SEEDS[j]}: {corr:.6f}')

    # ── 종합 비교 ──
    print('\n' + '=' * 70)
    print('종합 비교')
    print('=' * 70)
    print(f'  {"Config":<20s}  {"Meta MAE":>10s}  {"pred_std":>10s}  {"pred_max":>10s}')
    print(f'  {"-"*20}  {"-"*10}  {"-"*10}  {"-"*10}')
    print(f'  {"baseline(m34-B)":<20s}  {"8.4803":>10s}  {"16.15":>10s}  {"~100":>10s}')
    for r in seed_results:
        print(f'  {"seed="+str(r["seed"]):<20s}  {r["meta_mae"]:>10.4f}  {r["pred_std"]:>10.2f}  {r["pred_max"]:>10.2f}')
    print(f'  {"multi-seed avg":<20s}  {avg_oof_mae:>10.4f}  {avg_std:>10.2f}  {avg_max:>10.2f}')

    # ── 추가 블렌드: multi-seed avg × baseline blend ──
    print('\n[추가] Multi-seed × baseline 블렌드')
    baseline = pd.read_csv(os.path.join(SUB_DIR, 'blend_m33m34_w80.csv'))
    base_pred = baseline[TARGET].values
    for w in [0.3, 0.5, 0.7]:
        blended = w * avg_test + (1-w) * base_pred
        blended = np.maximum(blended, 0)
        sub = ref_df.copy()
        sub[TARGET] = blended
        fname = f'model39_blend_w{int(w*100)}.csv'
        sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f'  w={w:.1f}: mean={blended.mean():.2f}, std={blended.std():.2f} → {fname}')

    elapsed = time.time() - t0
    print(f'\n총 소요 시간: {elapsed/60:.1f}분')
    print('핵심: seed 간 상관이 0.999+ → 개선 Δ<0.001 예상 (안정적이나 극적이지 않음)')
    print('※ USER가 제출하여 실제 Public 스코어 확인 필요')


if __name__ == '__main__':
    main()
