"""
전략 4: Pseudo-labeling (High Confidence 선별) — model38
================================================================
근거:
  - 배율(Public/CV) 개선이 핵심 과제 (현재 1.1565)
  - test 시나리오가 train과 다른 분포 → pseudo-label로 test 분포 학습
  - high-confidence 구간만 선별하여 극값 오염 방지

접근법:
  1. Best 제출(blend_m33m34_w80)의 test 예측을 pseudo-label로 사용
  2. 모델 간 예측 편차가 작은 시나리오만 선별 (high confidence)
  3. pseudo-label 데이터를 train에 concat (sample_weight=0.5)
  4. 5모델 full 재학습 → meta 학습 → 제출

실행: python src/run_model38_pseudo_label.py
예상 시간: ~15분 (5모델 재학습)
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
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model38_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

TARGET = 'avg_delay_minutes_next_30m'

# ── 하이퍼파라미터 (model34 동일) ──
BEST_LGBM_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

CB_PARAMS = {
    'depth': 6, 'learning_rate': 0.05,
    'iterations': 3000, 'l2_leaf_reg': 3.0,
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


# ── FE ──
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
            if c not in {'ID', 'scenario_id', 'layout_id', TARGET}
            and df[c].dtype != object]


# ── Pseudo-label 선별 ──
def select_pseudo_labels(test_df, feat_cols):
    """
    High-confidence pseudo-label 선별:
    1. 여러 제출 파일의 예측값 로드
    2. 시나리오별 모델 간 std가 낮은 시나리오 선별
    3. 예측값 5~30 범위 (중앙 구간) 필터
    """
    # 여러 모델의 예측값 로드
    pred_files = {
        'm34':  'model34_6asym20.csv',
        'm33':  'model33_6model_asym.csv',
        'm31':  'model31_selected_fe.csv',
        'm30':  'model30_combined.csv',
        'bw80': 'blend_m33m34_w80.csv',
    }

    preds = {}
    for key, fname in pred_files.items():
        fpath = os.path.join(SUB_DIR, fname)
        if os.path.exists(fpath):
            preds[key] = pd.read_csv(fpath)[TARGET].values
            print(f'  로드: {fname}')

    if len(preds) < 3:
        print(f'  ⚠️ 제출 파일 {len(preds)}개만 발견 — 최소 3개 필요')
        return None, None

    # 모델 간 편차 계산 (시나리오 단위)
    pred_matrix = np.column_stack(list(preds.values()))  # (50000, N)
    row_std = pred_matrix.std(axis=1)  # 각 행의 모델 간 std
    row_mean = pred_matrix.mean(axis=1)  # 앙상블 평균

    # 시나리오별 평균 std
    test_scenarios = test_df['scenario_id'].values
    sc_ids = np.unique(test_scenarios)
    sc_std = {}
    for sc in sc_ids:
        mask = test_scenarios == sc
        sc_std[sc] = row_std[mask].mean()

    # 선별 조건:
    # 1) 시나리오 평균 std < 전체 중앙값 (모델 간 합의도 높은 시나리오)
    # 2) 시나리오 평균 예측값 5~30 (중앙 구간 — 극값 배제)
    median_sc_std = np.median(list(sc_std.values()))

    selected_rows = np.zeros(len(test_df), dtype=bool)
    for sc in sc_ids:
        mask = test_scenarios == sc
        sc_mean_pred = row_mean[mask].mean()
        if sc_std[sc] < median_sc_std and 5 <= sc_mean_pred <= 30:
            selected_rows[mask] = True

    n_selected = selected_rows.sum()
    n_scenarios = len([sc for sc in sc_ids
                       if sc_std[sc] < median_sc_std
                       and 5 <= row_mean[test_scenarios == sc].mean() <= 30])

    print(f'\n  [Pseudo-label 선별]')
    print(f'  전체 test: {len(test_df)} rows, {len(sc_ids)} scenarios')
    print(f'  모델 간 std 중앙값: {median_sc_std:.3f}')
    print(f'  선별 기준: sc_std < {median_sc_std:.3f} AND 5 ≤ mean_pred ≤ 30')
    print(f'  선별 결과: {n_selected} rows ({n_selected/len(test_df)*100:.1f}%), {n_scenarios} scenarios')
    print(f'  선별 구간 예측값: mean={row_mean[selected_rows].mean():.2f}, '
          f'std={row_mean[selected_rows].std():.2f}')

    pseudo_labels = row_mean  # 앙상블 평균을 pseudo-label로 사용

    return selected_rows, pseudo_labels


# ── 학습 함수 (pseudo-label 통합) ──
def train_model_with_pseudo(model_name, X_orig, y_orig_log, groups_orig,
                            X_pseudo, y_pseudo_log, groups_pseudo,
                            X_test, feat_cols, pseudo_weight=0.5):
    """
    원본 train + pseudo-label 통합 학습.
    OOF는 원본 train에 대해서만 계산 (CV 비교용).
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    n_orig = len(X_orig)
    oof = np.zeros(n_orig)
    test_pred = np.zeros(len(X_test))

    X_tr_feat = X_orig[feat_cols].fillna(0)
    X_te_feat = X_test[feat_cols].fillna(0)
    X_ps_feat = X_pseudo[feat_cols].fillna(0)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_feat, y_orig_log, groups_orig)):
        # train = 원본 train fold + 전체 pseudo-label
        X_fold_train = pd.concat([X_tr_feat.iloc[tr_idx], X_ps_feat], ignore_index=True)
        y_fold_train = np.concatenate([y_orig_log.iloc[tr_idx].values, y_pseudo_log])

        # sample weight: 원본=1.0, pseudo=pseudo_weight
        weights = np.concatenate([
            np.ones(len(tr_idx)),
            np.full(len(y_pseudo_log), pseudo_weight)
        ])

        if model_name == 'lgbm':
            dtrain = lgb.Dataset(X_fold_train, label=y_fold_train, weight=weights)
            dval = lgb.Dataset(X_tr_feat.iloc[va_idx], label=y_orig_log.iloc[va_idx].values)
            params = {k: v for k, v in BEST_LGBM_PARAMS.items() if k != 'n_estimators'}
            bst = lgb.train(params, dtrain, num_boost_round=BEST_LGBM_PARAMS['n_estimators'],
                           valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False),
                                                         lgb.log_evaluation(0)])
            oof[va_idx] = bst.predict(X_tr_feat.iloc[va_idx])
            test_pred += bst.predict(X_te_feat) / N_SPLITS
            itr = bst.best_iteration
            del bst

        elif model_name == 'cb':
            train_pool = cb.Pool(X_fold_train, label=y_fold_train, weight=weights)
            val_pool = cb.Pool(X_tr_feat.iloc[va_idx], label=y_orig_log.iloc[va_idx].values)
            m = cb.CatBoostRegressor(**CB_PARAMS)
            m.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)
            oof[va_idx] = m.predict(X_tr_feat.iloc[va_idx])
            test_pred += m.predict(X_te_feat) / N_SPLITS
            itr = m.best_iteration_
            del m, train_pool, val_pool

        elif model_name == 'tw15':
            dtrain = lgb.Dataset(X_fold_train, label=y_fold_train.clip(min=0), weight=weights)
            dval = lgb.Dataset(X_tr_feat.iloc[va_idx],
                              label=y_orig_log.iloc[va_idx].values.clip(min=0))
            params = {k: v for k, v in TW_PARAMS.items() if k != 'n_estimators'}
            bst = lgb.train(params, dtrain, num_boost_round=TW_PARAMS['n_estimators'],
                           valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False),
                                                         lgb.log_evaluation(0)])
            oof[va_idx] = bst.predict(X_tr_feat.iloc[va_idx])
            test_pred += bst.predict(X_te_feat) / N_SPLITS
            itr = bst.best_iteration
            del bst

        elif model_name == 'et':
            m = ExtraTreesRegressor(n_estimators=500, max_depth=20,
                                    min_samples_leaf=5, n_jobs=-1,
                                    random_state=RANDOM_STATE)
            m.fit(X_fold_train, y_fold_train, sample_weight=weights)
            oof[va_idx] = m.predict(X_tr_feat.iloc[va_idx])
            test_pred += m.predict(X_te_feat) / N_SPLITS
            itr = 500
            del m

        elif model_name == 'rf':
            m = RandomForestRegressor(n_estimators=500, max_depth=20,
                                      min_samples_leaf=5, n_jobs=-1,
                                      random_state=RANDOM_STATE)
            m.fit(X_fold_train, y_fold_train, sample_weight=weights)
            oof[va_idx] = m.predict(X_tr_feat.iloc[va_idx])
            test_pred += m.predict(X_te_feat) / N_SPLITS
            itr = 500
            del m

        elif model_name == 'asym20':
            dtrain = lgb.Dataset(X_fold_train, label=y_fold_train, weight=weights)
            dval = lgb.Dataset(X_tr_feat.iloc[va_idx], label=y_orig_log.iloc[va_idx].values)
            params = {k: v for k, v in ASYM_LGBM_PARAMS.items() if k != 'n_estimators'}
            params['objective'] = asymmetric_mae_objective
            bst = lgb.train(params, dtrain, num_boost_round=ASYM_LGBM_PARAMS['n_estimators'],
                           valid_sets=[dval], feval=asymmetric_mae_metric,
                           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_idx] = bst.predict(X_tr_feat.iloc[va_idx])
            test_pred += bst.predict(X_te_feat) / N_SPLITS
            itr = bst.best_iteration
            del bst

        # TW1.5는 raw space, 나머지는 log1p space
        if model_name == 'tw15':
            mae = np.abs(oof[va_idx] - np.expm1(y_orig_log.iloc[va_idx].values)).mean()
        else:
            mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_orig_log.iloc[va_idx].values)).mean()
        print(f'  [{model_name.upper():>6s}] Fold {fold+1}  MAE={mae:.4f}  iter={itr}')
        gc.collect()

    return oof, test_pred


def main():
    t0 = time.time()
    print('=' * 70)
    print('model38: Pseudo-labeling (High Confidence 선별)')
    print('  기준: blend_m33m34_w80 Public=9.8073')
    print('  목표: test 분포 학습으로 배율 개선')
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

    # ── Pseudo-label 선별 ──
    print('\n' + '─' * 70)
    print('[Pseudo-label 선별]')
    print('─' * 70)

    selected_mask, pseudo_labels = select_pseudo_labels(test, feat_cols)
    if selected_mask is None:
        print('Pseudo-label 선별 실패 — 중단')
        return

    # pseudo-label 데이터 구성
    test_pseudo = test[selected_mask].copy()
    test_pseudo[TARGET] = pseudo_labels[selected_mask]
    pseudo_y_raw = test_pseudo[TARGET].values
    pseudo_y_log = np.log1p(pseudo_y_raw)

    # 가짜 scenario_id (test 시나리오와 동일)
    pseudo_groups = test_pseudo['scenario_id'].values

    print(f'\n  Pseudo-label 데이터: {len(test_pseudo)} rows')
    print(f'  Pseudo-label target: mean={pseudo_y_raw.mean():.2f}, '
          f'std={pseudo_y_raw.std():.2f}, max={pseudo_y_raw.max():.2f}')

    # ── 6모델 학습 (원본 + pseudo) ──
    print('\n' + '─' * 70)
    print('[Layer 1] 6모델 학습 (train + pseudo-label)')
    print(f'  pseudo_weight=0.5 (원본 대비 절반 가중치)')
    print('─' * 70)

    model_names = ['lgbm', 'cb', 'tw15', 'et', 'rf', 'asym20']
    oof_dict = {}
    test_dict = {}

    for name in model_names:
        print(f'\n  ── {name.upper()} ──')
        oof_dict[name], test_dict[name] = train_model_with_pseudo(
            name, train, y_log, groups,
            test_pseudo, pseudo_y_log, pseudo_groups,
            test, feat_cols, pseudo_weight=0.5)

        # 전체 OOF MAE
        if name == 'tw15':
            oof_mae = np.abs(oof_dict[name] - y_raw.values).mean()
        else:
            oof_mae = np.abs(np.expm1(oof_dict[name]) - y_raw.values).mean()
        print(f'  [{name.upper()}] 전체 OOF MAE={oof_mae:.4f}')

    # ── 상관 분석 ──
    print('\n' + '─' * 70)
    print('[다양성 분석]')
    print('─' * 70)
    corr_pairs = [('lgbm','cb'), ('lgbm','tw15'), ('lgbm','et'), ('lgbm','asym20'), ('tw15','et')]
    for n1, n2 in corr_pairs:
        o1 = oof_dict[n1] if n1 != 'tw15' else np.log1p(np.maximum(oof_dict[n1], 0))
        o2 = oof_dict[n2] if n2 != 'tw15' else np.log1p(np.maximum(oof_dict[n2], 0))
        corr = np.corrcoef(o1, o2)[0, 1]
        print(f'  {n1.upper()}-{n2.upper()}: {corr:.4f}')

    # ── 메타 학습 ──
    print('\n' + '─' * 70)
    print('[Layer 2] 메타 스태킹 (LGBM-meta)')
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

    meta_train_arr = np.column_stack(meta_oof_list)
    meta_test_arr  = np.column_stack(meta_test_list)

    # 메타 학습 (원본 train의 OOF만 사용)
    gkf = GroupKFold(n_splits=N_SPLITS)
    meta_oof = np.zeros(len(y_raw)); meta_test_pred = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train_arr, y_raw, groups)):
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(meta_train_arr[tr_idx], np.log1p(y_raw.iloc[tr_idx].values),
              eval_set=[(meta_train_arr[va_idx], np.log1p(y_raw.iloc[va_idx].values))],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        meta_oof[va_idx] = np.expm1(m.predict(meta_train_arr[va_idx]))
        meta_test_pred += np.expm1(m.predict(meta_test_arr)) / N_SPLITS
        mae = np.abs(meta_oof[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [Meta] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    meta_mae = np.abs(meta_oof - y_raw.values).mean()
    test_pred_final = np.maximum(meta_test_pred, 0)
    test_std = test_pred_final.std()
    print(f'\n  [Meta] OOF MAE={meta_mae:.4f} | test_std={test_std:.2f} | test_max={test_pred_final.max():.2f}')

    # ── 제출 파일 ──
    print('\n' + '─' * 70)
    print('[제출 파일]')
    print('─' * 70)

    sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sub[TARGET] = test_pred_final
    fname = 'model38_pseudo.csv'
    sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
    print(f'  {fname}: CV={meta_mae:.4f}, std={test_std:.2f}, max={test_pred_final.max():.2f}')

    # ── 종합 ──
    elapsed = (time.time() - t0) / 60
    print(f'\n{"=" * 70}')
    print(f'model38 결과 ({elapsed:.1f}분 소요)')
    print(f'{"=" * 70}')
    print(f'  기준 (model34-B): CV=8.4803 / Public=9.8078')
    print(f'  Pseudo-label:     CV={meta_mae:.4f}  (Δ{meta_mae - 8.4803:+.4f})')
    print(f'  test_std: {test_std:.2f} (model34-B: 16.15)')
    print(f'  기대 Public (×1.157): {meta_mae * 1.157:.4f}')
    print(f'  Pseudo rows: {selected_mask.sum()} / {len(test)} ({selected_mask.sum()/len(test)*100:.0f}%)')

    if meta_mae < 8.48:
        print(f'\n  ✅ CV 개선 → 제출 강력 추천')
    elif meta_mae < 8.50:
        print(f'\n  △ CV 미세 변화 — 배율 확인 후 판단')
    else:
        print(f'\n  ⚠️ CV 악화 — pseudo-label 노이즈 가능성')
        print(f'     test_std 확인: 높으면 model29A 패턴(배율↓) 기대 가능')


if __name__ == '__main__':
    main()
