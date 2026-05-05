"""
model35: Asym α=2.5 탐색 — 6모델 스태킹
================================================================
근거:
  - model34 B(Asym α=2.0): Public **9.8078** (역대 최고, 배율 1.1565)
  - Asym α=2.0의 높은 pred_std(16.15)가 일반화에 기여
  - §3 ablation: α=3.0은 collapse (MAE 8.984) → α=2.5가 한계점
  - 가설: α=2.5로 pred_std를 더 확장하면 배율 추가 개선 가능

모델 구성 (6모델):
  [유지] LGBM, CB, ET, RF, TW1.5 — model34 체크포인트
  [교체] Asym α=2.0 → α=2.5

실행: python src/run_model35_asym25.py
예상 시간: ~5분 (Asym2.5만 신규 학습, 나머지 체크포인트)
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
CKPT_35  = os.path.join(_BASE, '..', 'docs', 'model35_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# ── 파라미터 ──
META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# ★ Asym α=2.5 (model34의 α=2.0에서 상향)
ASYM_ALPHA = 2.5
ASYM_LGBM_PARAMS = {
    'num_leaves': 127, 'learning_rate': 0.015,
    'feature_fraction': 0.50, 'bagging_fraction': 0.90,
    'min_child_samples': 35, 'reg_alpha': 2.0, 'reg_lambda': 1.0,
    'n_estimators': 3000, 'bagging_freq': 1,
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


# ── Asym α=2.5 학습 ──
def train_asym25_oof(X_train, X_test, y_log, groups, feat_cols):
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
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | pred_std={np.maximum(oof,0).std():.2f}')
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
    print('model35: Asym α=2.5 탐색 (6모델 스태킹)')
    print('  변경: Asym α=2.0 → α=2.5 (더 대담한 극값 예측)')
    print('  기준: model34-B 6model+Asym2.0 CV=8.4803 / Public=9.8078')
    print('  위험: §3 ablation α=3.0은 MAE 8.984 collapse')
    print('=' * 70)

    os.makedirs(CKPT_35, exist_ok=True)
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

    # ── 5종 체크포인트 로드 (model31 + model34) ──
    print('\n' + '─' * 70)
    print('[Layer 1] 체크포인트 로드 (LGBM, CB, ET, RF, TW1.5)')
    print('─' * 70)

    # model31: LGBM, CB, ET, RF
    for name in ['lgbm', 'cb', 'et', 'rf']:
        if ckpt_exists(CKPT_31, name):
            oof_dict[name], test_dict[name] = load_ckpt(CKPT_31, name)
            if name == 'tw15':
                mae = np.abs(oof_dict[name] - y_raw.values).mean()
            else:
                mae = np.abs(np.expm1(oof_dict[name]) - y_raw.values).mean()
            print(f'  [{name.upper()}] ckpt 로드 → OOF MAE={mae:.4f}')
        else:
            print(f'  ⚠️ [{name.upper()}] 체크포인트 없음!')

    # model34: TW1.5
    if ckpt_exists(CKPT_34, 'tw15'):
        oof_dict['tw15'], test_dict['tw15'] = load_ckpt(CKPT_34, 'tw15')
        mae = np.abs(oof_dict['tw15'] - y_raw.values).mean()
        print(f'  [TW1.5] ckpt 로드 → OOF MAE={mae:.4f}')
    else:
        print(f'  ⚠️ [TW1.5] 체크포인트 없음!')

    # ── Asym α=2.5 학습 ──
    print('\n' + '─' * 70)
    print(f'[Layer 1B] Asym α={ASYM_ALPHA} 학습')
    print('─' * 70)

    if ckpt_exists(CKPT_35, 'asym25'):
        print(f'  [Asym2.5] 체크포인트 로드')
        oof_dict['asym25'], test_dict['asym25'] = load_ckpt(CKPT_35, 'asym25')
    else:
        oof_dict['asym25'], test_dict['asym25'] = train_asym25_oof(
            train, test, y_log, groups, feat_cols)
        save_ckpt(CKPT_35, 'asym25', oof_dict['asym25'], test_dict['asym25'])

    asym25_oof_raw = np.expm1(oof_dict['asym25'])
    asym25_mae = np.abs(asym25_oof_raw - y_raw.values).mean()
    print(f'  Asym2.5 OOF MAE: {asym25_mae:.4f}')

    # model34 Asym2.0 체크포인트도 로드 (비교용)
    if ckpt_exists(CKPT_34, 'asym20'):
        oof_asym20, test_asym20 = load_ckpt(CKPT_34, 'asym20')
        asym20_mae = np.abs(np.expm1(oof_asym20) - y_raw.values).mean()
        print(f'  (참고) Asym2.0 OOF MAE: {asym20_mae:.4f}')

        # 상관 비교
        corr_25_20 = np.corrcoef(oof_dict['asym25'], oof_asym20)[0, 1]
        corr_25_lgbm = np.corrcoef(oof_dict['asym25'], oof_dict['lgbm'])[0, 1]
        corr_20_lgbm = np.corrcoef(oof_asym20, oof_dict['lgbm'])[0, 1]
        print(f'\n  [다양성] Asym2.5-Asym2.0: {corr_25_20:.4f}')
        print(f'  [다양성] Asym2.5-LGBM:    {corr_25_lgbm:.4f}')
        print(f'  [다양성] Asym2.0-LGBM:    {corr_20_lgbm:.4f} (참고)')

    # ── 극값 분석 ──
    print('\n[극값 분석] Asym2.5 vs Asym2.0')
    tail_mask = y_raw.values >= 80
    if tail_mask.sum() > 0:
        pr25 = asym25_oof_raw[tail_mask].mean() / y_raw.values[tail_mask].mean()
        mae25 = np.abs(asym25_oof_raw[tail_mask] - y_raw.values[tail_mask]).mean()
        print(f'  Asym2.5 [80+]: pred/actual={pr25:.3f}, MAE={mae25:.2f}')
        if ckpt_exists(CKPT_34, 'asym20'):
            asym20_raw = np.expm1(oof_asym20)
            pr20 = asym20_raw[tail_mask].mean() / y_raw.values[tail_mask].mean()
            mae20 = np.abs(asym20_raw[tail_mask] - y_raw.values[tail_mask]).mean()
            print(f'  Asym2.0 [80+]: pred/actual={pr20:.3f}, MAE={mae20:.2f}')

    # ── Collapse 체크 ──
    asym25_std = asym25_oof_raw.std()
    print(f'\n  Asym2.5 pred_std={asym25_std:.2f} (Asym2.0은 ~16.15)')
    if asym25_mae > 9.0:
        print(f'  ⚠️ Asym2.5 OOF MAE={asym25_mae:.4f} > 9.0 — Collapse 징후!')
        print(f'  → α=2.5가 한계 초과. 6모델 스태킹 진행하되 유의')

    # ── 6모델 메타 (5base + Asym2.5) ──
    print('\n' + '─' * 70)
    print('[Layer 2] 6모델 메타 스태킹')
    print('─' * 70)

    # log1p 공간 OOF → 메타 입력
    meta_names = ['lgbm', 'cb', 'tw15', 'et', 'rf', 'asym25']
    meta_oof_list = []
    meta_test_list = []

    for name in meta_names:
        if name == 'tw15':
            # raw space → log1p 변환
            meta_oof_list.append(np.log1p(np.maximum(oof_dict[name], 0)))
            meta_test_list.append(np.log1p(np.maximum(test_dict[name], 0)))
        elif name == 'asym25':
            # 이미 log1p space
            meta_oof_list.append(oof_dict[name])
            meta_test_list.append(test_dict[name])
        else:
            meta_oof_list.append(oof_dict[name])
            meta_test_list.append(test_dict[name])

    meta_train_arr = np.column_stack(meta_oof_list)
    meta_test_arr  = np.column_stack(meta_test_list)

    oof_6m, test_6m, mae_6m = run_meta(
        meta_train_arr, meta_test_arr, y_raw, groups, label='6model+Asym2.5')
    segment_analysis(oof_6m, y_raw.values, '6model+Asym2.5')

    # ── model34-B 재현 (Asym2.0 사용) 비교 ──
    if ckpt_exists(CKPT_34, 'asym20'):
        print('\n[비교] model34-B(Asym2.0) 재현')
        meta_oof_20 = meta_oof_list.copy()
        meta_test_20 = meta_test_list.copy()
        # asym25 → asym20으로 교체
        meta_oof_20[-1] = oof_asym20  # 이미 log1p space
        meta_test_20[-1] = test_asym20
        meta_train_20 = np.column_stack(meta_oof_20)
        meta_test_20_arr = np.column_stack(meta_test_20)

        oof_20, test_20, mae_20 = run_meta(
            meta_train_20, meta_test_20_arr, y_raw, groups, label='6model+Asym2.0(재현)')

    # ── 제출 파일 ──
    print('\n' + '─' * 70)
    print('[제출 파일]')
    print('─' * 70)

    sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sub['avg_delay_minutes_next_30m'] = np.maximum(test_6m, 0)
    fname = 'model35_6asym25.csv'
    sub.to_csv(os.path.join(SUB_DIR, fname), index=False)

    pred_std = np.maximum(test_6m, 0).std()
    pred_max = np.maximum(test_6m, 0).max()
    print(f'  {fname}: CV={mae_6m:.4f}, test_std={pred_std:.2f}, test_max={pred_max:.2f}')

    # ── 종합 ──
    elapsed = (time.time() - t0) / 60
    print(f'\n{"=" * 70}')
    print(f'model35 결과 ({elapsed:.1f}분 소요)')
    print(f'{"=" * 70}')
    print(f'  6model+Asym2.5: CV={mae_6m:.4f} (model34-B: 8.4803)')
    if ckpt_exists(CKPT_34, 'asym20'):
        delta = mae_6m - mae_20
        print(f'  vs Asym2.0 재현: Δ={delta:+.4f}')
    print(f'  Asym2.5 단독 OOF: {asym25_mae:.4f} (Asym2.0: 8.7699)')
    print(f'  pred_std: {pred_std:.2f} (model34-B: 16.15)')
    print(f'  기대 Public (×1.157): {mae_6m * 1.157:.4f}')

    if mae_6m > 8.55:
        print(f'\n  ⚠️ CV 악화 심함 → 제출 신중히 판단')
        print(f'  model34-B(Public 9.8078)가 여전히 최고일 가능성')
    elif mae_6m > mae_20 if ckpt_exists(CKPT_34, 'asym20') else True:
        print(f'\n  △ CV 악화이나 model29A 패턴(CV↓→Public↑) 가능')
        print(f'  pred_std가 16.15보다 높다면 제출 가치 있음')
    else:
        print(f'\n  ✅ CV도 개선 → 제출 추천')


if __name__ == '__main__':
    main()
