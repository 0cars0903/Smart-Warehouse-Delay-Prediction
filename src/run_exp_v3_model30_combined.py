"""
모델실험30: model29A 피처(422) + model29B Optuna 파라미터
=============================================================
두 축의 개선을 결합:
  - model29A: Tier 2 비율 피처 확장 → Public 9.8312, 배율 1.1567 (역대 최고)
  - model29B: Optuna LGBM+CB 재튜닝 → CV 8.4723 (model28A 대비 Δ-0.002)

가설:
  - 29A의 Tier 2 비율 피처가 일반화(배율)에 기여
  - 29B의 Optuna 파라미터가 CV에 기여
  - 결합 시 CV 개선 + 배율 유지/개선 가능성

핵심 변경점 (model29A 대비):
  - LGBM: num_leaves 181→129, lr 0.0206→0.0102, feat_frac 0.51→0.47,
          reg_alpha 0.38→1.47, reg_lambda 0.36→0.40, min_child 26→30
  - CB:   depth 6→9, lr 0.05→0.011, l2_leaf_reg 3.0→1.56,
          + random_strength 1.36, bagging_temperature 0.29
  - TW/ET/RF: 변경 없음 (동일 파라미터)

기대:
  - model29A CV 8.4989에서 Optuna 파라미터로 CV 개선
  - model29A 배율 1.1567 유지 (Tier 2 피처 효과)
  - 최적 결과: CV ~8.47 × 배율 ~1.157 = Public ~9.80

실행: python src/run_exp_v3_model30_combined.py
예상 시간: ~30분 (5모델 × 5fold, Optuna 탐색 없음)
출력: submissions/model30_combined.csv
체크포인트: docs/model30_ckpt/
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import warnings, gc, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features, get_feature_cols

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model30_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# ★ LGBM: model29B Optuna 최적 파라미터
# (model29A 대비: num_leaves↓, lr↓, feat_frac↓, reg↑↑ → 정규화 강화)
# ─────────────────────────────────────────────
LGBM_PARAMS = {
    'num_leaves':       129,        # 29A: 181 → 29B Optuna
    'learning_rate':    0.01021,    # 29A: 0.020616 → 절반 수준
    'feature_fraction': 0.465,      # 29A: 0.5122 → 더 강한 드롭아웃
    'bagging_fraction': 0.947,      # 29A: 0.9049
    'min_child_samples': 30,        # 29A: 26
    'reg_alpha':        1.468,      # 29A: 0.3805 → 3.9배 강화
    'reg_lambda':       0.396,      # 29A: 0.3630
    'objective': 'regression_l1',
    'n_estimators': 3000,
    'bagging_freq': 1,
    'random_state': RANDOM_STATE,
    'verbosity': -1,
    'n_jobs': -1,
}

# ★ CB: model29B Optuna 최적 파라미터
# (model29A 대비: depth↑, lr↓, + random_strength/bagging_temperature 추가)
CB_PARAMS = {
    'iterations': 3000,
    'learning_rate':       0.01144,   # 29A: 0.05 → 4.4배 느린 학습
    'depth':               9,         # 29A: 6 → 더 깊은 트리
    'l2_leaf_reg':         1.561,     # 29A: 3.0 → 절반
    'random_strength':     1.359,     # 29A: 없음 (신규)
    'bagging_temperature': 0.285,     # 29A: 없음 (신규)
    'loss_function': 'MAE',
    'random_seed': RANDOM_STATE,
    'verbose': 0,
    'early_stopping_rounds': 50,
}

# TW/ET/RF: model29A와 동일
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
    np.save(os.path.join(CKPT_DIR, f'{name}_oof.npy'), oof)
    np.save(os.path.join(CKPT_DIR, f'{name}_test.npy'), test_pred)

def load_ckpt(name):
    return (np.load(os.path.join(CKPT_DIR, f'{name}_oof.npy')),
            np.load(os.path.join(CKPT_DIR, f'{name}_test.npy')))

def ckpt_exists(name):
    return (os.path.exists(os.path.join(CKPT_DIR, f'{name}_oof.npy')) and
            os.path.exists(os.path.join(CKPT_DIR, f'{name}_test.npy')))


# ─────────────────────────────────────────────
# 시나리오 집계 피처 (model22: 11통계)
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


# ─────────────────────────────────────────────
# ★ 비율 피처: Tier 1 (model28A 동일 5종)
# ─────────────────────────────────────────────
def add_layout_ratio_features_tier1(df):
    """model28A 동일 비율 피처 5종"""
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


# ─────────────────────────────────────────────
# ★ 비율 피처: Tier 2 (model29A 동일 7종)
# ─────────────────────────────────────────────
def add_layout_ratio_features_tier2(df):
    """model29A 비율 피처 7종 — layout capacity 정규화"""
    def safe_div(a, b, fill=0):
        return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

    # 6. 교차 스트레스
    if all(c in df.columns for c in ['sc_congestion_score_mean', 'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'],
            df['robot_total'] ** 2)

    # 7. 로봇 밀도
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(
            df['robot_total'], df['floor_area_sqm'] / 100)

    # 8. 패킹 밀도
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(
            df['pack_station_count'], df['floor_area_sqm'] / 1000)

    # 9. 충전 경쟁
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(
            df['sc_robot_charging_mean'], df['charger_count'])

    # 10. 배터리 효율
    if 'sc_battery_mean_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_battery_per_robot'] = safe_div(
            df['sc_battery_mean_mean'] * df['robot_total'],
            df['robot_total'])
        if 'sc_robot_utilization_mean' in df.columns and 'charger_count' in df.columns:
            df['ratio_battery_per_robot'] = safe_div(
                df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'],
                df['charger_count'])

    # 11. 통로 혼잡률
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(
            df['sc_congestion_score_mean'], df['aisle_width_avg'])

    # 12. 유휴 비율
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

    # Tier 1 비율 피처 (model28A 동일)
    train = add_layout_ratio_features_tier1(train)
    test  = add_layout_ratio_features_tier1(test)

    # Tier 2 비율 피처 (model29A 동일)
    train = add_layout_ratio_features_tier2(train)
    test  = add_layout_ratio_features_tier2(test)

    ratio_cols = [c for c in train.columns if c.startswith('ratio_')]
    return train, test, ratio_cols


def get_feat_cols(train):
    return [c for c in train.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and train[c].dtype != object]


# ─────────────────────────────────────────────
# Base Learner 학습 함수
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
        print(f'  [LGBM-Optuna] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
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
        print(f'  [CB-Optuna] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
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


def main():
    t0 = time.time()
    print('=' * 60)
    print('모델실험30: model29A 피처(422) + model29B Optuna 파라미터')
    print('기준: Model29A CV 8.4989 / Public 9.8312 (배율 1.1567)')
    print('      Model29B CV 8.4723 / Public 9.8356 (배율 1.1609)')
    print('변경: 29A 피처셋 + 29B LGBM+CB Optuna 파라미터')
    print('=' * 60)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    train, test, ratio_cols = load_data()
    feat_cols = get_feat_cols(train)
    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']

    print(f'피처 수: {len(feat_cols)} (model29A: 422)')
    print(f'비율 피처: {len(ratio_cols)}종 (Tier1: 5 + Tier2: 7)')

    # 비율 피처 shift 분석
    print(f'\n비율 피처 train vs test shift:')
    for col in ratio_cols:
        tr_m = train[col].mean(); te_m = test[col].mean()
        tr_s = train[col].std()
        shift = abs(tr_m - te_m) / (tr_s + 1e-8)
        marker = '✅' if shift < 0.4 else '⚠️'
        print(f'  {col:40s}: shift={shift:.3f}σ {marker}  '
              f'(train={tr_m:.4f}, test={te_m:.4f})')

    # 파라미터 변경 비교
    print('\n' + '─' * 60)
    print('[파라미터 변경 비교]')
    print('─' * 60)
    print('  LGBM: num_leaves 181→129, lr 0.0206→0.0102, feat_frac 0.51→0.47')
    print('        reg_alpha 0.38→1.47 (3.9×), min_child 26→30')
    print('  CB:   depth 6→9, lr 0.05→0.011, l2_leaf_reg 3.0→1.56')
    print('        + random_strength=1.36, bagging_temp=0.29')

    # ── Layer 1: Base Learner ──
    print('\n' + '─' * 60)
    print('[Layer 1] Base Learner OOF 생성')
    print('─' * 60)

    # LGBM — 29B Optuna 파라미터로 재학습
    if ckpt_exists('lgbm'):
        print('\n[LGBM-Optuna] 체크포인트 로드'); oof_lg, test_lg = load_ckpt('lgbm')
    else:
        print('\n[LGBM-Optuna] 학습 시작...')
        oof_lg, test_lg = train_lgbm_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('lgbm', oof_lg, test_lg)
    lgbm_mae = np.abs(np.expm1(oof_lg) - y_raw.values).mean()
    print(f'  LGBM OOF MAE={lgbm_mae:.4f}  (29A: 8.5545, 29B: 8.5379)')

    # TW1.8 — 파라미터 변경 없음
    if ckpt_exists('tw18'):
        print('\n[TW1.8] 체크포인트 로드'); oof_tw, test_tw = load_ckpt('tw18')
    else:
        print('\n[TW1.8] 학습 시작...')
        oof_tw, test_tw = train_tw18_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt('tw18', oof_tw, test_tw)
    print(f'  TW1.8 OOF MAE={np.abs(oof_tw - y_raw.values).mean():.4f}  (29A: 8.7679)')

    # CB — 29B Optuna 파라미터로 재학습
    if ckpt_exists('cb'):
        print('\n[CB-Optuna] 체크포인트 로드'); oof_cb, test_cb = load_ckpt('cb')
    else:
        print('\n[CB-Optuna] 학습 시작...')
        oof_cb, test_cb = train_cb_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('cb', oof_cb, test_cb)
    cb_mae = np.abs(np.expm1(oof_cb) - y_raw.values).mean()
    print(f'  CB OOF MAE={cb_mae:.4f}  (29A: 8.6398, 29B: 8.5952)')

    # ET — 파라미터 변경 없음
    if ckpt_exists('et'):
        print('\n[ET] 체크포인트 로드'); oof_et, test_et = load_ckpt('et')
    else:
        print('\n[ET] 학습 시작...')
        oof_et, test_et = train_et_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('et', oof_et, test_et)
    print(f'  ET OOF MAE={np.abs(np.expm1(oof_et) - y_raw.values).mean():.4f}  (29A: 8.6962)')

    # RF — 파라미터 변경 없음
    if ckpt_exists('rf'):
        print('\n[RF] 체크포인트 로드'); oof_rf, test_rf = load_ckpt('rf')
    else:
        print('\n[RF] 학습 시작...')
        oof_rf, test_rf = train_rf_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('rf', oof_rf, test_rf)
    print(f'  RF OOF MAE={np.abs(np.expm1(oof_rf) - y_raw.values).mean():.4f}  (29A: 8.7044)')

    # ── 상관관계 ──
    print('\n' + '─' * 60)
    print('[다양성] OOF 상관관계')
    print('─' * 60)
    oof_raw = {
        'LGBM': np.expm1(oof_lg), 'TW': oof_tw, 'CB': np.expm1(oof_cb),
        'ET': np.expm1(oof_et), 'RF': np.expm1(oof_rf)
    }
    names = list(oof_raw.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = np.corrcoef(oof_raw[names[i]], oof_raw[names[j]])[0,1]
            print(f'  {names[i]:4s}-{names[j]:4s}: {c:.4f}')

    # ── 가중 앙상블 ──
    arrs = [oof_raw['LGBM'], oof_raw['CB'], oof_raw['TW'], oof_raw['ET'], oof_raw['RF']]
    def loss5(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        return np.mean(np.abs(sum(w[i]*arrs[i] for i in range(5)) - y_raw.values))
    best_loss, best_w = np.inf, np.ones(5)/5
    for _ in range(500):
        w0 = np.random.dirichlet(np.ones(5))
        res = minimize(loss5, w0, method='Nelder-Mead')
        if res.fun < best_loss:
            best_loss = res.fun; best_w = np.abs(res.x) / np.abs(res.x).sum()
    print(f'\n  가중 앙상블 CV MAE: {best_loss:.4f}')
    print(f'    LGBM={best_w[0]:.3f}, CB={best_w[1]:.3f}, TW={best_w[2]:.3f}, '
          f'ET={best_w[3]:.3f}, RF={best_w[4]:.3f}')

    # ── Layer 2: 메타 학습기 ──
    print('\n' + '─' * 60)
    print('[Layer 2] 5모델 LGBM 메타 학습기')
    print('─' * 60)

    test_tw_clipped = np.maximum(test_tw, 0)
    meta_train = np.column_stack([oof_lg, oof_cb,
                                   np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf])
    meta_test  = np.column_stack([test_lg, test_cb,
                                   np.log1p(test_tw_clipped), test_et, test_rf])

    oof_meta, test_meta, mae_meta = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    # 제출 파일
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    sub_path = os.path.join(SUB_DIR, 'model30_combined.csv')
    sample.to_csv(sub_path, index=False)
    print(f'\n제출 파일: {sub_path}')

    # ── 분석 ──
    print('\n' + '─' * 60)
    print('[분석] 타겟 구간별 MAE')
    print('─' * 60)
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    for lo, hi in bins:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(oof_meta[mask] - y_raw.values[mask]).mean()
            print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d}  MAE={seg_mae:6.2f}')

    print('\n' + '─' * 60)
    print('[분석] 예측 분포')
    print('─' * 60)
    print(f'  OOF:  mean={oof_meta.mean():.2f}, std={oof_meta.std():.2f}, max={oof_meta.max():.2f}')
    print(f'  test: mean={test_meta.mean():.2f}, std={test_meta.std():.2f}, max={test_meta.max():.2f}')

    # ── 29A vs 30 비교 ──
    print('\n' + '─' * 60)
    print('[비교] model29A vs model30')
    print('─' * 60)
    print(f'  LGBM OOF:   29A=8.5545  →  30={lgbm_mae:.4f}  Δ={lgbm_mae-8.5545:+.4f}')
    print(f'  CB OOF:     29A=8.6398  →  30={cb_mae:.4f}  Δ={cb_mae-8.6398:+.4f}')
    print(f'  메타 CV:    29A=8.4989  →  30={mae_meta:.4f}  Δ={mae_meta-8.4989:+.4f}')
    print(f'  test std:   29A=15.94   →  30={test_meta.std():.2f}')

    # 최종 요약
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 60)
    print(f'모델실험30 결과 ({elapsed:.1f}분 소요)')
    print('=' * 60)
    print(f'  피처 수      : {len(feat_cols)} (model29A: 422)')
    print(f'  메타 LGBM    : {mae_meta:.4f}  pred_std={oof_meta.std():.2f}')
    print(f'  test pred    : mean={test_meta.mean():.2f}, std={test_meta.std():.2f}')
    print(f'  Model29A 기준: CV 8.4989 / Public 9.8312 (배율 1.1567)')
    print(f'  Model29B 기준: CV 8.4723 / Public 9.8356 (배율 1.1609)')
    print(f'  Model30 변화 : {mae_meta - 8.4989:+.4f} (vs 29A)')
    print(f'  기대 Public (×1.157): {mae_meta * 1.157:.4f}')
    print(f'  기대 Public (×1.163): {mae_meta * 1.163:.4f}')

    if test_meta.std() > 15.5:
        print(f'\n  ✅ test std={test_meta.std():.2f} > 15.5 (분포 유지)')
    else:
        print(f'\n  ⚠️ test std={test_meta.std():.2f} < 15.5 (29A 15.94 대비 압축 주의)')

    # 판정 가이드
    print('\n[판정 가이드]')
    if mae_meta < 8.4989:
        print(f'  ✅ CV 개선 ({mae_meta:.4f} < 29A 8.4989)')
        if test_meta.std() >= 15.5:
            print(f'  ✅ 분포 유지 → 제출 강력 추천')
        else:
            print(f'  ⚠️ 분포 압축 → 제출하되 배율 주시 (29A 배율 1.1567 기준)')
    else:
        print(f'  ⚠️ CV 악화 ({mae_meta:.4f} > 29A 8.4989)')
        print(f'  → 29A 교훈: CV 악화도 제출 가치 있음. shift/배율 확인 후 판단')

    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
