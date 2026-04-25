"""
모델실험28A: Layout-aware + 분포 이동 보정 (축1+2)
=============================================================
model22(Public 최고 9.9385) 기반, 배율(1.168) 개선 목표.

핵심 가설:
  - test의 40%가 unseen layout, order_inflow가 train 대비 +39%
  - 절대값 피처는 분포 이동에 취약 → "용량 대비 부하" 비율 피처 추가
  - 비율 피처는 분포 이동에 invariant (demand 200/robot 80 ≈ demand 100/robot 40)

핵심 변경 (model22 대비):
  1. Layout-capacity-normalized 비율 피처 5종 추가
     - demand_per_robot = sc_order_inflow_mean / robot_total
     - congestion_per_intersection = sc_congestion_mean / (intersection_count + 1)
     - battery_stress = sc_low_battery_mean * charge_queue / (charger_count + 1)
     - packing_pressure = sc_order_inflow_mean / (pack_station_count + 1)
     - utilization_vs_capacity = sc_robot_utilization_mean * robot_total
  2. model22 11-stat sc_agg 유지 (198 sc피처)
  3. 5모델 스태킹 유지

기대:
  - 비율 피처가 test 분포 이동 구간에서 더 안정적 예측
  - 배율 1.168 → 1.163 접근 (v1 최저 배율 수준)
  - 피처 5종만 추가 → 과적합 위험 최소

실행: python src/run_exp_model28A_layout_robust.py
예상 시간: ~90분 (5모델 × 5fold + 메타, 체크포인트 없으면 전체 재학습)
출력: submissions/model28A_layout_robust.csv
체크포인트: docs/model28A_ckpt/
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
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model28A_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 시나리오 집계 피처 대상 (18종 — model22 동일)
# ─────────────────────────────────────────────
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

# ─────────────────────────────────────────────
# 모델 하이퍼파라미터 (model22 동일)
# ─────────────────────────────────────────────
LGBM_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

TW18_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3.0,
    'loss_function': 'Tweedie:variance_power=1.8',
    'random_seed': RANDOM_STATE, 'verbose': 0,
    'early_stopping_rounds': 50,
}

CB_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3.0,
    'loss_function': 'MAE',
    'random_seed': RANDOM_STATE, 'verbose': 0,
    'early_stopping_rounds': 50,
}

ET_PARAMS = {
    'n_estimators': 500, 'max_features': 0.5,
    'min_samples_leaf': 26, 'n_jobs': -1,
    'random_state': RANDOM_STATE,
}

RF_PARAMS = {
    'n_estimators': 500, 'max_features': 0.33,
    'min_samples_leaf': 26, 'n_jobs': -1,
    'random_state': RANDOM_STATE,
}

META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────
def save_ckpt(name, oof, test_pred):
    np.save(os.path.join(CKPT_DIR, f'{name}_oof.npy'), oof)
    np.save(os.path.join(CKPT_DIR, f'{name}_test.npy'), test_pred)

def load_ckpt(name):
    oof  = np.load(os.path.join(CKPT_DIR, f'{name}_oof.npy'))
    test = np.load(os.path.join(CKPT_DIR, f'{name}_test.npy'))
    return oof, test

def ckpt_exists(name):
    return (os.path.exists(os.path.join(CKPT_DIR, f'{name}_oof.npy'))
            and os.path.exists(os.path.join(CKPT_DIR, f'{name}_test.npy')))


# ─────────────────────────────────────────────
# 시나리오 집계 피처 (model22 동일 — 11통계)
# ─────────────────────────────────────────────
def add_scenario_agg_features(df):
    """11통계 시나리오 집계 broadcast (model22 동일)"""
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
# ★ 신규: Layout-capacity-normalized 비율 피처
# ─────────────────────────────────────────────
def add_layout_ratio_features(df):
    """
    용량 대비 부하 비율 피처 (5종)

    핵심 원리: 절대값(order_inflow=200)은 분포 이동에 취약하지만
    비율(order_inflow/robot_total=2.5)은 "스트레스 수준"을 나타내므로
    train과 test에서 동일한 의미를 가짐.

    test가 train보다 busy한 이유: 더 큰 layout(robot 多)에서 더 많은 주문.
    → 절대값은 shift, 비율은 stable.
    """
    df = df.copy()

    # 안전한 나눗셈 헬퍼
    def safe_div(a, b, fill=0):
        return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

    # 1. 수요/로봇 용량 비율: "로봇 1대당 주문 부하"
    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = safe_div(
            df['sc_order_inflow_15m_mean'], df['robot_total'])

    # 2. 혼잡/교차로 비율: "교차로당 혼잡 강도"
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = safe_div(
            df['sc_congestion_score_mean'], df['intersection_count'])

    # 3. 배터리 스트레스: "충전기당 배터리 부족 압력"
    if 'sc_low_battery_ratio_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_stress'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df.get('sc_charge_queue_length_mean', 0),
            df['charger_count'])

    # 4. 패킹 압력: "패킹 스테이션당 주문량"
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = safe_div(
            df['sc_order_inflow_15m_mean'], df['pack_station_count'])

    # 5. 가용률 × 용량: "절대 가용 로봇 수 추정"
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']

    ratio_cols = [c for c in df.columns if c.startswith('ratio_')]
    return df, ratio_cols


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
def load_data():
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    # FE v1 파이프라인
    train, test = build_features(
        train, test, layout,
        lag_lags=[1,2,3,4,5,6],
        rolling_windows=[3,5,10],
    )

    # 시나리오 집계 피처 (11통계 — model22 동일)
    train = add_scenario_agg_features(train)
    test  = add_scenario_agg_features(test)
    sc_feats = [c for c in train.columns if c.startswith('sc_')]
    print(f'시나리오 집계 피처: {len(sc_feats)}종')

    # ★ 신규: Layout-capacity-normalized 비율 피처
    train, ratio_cols = add_layout_ratio_features(train)
    test, _           = add_layout_ratio_features(test)
    print(f'Layout 비율 피처: {len(ratio_cols)}종 → {ratio_cols}')

    return train, test


def get_feat_cols(train):
    return [c for c in train.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and train[c].dtype != object]


# ─────────────────────────────────────────────
# Layer 1: Base Learner OOF 생성 (model22 동일)
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
        print(f'  [LGBM] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
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
        print(f'  [CB] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
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


# ─────────────────────────────────────────────
# Layer 2: 메타 학습기 (model22 동일)
# ─────────────────────────────────────────────
def run_meta_lgbm(meta_train, meta_test, y_raw, groups, label='LGBM-meta'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_meta = np.zeros(len(y_raw)); test_meta = np.zeros(meta_test.shape[0])
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
        y_tr_log = np.log1p(y_raw.iloc[tr_idx].values)
        y_va_log = np.log1p(y_raw.iloc[va_idx].values)
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(X_tr, y_tr_log,
              eval_set=[(X_va, y_va_log)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[va_idx] = np.expm1(m.predict(X_va))
        test_meta += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    oof_mae = np.abs(oof_meta - y_raw.values).mean()
    pred_std = oof_meta.std()
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | pred_std={pred_std:.2f}')
    return oof_meta, test_meta, oof_mae


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 60)
    print('모델실험28A: Layout-aware + 분포 이동 보정')
    print('기준: Model22 CV ~8.51 / Public 9.9385 (배율 ~1.168)')
    print('변경: Layout-capacity 비율 피처 5종 추가')
    print('가설: 비율 피처가 test 분포 이동에 robust → 배율 개선')
    print('=' * 60)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    # 데이터 로드
    train, test = load_data()
    feat_cols = get_feat_cols(train)
    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'피처 수: {len(feat_cols)} (model22: 198 sc + 212 FE v1 = ~410, +5 ratio)')

    # 비율 피처 train vs test 분포 확인
    ratio_cols = [c for c in train.columns if c.startswith('ratio_')]
    print(f'\n비율 피처 train vs test 분포:')
    for col in ratio_cols:
        tr_m = train[col].mean()
        te_m = test[col].mean()
        tr_s = train[col].std()
        shift = abs(tr_m - te_m) / (tr_s + 1e-8)
        print(f'  {col:40s}: train={tr_m:.4f} test={te_m:.4f} shift={shift:.3f}σ')

    # 비교: 원본 피처의 shift
    orig_shifts = {
        'order_inflow_15m': 0.481,
        'congestion_score': 0.139,
        'low_battery_ratio': 0.197,
    }
    print(f'\n원본 피처 shift (비교):')
    for col, s in orig_shifts.items():
        print(f'  {col:40s}: shift={s:.3f}σ')

    # ══════════════════════════════════════════
    # Layer 1: 5모델 Base Learner OOF
    # ══════════════════════════════════════════
    print('\n' + '─' * 60)
    print('[Layer 1] Base Learner OOF 생성')
    print('─' * 60)

    # ── LGBM ──
    if ckpt_exists('lgbm'):
        print('\n[LGBM] 체크포인트 로드')
        oof_lg, test_lg = load_ckpt('lgbm')
    else:
        print('\n[LGBM] 학습 시작...')
        oof_lg, test_lg = train_lgbm_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('lgbm', oof_lg, test_lg)
    mae_lg = np.abs(np.expm1(oof_lg) - y_raw.values).mean()
    print(f'  LGBM OOF MAE={mae_lg:.4f}')

    # ── TW1.8 ──
    if ckpt_exists('tw18'):
        print('\n[TW1.8] 체크포인트 로드')
        oof_tw, test_tw = load_ckpt('tw18')
    else:
        print('\n[TW1.8] 학습 시작...')
        oof_tw, test_tw = train_tw18_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt('tw18', oof_tw, test_tw)
    mae_tw = np.abs(oof_tw - y_raw.values).mean()
    print(f'  TW1.8 OOF MAE={mae_tw:.4f}')

    # ── CatBoost ──
    if ckpt_exists('cb'):
        print('\n[CB] 체크포인트 로드')
        oof_cb, test_cb = load_ckpt('cb')
    else:
        print('\n[CB] 학습 시작...')
        oof_cb, test_cb = train_cb_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('cb', oof_cb, test_cb)
    mae_cb = np.abs(np.expm1(oof_cb) - y_raw.values).mean()
    print(f'  CB OOF MAE={mae_cb:.4f}')

    # ── ExtraTrees ──
    if ckpt_exists('et'):
        print('\n[ET] 체크포인트 로드')
        oof_et, test_et = load_ckpt('et')
    else:
        print('\n[ET] 학습 시작...')
        oof_et, test_et = train_et_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('et', oof_et, test_et)
    mae_et = np.abs(np.expm1(oof_et) - y_raw.values).mean()
    print(f'  ET OOF MAE={mae_et:.4f}')

    # ── RandomForest ──
    if ckpt_exists('rf'):
        print('\n[RF] 체크포인트 로드')
        oof_rf, test_rf = load_ckpt('rf')
    else:
        print('\n[RF] 학습 시작...')
        oof_rf, test_rf = train_rf_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('rf', oof_rf, test_rf)
    mae_rf = np.abs(np.expm1(oof_rf) - y_raw.values).mean()
    print(f'  RF OOF MAE={mae_rf:.4f}')

    # ══════════════════════════════════════════
    # OOF 상관관계
    # ══════════════════════════════════════════
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

    # ══════════════════════════════════════════
    # 가중 앙상블
    # ══════════════════════════════════════════
    arrs = [oof_raw['LGBM'], oof_raw['CB'], oof_raw['TW'], oof_raw['ET'], oof_raw['RF']]
    def loss5(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        return np.mean(np.abs(sum(w[i]*arrs[i] for i in range(5)) - y_raw.values))
    best_loss, best_w = np.inf, np.ones(5)/5
    for _ in range(500):
        w0 = np.random.dirichlet(np.ones(5))
        res = minimize(loss5, w0, method='Nelder-Mead')
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = np.abs(res.x) / np.abs(res.x).sum()
    print(f'\n  가중 앙상블 CV MAE: {best_loss:.4f}')
    print(f'    LGBM={best_w[0]:.3f}, CB={best_w[1]:.3f}, TW={best_w[2]:.3f}, '
          f'ET={best_w[3]:.3f}, RF={best_w[4]:.3f}')

    # ══════════════════════════════════════════
    # Layer 2: 메타 학습기
    # ══════════════════════════════════════════
    print('\n' + '─' * 60)
    print('[Layer 2] 5모델 LGBM 메타 학습기')
    print('─' * 60)

    test_tw_clipped = np.maximum(test_tw, 0)
    meta_train = np.column_stack([oof_lg, np.expm1(oof_cb) if False else oof_cb,
                                   np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf])
    meta_test  = np.column_stack([test_lg, test_cb,
                                   np.log1p(test_tw_clipped), test_et, test_rf])

    oof_meta, test_meta, mae_meta = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    # ── 제출 파일 ──
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    sub_path = os.path.join(SUB_DIR, 'model28A_layout_robust.csv')
    sample.to_csv(sub_path, index=False)
    print(f'\n제출 파일: {sub_path}')

    # ══════════════════════════════════════════
    # 타겟 구간별 분석
    # ══════════════════════════════════════════
    print('\n' + '─' * 60)
    print('[분석] 타겟 구간별 MAE')
    print('─' * 60)
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    for lo, hi in bins:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(oof_meta[mask] - y_raw.values[mask]).mean()
            print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d}  MAE={seg_mae:6.2f}')

    # ══════════════════════════════════════════
    # 예측 분포 분석
    # ══════════════════════════════════════════
    print('\n' + '─' * 60)
    print('[분석] 예측 분포')
    print('─' * 60)
    print(f'  OOF:  mean={oof_meta.mean():.2f}, std={oof_meta.std():.2f}, '
          f'min={oof_meta.min():.2f}, max={oof_meta.max():.2f}')
    print(f'  test: mean={test_meta.mean():.2f}, std={test_meta.std():.2f}, '
          f'min={test_meta.min():.2f}, max={test_meta.max():.2f}')

    # ══════════════════════════════════════════
    # 최종 요약
    # ══════════════════════════════════════════
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 60)
    print(f'모델실험28A 결과 ({elapsed:.1f}분 소요)')
    print('=' * 60)
    print(f'  LGBM  OOF MAE: {mae_lg:.4f}')
    print(f'  TW1.8 OOF MAE: {mae_tw:.4f}')
    print(f'  CB    OOF MAE: {mae_cb:.4f}')
    print(f'  ET    OOF MAE: {mae_et:.4f}')
    print(f'  RF    OOF MAE: {mae_rf:.4f}')
    print(f'  가중 앙상블    : {best_loss:.4f}')
    print(f'  메타 LGBM     : {mae_meta:.4f}  pred_std={oof_meta.std():.2f}')
    print(f'  test pred     : mean={test_meta.mean():.2f}, std={test_meta.std():.2f}')
    print()
    print(f'  Model22 (기준): ~8.51 / Public 9.9385 (배율 ~1.168)')
    print(f'  Model28A 변화 : {mae_meta - 8.51:+.4f}')
    print(f'  기대 Public (×1.168): {mae_meta * 1.168:.4f}')
    print(f'  기대 Public (×1.163): {mae_meta * 1.163:.4f}')

    # 비율 피처 효과 판정
    if test_meta.std() > 15.0:
        print(f'\n  ✅ test pred_std={test_meta.std():.2f} > 15.0 — model22(15.27) 대비 유지/확장')
    else:
        print(f'\n  ⚠️ test pred_std={test_meta.std():.2f} < 15.0 — 압축 우려')

    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
