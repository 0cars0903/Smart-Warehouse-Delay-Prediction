"""
모델실험29A: 비율 피처 확장 (model28A 기반)
=============================================================
model28A(Public 9.8525 🏆)의 5종 비율 피처를 확장.
"용량 대비 부하" 원리를 추가 차원으로 적용.

핵심 원칙 (model28A에서 확인):
  - 비율 피처는 분포 이동에 invariant → 배율 개선
  - base learner 수준에서 효과 → CV 개선 + test 분포 확장 동시
  - 피처 수 최소화 → 과적합 방지 (model28A: +5종으로 Δ-0.065 CV)

확장 전략 (보수적 — LAW-4 "FE 확장=배율 악화" 위험 인지):
  Tier 1 (높은 확신, 5종):
    - 기존 model28A 비율 5종 유지
  Tier 2 (중간 확신, +7종 신규):
    - 교차 비율: demand × congestion / capacity
    - 면적 밀도: robot/area, pack/area
    - 시간적 비율: current_inflow / scenario_mean_inflow
    - 배터리 효율: battery_mean / robot_total
  총 12종 비율 → 약 420~425 피처 예상

⚠️ 과적합 방어:
  - 비율 피처만 추가 (절대값 피처 추가 없음)
  - 모든 비율은 layout capacity로 정규화 → shift invariant 보장
  - 신규 비율의 train-test shift 출력으로 사전 검증

실행: python src/run_exp_v3_model29A_ratio_expand.py
예상 시간: ~90분 (5모델 × 5fold 전체 재학습)
출력: submissions/model29A_ratio_expand.csv
체크포인트: docs/model29A_ckpt/
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
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model29A_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

# model28A 동일 하이퍼파라미터
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


def save_ckpt(name, oof, test_pred):
    np.save(os.path.join(CKPT_DIR, f'{name}_oof.npy'), oof)
    np.save(os.path.join(CKPT_DIR, f'{name}_test.npy'), test_pred)

def load_ckpt(name):
    return (np.load(os.path.join(CKPT_DIR, f'{name}_oof.npy')),
            np.load(os.path.join(CKPT_DIR, f'{name}_test.npy')))

def ckpt_exists(name):
    return (os.path.exists(os.path.join(CKPT_DIR, f'{name}_oof.npy'))
            and os.path.exists(os.path.join(CKPT_DIR, f'{name}_test.npy')))


# ─────────────────────────────────────────────
# 시나리오 집계 피처 (model28A 동일 — 11통계)
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
# ★ 비율 피처: Tier 2 (신규 확장 7종)
# ─────────────────────────────────────────────
def add_layout_ratio_features_tier2(df):
    """
    신규 비율 피처 7종 — 모두 layout capacity 정규화
    """
    def safe_div(a, b, fill=0):
        return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

    # 6. 교차 스트레스: "혼잡 × 수요 / 로봇 용량" — 복합 부하 지표
    if all(c in df.columns for c in ['sc_congestion_score_mean', 'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'],
            df['robot_total'] ** 2)  # 제곱으로 스케일 보정

    # 7. 로봇 밀도: "면적당 로봇 수" — 물리적 혼잡 지표
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(
            df['robot_total'], df['floor_area_sqm'] / 100)  # 100㎡당

    # 8. 패킹 밀도: "면적당 패킹 스테이션" — 처리 인프라 밀도
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(
            df['pack_station_count'], df['floor_area_sqm'] / 1000)  # 1000㎡당

    # 9. 충전 여유율: "충전기당 충전 중 로봇 수" — 충전 경쟁 강도
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(
            df['sc_robot_charging_mean'], df['charger_count'])

    # 10. 배터리 효율: "로봇당 평균 배터리" — 운영 효율 지표
    if 'sc_battery_mean_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_battery_per_robot'] = safe_div(
            df['sc_battery_mean_mean'] * df['robot_total'],
            df['robot_total'])  # = sc_battery_mean_mean (자체가 비율이지만 layout과 결합)
        # 실제: battery_mean * utilization / charger — 복합 효율
        if 'sc_robot_utilization_mean' in df.columns and 'charger_count' in df.columns:
            df['ratio_battery_per_robot'] = safe_div(
                df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'],
                df['charger_count'])

    # 11. 통로 혼잡률: "혼잡도 / 통로 폭" — 좁은 통로일수록 혼잡 영향 큼
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(
            df['sc_congestion_score_mean'], df['aisle_width_avg'])

    # 12. 유휴 비율: "유휴 로봇 / 전체 로봇" — 여유 용량 비율
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

    # Tier 2 비율 피처 (신규)
    train = add_layout_ratio_features_tier2(train)
    test  = add_layout_ratio_features_tier2(test)

    ratio_cols = [c for c in train.columns if c.startswith('ratio_')]
    return train, test, ratio_cols


def get_feat_cols(train):
    return [c for c in train.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and train[c].dtype != object]


# ─────────────────────────────────────────────
# Base Learner 학습 함수 (model28A 동일)
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
    print('모델실험29A: 비율 피처 확장')
    print('기준: Model28A CV 8.4743 / Public 9.8525 (배율 1.1626)')
    print('변경: Tier 2 비율 피처 +7종 추가 (총 12종)')
    print('=' * 60)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    train, test, ratio_cols = load_data()
    feat_cols = get_feat_cols(train)
    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']

    print(f'피처 수: {len(feat_cols)} (model28A: 415)')
    print(f'비율 피처: {len(ratio_cols)}종')

    # 비율 피처 shift 분석
    print(f'\n비율 피처 train vs test shift:')
    for col in ratio_cols:
        tr_m = train[col].mean(); te_m = test[col].mean()
        tr_s = train[col].std()
        shift = abs(tr_m - te_m) / (tr_s + 1e-8)
        marker = '✅' if shift < 0.4 else '⚠️'
        print(f'  {col:40s}: shift={shift:.3f}σ {marker}  '
              f'(train={tr_m:.4f}, test={te_m:.4f})')

    # ── Layer 1: Base Learner ──
    print('\n' + '─' * 60)
    print('[Layer 1] Base Learner OOF 생성')
    print('─' * 60)

    if ckpt_exists('lgbm'):
        print('\n[LGBM] 체크포인트 로드'); oof_lg, test_lg = load_ckpt('lgbm')
    else:
        print('\n[LGBM] 학습 시작...')
        oof_lg, test_lg = train_lgbm_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('lgbm', oof_lg, test_lg)
    print(f'  LGBM OOF MAE={np.abs(np.expm1(oof_lg) - y_raw.values).mean():.4f}')

    if ckpt_exists('tw18'):
        print('\n[TW1.8] 체크포인트 로드'); oof_tw, test_tw = load_ckpt('tw18')
    else:
        print('\n[TW1.8] 학습 시작...')
        oof_tw, test_tw = train_tw18_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt('tw18', oof_tw, test_tw)
    print(f'  TW1.8 OOF MAE={np.abs(oof_tw - y_raw.values).mean():.4f}')

    if ckpt_exists('cb'):
        print('\n[CB] 체크포인트 로드'); oof_cb, test_cb = load_ckpt('cb')
    else:
        print('\n[CB] 학습 시작...')
        oof_cb, test_cb = train_cb_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('cb', oof_cb, test_cb)
    print(f'  CB OOF MAE={np.abs(np.expm1(oof_cb) - y_raw.values).mean():.4f}')

    if ckpt_exists('et'):
        print('\n[ET] 체크포인트 로드'); oof_et, test_et = load_ckpt('et')
    else:
        print('\n[ET] 학습 시작...')
        oof_et, test_et = train_et_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('et', oof_et, test_et)
    print(f'  ET OOF MAE={np.abs(np.expm1(oof_et) - y_raw.values).mean():.4f}')

    if ckpt_exists('rf'):
        print('\n[RF] 체크포인트 로드'); oof_rf, test_rf = load_ckpt('rf')
    else:
        print('\n[RF] 학습 시작...')
        oof_rf, test_rf = train_rf_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('rf', oof_rf, test_rf)
    print(f'  RF OOF MAE={np.abs(np.expm1(oof_rf) - y_raw.values).mean():.4f}')

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
    sub_path = os.path.join(SUB_DIR, 'model29A_ratio_expand.csv')
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

    # 최종 요약
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 60)
    print(f'모델실험29A 결과 ({elapsed:.1f}분 소요)')
    print('=' * 60)
    print(f'  피처 수      : {len(feat_cols)} (model28A: 415)')
    print(f'  메타 LGBM    : {mae_meta:.4f}  pred_std={oof_meta.std():.2f}')
    print(f'  test pred    : mean={test_meta.mean():.2f}, std={test_meta.std():.2f}')
    print(f'  Model28A 기준: CV 8.4743 / Public 9.8525 (배율 1.1626)')
    print(f'  Model29A 변화: {mae_meta - 8.4743:+.4f}')
    print(f'  기대 Public (×1.163): {mae_meta * 1.163:.4f}')
    print(f'  기대 Public (×1.168): {mae_meta * 1.168:.4f}')
    if test_meta.std() > 16.0:
        print(f'\n  ✅ test std={test_meta.std():.2f} > 16.0 (model28A 수준 유지)')
    else:
        print(f'\n  ⚠️ test std={test_meta.std():.2f} < 16.0 (model28A 16.28 대비 압축)')
    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
