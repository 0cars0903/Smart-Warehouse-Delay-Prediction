"""
model45: LDS 기반 극값 개선 3종 전략
================================================================
배경:
  target≥80 구간이 전체 MAE의 27.6%를 차지하지만,
  모든 base learner가 실제의 32~40%만 예측 (pred/actual = 0.32~0.40).
  이 스크립트는 극값 예측 개선을 위한 세 가지 독립 전략을 실험한다.

전략 A — LDS sample_weight (기존 파이프라인 재학습)
  - Label Distribution Smoothing으로 희귀 타깃(극값) 샘플에 역빈도 가중치 부여
  - LGBM, CatBoost의 weight/sample_weight 파라미터 활용
  - 기존 6모델 스태킹 파이프라인 구조 유지

전략 B — MLP + LDS 가중 재샘플링
  - sklearn MLPRegressor (sample_weight 미지원) → LDS 가중치 기반 오버샘플링
  - 극값 구간 샘플을 weight 비례로 복제 → 학습 분포 재조정
  - 기존 model43 대비 개선 가능성 확인

전략 C — High-Quantile 전용 모델 + Pareto-inspired Loss 분석
  - LGBM Quantile (q=0.85, 0.90, 0.95): 우측 꼬리 집중 예측
  - Pareto-weighted MAE: 극값에 크기 비례 가중치 부여하는 커스텀 loss
  - 새 base learner로 meta stacking에 투입, 다양성 기여 측정

기준: model34 Config B (CV 8.4803 / Public 9.8078)
목표: [80+] MAE 현재 81~91 → 70 이하 | 전체 CV 개선
실행: python src/run_model45_lds_extreme.py [--strategy A] [--strategy B] [--strategy C] [--all]
예상 시간: A≈25분, B≈30분, C≈20분
"""

import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings, gc, os, sys, time

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_BASE, '..', 'data')
SUB_DIR   = os.path.join(_BASE, '..', 'submissions')
CKPT_BASE = os.path.join(_BASE, '..', 'docs', 'model45_ckpt')
RANDOM_STATE = 42
N_SPLITS     = 5

# ── 파라미터 (model34 Config B 동일) ─────────────────────────────────────────
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
    'iterations': 3000, 'learning_rate': 0.05, 'depth': 6,
    'l2_leaf_reg': 3.0, 'loss_function': 'Tweedie:variance_power=1.5',
    'random_seed': RANDOM_STATE, 'verbose': 0, 'early_stopping_rounds': 50,
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


# ═══════════════════════════════════════════════════════════════════════════════
# LDS (Label Distribution Smoothing) 구현
# ═══════════════════════════════════════════════════════════════════════════════
def compute_lds_weights(y, num_bins=100, sigma=2.0, clip_pct=95, verbose=True):
    """
    Yang et al. (ICML 2021) LDS 가중치 계산
    - 레이블 히스토그램에 Gaussian 커널을 적용해 유효 밀도 추정
    - weight = 1 / effective_density (희귀 구간에 높은 가중치)

    Parameters
    ----------
    y        : 타깃 배열 (원본 스케일)
    num_bins : 히스토그램 구간 수
    sigma    : Gaussian 커널 표준편차 (클수록 더 많이 스무딩)
    clip_pct : 최대 가중치 클리핑 퍼센타일 (이상치 방지)
    """
    y_min, y_max = y.min(), y.max()
    bins = np.linspace(y_min, y_max, num_bins + 1)
    hist, _ = np.histogram(y, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # Gaussian kernel (scipy 없이 numpy로 구현)
    ks = int(4 * sigma) * 2 + 1  # 커널 크기 (홀수)
    x = np.arange(ks) - ks // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    pad = ks // 2
    hist_padded = np.pad(hist.astype(float), pad, mode='edge')
    smoothed = np.convolve(hist_padded, kernel, mode='valid')[:num_bins]
    smoothed = np.maximum(smoothed, 1e-8)

    # 각 샘플의 bin 인덱스 찾기 → smoothed density 보간
    bin_idx = np.searchsorted(bins[1:], y, side='left')
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    density = smoothed[bin_idx]

    # weight = 1 / density, 클리핑 후 정규화
    weights = 1.0 / density
    upper = np.percentile(weights, clip_pct)
    weights = np.clip(weights, weights.min(), upper)
    weights = weights / weights.mean()  # 평균 = 1.0으로 정규화

    if verbose:
        thr = np.percentile(y, 90)
        w_tail = weights[y >= thr].mean()
        w_body = weights[y <  thr].mean()
        print(f"  LDS 가중치 — 꼬리(≥p90) 평균: {w_tail:.3f} | 몸통(<p90) 평균: {w_body:.3f}")
        print(f"  가중치 범위: [{weights.min():.3f}, {weights.max():.3f}] | 평균: {weights.mean():.3f}")
        print(f"  극값(≥80) 샘플 수: {(y>=80).sum()} | 평균 가중치: {weights[y>=80].mean():.3f}")
    return weights


# ═══════════════════════════════════════════════════════════════════════════════
# FE (model34 동일)
# ═══════════════════════════════════════════════════════════════════════════════
def add_scenario_agg(df):
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
        df[f'sc_{col}_cv']       = (df[f'sc_{col}_std'] /
                                    (df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)
    return df

def safe_div(a, b, fill=0):
    return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

def add_ratio_features(df):
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
    if all(c in df.columns for c in ['sc_congestion_score_mean', 'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'], df['robot_total'] ** 2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm'] / 100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(df['pack_station_count'], df['floor_area_sqm'] / 1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns:
        df['ss_congestion_x_battery'] = (
            df.get('sc_congestion_score_mean', 0) * df.get('sc_low_battery_ratio_mean', 0))
        df['ss_order_x_util'] = (
            df.get('sc_order_inflow_15m_mean', 0) * df.get('sc_robot_utilization_mean', 0))
        df['ss_demand_x_congestion'] = (
            df.get('ratio_demand_per_robot', 0) * df.get('ratio_congestion_per_intersection', 0))
        df['ss_stress_x_pressure'] = (
            df.get('ratio_battery_stress', 0) * df.get('ratio_packing_pressure', 0))
        df['ss_idle_x_demand'] = (
            df.get('ratio_idle_fraction', 0) * df.get('ratio_demand_per_robot', 0))
    return df


# ── 체크포인트 ────────────────────────────────────────────────────────────────
def save_ckpt(d, name, oof, test_pred):
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f'{name}_oof.npy'),  oof)
    np.save(os.path.join(d, f'{name}_test.npy'), test_pred)

def load_ckpt(d, name):
    return (np.load(os.path.join(d, f'{name}_oof.npy')),
            np.load(os.path.join(d, f'{name}_test.npy')))

def ckpt_exists(d, name):
    return (os.path.exists(os.path.join(d, f'{name}_oof.npy')) and
            os.path.exists(os.path.join(d, f'{name}_test.npy')))


def load_model34_config_b():
    """
    model34 Config B 체크포인트 로드
    (여러 체크포인트 폴더에 분산 저장된 6개 base learner OOF/test 복원)
      lgbm, cb, et, rf  ← model31_ckpt
      tw15              ← model34_ckpt
      asym (α=2.0)      ← model34_ckpt (파일명: asym20)
    """
    docs = os.path.join(_BASE, '..', 'docs')
    m31  = os.path.join(docs, 'model31_ckpt')
    m34  = os.path.join(docs, 'model34_ckpt')

    mapping = {
        'lgbm': (m31, 'lgbm'),
        'cb':   (m31, 'cb'),
        'et':   (m31, 'et'),
        'rf':   (m31, 'rf'),
        'tw15': (m34, 'tw15'),
        'asym': (m34, 'asym20'),   # asym20_oof.npy → key: 'asym'
    }

    oof_dict  = {}
    test_dict = {}
    for key, (d, fname) in mapping.items():
        if ckpt_exists(d, fname):
            oof_dict[key], test_dict[key] = load_ckpt(d, fname)
        else:
            print(f"  ⚠️  model34 ckpt 없음: {fname} in {os.path.basename(d)}")

    if oof_dict:
        print(f"  model34 Config B 로드 완료: {list(oof_dict.keys())}")
    return oof_dict, test_dict


# ── 데이터 로드 ──────────────────────────────────────────────────────────────
def load_data():
    print("데이터 로드 중...")
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    train_fe, test_fe = build_features(train, test, layout, verbose=True)
    train_fe = add_scenario_agg(train_fe)
    test_fe  = add_scenario_agg(test_fe)
    train_fe = add_ratio_features(train_fe)
    test_fe  = add_ratio_features(test_fe)

    drop_cols = {'id', 'ID', 'target', 'scenario_id', 'timestamp',
                 'layout_id', 'avg_delay_minutes_next_30m'}
    feat_cols = [c for c in train_fe.columns
                 if c not in drop_cols
                 and c in test_fe.columns
                 and train_fe[c].dtype != object]

    target_col = ('avg_delay_minutes_next_30m'
                  if 'avg_delay_minutes_next_30m' in train_fe.columns else 'target')

    X_tr = np.nan_to_num(train_fe[feat_cols].values.astype(np.float32), nan=0.0)
    X_te = np.nan_to_num(test_fe[feat_cols].values.astype(np.float32),  nan=0.0)
    y_tr = train_fe[target_col].values.astype(np.float32)
    grp  = train_fe['scenario_id'].values

    print(f"  피처: {len(feat_cols)} | train: {X_tr.shape} | NaN→0 처리 완료")
    return X_tr, X_te, y_tr, grp, sample, feat_cols


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
def print_segment_mae(oof, y_tr, label=""):
    segs = [(0,5), (5,20), (20,50), (50,80), (80,800)]
    print(f"  구간별 MAE ({label}):")
    for lo, hi in segs:
        mask = (y_tr >= lo) & (y_tr < hi)
        if mask.sum() == 0: continue
        mae  = np.abs(oof[mask] - y_tr[mask]).mean()
        pr   = oof[mask].mean() / (y_tr[mask].mean() + 1e-8)
        print(f"    [{lo:3d},{hi:3d}) n={mask.sum():5d}  MAE={mae:.2f}  pred/actual={pr:.3f}")

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

def train_meta(oof_dict, test_dict, y_tr, grp, label=""):
    names = list(oof_dict.keys())
    X_meta_tr = np.column_stack([oof_dict[n] for n in names])
    X_meta_te = np.column_stack([test_dict[n] for n in names])
    y_log = np.log1p(y_tr)
    oof_meta  = np.zeros(len(y_tr))
    test_preds = []
    kf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_meta_tr, y_log, grp)):
        m = lgb.LGBMRegressor(**META_PARAMS)
        m.fit(X_meta_tr[tr_idx], y_log[tr_idx],
              eval_set=[(X_meta_tr[va_idx], y_log[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        oof_meta[va_idx] = np.expm1(m.predict(X_meta_tr[va_idx]))
        test_preds.append(np.expm1(m.predict(X_meta_te)))
    test_meta = np.mean(test_preds, axis=0)
    cv_mae = np.abs(oof_meta - y_tr).mean()
    print(f"  [{label}] 메타 CV MAE = {cv_mae:.4f} | pred_std = {test_meta.std():.2f}")
    return cv_mae, oof_meta, test_meta


# ═══════════════════════════════════════════════════════════════════════════════
# 전략 A — LDS sample_weight 재학습
# ═══════════════════════════════════════════════════════════════════════════════
def strategy_a(X_tr, X_te, y_tr, grp, sample):
    print(f"\n{'='*65}")
    print(f"  전략 A: LDS sample_weight 기반 6모델 재학습")
    print(f"{'='*65}")
    ckpt_dir = os.path.join(CKPT_BASE, 'strat_a')
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.time()

    print("\n  LDS 가중치 계산...")
    lds_w = compute_lds_weights(y_tr, num_bins=100, sigma=2.0, clip_pct=95)
    np.save(os.path.join(ckpt_dir, 'lds_weights.npy'), lds_w)

    y_log = np.log1p(y_tr)
    kf = GroupKFold(n_splits=N_SPLITS)
    oof_dict  = {}
    test_dict = {}

    for name in ['lgbm', 'cb', 'tw15', 'et', 'rf', 'asym']:
        if ckpt_exists(ckpt_dir, name):
            print(f"  [{name}] 체크포인트 로드")
            oof_dict[name], test_dict[name] = load_ckpt(ckpt_dir, name)
            continue

        print(f"\n  [{name}] LDS 가중치 적용 학습...")
        oof = np.zeros(len(y_tr))
        test_preds = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr, y_log, grp)):
            Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
            ytr, yva = y_log[tr_idx], y_log[va_idx]
            wtr      = lds_w[tr_idx]  # ← LDS 가중치

            if name == 'lgbm':
                dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
                dva = lgb.Dataset(Xva, label=yva)
                p = {k: v for k, v in LGBM_PARAMS.items()
                     if k not in ('n_estimators',)}
                p['n_jobs'] = -1
                m = lgb.train(p, dtr, num_boost_round=LGBM_PARAMS['n_estimators'],
                              valid_sets=[dva],
                              callbacks=[lgb.early_stopping(50, verbose=False),
                                         lgb.log_evaluation(-1)])
                oof[va_idx]    = np.expm1(m.predict(Xva))
                test_preds.append(np.expm1(m.predict(X_te)))

            elif name == 'cb':
                m = cb.CatBoostRegressor(**CB_PARAMS)
                m.fit(Xtr, ytr, sample_weight=wtr,
                      eval_set=(Xva, yva), use_best_model=True, verbose=0)
                oof[va_idx]    = np.expm1(m.predict(Xva))
                test_preds.append(np.expm1(m.predict(X_te)))

            elif name == 'tw15':
                # TW15: raw target (CatBoost Tweedie)
                ytr_raw = y_tr[tr_idx]
                yva_raw = y_tr[va_idx]
                m = cb.CatBoostRegressor(**TW15_PARAMS)
                m.fit(Xtr, ytr_raw, sample_weight=wtr,
                      eval_set=(Xva, yva_raw), use_best_model=True, verbose=0)
                oof[va_idx]    = m.predict(Xva)
                test_preds.append(m.predict(X_te))

            elif name == 'et':
                # ET/RF: sample_weight 지원 ✅
                m = ExtraTreesRegressor(**ET_PARAMS)
                m.fit(Xtr, ytr, sample_weight=wtr)
                oof[va_idx]    = np.expm1(m.predict(Xva))
                test_preds.append(np.expm1(m.predict(X_te)))

            elif name == 'rf':
                m = RandomForestRegressor(**RF_PARAMS)
                m.fit(Xtr, ytr, sample_weight=wtr)
                oof[va_idx]    = np.expm1(m.predict(Xva))
                test_preds.append(np.expm1(m.predict(X_te)))

            elif name == 'asym':
                raw_p = dict(ASYM_PARAMS)
                raw_p['objective'] = asym_obj
                num_boost = raw_p.pop('n_estimators')
                dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
                dva = lgb.Dataset(Xva, label=yva)
                m = lgb.train(raw_p, dtr, num_boost_round=num_boost,
                              valid_sets=[dva], feval=asym_metric,
                              callbacks=[lgb.early_stopping(50, verbose=False),
                                         lgb.log_evaluation(-1)])
                oof[va_idx]    = np.expm1(m.predict(Xva))
                test_preds.append(np.expm1(m.predict(X_te)))

            fold_mae = np.abs(oof[va_idx] - y_tr[va_idx]).mean()
            print(f"    fold {fold+1}/{N_SPLITS}  MAE={fold_mae:.4f}")
            gc.collect()

        test_pred = np.mean(test_preds, axis=0)
        oof_mae   = np.abs(oof - y_tr).mean()
        print(f"  [{name}] OOF={oof_mae:.4f} | test_std={test_pred.std():.2f}")
        oof_dict[name]  = oof
        test_dict[name] = test_pred
        save_ckpt(ckpt_dir, name, oof, test_pred)

    cv_mae, oof_meta, test_meta = train_meta(oof_dict, test_dict, y_tr, grp, label="A-LDS")
    print_segment_mae(oof_meta, y_tr, "A-LDS 메타")
    print(f"  경과 시간: {(time.time()-t0)/60:.1f}분")

    sub = sample.copy()
    pred_col = [c for c in sub.columns if c != 'ID'][0]
    sub[pred_col] = np.clip(test_meta, 0, None)
    sub_path = os.path.join(SUB_DIR, f'model45a_lds_cv{cv_mae:.4f}.csv')
    sub.to_csv(sub_path, index=False)
    print(f"  저장: {os.path.basename(sub_path)}")
    return cv_mae, test_meta.std()


# ═══════════════════════════════════════════════════════════════════════════════
# 전략 B — MLP + LDS 오버샘플링
# ═══════════════════════════════════════════════════════════════════════════════
def strategy_b(X_tr, X_te, y_tr, grp, sample):
    """
    sklearn MLPRegressor는 sample_weight 미지원.
    → LDS 가중치를 활용한 오버샘플링: weight 비례로 극값 샘플 복제.
    → 기존 model34 OOF(lgbm/cb/tw15/et/rf/asym)와 결합해 7모델 스태킹 테스트.
    """
    print(f"\n{'='*65}")
    print(f"  전략 B: MLP + LDS 가중 오버샘플링")
    print(f"{'='*65}")
    ckpt_dir = os.path.join(CKPT_BASE, 'strat_b')
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.time()

    lds_w_path = os.path.join(CKPT_BASE, 'strat_a', 'lds_weights.npy')
    if os.path.exists(lds_w_path):
        lds_w = np.load(lds_w_path)
        print("  LDS 가중치 로드 (전략 A에서)")
    else:
        print("  LDS 가중치 새로 계산...")
        lds_w = compute_lds_weights(y_tr, verbose=True)

    if ckpt_exists(ckpt_dir, 'mlp'):
        print("  [mlp] 체크포인트 로드")
        mlp_oof, mlp_test = load_ckpt(ckpt_dir, 'mlp')
    else:
        print("\n  [mlp] LDS 오버샘플링 + MLP 학습...")

        # 스케일링
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        y_log   = np.log1p(y_tr)

        mlp_oof    = np.zeros(len(y_tr))
        test_preds = []
        kf = GroupKFold(n_splits=N_SPLITS)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr_sc, y_log, grp)):
            Xtr, Xva = X_tr_sc[tr_idx], X_tr_sc[va_idx]
            ytr, yva = y_log[tr_idx],   y_log[va_idx]
            wtr      = lds_w[tr_idx]

            # ── LDS 오버샘플링: 가중치 비례로 인덱스 샘플링 ──
            # 정수 반올림으로 복제 횟수 결정 (최대 5배 제한)
            rep_counts = np.clip(np.round(wtr).astype(int), 1, 5)
            aug_idx    = np.repeat(np.arange(len(Xtr)), rep_counts)
            Xtr_aug    = Xtr[aug_idx]
            ytr_aug    = ytr[aug_idx]

            print(f"    fold {fold+1}  원본: {len(Xtr)} → 증강 후: {len(Xtr_aug)}"
                  f" ({len(Xtr_aug)/len(Xtr):.1f}x)")

            mlp = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                max_iter=200,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, tol=1e-4,
                random_state=RANDOM_STATE + fold,
                verbose=False,
            )
            mlp.fit(Xtr_aug, ytr_aug)

            mlp_oof[va_idx] = np.expm1(mlp.predict(Xva))
            test_preds.append(np.expm1(mlp.predict(X_te_sc)))

            fold_mae = np.abs(mlp_oof[va_idx] - y_tr[va_idx]).mean()
            print(f"    fold {fold+1}  MAE={fold_mae:.4f} | iter={mlp.n_iter_}")
            gc.collect()

        mlp_test = np.mean(test_preds, axis=0)
        mlp_oof_mae = np.abs(mlp_oof - y_tr).mean()
        print(f"\n  [mlp] OOF MAE={mlp_oof_mae:.4f} | test_std={mlp_test.std():.2f}")
        save_ckpt(ckpt_dir, 'mlp', mlp_oof, mlp_test)

    mlp_oof_mae = np.abs(mlp_oof - y_tr).mean()
    print_segment_mae(mlp_oof, y_tr, "MLP+LDS OOF")

    # ── MLP와 기존 model34 base learners 상관 확인 ──
    oof_dict, test_dict = load_model34_config_b()
    if 'lgbm' in oof_dict:
        lgbm_oof = oof_dict['lgbm']
        corr = np.corrcoef(mlp_oof, lgbm_oof)[0, 1]
        print(f"\n  MLP-LGBM 상관: {corr:.4f}  (< 0.92이면 다양성 유효)")

        if corr < 0.92 and mlp_oof_mae < 9.5:
            print("  ✅ 다양성 기준 통과 → 7모델 스태킹 진행")
            oof_dict['mlp']  = mlp_oof
            test_dict['mlp'] = mlp_test
            cv_mae, oof_meta, test_meta = train_meta(
                oof_dict, test_dict, y_tr, grp, label="B-MLP7모델")
            print_segment_mae(oof_meta, y_tr, "B-MLP 7모델 메타")

            sub = sample.copy()
            pred_col = [c for c in sub.columns if c != 'ID'][0]
            sub[pred_col] = np.clip(test_meta, 0, None)
            sub_path = os.path.join(SUB_DIR, f'model45b_mlp7_cv{cv_mae:.4f}.csv')
            sub.to_csv(sub_path, index=False)
            print(f"  저장: {os.path.basename(sub_path)}")
        else:
            print(f"  ❌ 다양성 기준 미달 (corr={corr:.4f}, MAE={mlp_oof_mae:.4f}) → 스태킹 생략")
            cv_mae = mlp_oof_mae
    else:
        print("  model34 체크포인트 없음 — MLP 단독 OOF만 기록")
        cv_mae = mlp_oof_mae

    print(f"  경과 시간: {(time.time()-t0)/60:.1f}분")
    return cv_mae


# ═══════════════════════════════════════════════════════════════════════════════
# 전략 C — High-Quantile 모델 + Pareto-weighted MAE 분석
# ═══════════════════════════════════════════════════════════════════════════════
def strategy_c(X_tr, X_te, y_tr, grp, sample):
    """
    C-1. High-Quantile 전용 모델 (q=0.85, 0.90, 0.95)
         - 우측 꼬리 예측에 특화된 base learner
         - model34 OOF와 함께 meta stacking → 극값 coverage 개선 가능성

    C-2. Pareto-weighted MAE Custom Loss
         - weight(y) = (y / threshold)^alpha  for y > threshold
         - 크기에 비례한 손실 가중치 → Pareto 분포 heavy-tail 모방
         - alpha=0.5, 1.0, 1.5 탐색
    """
    print(f"\n{'='*65}")
    print(f"  전략 C: High-Quantile 모델 + Pareto-weighted Loss")
    print(f"{'='*65}")
    ckpt_dir = os.path.join(CKPT_BASE, 'strat_c')
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.time()

    y_log = np.log1p(y_tr)
    kf = GroupKFold(n_splits=N_SPLITS)

    # ── C-1: High-Quantile 모델 ────────────────────────────────────────────
    quantiles = [0.85, 0.90, 0.95]
    q_oof_dict  = {}
    q_test_dict = {}

    for q in quantiles:
        qname = f'q{int(q*100)}'
        if ckpt_exists(ckpt_dir, qname):
            print(f"  [{qname}] 체크포인트 로드")
            q_oof_dict[qname], q_test_dict[qname] = load_ckpt(ckpt_dir, qname)
            continue

        print(f"\n  [{qname}] Quantile Regression (q={q}) 학습...")
        qparams = {
            'num_leaves': 127, 'learning_rate': 0.02,
            'feature_fraction': 0.50, 'bagging_fraction': 0.90,
            'min_child_samples': 30, 'reg_alpha': 1.0, 'reg_lambda': 0.5,
            'objective': 'quantile', 'alpha': q,
            'n_estimators': 2000, 'bagging_freq': 1,
            'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
        }
        oof = np.zeros(len(y_tr))
        test_preds = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr, y_log, grp)):
            Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
            ytr, yva = y_log[tr_idx], y_log[va_idx]
            m = lgb.LGBMRegressor(**qparams)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
            oof[va_idx]    = np.expm1(m.predict(Xva))
            test_preds.append(np.expm1(m.predict(X_te)))
            fold_mae = np.abs(oof[va_idx] - y_tr[va_idx]).mean()
            print(f"    fold {fold+1}/{N_SPLITS}  MAE={fold_mae:.4f}")
            gc.collect()

        test_pred = np.mean(test_preds, axis=0)
        oof_mae   = np.abs(oof - y_tr).mean()
        print(f"  [{qname}] OOF={oof_mae:.4f} | test_std={test_pred.std():.2f}")
        save_ckpt(ckpt_dir, qname, oof, test_pred)
        q_oof_dict[qname]  = oof
        q_test_dict[qname] = test_pred

    # Quantile 모델 분석
    print(f"\n  ── Quantile 모델 구간별 분석 ──")
    for qname, oof in q_oof_dict.items():
        oof_mae = np.abs(oof - y_tr).mean()
        tail_mask = y_tr >= 80
        tail_pr   = oof[tail_mask].mean() / (y_tr[tail_mask].mean() + 1e-8) if tail_mask.sum() > 0 else 0
        print(f"  {qname}: OOF={oof_mae:.4f} | [80+] pred/actual={tail_pr:.3f}")

    # model34 OOF와 상관 및 다양성 확인
    m34_oof_dict, m34_test_dict = load_model34_config_b()
    if 'lgbm' in m34_oof_dict:
        lgbm_oof = m34_oof_dict['lgbm']
        print(f"\n  ── model34 LGBM과 상관 ──")
        for qname, oof in q_oof_dict.items():
            corr = np.corrcoef(oof, lgbm_oof)[0, 1]
            print(f"  LGBM-{qname} 상관: {corr:.4f}")

        # 가장 유망한 quantile 모델로 7모델 스태킹
        best_q = min(q_oof_dict.keys(),
                     key=lambda k: np.abs(q_oof_dict[k] - y_tr).mean())
        print(f"\n  최적 quantile 모델: {best_q} → 7모델 스태킹 진행")
        oof_dict, test_dict = load_model34_config_b()
        oof_dict[best_q]  = q_oof_dict[best_q]
        test_dict[best_q] = q_test_dict[best_q]
        cv_q, oof_meta_q, test_meta_q = train_meta(
            oof_dict, test_dict, y_tr, grp, label=f"C-Q7({best_q})")
        print_segment_mae(oof_meta_q, y_tr, f"C-Q7({best_q}) 메타")

        sub = sample.copy()
        pred_col = [c for c in sub.columns if c != 'ID'][0]
        sub[pred_col] = np.clip(test_meta_q, 0, None)
        sub_path = os.path.join(SUB_DIR, f'model45c_q7_{best_q}_cv{cv_q:.4f}.csv')
        sub.to_csv(sub_path, index=False)
        print(f"  저장: {os.path.basename(sub_path)}")

    # ── C-2: Pareto-weighted MAE 분석 ─────────────────────────────────────
    print(f"\n  ── Pareto-weighted MAE Loss 분석 ──")
    threshold_raw = np.percentile(y_tr, 90)   # raw 스케일 p90
    threshold_log = np.log1p(threshold_raw)   # log1p 스케일로 변환 (dtrain label과 동일 단위)
    print(f"  임계값 (p90): raw={threshold_raw:.2f} | log1p={threshold_log:.4f}")

    # ── make_pareto_obj: dtrain label은 log1p 스케일임에 유의 ──────────────
    # y_true (log1p) → expm1 → raw 비교 → weight 계산 → log1p gradient 적용
    def make_pareto_obj(a, thr_raw):
        def pareto_obj(y_pred, dtrain):
            y_true  = dtrain.get_label()          # log1p 스케일
            raw_y   = np.expm1(y_true)            # raw 스케일로 복원
            residual = y_true - y_pred            # gradient 계산은 log1p 공간
            # 임계값 비교는 raw 스케일에서 수행
            w = np.where(raw_y > thr_raw,
                         (raw_y / (thr_raw + 1e-8)) ** a, 1.0)
            w = np.clip(w, 1.0, 10.0)
            grad = np.where(residual > 0, -w, w)
            hess = w
            return grad, hess
        return pareto_obj

    def pareto_metric(y_pred, dtrain):
        y_true = dtrain.get_label()               # log1p 스케일
        mae = np.abs(np.expm1(y_pred) - np.expm1(y_true)).mean()
        return 'pareto_mae', mae, False

    for alpha_p in [0.5, 1.0, 1.5]:
        pname = f'pareto_a{int(alpha_p*10)}'
        if ckpt_exists(ckpt_dir, pname):
            p_oof, _ = load_ckpt(ckpt_dir, pname)
            p_mae = np.abs(p_oof - y_tr).mean()
            tail_pr = p_oof[y_tr >= 80].mean() / (y_tr[y_tr >= 80].mean() + 1e-8)
            print(f"  [pareto α={alpha_p}] 체크포인트: OOF={p_mae:.4f} | [80+] pred/actual={tail_pr:.3f}")
            continue

        print(f"\n  [pareto α={alpha_p}] 학습...")
        # params['objective'] = callable 방식 (lgb.train 현재 버전 기준)
        p_params = {
            'num_leaves': 129, 'learning_rate': 0.015,
            'feature_fraction': 0.50, 'bagging_fraction': 0.90,
            'min_child_samples': 30, 'reg_alpha': 1.5, 'reg_lambda': 0.5,
            'n_estimators': 3000, 'bagging_freq': 1,
            'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
            'objective': make_pareto_obj(alpha_p, threshold_raw),  # callable 직접 주입
        }

        p_oof = np.zeros(len(y_tr))
        p_test_preds = []
        num_boost = p_params.pop('n_estimators')

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr, y_log, grp)):
            dtr = lgb.Dataset(X_tr[tr_idx], label=y_log[tr_idx])
            dva = lgb.Dataset(X_tr[va_idx], label=y_log[va_idx])
            m = lgb.train(p_params, dtr, num_boost_round=num_boost,
                          valid_sets=[dva], feval=pareto_metric,
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(-1)])
            p_oof[va_idx] = np.expm1(m.predict(X_tr[va_idx]))
            p_test_preds.append(np.expm1(m.predict(X_te)))
            fold_mae = np.abs(p_oof[va_idx] - y_tr[va_idx]).mean()
            print(f"    fold {fold+1}/{N_SPLITS}  MAE={fold_mae:.4f}")
            gc.collect()

        p_test = np.mean(p_test_preds, axis=0)
        p_mae  = np.abs(p_oof - y_tr).mean()
        tail_pr = p_oof[y_tr >= 80].mean() / (y_tr[y_tr >= 80].mean() + 1e-8)
        print(f"  [pareto α={alpha_p}] OOF={p_mae:.4f} | [80+] pred/actual={tail_pr:.3f} | std={p_test.std():.2f}")
        save_ckpt(ckpt_dir, pname, p_oof, p_test)

    print(f"\n  경과 시간: {(time.time()-t0)/60:.1f}분")

    # 최종 분석 리포트
    print(f"\n  ── 전략 C 최종 분석 ──")
    print(f"  기준 model34 [80+] pred/actual ≈ 0.40 (MAE ≈ 81)")
    print(f"  개선 목표: pred/actual ≥ 0.50 또는 MAE ≤ 70")


# ═══════════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', action='append',
                        choices=['A', 'B', 'C'],
                        help='실행할 전략 (A/B/C, 여러 번 사용 가능)')
    parser.add_argument('--all', action='store_true', help='A+B+C 전체 실행')
    args = parser.parse_args()

    strategies = args.strategy if args.strategy else []
    if args.all or not strategies:
        strategies = ['A', 'B', 'C']
    strategies = sorted(set(strategies))

    print(f"실행 전략: {strategies}")
    print(f"기준: model34 Config B | CV 8.4803 | Public 9.8078 | [80+] MAE ≈ 81")

    X_tr, X_te, y_tr, grp, sample, _ = load_data()

    results = {}
    if 'A' in strategies:
        cv, std = strategy_a(X_tr, X_te, y_tr, grp, sample)
        results['A'] = {'cv': cv, 'std': std}

    if 'B' in strategies:
        cv = strategy_b(X_tr, X_te, y_tr, grp, sample)
        results['B'] = {'cv': cv}

    if 'C' in strategies:
        strategy_c(X_tr, X_te, y_tr, grp, sample)

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  model45 전략별 요약")
    print(f"{'='*65}")
    print(f"  기준 (model34 Config B): CV=8.4803 | pred_std≈16.15")
    if 'A' in results:
        r = results['A']
        diff = r['cv'] - 8.4803
        mark = "✅" if diff < 0 else "❌"
        print(f"  {mark} 전략A (LDS 가중치): CV={r['cv']:.4f} (Δ{diff:+.4f}) | pred_std={r['std']:.2f}")
    if 'B' in results:
        r = results['B']
        print(f"  전략B (MLP+LDS): MLP OOF={r['cv']:.4f}")
    print(f"\n  제출 파일: submissions/model45[a/b/c]_*.csv")
    print(f"{'='*65}")
