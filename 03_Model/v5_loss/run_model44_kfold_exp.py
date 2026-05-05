"""
model44: GroupKFold k=3 / k=5 / k=10 비교 실험
================================================================
핵심 가설:
  k=3  → test 예측 = 3개 모델 평균 → 압축 최소 → pred_std ↑ → 배율 개선 가능
  k=5  → 현재 기준 (model34 Config B, Public 9.8078)
  k=10 → test 예측 = 10개 모델 평균 → 압축 최대 → pred_std ↓ → 배율 악화 예상

검증 지표:
  1. CV MAE     : OOF 성능 (낮을수록 좋음)
  2. pred_std   : 제출 예측 표준편차 (≥15.5가 목표)
  3. 배율 추정  : pred_std / 실제 target std (27.4)로 일반화 품질 간접 측정

실행: python src/run_model44_kfold_exp.py [--k 3] [--k 5] [--k 10] [--all]
기본: --all (k=3,5,10 전체 실행)

예상 시간:
  k=3  : ~12분 (신규 학습 6모델×3fold + 메타)
  k=5  : ~18분 (신규 학습 6모델×5fold + 메타)  ← k=5 ckpt 있으면 스킵
  k=10 : ~35분 (신규 학습 6모델×10fold + 메타)
"""

import argparse
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
CKPT_BASE = os.path.join(_BASE, '..', 'docs', 'model44_ckpt')
RANDOM_STATE = 42

# ── 파라미터 (model34 Config B 기준) ──────────────────────────────────────
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
META_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}
ASYM_ALPHA = 2.0
ASYM_PARAMS = {
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


# ── Custom Loss ──────────────────────────────────────────────────────────────
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


# ── FE ────────────────────────────────────────────────────────────────────────
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

def add_shift_safe(df):
    """shift-safe cross 피처 (model31 동일)"""
    sc = [c for c in df.columns if c.startswith('sc_') and c.endswith('_mean')]
    ratio = [c for c in df.columns if c.startswith('ratio_')]
    if len(sc) >= 2:
        df['ss_congestion_x_battery'] = (
            df.get('sc_congestion_score_mean', 0) * df.get('sc_low_battery_ratio_mean', 0))
        df['ss_order_x_util'] = (
            df.get('sc_order_inflow_15m_mean', 0) * df.get('sc_robot_utilization_mean', 0))
        df['ss_idle_x_battery'] = (
            df.get('sc_robot_idle_mean', 0) * df.get('sc_battery_mean_mean', 0))
    if ratio:
        df['ss_demand_x_congestion'] = (
            df.get('ratio_demand_per_robot', 0) * df.get('ratio_congestion_per_intersection', 0))
        df['ss_stress_x_pressure'] = (
            df.get('ratio_battery_stress', 0) * df.get('ratio_packing_pressure', 0))
        df['ss_density_x_competition'] = (
            df.get('ratio_robot_density', 0) * df.get('ratio_charge_competition', 0))
        df['ss_idle_x_demand'] = (
            df.get('ratio_idle_fraction', 0) * df.get('ratio_demand_per_robot', 0))
    return df


# ── 데이터 로드 & FE ─────────────────────────────────────────────────────────
def load_data():
    print("데이터 로드 중...")
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    # build_features: (train, test, layout) → (train_fe, test_fe) 동시 반환
    train_fe, test_fe = build_features(train, test, layout, verbose=True)

    train_fe = add_scenario_agg(train_fe)
    test_fe  = add_scenario_agg(test_fe)
    train_fe = add_ratio_tier1(train_fe)
    test_fe  = add_ratio_tier1(test_fe)
    train_fe = add_ratio_tier2(train_fe)
    test_fe  = add_ratio_tier2(test_fe)
    train_fe = add_shift_safe(train_fe)
    test_fe  = add_shift_safe(test_fe)

    drop_cols = {'id', 'ID', 'target', 'scenario_id', 'timestamp',
                 'layout_id', 'avg_delay_minutes_next_30m'}
    feat_cols = [c for c in train_fe.columns
                 if c not in drop_cols
                 and c in test_fe.columns
                 and train_fe[c].dtype != object]

    # target 컬럼명 자동 탐지
    target_col = 'avg_delay_minutes_next_30m' if 'avg_delay_minutes_next_30m' in train_fe.columns else 'target'

    X_tr = np.nan_to_num(train_fe[feat_cols].values.astype(np.float32), nan=0.0)
    X_te = np.nan_to_num(test_fe[feat_cols].values.astype(np.float32),  nan=0.0)
    y_tr = train_fe[target_col].values.astype(np.float32)
    grp  = train_fe['scenario_id'].values

    nan_cnt = np.isnan(train_fe[feat_cols].values).sum()
    print(f"  train: {X_tr.shape} | test: {X_te.shape} | 피처: {len(feat_cols)} | NaN→0 처리: {nan_cnt}개")
    return X_tr, X_te, y_tr, grp, sample


# ── Base Learner 학습 ────────────────────────────────────────────────────────
def train_base_learner(name, X_tr, y_tr, X_te, grp, n_splits, ckpt_dir):
    if ckpt_exists(ckpt_dir, name):
        print(f"  [{name}] 체크포인트 로드")
        return load_ckpt(ckpt_dir, name)

    y_log = np.log1p(y_tr)
    oof   = np.zeros(len(y_tr))
    test_preds = []
    kf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr, y_log, grp)):
        X_train, X_val = X_tr[tr_idx], X_tr[va_idx]
        y_train, y_val = y_log[tr_idx], y_log[va_idx]

        if name == 'lgbm':
            p = dict(LGBM_PARAMS)
            mdl = lgb.LGBMRegressor(**p)
            mdl.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                                lgb.log_evaluation(-1)])
            oof[va_idx]    = np.expm1(mdl.predict(X_val))
            test_preds.append(np.expm1(mdl.predict(X_te)))

        elif name == 'cb':
            mdl = cb.CatBoostRegressor(**CB_PARAMS)
            mdl.fit(X_train, y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True, verbose=0)
            oof[va_idx]    = np.expm1(mdl.predict(X_val))
            test_preds.append(np.expm1(mdl.predict(X_te)))

        elif name == 'tw15':
            mdl = cb.CatBoostRegressor(**TW15_PARAMS)
            y_raw_train = y_tr[tr_idx]
            y_raw_val   = y_tr[va_idx]
            mdl.fit(X_train, y_raw_train,
                    eval_set=(X_val, y_raw_val),
                    use_best_model=True, verbose=0)
            oof[va_idx]    = mdl.predict(X_val)
            test_preds.append(mdl.predict(X_te))

        elif name == 'et':
            mdl = ExtraTreesRegressor(**ET_PARAMS)
            mdl.fit(X_train, y_train)
            oof[va_idx]    = np.expm1(mdl.predict(X_val))
            test_preds.append(np.expm1(mdl.predict(X_te)))

        elif name == 'rf':
            mdl = RandomForestRegressor(**RF_PARAMS)
            mdl.fit(X_train, y_train)
            oof[va_idx]    = np.expm1(mdl.predict(X_val))
            test_preds.append(np.expm1(mdl.predict(X_te)))

        elif name == 'asym':
            raw_p = dict(ASYM_PARAMS)
            raw_p['objective'] = asym_obj   # lgb 4.x 방식
            dtr = lgb.Dataset(X_train, label=y_train)
            dva = lgb.Dataset(X_val,   label=y_val)
            num_boost = raw_p.pop('n_estimators')
            mdl = lgb.train(raw_p, dtr, num_boost_round=num_boost,
                            valid_sets=[dva],
                            feval=asym_metric,
                            callbacks=[lgb.early_stopping(50, verbose=False),
                                       lgb.log_evaluation(-1)])
            oof[va_idx]    = np.expm1(mdl.predict(X_val))
            test_preds.append(np.expm1(mdl.predict(X_te)))

        mae = np.abs(oof[va_idx] - y_tr[va_idx]).mean()
        print(f"    fold {fold+1}/{n_splits}  MAE={mae:.4f}")
        del X_train, X_val, y_train, y_val; gc.collect()

    test_pred = np.mean(test_preds, axis=0)
    oof_mae = np.abs(oof - y_tr).mean()
    print(f"  [{name}] OOF MAE = {oof_mae:.4f} | test_std = {test_pred.std():.2f}")

    save_ckpt(ckpt_dir, name, oof, test_pred)
    return oof, test_pred


# ── 메타 학습 ────────────────────────────────────────────────────────────────
def train_meta(oof_dict, test_dict, y_tr, grp, n_splits, ckpt_dir):
    names = list(oof_dict.keys())
    X_meta_tr = np.column_stack([oof_dict[n] for n in names])
    X_meta_te = np.column_stack([test_dict[n] for n in names])
    y_log = np.log1p(y_tr)

    oof_meta = np.zeros(len(y_tr))
    test_preds = []
    kf = GroupKFold(n_splits=n_splits)
    fold_maes = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_meta_tr, y_log, grp)):
        X_train, X_val = X_meta_tr[tr_idx], X_meta_tr[va_idx]
        y_train, y_val = y_log[tr_idx], y_log[va_idx]

        mdl = lgb.LGBMRegressor(**META_PARAMS)
        mdl.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)])
        oof_meta[va_idx] = np.expm1(mdl.predict(X_val))
        test_preds.append(np.expm1(mdl.predict(X_meta_te)))

        mae = np.abs(oof_meta[va_idx] - y_tr[va_idx]).mean()
        fold_maes.append(mae)
        del X_train, X_val, y_train, y_val; gc.collect()

    test_pred = np.mean(test_preds, axis=0)
    cv_mae = np.abs(oof_meta - y_tr).mean()
    return cv_mae, fold_maes, test_pred


# ── 단일 k 실험 ──────────────────────────────────────────────────────────────
def run_experiment(k, X_tr, X_te, y_tr, grp, sample):
    print(f"\n{'='*60}")
    print(f"  GroupKFold k={k} 실험 시작")
    print(f"{'='*60}")

    ckpt_dir = os.path.join(CKPT_BASE, f'k{k}')
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.time()

    # ── Base Learner 학습 ──
    base_names = ['lgbm', 'cb', 'tw15', 'et', 'rf', 'asym']
    oof_dict  = {}
    test_dict = {}
    for name in base_names:
        print(f"\n  [{name}] 학습 시작...")
        oof, test_pred = train_base_learner(
            name, X_tr, y_tr, X_te, grp, k, ckpt_dir)
        oof_dict[name]  = oof
        test_dict[name] = test_pred

    # ── 상관관계 출력 ──
    print(f"\n  상관관계 (OOF 기준):")
    corr_mat = np.corrcoef(np.column_stack([oof_dict[n] for n in base_names]).T)
    pairs = [('lgbm','cb'), ('lgbm','tw15'), ('lgbm','et'), ('lgbm','rf'),
             ('lgbm','asym'), ('tw15','et'), ('cb','et')]
    for a, b in pairs:
        i, j = base_names.index(a), base_names.index(b)
        print(f"    {a}-{b}: {corr_mat[i,j]:.4f}")

    # ── 메타 학습 ──
    print(f"\n  메타 LGBM 학습 (k={k})...")
    cv_mae, fold_maes, test_meta = train_meta(
        oof_dict, test_dict, y_tr, grp, k, ckpt_dir)

    fold_str = " / ".join(f"{m:.4f}" for m in fold_maes)
    test_std = test_meta.std()
    test_mean = test_meta.mean()
    test_max  = test_meta.max()

    # ── 예측 통계 ──
    TARGET_STD = 27.4  # 실제 train target std (배율 간접 추정용)
    ratio_est = test_std / (TARGET_STD * 0.58)  # 경험적 보정 계수 (past experiments)

    print(f"\n  ─── k={k} 결과 ───")
    print(f"  CV MAE      : {cv_mae:.4f}")
    print(f"  Fold MAE    : {fold_str}")
    print(f"  test_std    : {test_std:.2f}")
    print(f"  test_mean   : {test_mean:.2f}")
    print(f"  test_max    : {test_max:.2f}")
    print(f"  경과 시간   : {(time.time()-t0)/60:.1f}분")

    if test_std < 14.0:
        print(f"  ⚠️  pred_std {test_std:.2f} < 14.0 — 배율 악화 위험!")
    elif test_std >= 15.5:
        print(f"  ✅ pred_std {test_std:.2f} ≥ 15.5 — 배율 양호 구간")
    else:
        print(f"  △  pred_std {test_std:.2f} — 경계 구간")

    # ── 제출 파일 생성 ──
    sub = sample.copy()
    pred_col = [c for c in sub.columns if c != 'ID'][0]  # 'avg_delay_minutes_next_30m'
    sub[pred_col] = np.clip(test_meta, 0, None)
    sub_path = os.path.join(SUB_DIR, f'model44_k{k}_cv{cv_mae:.4f}.csv')
    sub.to_csv(sub_path, index=False)
    print(f"  저장: {os.path.basename(sub_path)}")

    return {
        'k': k,
        'cv_mae': cv_mae,
        'fold_maes': fold_maes,
        'test_std': test_std,
        'test_mean': test_mean,
        'test_max': test_max,
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, action='append',
                        help='실험할 k값 (여러 번 사용 가능: --k 3 --k 5)')
    parser.add_argument('--all', action='store_true',
                        help='k=3, 5, 10 전체 실행 (기본)')
    args = parser.parse_args()

    k_list = args.k if args.k else []
    if args.all or not k_list:
        k_list = [3, 5, 10]
    k_list = sorted(set(k_list))

    print(f"실험 k값: {k_list}")
    print(f"기준 (model34 Config B, k=5): CV 8.4803 / Public 9.8078 / pred_std ≈ 16.15")

    # 데이터 로드 (한 번만)
    X_tr, X_te, y_tr, grp, sample = load_data()

    results = []
    for k in k_list:
        res = run_experiment(k, X_tr, X_te, y_tr, grp, sample)
        results.append(res)

    # ── 최종 비교 테이블 ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  GroupKFold 비교 결과 요약")
    print(f"{'='*70}")
    print(f"  {'k':>4}  {'CV MAE':>8}  {'pred_std':>9}  {'test_max':>9}  판정")
    print(f"  {'-'*55}")

    # 기준값 (model34 Config B, k=5 실측)
    baseline = {'k': 5, 'cv_mae': 8.4803, 'test_std': 16.15, 'test_max': None}
    print(f"  {'5(기준)':>8}  {baseline['cv_mae']:>8.4f}  {baseline['test_std']:>9.2f}"
          f"  {'(기준)':>9}  [Public 9.8078]")

    for r in results:
        cv_diff   = r['cv_mae'] - baseline['cv_mae']
        std_diff  = r['test_std'] - baseline['test_std']
        cv_mark   = f"Δ{cv_diff:+.4f}"
        std_mark  = f"Δ{std_diff:+.2f}"

        if r['test_std'] >= 15.5 and r['cv_mae'] <= 8.50:
            status = "✅ 제출 권장"
        elif r['test_std'] < 14.0:
            status = "❌ pred_std 압축"
        elif r['cv_mae'] > 8.55:
            status = "⚠️ CV 악화"
        else:
            status = "△ 제출 검토"

        print(f"  k={r['k']:>2}   {r['cv_mae']:>8.4f}  {r['test_std']:>9.2f}"
              f"  {r['test_max']:>9.2f}  {status}")
        print(f"  {'':>5}  ({cv_mark})  ({std_mark})")

    print(f"\n  가설 검증:")
    if len(results) >= 2:
        k3 = next((r for r in results if r['k'] == 3), None)
        k10 = next((r for r in results if r['k'] == 10), None)
        if k3 and k10:
            if k3['test_std'] > k10['test_std']:
                print(f"  ✅ k=3 pred_std({k3['test_std']:.2f}) > k=10({k10['test_std']:.2f})"
                      f" — 가설 확인: 적은 fold = 낮은 압축")
            else:
                print(f"  ❌ k=3 pred_std({k3['test_std']:.2f}) ≤ k=10({k10['test_std']:.2f})"
                      f" — 가설 기각")

    print(f"\n  제출 파일 위치: submissions/model44_k[k]_cv[mae].csv")
    print(f"{'='*70}")
