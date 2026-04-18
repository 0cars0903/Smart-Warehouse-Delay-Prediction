"""
run_optuna_full.py
==================
LGBM · CatBoost · XGBoost 3종 모두 Optuna 튜닝 후 최적 앙상블

변경점 (vs run_ensemble.py):
  - CatBoost: 고정 파라미터 → Optuna 20 trials
  - XGBoost:  고정 파라미터 → Optuna 20 trials
  - LGBM:     기존 최적 파라미터 재사용 (추가 튜닝 불필요)
  - 탐색 fold: 2-fold (속도), 최종 CV: 5-fold

실행: python src/run_optuna_full.py
예상 시간: 40~60분 (Optuna 40 trials + 5-fold × 3모델)
"""

import pandas as pd
import numpy as np
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from feature_engineering import (
    merge_layout, encode_categoricals, add_ts_features,
    add_lag_features, add_rolling_features, add_domain_features,
    get_feature_cols
)

# ─── 설정 ────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'
SUB_PATH  = os.path.join(os.path.dirname(__file__), '..', 'submissions') + '/'
TARGET    = 'avg_delay_minutes_next_30m'
SEED      = 42
N_SPLITS  = 5
N_TRIALS  = 20   # 모델당 Optuna trial 수

KEY_COLS_EXT = [
    'low_battery_ratio', 'battery_mean', 'charge_queue_length',
    'robot_idle', 'order_inflow_15m', 'congestion_score',
    'max_zone_density', 'avg_trip_distance',
    'robot_utilization', 'task_reassign_15m', 'blocked_path_15m',
    'urgent_order_ratio', 'fault_count_15m', 'avg_recovery_time',
]

# 기존 최적 LGBM 파라미터 (CLAUDE.md — 재사용)
BEST_LGBM_PARAMS = {
    'num_leaves'       : 181,
    'learning_rate'    : 0.020616,
    'feature_fraction' : 0.5122,
    'bagging_fraction' : 0.9049,
    'bagging_freq'     : 1,
    'min_child_samples': 26,
    'reg_alpha'        : 0.3805,
    'reg_lambda'       : 0.3630,
    'objective'        : 'regression_l1',
    'metric'           : 'mae',
    'n_estimators'     : 3000,
    'random_state'     : SEED,
    'verbosity'        : -1,
}


# ─── Step 1: 데이터 로드 & 피처 엔지니어링 ──────────────────

print("\n" + "="*60)
print(" STEP 1: 데이터 로드 & 피처 엔지니어링")
print("="*60)
t0 = time.time()

train_raw     = pd.read_csv(DATA_PATH + 'train.csv')
test_raw      = pd.read_csv(DATA_PATH + 'test.csv')
layout        = pd.read_csv(DATA_PATH + 'layout_info.csv')
test_orig_ids = test_raw['ID'].values.copy()
print(f"Train: {train_raw.shape}, Test: {test_raw.shape}")

train, test = merge_layout(train_raw.copy(), test_raw.copy(), layout)
train, test = encode_categoricals(train, test, TARGET)
train = add_ts_features(train)
test  = add_ts_features(test)
train, test = add_lag_features(train, test, key_cols=KEY_COLS_EXT, lags=[1,2,3,4,5,6])
train, test = add_rolling_features(train, test, key_cols=KEY_COLS_EXT, windows=[3,5,10])
train = add_domain_features(train)
test  = add_domain_features(test)

assert (test['ID'].values == test_orig_ids).all(), "❌ ID 순서 오류!"
feat_cols = get_feature_cols(train, TARGET)
print(f"✅ 전처리 완료 ({time.time()-t0:.0f}s) | 피처={len(feat_cols)}")

X      = train[feat_cols].values.astype(np.float32)
y      = train[TARGET].values.astype(np.float32)
y_log  = np.log1p(y)
X_test = test[feat_cols].values.astype(np.float32)
groups = train['scenario_id'].values

gkf  = GroupKFold(n_splits=N_SPLITS)
gkf2 = GroupKFold(n_splits=2)   # Optuna 탐색용


# ─── Step 2: CatBoost Optuna ─────────────────────────────────

print("\n" + "="*60)
print(f" STEP 2: CatBoost Optuna ({N_TRIALS} trials, 2-fold)")
print("="*60)

def cb_objective(trial):
    params = dict(
        iterations          = 2000,
        learning_rate       = trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        depth               = trial.suggest_int('depth', 4, 10),
        l2_leaf_reg         = trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        bagging_temperature = trial.suggest_float('bagging_temperature', 0.0, 1.0),
        random_strength     = trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        border_count        = trial.suggest_int('border_count', 32, 255),
        loss_function       = 'MAE',
        eval_metric         = 'MAE',
        random_seed         = SEED,
        early_stopping_rounds = 50,
        verbose             = 0,
        allow_writing_files = False,
    )
    oof = np.zeros(len(X))
    for tr_idx, val_idx in gkf2.split(X, y_log, groups=groups):
        m = cb.CatBoostRegressor(**params)
        m.fit(X[tr_idx], y_log[tr_idx],
              eval_set=(X[val_idx], y_log[val_idx]))
        oof[val_idx] = np.expm1(m.predict(X[val_idx])).clip(0)
    return mean_absolute_error(y, oof)

t0 = time.time()
study_cb = optuna.create_study(direction='minimize',
                               sampler=optuna.samplers.TPESampler(seed=SEED))
study_cb.optimize(cb_objective, n_trials=N_TRIALS)
print(f"✅ CatBoost Optuna 완료 ({time.time()-t0:.0f}s) | Best MAE: {study_cb.best_value:.4f}")

best_cb_params = dict(study_cb.best_params)
best_cb_params.update(dict(
    iterations          = 3000,
    loss_function       = 'MAE',
    eval_metric         = 'MAE',
    random_seed         = SEED,
    early_stopping_rounds = 100,
    verbose             = 0,
    allow_writing_files = False,
))
print(f"Best CB params: {study_cb.best_params}")


# ─── Step 3: XGBoost Optuna ──────────────────────────────────

print("\n" + "="*60)
print(f" STEP 3: XGBoost Optuna ({N_TRIALS} trials, 2-fold)")
print("="*60)

def xgb_objective(trial):
    params = dict(
        n_estimators      = 1000,
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        max_depth         = trial.suggest_int('max_depth', 4, 10),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 20),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        gamma             = trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        objective         = 'reg:absoluteerror',
        eval_metric       = 'mae',
        random_state      = SEED,
        tree_method       = 'hist',
        early_stopping_rounds = 50,
        verbosity         = 0,
    )
    oof = np.zeros(len(X))
    for tr_idx, val_idx in gkf2.split(X, y_log, groups=groups):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr_idx], y_log[tr_idx],
              eval_set=[(X[val_idx], y_log[val_idx])],
              verbose=False)
        oof[val_idx] = np.expm1(m.predict(X[val_idx])).clip(0)
    return mean_absolute_error(y, oof)

t0 = time.time()
study_xgb = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(xgb_objective, n_trials=N_TRIALS)
print(f"✅ XGBoost Optuna 완료 ({time.time()-t0:.0f}s) | Best MAE: {study_xgb.best_value:.4f}")

best_xgb_params = dict(study_xgb.best_params)
best_xgb_params.update(dict(
    n_estimators          = 3000,
    objective             = 'reg:absoluteerror',
    eval_metric           = 'mae',
    random_state          = SEED,
    tree_method           = 'hist',
    early_stopping_rounds = 100,
    verbosity             = 0,
))
print(f"Best XGB params: {study_xgb.best_params}")


# ─── Step 4: LGBM Full 5-fold (기존 최적 파라미터 재사용) ────

print("\n" + "="*60)
print(" STEP 4: LightGBM Full 5-fold CV (기존 최적 파라미터)")
print("="*60)

oof_lgbm  = np.zeros(len(X))
test_lgbm = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    m = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
    m.fit(X[tr_idx], y_log[tr_idx],
          eval_set=[(X[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False),
                     lgb.log_evaluation(-1)])
    oof_lgbm[val_idx] = np.expm1(m.predict(X[val_idx])).clip(0)
    test_lgbm += np.expm1(m.predict(X_test)).clip(0) / N_SPLITS
    print(f"  Fold {fold+1}: MAE={mean_absolute_error(y[val_idx], oof_lgbm[val_idx]):.4f}"
          f" ({time.time()-t0:.0f}s)")

lgbm_mae = mean_absolute_error(y, oof_lgbm)
print(f"✅ LGBM OOF MAE: {lgbm_mae:.4f}")


# ─── Step 5: CatBoost Full 5-fold ────────────────────────────

print("\n" + "="*60)
print(" STEP 5: CatBoost Full 5-fold CV (Optuna 최적 파라미터)")
print("="*60)

oof_cb  = np.zeros(len(X))
test_cb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    m = cb.CatBoostRegressor(**best_cb_params)
    m.fit(X[tr_idx], y_log[tr_idx],
          eval_set=(X[val_idx], y_log[val_idx]))
    oof_cb[val_idx] = np.expm1(m.predict(X[val_idx])).clip(0)
    test_cb += np.expm1(m.predict(X_test)).clip(0) / N_SPLITS
    print(f"  Fold {fold+1}: MAE={mean_absolute_error(y[val_idx], oof_cb[val_idx]):.4f}"
          f" ({time.time()-t0:.0f}s)")

cb_mae = mean_absolute_error(y, oof_cb)
print(f"✅ CatBoost OOF MAE: {cb_mae:.4f}")


# ─── Step 6: XGBoost Full 5-fold ─────────────────────────────

print("\n" + "="*60)
print(" STEP 6: XGBoost Full 5-fold CV (Optuna 최적 파라미터)")
print("="*60)

oof_xgb  = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    m = xgb.XGBRegressor(**best_xgb_params)
    m.fit(X[tr_idx], y_log[tr_idx],
          eval_set=[(X[val_idx], y_log[val_idx])],
          verbose=False)
    oof_xgb[val_idx] = np.expm1(m.predict(X[val_idx])).clip(0)
    test_xgb += np.expm1(m.predict(X_test)).clip(0) / N_SPLITS
    print(f"  Fold {fold+1}: MAE={mean_absolute_error(y[val_idx], oof_xgb[val_idx]):.4f}"
          f" ({time.time()-t0:.0f}s)")

xgb_mae = mean_absolute_error(y, oof_xgb)
print(f"✅ XGBoost OOF MAE: {xgb_mae:.4f}")


# ─── Step 7: 앙상블 가중치 최적화 ────────────────────────────

print("\n" + "="*60)
print(" STEP 7: 최적 앙상블 가중치 탐색")
print("="*60)

names = ['LGBM', 'CatBoost', 'XGBoost']
maes  = [lgbm_mae, cb_mae, xgb_mae]
oofs  = np.stack([oof_lgbm, oof_cb, oof_xgb], axis=1)

def neg_mae(w):
    w = np.abs(w) / np.abs(w).sum()
    return mean_absolute_error(y, (oofs * w).sum(axis=1))

res = minimize(neg_mae, [1/3, 1/3, 1/3], method='Nelder-Mead',
               options={'maxiter': 3000, 'xatol': 1e-8})
opt_w = np.abs(res.x) / np.abs(res.x).sum()

print(f"\n{'모델':<12} {'OOF MAE':>10}  {'기존 MAE':>10}  {'가중치':>8}")
print("-" * 46)
prev_maes = {'LGBM': 8.8895, 'CatBoost': None, 'XGBoost': None}
for nm, mae_v, w in zip(names, maes, opt_w):
    prev = f"{prev_maes.get(nm):.4f}" if prev_maes.get(nm) else "  (고정파라미터)"
    print(f"  {nm:<10} {mae_v:>10.4f}  {prev:>10}  {w:>8.3f}")
print("-" * 46)
print(f"  균등 앙상블       {neg_mae([1/3,1/3,1/3]):>10.4f}")
print(f"  최적 앙상블       {res.fun:>10.4f}  ← {'↓' if res.fun < 8.8703 else '↑'}{abs(res.fun - 8.8703):.4f} vs 이전 최고(8.8703)")


# ─── Step 8: 제출 파일 생성 ──────────────────────────────────

print("\n" + "="*60)
print(" STEP 8: 제출 파일 생성")
print("="*60)

sample_sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')

# 최적 가중치 앙상블
test_ensemble = (test_lgbm * opt_w[0] +
                 test_cb   * opt_w[1] +
                 test_xgb  * opt_w[2]).clip(0)

sub1 = sample_sub.copy()
sub1[TARGET] = test_ensemble
out1 = SUB_PATH + 'ensemble_optuna_all3.csv'
sub1.to_csv(out1, index=False)
print(f"✅ 3모델 최적 앙상블: {out1}")
print(f"   예측: mean={test_ensemble.mean():.2f}, std={test_ensemble.std():.2f}")

# 균등 앙상블 (비교용)
test_equal = ((test_lgbm + test_cb + test_xgb) / 3).clip(0)
sub2 = sample_sub.copy()
sub2[TARGET] = test_equal
out2 = SUB_PATH + 'ensemble_optuna_all3_equal.csv'
sub2.to_csv(out2, index=False)
print(f"✅ 균등 앙상블 (비교용): {out2}")

# 개별 최고 모델
best_idx  = int(np.argmin(maes))
best_pred = [test_lgbm, test_cb, test_xgb][best_idx]
sub3 = sample_sub.copy()
sub3[TARGET] = best_pred.clip(0)
out3 = SUB_PATH + f'best_single_{names[best_idx].lower()}_optuna_full.csv'
sub3.to_csv(out3, index=False)
print(f"✅ 단일 최고 ({names[best_idx]}, MAE={maes[best_idx]:.4f}): {out3}")


# ─── 결과 요약 저장 ──────────────────────────────────────────

import datetime
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
summary = pd.DataFrame([
    {'모델': nm, 'OOF_MAE': mae_v, '가중치': w}
    for nm, mae_v, w in zip(names, maes, opt_w)
] + [
    {'모델': '균등앙상블',  'OOF_MAE': neg_mae([1/3,1/3,1/3]), '가중치': None},
    {'모델': '최적앙상블',  'OOF_MAE': res.fun,                 '가중치': None},
])
summary_path = os.path.join(os.path.dirname(__file__), '..', 'docs',
                             f'optuna_full_results_{ts}.csv')
summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

print(f"\n결과 저장: {summary_path}")
print("\n" + "="*60)
print(f"  이전 최고 (앙상블) OOF MAE : 8.8703")
print(f"  이전 최고 (앙상블) Public  : 10.3349")
print(f"  신규 최적 앙상블   OOF MAE : {res.fun:.4f}")
print("="*60)
print("제출 우선순위: ensemble_optuna_all3.csv → best_single → ensemble_equal")
