"""
run_ensemble.py
===============
LightGBM (Optuna) + CatBoost + XGBoost 앙상블 실험
- Optuna: 15 trials, 2-fold (탐색 속도 최적화)
- Full CV: 5-fold (최종 모델)
- log1p 타깃 변환
- 확장 lag(1~6) + rolling(3,5,10)
"""

import pandas as pd
import numpy as np
import sys, warnings, os, time
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

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'
SUB_PATH   = os.path.join(os.path.dirname(__file__), '..', 'submissions') + '/'
TARGET     = 'avg_delay_minutes_next_30m'
SEED       = 42
N_SPLITS   = 5

# ─────────────────────────────────────────────
# 1. 데이터 로드 & 피처 엔지니어링
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 1: 데이터 로드 & 확장 피처 엔지니어링")
print("="*60)

train_raw = pd.read_csv(DATA_PATH + 'train.csv')
test_raw  = pd.read_csv(DATA_PATH + 'test.csv')
layout    = pd.read_csv(DATA_PATH + 'layout_info.csv')
test_orig_ids = test_raw['ID'].values.copy()
print(f"Train: {train_raw.shape}, Test: {test_raw.shape}")

# 파이프라인 적용
train, test = merge_layout(train_raw.copy(), test_raw.copy(), layout)
train, test = encode_categoricals(train, test, TARGET)
train = add_ts_features(train)
test  = add_ts_features(test)

# 확장 피처셋: 핵심 14개 컬럼, lag 1~6, rolling 3/5/10
KEY_COLS_EXT = [
    'low_battery_ratio', 'battery_mean', 'charge_queue_length',
    'robot_idle', 'order_inflow_15m', 'congestion_score',
    'max_zone_density', 'avg_trip_distance',
    'robot_utilization', 'task_reassign_15m', 'blocked_path_15m',
    'urgent_order_ratio', 'fault_count_15m', 'avg_recovery_time',
]
train, test = add_lag_features(train, test, key_cols=KEY_COLS_EXT, lags=[1,2,3,4,5,6])
train, test = add_rolling_features(train, test, key_cols=KEY_COLS_EXT, windows=[3,5,10])
train = add_domain_features(train)
test  = add_domain_features(test)

assert (test['ID'].values == test_orig_ids).all(), "❌ ID 순서 오류!"
feat_cols = get_feature_cols(train, TARGET)
print(f"✅ 피처 수: {len(feat_cols)}")

X      = train[feat_cols].values.astype(np.float32)
y      = train[TARGET].values.astype(np.float32)
y_log  = np.log1p(y)
X_test = test[feat_cols].values.astype(np.float32)
groups = train['scenario_id'].values
gkf    = GroupKFold(n_splits=N_SPLITS)

print(f"y: mean={y.mean():.2f}, median={np.median(y):.2f}, std={y.std():.2f}")

# ─────────────────────────────────────────────
# 2. Optuna로 LightGBM 하이퍼파라미터 탐색
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 2: Optuna LightGBM 탐색 (15 trials, 2-fold)")
print("="*60)

gkf2 = GroupKFold(n_splits=2)  # 탐색용 2-fold

def lgbm_objective(trial):
    params = {
        'objective'        : 'regression_l1',
        'metric'           : 'mae',
        'verbosity'        : -1,
        'boosting_type'    : 'gbdt',
        'n_estimators'     : 1000,
        'num_leaves'       : trial.suggest_int('num_leaves', 31, 255),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        'feature_fraction' : trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq'     : 1,
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state'     : SEED,
    }
    oof = np.zeros(len(X))
    for tr_idx, val_idx in gkf2.split(X, y_log, groups=groups):
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr_idx], y_log[tr_idx],
              eval_set=[(X[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[val_idx] = np.expm1(m.predict(X[val_idx])).clip(0)
    return mean_absolute_error(y, oof)

t0 = time.time()
study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(lgbm_objective, n_trials=15)
print(f"✅ Optuna 완료 ({time.time()-t0:.0f}초) | Best MAE: {study.best_value:.4f}")

best_lgbm_params = dict(study.best_params)
best_lgbm_params.update({
    'objective': 'regression_l1', 'metric': 'mae',
    'verbosity': -1, 'boosting_type': 'gbdt',
    'n_estimators': 3000, 'bagging_freq': 1,
    'random_state': SEED,
})
print(f"Best params: {best_lgbm_params}")

# ─────────────────────────────────────────────
# 3. LightGBM Full 5-fold CV
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 3: LightGBM Full 5-fold CV")
print("="*60)

oof_lgbm  = np.zeros(len(X))
test_lgbm = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    m = lgb.LGBMRegressor(**best_lgbm_params)
    m.fit(X[tr_idx], y_log[tr_idx],
          eval_set=[(X[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False),
                     lgb.log_evaluation(-1)])
    oof_lgbm[val_idx] = np.expm1(m.predict(X[val_idx])).clip(0)
    test_lgbm += np.expm1(m.predict(X_test)).clip(0) / N_SPLITS
    mae = mean_absolute_error(y[val_idx], oof_lgbm[val_idx])
    print(f"  Fold {fold+1}: MAE={mae:.4f} ({time.time()-t0:.0f}s)")

lgbm_mae = mean_absolute_error(y, oof_lgbm)
print(f"✅ LightGBM OOF MAE: {lgbm_mae:.4f}")

# ─────────────────────────────────────────────
# 4. CatBoost Full 5-fold CV
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 4: CatBoost Full 5-fold CV")
print("="*60)

cb_params = dict(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=SEED,
    early_stopping_rounds=100,
    verbose=0,
    allow_writing_files=False,
)

oof_cb  = np.zeros(len(X))
test_cb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    m = cb.CatBoostRegressor(**cb_params)
    m.fit(X[tr_idx], y_log[tr_idx],
          eval_set=(X[val_idx], y_log[val_idx]))
    oof_cb[val_idx] = np.expm1(m.predict(X[val_idx])).clip(0)
    test_cb += np.expm1(m.predict(X_test)).clip(0) / N_SPLITS
    mae = mean_absolute_error(y[val_idx], oof_cb[val_idx])
    print(f"  Fold {fold+1}: MAE={mae:.4f} ({time.time()-t0:.0f}s)")

cb_mae = mean_absolute_error(y, oof_cb)
print(f"✅ CatBoost OOF MAE: {cb_mae:.4f}")

# ─────────────────────────────────────────────
# 5. XGBoost Full 5-fold CV
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 5: XGBoost Full 5-fold CV")
print("="*60)

xgb_params = dict(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='reg:absoluteerror',
    eval_metric='mae',
    random_state=SEED,
    tree_method='hist',
    early_stopping_rounds=100,
    verbosity=0,
)

oof_xgb  = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X[tr_idx], y_log[tr_idx],
          eval_set=[(X[val_idx], y_log[val_idx])],
          verbose=False)
    oof_xgb[val_idx] = np.expm1(m.predict(X[val_idx])).clip(0)
    test_xgb += np.expm1(m.predict(X_test)).clip(0) / N_SPLITS
    mae = mean_absolute_error(y[val_idx], oof_xgb[val_idx])
    print(f"  Fold {fold+1}: MAE={mae:.4f} ({time.time()-t0:.0f}s)")

xgb_mae = mean_absolute_error(y, oof_xgb)
print(f"✅ XGBoost OOF MAE: {xgb_mae:.4f}")

# ─────────────────────────────────────────────
# 6. 최적 앙상블 가중치 탐색
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 6: 최적 앙상블 가중치 탐색")
print("="*60)

oofs  = np.stack([oof_lgbm, oof_cb, oof_xgb], axis=1)
names = ['LightGBM', 'CatBoost', 'XGBoost']

def neg_mae(w):
    w = np.abs(w) / np.abs(w).sum()
    return mean_absolute_error(y, (oofs * w).sum(axis=1))

res = minimize(neg_mae, [1/3, 1/3, 1/3], method='Nelder-Mead',
               options={'maxiter': 2000, 'xatol': 1e-7})
opt_w = np.abs(res.x) / np.abs(res.x).sum()

print(f"\n{'모델':<12} {'OOF MAE':>10}  {'가중치':>8}")
print("-" * 34)
for nm, mae_v, w in zip(names, [lgbm_mae, cb_mae, xgb_mae], opt_w):
    print(f"  {nm:<10} {mae_v:>10.4f}  {w:>8.3f}")
print("-" * 34)
print(f"  균등 앙상블  {neg_mae([1/3,1/3,1/3]):>10.4f}")
print(f"  최적 앙상블  {res.fun:>10.4f}")

# ─────────────────────────────────────────────
# 7. 제출 파일 생성
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 7: 제출 파일 생성")
print("="*60)

test_ensemble = (test_lgbm * opt_w[0] +
                 test_cb   * opt_w[1] +
                 test_xgb  * opt_w[2]).clip(0)

submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
submission['avg_delay_minutes_next_30m'] = test_ensemble
out1 = SUB_PATH + 'ensemble_lgbm_cb_xgb_optuna.csv'
submission.to_csv(out1, index=False)
print(f"✅ 앙상블 제출 파일: {out1}")
print(f"   예측: mean={test_ensemble.mean():.2f}, std={test_ensemble.std():.2f}, max={test_ensemble.max():.2f}")

# 단일 최고 모델도 저장
best_idx  = np.argmin([lgbm_mae, cb_mae, xgb_mae])
best_name = names[best_idx]
best_pred = [test_lgbm, test_cb, test_xgb][best_idx]
sub2 = pd.read_csv(DATA_PATH + 'sample_submission.csv')
sub2['avg_delay_minutes_next_30m'] = best_pred.clip(0)
out2 = SUB_PATH + f'best_single_{best_name.lower()}_optuna.csv'
sub2.to_csv(out2, index=False)
print(f"✅ 단일 최고 모델 ({best_name}): {out2}")

print("\n" + "="*60)
print(f"  이전 Public Score  : 10.4936")
print(f"  앙상블 OOF MAE     : {res.fun:.4f}")
print(f"  단일최고 OOF MAE   : {[lgbm_mae, cb_mae, xgb_mae][best_idx]:.4f}")
print("="*60)
print("완료! 두 파일 모두 제출하여 Public Score 비교 추천")
