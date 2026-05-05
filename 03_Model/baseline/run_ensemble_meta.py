"""
run_ensemble_meta.py
====================
P_extreme 메타 피처 + LGBM·CatBoost·XGBoost 앙상블 (Option A)

설계 원칙
---------
2-Stage 단독 실패 원인: 극값 전용 회귀 모델의 불안정성
해결책: 분류기 확률(AUC=0.8754)을 "부드러운 신호 피처"로 변환
  → 단일 강력한 모델이 P_extreme을 활용해 직접 최적화

리크 방지
---------
P_extreme은 GroupKFold OOF로 생성:
  - train[i]의 P_extreme = 해당 시나리오를 제외한 fold에서 학습한 분류기 예측
  - test의 P_extreme = 5개 fold 분류기 평균 (stacking 표준 방식)
  - 동일한 GroupKFold splits 재사용 → fold 간 정렬 보장

비교 기준
---------
  기준 LGBM       : CV 8.8836 / Public 10.3347 (앙상블)
  2-Stage 혼합    : CV 8.8745 (개선은 하지만 미미)
  기대 목표       : CV < 8.87 → Public < 10.30

실행
----
  python src/run_ensemble_meta.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.optimize import minimize

from feature_engineering import (
    merge_layout, encode_categoricals, add_ts_features,
    add_lag_features, add_rolling_features, add_domain_features,
    get_feature_cols,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'
SUB_PATH  = os.path.join(os.path.dirname(__file__), '..', 'submissions') + '/'
TARGET    = 'avg_delay_minutes_next_30m'
SEED      = 42
N_SPLITS  = 5
P90_THRESHOLD = 45.2

CLF_PARAMS = {
    'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 1,
    'min_child_samples': 30, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'objective': 'binary', 'metric': 'auc',
    'n_estimators': 1000, 'random_state': SEED,
    'verbosity': -1, 'n_jobs': -1,
}


# ─────────────────────────────────────────────
# 1. 데이터 로드 & 피처 엔지니어링
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 1: 데이터 로드 & 피처 엔지니어링")
print("="*60)

train_raw     = pd.read_csv(DATA_PATH + 'train.csv')
test_raw      = pd.read_csv(DATA_PATH + 'test.csv')
layout        = pd.read_csv(DATA_PATH + 'layout_info.csv')
test_orig_ids = test_raw['ID'].values.copy()

train, test = merge_layout(train_raw.copy(), test_raw.copy(), layout)
train, test = encode_categoricals(train, test, TARGET)
train = add_ts_features(train)
test  = add_ts_features(test)

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
feat_cols_base = get_feature_cols(train, TARGET)
print(f"✅ 기본 피처 수: {len(feat_cols_base)}")

y      = train[TARGET].values.astype(np.float32)
y_log  = np.log1p(y)
y_bin  = (y > P90_THRESHOLD).astype(np.int8)
groups = train['scenario_id'].values
gkf    = GroupKFold(n_splits=N_SPLITS)


# ─────────────────────────────────────────────
# 2. 분류기 OOF → P_extreme 메타 피처 생성
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 2: 분류기 OOF → P_extreme 메타 피처")
print("="*60)

X_base      = train[feat_cols_base].values.astype(np.float32)
X_test_base = test[feat_cols_base].values.astype(np.float32)

oof_p_ext  = np.zeros(len(y))
test_p_ext = np.zeros(len(X_test_base))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_base, y_bin, groups=groups)):
    t0 = time.time()
    clf = lgb.LGBMClassifier(**CLF_PARAMS)
    clf.fit(X_base[tr_idx], y_bin[tr_idx],
            eval_set=[(X_base[val_idx], y_bin[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)])
    oof_p_ext[val_idx]  = clf.predict_proba(X_base[val_idx])[:, 1]
    test_p_ext          += clf.predict_proba(X_test_base)[:, 1] / N_SPLITS
    auc = roc_auc_score(y_bin[val_idx], oof_p_ext[val_idx])
    print(f"  Fold {fold+1}: AUC={auc:.4f} ({time.time()-t0:.0f}s)")

clf_auc = roc_auc_score(y_bin, oof_p_ext)
print(f"✅ 분류기 OOF AUC: {clf_auc:.4f}")

# 메타 피처를 DataFrame에 추가
train['p_extreme'] = oof_p_ext
test['p_extreme']  = test_p_ext

# 메타 피처 포함 feature_cols
feat_cols = feat_cols_base + ['p_extreme']
X      = train[feat_cols].values.astype(np.float32)
X_test = test[feat_cols].values.astype(np.float32)
print(f"  메타 피처 추가 후 총 피처 수: {len(feat_cols)} (+1)")
print(f"  p_extreme: train mean={oof_p_ext.mean():.3f}, test mean={test_p_ext.mean():.3f}")


# ─────────────────────────────────────────────
# 3. Optuna LightGBM 탐색
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 3: Optuna LightGBM 탐색 (15 trials, 2-fold)")
print("="*60)

gkf2 = GroupKFold(n_splits=2)

def lgbm_objective(trial):
    params = {
        'objective': 'regression_l1', 'metric': 'mae',
        'verbosity': -1, 'boosting_type': 'gbdt',
        'n_estimators': 1000,
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
print(f"✅ Optuna 완료 ({time.time()-t0:.0f}초) | Best: {study.best_value:.4f}")

best_lgbm_params = dict(study.best_params)
best_lgbm_params.update({
    'objective': 'regression_l1', 'metric': 'mae',
    'verbosity': -1, 'boosting_type': 'gbdt',
    'n_estimators': 3000, 'bagging_freq': 1,
    'random_state': SEED,
})


# ─────────────────────────────────────────────
# 4. LightGBM 5-fold CV (메타 피처 포함)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 4: LightGBM 5-fold CV")
print("="*60)

oof_lgbm  = np.zeros(len(y))
test_lgbm = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    m = lgb.LGBMRegressor(**best_lgbm_params)
    m.fit(X[tr_idx], y_log[tr_idx],
          eval_set=[(X[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False),
                     lgb.log_evaluation(-1)])
    oof_lgbm[val_idx]  = np.expm1(m.predict(X[val_idx])).clip(0)
    test_lgbm          += np.expm1(m.predict(X_test)).clip(0) / N_SPLITS
    print(f"  Fold {fold+1}: MAE={mean_absolute_error(y[val_idx], oof_lgbm[val_idx]):.4f}"
          f" ({time.time()-t0:.0f}s)")

lgbm_mae = mean_absolute_error(y, oof_lgbm)
print(f"✅ LGBM OOF MAE: {lgbm_mae:.4f}")


# ─────────────────────────────────────────────
# 5. CatBoost 5-fold CV
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 5: CatBoost 5-fold CV")
print("="*60)

cb_params = dict(
    iterations=2000, learning_rate=0.05, depth=8,
    l2_leaf_reg=3.0, loss_function='MAE', eval_metric='MAE',
    random_seed=SEED, early_stopping_rounds=100,
    verbose=0, allow_writing_files=False,
)

oof_cb  = np.zeros(len(y))
test_cb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    m = cb.CatBoostRegressor(**cb_params)
    m.fit(X[tr_idx], y_log[tr_idx], eval_set=(X[val_idx], y_log[val_idx]))
    oof_cb[val_idx]  = np.expm1(m.predict(X[val_idx])).clip(0)
    test_cb          += np.expm1(m.predict(X_test)).clip(0) / N_SPLITS
    print(f"  Fold {fold+1}: MAE={mean_absolute_error(y[val_idx], oof_cb[val_idx]):.4f}"
          f" ({time.time()-t0:.0f}s)")

cb_mae = mean_absolute_error(y, oof_cb)
print(f"✅ CatBoost OOF MAE: {cb_mae:.4f}")


# ─────────────────────────────────────────────
# 6. XGBoost 5-fold CV
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 6: XGBoost 5-fold CV")
print("="*60)

xgb_params = dict(
    n_estimators=3000, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    objective='reg:absoluteerror', eval_metric='mae',
    random_state=SEED, tree_method='hist',
    early_stopping_rounds=100, verbosity=0,
)

oof_xgb  = np.zeros(len(y))
test_xgb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X[tr_idx], y_log[tr_idx],
          eval_set=[(X[val_idx], y_log[val_idx])], verbose=False)
    oof_xgb[val_idx]  = np.expm1(m.predict(X[val_idx])).clip(0)
    test_xgb          += np.expm1(m.predict(X_test)).clip(0) / N_SPLITS
    print(f"  Fold {fold+1}: MAE={mean_absolute_error(y[val_idx], oof_xgb[val_idx]):.4f}"
          f" ({time.time()-t0:.0f}s)")

xgb_mae = mean_absolute_error(y, oof_xgb)
print(f"✅ XGBoost OOF MAE: {xgb_mae:.4f}")


# ─────────────────────────────────────────────
# 7. 최적 앙상블 가중치
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 7: 앙상블 가중치 최적화")
print("="*60)

oofs  = np.stack([oof_lgbm, oof_cb, oof_xgb], axis=1)
names = ['LightGBM', 'CatBoost', 'XGBoost']

def neg_mae(w):
    w = np.abs(w) / np.abs(w).sum()
    return mean_absolute_error(y, (oofs * w).sum(axis=1))

res = minimize(neg_mae, [1/3, 1/3, 1/3], method='Nelder-Mead',
               options={'maxiter': 2000, 'xatol': 1e-7})
opt_w = np.abs(res.x) / np.abs(res.x).sum()

PREV_BEST_CV = 8.8674

print(f"\n{'모델':<12} {'OOF MAE':>10}  {'가중치':>8}")
print("-" * 34)
for nm, mae_v, w in zip(names, [lgbm_mae, cb_mae, xgb_mae], opt_w):
    print(f"  {nm:<10} {mae_v:>10.4f}  {w:>8.3f}")
print("-" * 34)
print(f"  균등 앙상블  {neg_mae([1/3,1/3,1/3]):>10.4f}")
print(f"  최적 앙상블  {res.fun:>10.4f}  ← {'✅ 개선' if res.fun < PREV_BEST_CV else '❌ 미개선'} vs {PREV_BEST_CV}")
print(f"  (이전 최고 CV: {PREV_BEST_CV})")


# ─────────────────────────────────────────────
# 8. 제출 파일 생성
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 8: 제출 파일 생성")
print("="*60)

os.makedirs(SUB_PATH, exist_ok=True)
sample = pd.read_csv(DATA_PATH + 'sample_submission.csv')

test_ensemble = (test_lgbm * opt_w[0] +
                 test_cb   * opt_w[1] +
                 test_xgb  * opt_w[2]).clip(0)

sub = sample.copy()
sub[TARGET] = test_ensemble
out = SUB_PATH + 'ensemble_meta_p_extreme.csv'
sub.to_csv(out, index=False)
print(f"✅ 제출 파일: {out}")
print(f"   예측: mean={test_ensemble.mean():.2f}, std={test_ensemble.std():.2f}")

print("\n" + "="*60)
print(f"  이전 최고 Public    : 10.3347  (CV {PREV_BEST_CV})")
print(f"  Meta 앙상블 CV MAE  : {res.fun:.4f}  (Δ{res.fun - PREV_BEST_CV:+.4f})")
print(f"  분류기 AUC          : {clf_auc:.4f}")
print("="*60)
print("완료! → ensemble_meta_p_extreme.csv 제출")
