"""
run_dart.py
===========
LGBM DART Boosting 비교 분석 (Option B)
+ P_extreme 메타 피처 포함 (Option A와 동일 feature set)

DART vs GBDT
-----------
GBDT: 이전 트리 잔차에만 집중 → 특정 트리에 과의존 → 과적합
DART: 매 round마다 기존 트리 일부를 dropout → 분산된 기여 → 일반화↑

현재 상황:
  CV 8.8674 → Public 10.3347  (갭 1.47)
  → CV-Public 갭이 크다 → 과적합 신호 → DART가 효과적일 가능성

DART 특성:
  - early_stopping 비권장 (dropout 특성상 마지막 트리가 중요)
  - n_estimators를 Optuna로 탐색
  - 주요 파라미터: drop_rate, skip_drop, max_drop

실행
----
  python src/run_dart.py
  (run_ensemble_meta.py 실행 결과 확인 후 실행 권장)
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

PREV_BEST_CV = 8.8674   # 현재 최고 앙상블 CV
PREV_META_CV = 8.9089   # run_ensemble_meta.py 결과 (A 실패 — 분포 이탈)

CLF_PARAMS = {
    'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 1,
    'min_child_samples': 30, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'objective': 'binary', 'metric': 'auc',
    'n_estimators': 1000, 'random_state': SEED,
    'verbosity': -1, 'n_jobs': -1,
}

# CatBoost / XGBoost는 run_ensemble_meta.py와 동일
CB_PARAMS = dict(
    iterations=2000, learning_rate=0.05, depth=8,
    l2_leaf_reg=3.0, loss_function='MAE', eval_metric='MAE',
    random_seed=SEED, early_stopping_rounds=100,
    verbose=0, allow_writing_files=False,
)
XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    objective='reg:absoluteerror', eval_metric='mae',
    random_state=SEED, tree_method='hist',
    early_stopping_rounds=100, verbosity=0,
)


# ─────────────────────────────────────────────
# 1. 데이터 로드 & FE
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
# 2. P_extreme 메타 피처 생성 (A와 동일)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 2: P_extreme 메타 피처 생성")
print("="*60)

X_base      = train[feat_cols_base].values.astype(np.float32)
X_test_base = test[feat_cols_base].values.astype(np.float32)

oof_p_ext  = np.zeros(len(y))
test_p_ext = np.zeros(len(X_test_base))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_base, y_bin, groups=groups)):
    clf = lgb.LGBMClassifier(**CLF_PARAMS)
    clf.fit(X_base[tr_idx], y_bin[tr_idx],
            eval_set=[(X_base[val_idx], y_bin[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)])
    oof_p_ext[val_idx]  = clf.predict_proba(X_base[val_idx])[:, 1]
    test_p_ext          += clf.predict_proba(X_test_base)[:, 1] / N_SPLITS

clf_auc = roc_auc_score(y_bin, oof_p_ext)
print(f"✅ 분류기 AUC: {clf_auc:.4f}")

train['p_extreme'] = oof_p_ext
test['p_extreme']  = test_p_ext
feat_cols = feat_cols_base + ['p_extreme']
X      = train[feat_cols].values.astype(np.float32)
X_test = test[feat_cols].values.astype(np.float32)
print(f"  총 피처 수: {len(feat_cols)}")


# ─────────────────────────────────────────────
# 3. DART 하이퍼파라미터 탐색
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 3: Optuna DART 파라미터 탐색 (20 trials, 2-fold)")
print("="*60)
print("  ※ DART는 early_stopping 미사용 → n_estimators를 직접 탐색")

gkf2 = GroupKFold(n_splits=2)

def dart_objective(trial):
    params = {
        'objective'        : 'regression_l1',
        'metric'           : 'mae',
        'verbosity'        : -1,
        'boosting_type'    : 'dart',
        # DART 전용 파라미터
        'drop_rate'        : trial.suggest_float('drop_rate', 0.05, 0.3),
        'skip_drop'        : trial.suggest_float('skip_drop', 0.3, 0.7),
        'max_drop'         : trial.suggest_int('max_drop', 20, 100),
        'uniform_drop'     : trial.suggest_categorical('uniform_drop', [True, False]),
        # 기본 파라미터 (GBDT 최적 범위 유지)
        'n_estimators'     : trial.suggest_int('n_estimators', 500, 2000),
        'num_leaves'       : trial.suggest_int('num_leaves', 63, 255),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.02, 0.08, log=True),
        'feature_fraction' : trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq'     : 1,
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
        'random_state'     : SEED,
    }
    oof = np.zeros(len(X))
    for tr_idx, val_idx in gkf2.split(X, y_log, groups=groups):
        m = lgb.LGBMRegressor(**params)
        # DART: early_stopping 없이 고정 n_estimators 사용
        m.fit(X[tr_idx], y_log[tr_idx],
              callbacks=[lgb.log_evaluation(-1)])
        oof[val_idx] = np.expm1(m.predict(X[val_idx])).clip(0)
    return mean_absolute_error(y, oof)

t0 = time.time()
study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(dart_objective, n_trials=20)
elapsed = time.time() - t0
print(f"✅ DART Optuna 완료 ({elapsed:.0f}초) | Best: {study.best_value:.4f}")

dart_params = dict(study.best_params)
dart_params.update({
    'objective': 'regression_l1', 'metric': 'mae',
    'verbosity': -1, 'boosting_type': 'dart',
    'random_state': SEED,
})
print(f"Best DART params: {dart_params}")


# ─────────────────────────────────────────────
# 4. DART LGBM 5-fold CV
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 4: DART LightGBM 5-fold CV")
print("="*60)

oof_dart  = np.zeros(len(y))
test_dart = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    m = lgb.LGBMRegressor(**dart_params)
    m.fit(X[tr_idx], y_log[tr_idx],
          callbacks=[lgb.log_evaluation(-1)])   # DART: no early_stopping
    oof_dart[val_idx]  = np.expm1(m.predict(X[val_idx])).clip(0)
    test_dart          += np.expm1(m.predict(X_test)).clip(0) / N_SPLITS
    print(f"  Fold {fold+1}: MAE={mean_absolute_error(y[val_idx], oof_dart[val_idx]):.4f}"
          f" ({time.time()-t0:.0f}s)")

dart_mae = mean_absolute_error(y, oof_dart)
print(f"✅ DART LightGBM OOF MAE: {dart_mae:.4f}")


# ─────────────────────────────────────────────
# 5. CatBoost & XGBoost CV (메타 피처 포함)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 5: CatBoost & XGBoost CV (메타 피처 포함)")
print("="*60)

oof_cb  = np.zeros(len(y)); test_cb  = np.zeros(len(X_test))
oof_xgb = np.zeros(len(y)); test_xgb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    # CatBoost
    m_cb = cb.CatBoostRegressor(**CB_PARAMS)
    m_cb.fit(X[tr_idx], y_log[tr_idx], eval_set=(X[val_idx], y_log[val_idx]))
    oof_cb[val_idx]  = np.expm1(m_cb.predict(X[val_idx])).clip(0)
    test_cb          += np.expm1(m_cb.predict(X_test)).clip(0) / N_SPLITS
    # XGBoost
    m_xgb = xgb.XGBRegressor(**XGB_PARAMS)
    m_xgb.fit(X[tr_idx], y_log[tr_idx],
              eval_set=[(X[val_idx], y_log[val_idx])], verbose=False)
    oof_xgb[val_idx]  = np.expm1(m_xgb.predict(X[val_idx])).clip(0)
    test_xgb          += np.expm1(m_xgb.predict(X_test)).clip(0) / N_SPLITS
    print(f"  Fold {fold+1}: CB={mean_absolute_error(y[val_idx], oof_cb[val_idx]):.4f}"
          f"  XGB={mean_absolute_error(y[val_idx], oof_xgb[val_idx]):.4f}"
          f" ({time.time()-t0:.0f}s)")

cb_mae  = mean_absolute_error(y, oof_cb)
xgb_mae = mean_absolute_error(y, oof_xgb)
print(f"✅ CatBoost MAE: {cb_mae:.4f} | XGBoost MAE: {xgb_mae:.4f}")


# ─────────────────────────────────────────────
# 6. DART + CB + XGB 앙상블 가중치 최적화
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 6: DART 앙상블 가중치 최적화")
print("="*60)

oofs  = np.stack([oof_dart, oof_cb, oof_xgb], axis=1)
names = ['DART-LGBM', 'CatBoost', 'XGBoost']

def neg_mae(w):
    w = np.abs(w) / np.abs(w).sum()
    return mean_absolute_error(y, (oofs * w).sum(axis=1))

res = minimize(neg_mae, [1/3, 1/3, 1/3], method='Nelder-Mead',
               options={'maxiter': 2000, 'xatol': 1e-7})
opt_w = np.abs(res.x) / np.abs(res.x).sum()
dart_ensemble_mae = res.fun

print(f"\n{'모델':<12} {'OOF MAE':>10}  {'가중치':>8}")
print("-" * 34)
for nm, mae_v, w in zip(names, [dart_mae, cb_mae, xgb_mae], opt_w):
    print(f"  {nm:<12} {mae_v:>10.4f}  {w:>8.3f}")
print("-" * 34)
print(f"  균등 앙상블  {neg_mae([1/3,1/3,1/3]):>10.4f}")
print(f"  최적 앙상블  {dart_ensemble_mae:>10.4f}")


# ─────────────────────────────────────────────
# 7. DART-GBDT 혼합 (DART 단독이 부족할 경우 보완)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 7: 전체 비교 요약")
print("="*60)

print(f"\n  {'구성':<35} {'CV MAE':>8}  {'vs 현재최고':>12}")
print(f"  {'-'*58}")
print(f"  {'이전 최고 앙상블 (GBDT, 메타 없음)':<35} {PREV_BEST_CV:>8.4f}  {'기준':>12}")
if PREV_META_CV:
    delta_meta = PREV_META_CV - PREV_BEST_CV
    verdict = f"{'✅' if delta_meta < 0 else '❌'} {delta_meta:+.4f}"
    print(f"  {'메타 앙상블 (GBDT + p_extreme)':<35} {PREV_META_CV:>8.4f}  {verdict:>12}")
print(f"  {'DART 단독':<35} {dart_mae:>8.4f}  {dart_mae - PREV_BEST_CV:>+12.4f}")
print(f"  {'DART 앙상블 (DART+CB+XGB)':<35} {dart_ensemble_mae:>8.4f}  "
      f"  {'✅' if dart_ensemble_mae < PREV_BEST_CV else '❌'} {dart_ensemble_mae - PREV_BEST_CV:>+.4f}")


# ─────────────────────────────────────────────
# 8. 제출 파일
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 8: 제출 파일 생성")
print("="*60)

os.makedirs(SUB_PATH, exist_ok=True)
sample = pd.read_csv(DATA_PATH + 'sample_submission.csv')

# DART 앙상블
test_dart_ens = (test_dart * opt_w[0] +
                 test_cb   * opt_w[1] +
                 test_xgb  * opt_w[2]).clip(0)

sub = sample.copy()
sub[TARGET] = test_dart_ens
out = SUB_PATH + 'ensemble_dart_meta.csv'
sub.to_csv(out, index=False)
print(f"✅ DART 앙상블: {out}")
print(f"   예측: mean={test_dart_ens.mean():.2f}, std={test_dart_ens.std():.2f}")

# 최고 CV 구성 별도 저장
if dart_mae < PREV_BEST_CV:
    sub2 = sample.copy()
    sub2[TARGET] = test_dart.clip(0)
    out2 = SUB_PATH + 'dart_single_best.csv'
    sub2.to_csv(out2, index=False)
    print(f"✅ DART 단독 (CV 개선): {out2}")

print("\n" + "="*60)
print(f"  이전 최고 Public    : 10.3347  (CV {PREV_BEST_CV})")
print(f"  DART 앙상블 CV      : {dart_ensemble_mae:.4f}  (Δ{dart_ensemble_mae - PREV_BEST_CV:+.4f})")
print(f"  DART Optuna best    : {study.best_value:.4f}")
print("="*60)
print("완료! → ensemble_dart_meta.csv 제출")
