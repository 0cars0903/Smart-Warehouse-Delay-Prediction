"""
run_v2.py
=========
개선 전략:
1. Target Encoding (layout_id, 교차검증 safe)
2. Sqrt transform (log1p보다 덜 압축 → 극값 예측 개선)
3. 확장 lag(1~6) + rolling(3,5,10)
4. LightGBM (Optuna best params 재사용) + CatBoost + XGBoost 앙상블
5. 최적 가중치 블렌딩
"""

import pandas as pd
import numpy as np
import sys, warnings, os, time
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from feature_engineering import (
    merge_layout, encode_categoricals, add_ts_features,
    add_lag_features, add_rolling_features, add_domain_features,
    get_feature_cols
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'
SUB_PATH  = os.path.join(os.path.dirname(__file__), '..', 'submissions') + '/'
TARGET    = 'avg_delay_minutes_next_30m'
SEED      = 42
N_SPLITS  = 5

# ─────────────────────────────────────────────
# 이전 Optuna 최적 파라미터 (재사용)
# ─────────────────────────────────────────────
BEST_LGBM_PARAMS = {
    'num_leaves': 181,
    'learning_rate': 0.020616343657609632,
    'feature_fraction': 0.5121950779327391,
    'bagging_fraction': 0.9048846816723592,
    'min_child_samples': 26,
    'reg_alpha': 0.3804619445237042,
    'reg_lambda': 0.3629852778670354,
    'objective': 'regression_l1',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'n_estimators': 3000,
    'bagging_freq': 1,
    'random_state': SEED,
}

# ─────────────────────────────────────────────
# 1. 데이터 로드 & 기본 피처
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 1: 데이터 로드 & 피처 엔지니어링")
print("="*60)

train_raw = pd.read_csv(DATA_PATH + 'train.csv')
test_raw  = pd.read_csv(DATA_PATH + 'test.csv')
layout    = pd.read_csv(DATA_PATH + 'layout_info.csv')
test_orig_ids = test_raw['ID'].values.copy()

# 기본 파이프라인
train, test = merge_layout(train_raw.copy(), test_raw.copy(), layout)
train, test = encode_categoricals(train, test, TARGET)
train = add_ts_features(train)
test  = add_ts_features(test)

# 확장 Lag/Rolling
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

assert (test['ID'].values == test_orig_ids).all()
print(f"기본 피처 완료")

# ─────────────────────────────────────────────
# 2. Target Encoding (GroupKFold safe)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 2: Target Encoding")
print("="*60)

gkf = GroupKFold(n_splits=N_SPLITS)
groups = train['scenario_id'].values

# layout_id Target Encoding (CV-safe: 검증 폴드 제외하고 계산)
train['layout_te'] = np.nan
for tr_idx, val_idx in gkf.split(train, groups=groups):
    tr_part = train.iloc[tr_idx]
    layout_means = tr_part.groupby('layout_id')[TARGET].mean()
    train.loc[train.index[val_idx], 'layout_te'] = \
        train.iloc[val_idx]['layout_id'].map(layout_means)

# 전체 통계로 test set 인코딩
layout_means_all = train.groupby('layout_id')[TARGET].mean()
test['layout_te'] = test['layout_id'].map(layout_means_all)

# scenario_id 타입 기반 통계 (layout × shift_hour)
train['layout_shift_te'] = np.nan
for tr_idx, val_idx in gkf.split(train, groups=groups):
    tr_part = train.iloc[tr_idx]
    grp = tr_part.groupby(['layout_id', 'shift_hour'])[TARGET].mean() \
          if 'shift_hour' in tr_part.columns else \
          tr_part.groupby('layout_id')[TARGET].mean()
    train.loc[train.index[val_idx], 'layout_shift_te'] = \
        train.iloc[val_idx].set_index(['layout_id', 'shift_hour']).index.map(
            grp.to_dict()
        ) if 'shift_hour' in train.columns else \
        train.iloc[val_idx]['layout_id'].map(
            tr_part.groupby('layout_id')[TARGET].mean()
        )

layout_shift_all = train.groupby(['layout_id', 'shift_hour'])[TARGET].mean() \
    if 'shift_hour' in train.columns else layout_means_all
test['layout_shift_te'] = test.set_index(['layout_id', 'shift_hour']).index.map(
    layout_shift_all.to_dict()
) if 'shift_hour' in test.columns else test['layout_id'].map(layout_means_all)

# NaN 채우기 (글로벌 평균)
global_mean = train[TARGET].mean()
for col in ['layout_te', 'layout_shift_te']:
    train[col] = train[col].fillna(global_mean)
    test[col]  = test[col].fillna(global_mean)

print(f"layout_te 분포: mean={train['layout_te'].mean():.2f}, std={train['layout_te'].std():.2f}")
print(f"layout_te 상관계수: {train['layout_te'].corr(train[TARGET]):.4f}")

# ─────────────────────────────────────────────
# 3. 피처 준비 (target encoding 포함)
# ─────────────────────────────────────────────
feat_cols = get_feature_cols(train, TARGET)
# target encoding 컬럼 추가
te_cols = ['layout_te', 'layout_shift_te']
for c in te_cols:
    if c in train.columns and c not in feat_cols:
        feat_cols.append(c)

print(f"\n총 피처 수: {len(feat_cols)}")

X      = train[feat_cols].values.astype(np.float32)
y      = train[TARGET].values.astype(np.float32)

# Sqrt transform (log1p보다 덜 압축)
y_sqrt = np.sqrt(y)
y_log  = np.log1p(y)

X_test = test[feat_cols].values.astype(np.float32)

print(f"y_sqrt: mean={y_sqrt.mean():.3f}, std={y_sqrt.std():.3f}, skew={pd.Series(y_sqrt).skew():.3f}")
print(f"y_log : mean={y_log.mean():.3f}, std={y_log.std():.3f}, skew={pd.Series(y_log).skew():.3f}")

# ─────────────────────────────────────────────
# 4. 빠른 비교: sqrt vs log1p (2-fold)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 3: Sqrt vs Log1p 변환 비교 (2-fold 빠른 검증)")
print("="*60)

gkf2 = GroupKFold(n_splits=2)

quick_params = dict(BEST_LGBM_PARAMS)
quick_params['n_estimators'] = 1000

for transform_name, y_t, inv_fn in [
    ('sqrt ', y_sqrt, lambda x: np.clip(x**2, 0, None)),
    ('log1p', y_log,  lambda x: np.expm1(np.clip(x, 0, None))),
]:
    oof = np.zeros(len(X))
    for tr_idx, val_idx in gkf2.split(X, y_t, groups=groups):
        m = lgb.LGBMRegressor(**quick_params)
        m.fit(X[tr_idx], y_t[tr_idx],
              eval_set=[(X[val_idx], y_t[val_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        oof[val_idx] = inv_fn(m.predict(X[val_idx]))
    mae = mean_absolute_error(y, oof)
    pred_std = oof.std()
    print(f"  {transform_name}: MAE={mae:.4f}, pred_std={pred_std:.2f} (true_std={y.std():.2f})")

# ─────────────────────────────────────────────
# 5. Full CV - LightGBM (sqrt + log1p 앙상블)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 4: LightGBM Full 5-fold (sqrt + log1p 블렌드)")
print("="*60)

oof_lgbm_sqrt = np.zeros(len(X))
oof_lgbm_log  = np.zeros(len(X))
test_lgbm_sqrt = np.zeros(len(X_test))
test_lgbm_log  = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_sqrt, groups=groups)):
    t0 = time.time()
    # sqrt 모델
    m_sqrt = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
    m_sqrt.fit(X[tr_idx], y_sqrt[tr_idx],
               eval_set=[(X[val_idx], y_sqrt[val_idx])],
               callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_lgbm_sqrt[val_idx] = np.clip(m_sqrt.predict(X[val_idx]), 0, None)**2
    test_lgbm_sqrt += np.clip(m_sqrt.predict(X_test), 0, None)**2 / N_SPLITS

    # log1p 모델
    m_log = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
    m_log.fit(X[tr_idx], y_log[tr_idx],
              eval_set=[(X[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_lgbm_log[val_idx] = np.expm1(m_log.predict(X[val_idx])).clip(0)
    test_lgbm_log += np.expm1(m_log.predict(X_test)).clip(0) / N_SPLITS

    mae_s = mean_absolute_error(y[val_idx], oof_lgbm_sqrt[val_idx])
    mae_l = mean_absolute_error(y[val_idx], oof_lgbm_log[val_idx])
    print(f"  Fold {fold+1}: sqrt={mae_s:.4f}, log={mae_l:.4f} ({time.time()-t0:.0f}s)")

# 두 LightGBM 블렌드
oof_lgbm  = 0.5 * oof_lgbm_sqrt + 0.5 * oof_lgbm_log
test_lgbm = 0.5 * test_lgbm_sqrt + 0.5 * test_lgbm_log
lgbm_mae  = mean_absolute_error(y, oof_lgbm)
print(f"\n✅ LightGBM (sqrt+log blend) OOF MAE: {lgbm_mae:.4f}")
print(f"   pred_std={oof_lgbm.std():.2f}, max={oof_lgbm.max():.2f}")

# ─────────────────────────────────────────────
# 6. CatBoost Full CV
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 5: CatBoost Full 5-fold (sqrt)")
print("="*60)

cb_params = dict(
    iterations=2000, learning_rate=0.05, depth=8,
    l2_leaf_reg=3.0, loss_function='MAE', eval_metric='MAE',
    random_seed=SEED, early_stopping_rounds=100,
    verbose=0, allow_writing_files=False,
)

oof_cb  = np.zeros(len(X))
test_cb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_sqrt, groups=groups)):
    t0 = time.time()
    m = cb.CatBoostRegressor(**cb_params)
    m.fit(X[tr_idx], y_sqrt[tr_idx], eval_set=(X[val_idx], y_sqrt[val_idx]))
    oof_cb[val_idx] = np.clip(m.predict(X[val_idx]), 0, None)**2
    test_cb += np.clip(m.predict(X_test), 0, None)**2 / N_SPLITS
    mae = mean_absolute_error(y[val_idx], oof_cb[val_idx])
    print(f"  Fold {fold+1}: MAE={mae:.4f} ({time.time()-t0:.0f}s)")

cb_mae = mean_absolute_error(y, oof_cb)
print(f"✅ CatBoost OOF MAE: {cb_mae:.4f}, pred_std={oof_cb.std():.2f}")

# ─────────────────────────────────────────────
# 7. XGBoost Full CV
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 6: XGBoost Full 5-fold (sqrt)")
print("="*60)

xgb_params = dict(
    n_estimators=3000, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    objective='reg:squarederror',   # sqrt 타깃에는 MSE 목적함수도 유효
    eval_metric='rmse',
    random_state=SEED, tree_method='hist',
    early_stopping_rounds=100, verbosity=0,
)

oof_xgb  = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_sqrt, groups=groups)):
    t0 = time.time()
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X[tr_idx], y_sqrt[tr_idx],
          eval_set=[(X[val_idx], y_sqrt[val_idx])], verbose=False)
    oof_xgb[val_idx] = np.clip(m.predict(X[val_idx]), 0, None)**2
    test_xgb += np.clip(m.predict(X_test), 0, None)**2 / N_SPLITS
    mae = mean_absolute_error(y[val_idx], oof_xgb[val_idx])
    print(f"  Fold {fold+1}: MAE={mae:.4f} ({time.time()-t0:.0f}s)")

xgb_mae = mean_absolute_error(y, oof_xgb)
print(f"✅ XGBoost OOF MAE: {xgb_mae:.4f}, pred_std={oof_xgb.std():.2f}")

# ─────────────────────────────────────────────
# 8. 최적 앙상블 가중치
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 7: 최적 앙상블 가중치 탐색")
print("="*60)

oofs  = np.stack([oof_lgbm, oof_cb, oof_xgb], axis=1)
names = ['LightGBM', 'CatBoost', 'XGBoost']

def neg_mae(w):
    w = np.abs(w) / np.abs(w).sum()
    return mean_absolute_error(y, (oofs * w).sum(axis=1))

res = minimize(neg_mae, [1/3,1/3,1/3], method='Nelder-Mead',
               options={'maxiter':2000, 'xatol':1e-7})
opt_w = np.abs(res.x) / np.abs(res.x).sum()

print(f"\n{'모델':<12} {'OOF MAE':>10}  {'가중치':>8}")
print("-"*34)
for nm, mv, w in zip(names, [lgbm_mae, cb_mae, xgb_mae], opt_w):
    print(f"  {nm:<10} {mv:>10.4f}  {w:>8.3f}")
print("-"*34)
print(f"  균등 앙상블  {neg_mae([1/3,1/3,1/3]):>10.4f}")
print(f"  최적 앙상블  {res.fun:>10.4f}")

# ─────────────────────────────────────────────
# 9. 제출 파일
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 8: 제출 파일 생성")
print("="*60)

test_ens = (test_lgbm * opt_w[0] +
            test_cb   * opt_w[1] +
            test_xgb  * opt_w[2]).clip(0)

# 이전 앙상블과 블렌딩 (v1 + v2 평균)
prev_ens = pd.read_csv(SUB_PATH + 'sub_ens.csv')['avg_delay_minutes_next_30m'].values
test_blend_prev = 0.5 * test_ens + 0.5 * prev_ens

sample = pd.read_csv(DATA_PATH + 'sample_submission.csv')

# v2 앙상블
sub_v2 = sample.copy()
sub_v2['avg_delay_minutes_next_30m'] = test_ens
sub_v2.to_csv(SUB_PATH + 'sub_v2.csv', index=False)

# v1+v2 블렌드
sub_blend = sample.copy()
sub_blend['avg_delay_minutes_next_30m'] = test_blend_prev
sub_blend.to_csv(SUB_PATH + 'sub_blend_v1v2.csv', index=False)

print(f"✅ sub_v2.csv        : mean={test_ens.mean():.2f}, std={test_ens.std():.2f}, max={test_ens.max():.2f}")
print(f"✅ sub_blend_v1v2.csv: mean={test_blend_prev.mean():.2f}, std={test_blend_prev.std():.2f}, max={test_blend_prev.max():.2f}")

print("\n" + "="*60)
print(f"  이전 Public Score      : 10.3349 (sub_ens.csv)")
print(f"  v2 OOF MAE             : {res.fun:.4f}")
print(f"  v1 OOF MAE (이전)      : 8.8703")
print("="*60)
print("  ▶ sub_v2.csv 먼저, 그 다음 sub_blend_v1v2.csv 제출 권장")
