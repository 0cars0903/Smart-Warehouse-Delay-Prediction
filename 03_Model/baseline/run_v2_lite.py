"""
run_v2_lite.py
==============
v2 경량화 버전 (OOM 방지)
- LightGBM 단독 (CatBoost/XGBoost 제거 → 메모리 절감)
- Target Encoding for layout_id (CV-safe)
- Sqrt + log1p 두 transform 비교 후 더 좋은 것 선택
- v1 앙상블(sub_ens.csv)과 블렌딩

실행 시간 목표: ~15분 이내
"""

import pandas as pd
import numpy as np
import sys, warnings, os, time, gc
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
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

BEST_LGBM_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616343657609632,
    'feature_fraction': 0.5121950779327391, 'bagging_fraction': 0.9048846816723592,
    'min_child_samples': 26, 'reg_alpha': 0.3804619445237042,
    'reg_lambda': 0.3629852778670354,
    'objective': 'regression_l1', 'metric': 'mae',
    'verbosity': -1, 'boosting_type': 'gbdt',
    'n_estimators': 3000, 'bagging_freq': 1, 'random_state': SEED,
}

# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 1: 데이터 로드 & 피처 엔지니어링")
print("="*60)

train_raw = pd.read_csv(DATA_PATH + 'train.csv')
test_raw  = pd.read_csv(DATA_PATH + 'test.csv')
layout    = pd.read_csv(DATA_PATH + 'layout_info.csv')
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

assert (test['ID'].values == test_orig_ids).all()
groups = train['scenario_id'].values
gkf    = GroupKFold(n_splits=N_SPLITS)

# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 2: Target Encoding (layout_id, CV-safe)")
print("="*60)

# layout_id Target Encoding
train['layout_te'] = np.nan
for tr_idx, val_idx in gkf.split(train, groups=groups):
    layout_means = train.iloc[tr_idx].groupby('layout_id')[TARGET].mean()
    train.loc[train.index[val_idx], 'layout_te'] = \
        train.iloc[val_idx]['layout_id'].map(layout_means)

layout_means_all = train.groupby('layout_id')[TARGET].mean()
test['layout_te'] = test['layout_id'].map(layout_means_all)

# layout_id std encoding (분산도 시그널)
train['layout_te_std'] = np.nan
for tr_idx, val_idx in gkf.split(train, groups=groups):
    layout_stds = train.iloc[tr_idx].groupby('layout_id')[TARGET].std()
    train.loc[train.index[val_idx], 'layout_te_std'] = \
        train.iloc[val_idx]['layout_id'].map(layout_stds)

layout_stds_all = train.groupby('layout_id')[TARGET].std()
test['layout_te_std'] = test['layout_id'].map(layout_stds_all)

# layout × ts_idx Target Encoding (layout 내 시간대별 패턴)
train['layout_ts_te'] = np.nan
for tr_idx, val_idx in gkf.split(train, groups=groups):
    grp = train.iloc[tr_idx].groupby(['layout_id', 'ts_idx'])[TARGET].mean()
    idx_val = list(zip(train.iloc[val_idx]['layout_id'], train.iloc[val_idx]['ts_idx']))
    train.loc[train.index[val_idx], 'layout_ts_te'] = [grp.get(k, np.nan) for k in idx_val]

grp_all = train.groupby(['layout_id', 'ts_idx'])[TARGET].mean()
idx_test = list(zip(test['layout_id'], test['ts_idx']))
test['layout_ts_te'] = [grp_all.get(k, np.nan) for k in idx_test]

global_mean = train[TARGET].mean()
for col in ['layout_te', 'layout_te_std', 'layout_ts_te']:
    train[col] = train[col].fillna(global_mean)
    test[col]  = test[col].fillna(global_mean)

print(f"layout_te 상관계수: {train['layout_te'].corr(train[TARGET]):.4f}")
print(f"layout_te_std 상관: {train['layout_te_std'].corr(train[TARGET]):.4f}")
print(f"layout_ts_te 상관: {train['layout_ts_te'].corr(train[TARGET]):.4f}")

# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 3: 피처 준비")
print("="*60)

feat_cols = get_feature_cols(train, TARGET)
for c in ['layout_te', 'layout_te_std', 'layout_ts_te']:
    if c not in feat_cols:
        feat_cols.append(c)

print(f"총 피처 수: {len(feat_cols)}")

X      = train[feat_cols].values.astype(np.float32)
y      = train[TARGET].values.astype(np.float32)
y_sqrt = np.sqrt(y)
y_log  = np.log1p(y)
X_test = test[feat_cols].values.astype(np.float32)

# 메모리 정리
del train_raw, test_raw, layout
gc.collect()
print(f"메모리 정리 완료. X: {X.shape}")

# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 4: Sqrt vs Log1p 2-fold 빠른 비교")
print("="*60)

gkf2 = GroupKFold(n_splits=2)
quick = dict(BEST_LGBM_PARAMS)
quick['n_estimators'] = 800

best_transform = None
best_quick_mae = float('inf')

for tname, y_t, inv_fn in [
    ('sqrt ', y_sqrt, lambda x: np.clip(x, 0, None)**2),
    ('log1p', y_log,  lambda x: np.expm1(np.clip(x, 0, None))),
]:
    oof = np.zeros(len(X))
    for tr_idx, val_idx in gkf2.split(X, y_t, groups=groups):
        m = lgb.LGBMRegressor(**quick)
        m.fit(X[tr_idx], y_t[tr_idx],
              eval_set=[(X[val_idx], y_t[val_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        oof[val_idx] = inv_fn(m.predict(X[val_idx]))
        del m; gc.collect()
    mae = mean_absolute_error(y, oof)
    pstd = oof.std()
    print(f"  {tname}: MAE={mae:.4f}, pred_std={pstd:.2f}")
    if mae < best_quick_mae:
        best_quick_mae = mae
        best_transform = tname.strip()

print(f"\n→ 선택된 transform: {best_transform}")

if best_transform == 'sqrt':
    y_use  = y_sqrt
    inv_fn = lambda x: np.clip(x, 0, None)**2
else:
    y_use  = y_log
    inv_fn = lambda x: np.expm1(np.clip(x, 0, None))

# ─────────────────────────────────────────────
print("\n" + "="*60)
print(f" STEP 5: LightGBM Full 5-fold ({best_transform})")
print("="*60)

oof_lgbm  = np.zeros(len(X))
test_lgbm = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_use, groups=groups)):
    t0 = time.time()
    m = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
    m.fit(X[tr_idx], y_use[tr_idx],
          eval_set=[(X[val_idx], y_use[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_lgbm[val_idx] = inv_fn(m.predict(X[val_idx]))
    test_lgbm        += inv_fn(m.predict(X_test)) / N_SPLITS
    mae = mean_absolute_error(y[val_idx], oof_lgbm[val_idx])
    print(f"  Fold {fold+1}: MAE={mae:.4f} ({time.time()-t0:.0f}s)")
    del m; gc.collect()

lgbm_mae = mean_absolute_error(y, oof_lgbm)
print(f"\n✅ LightGBM v2 OOF MAE : {lgbm_mae:.4f}")
print(f"   pred_std={oof_lgbm.std():.2f} (true={y.std():.2f})")
print(f"   pred_max={oof_lgbm.max():.1f}  (true={y.max():.1f})")

# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 6: 제출 파일 생성")
print("="*60)

sample = pd.read_csv(DATA_PATH + 'sample_submission.csv')

# v2 단독
sub_v2 = sample.copy()
sub_v2['avg_delay_minutes_next_30m'] = test_lgbm.clip(0)
sub_v2.to_csv(SUB_PATH + 'sub_v2_lite.csv', index=False)

# v1 앙상블과 블렌딩 (50:50)
prev = pd.read_csv(SUB_PATH + 'sub_ens.csv')['avg_delay_minutes_next_30m'].values
blend = 0.5 * test_lgbm.clip(0) + 0.5 * prev
sub_blend = sample.copy()
sub_blend['avg_delay_minutes_next_30m'] = blend
sub_blend.to_csv(SUB_PATH + 'sub_v2_blend.csv', index=False)

print(f"✅ sub_v2_lite.csv  : mean={test_lgbm.mean():.2f}, std={test_lgbm.std():.2f}, max={test_lgbm.max():.1f}")
print(f"✅ sub_v2_blend.csv : mean={blend.mean():.2f}, std={blend.std():.2f}, max={blend.max():.1f}")

print("\n" + "="*60)
print(f"  이전 최고 Public  : 10.3349 (sub_ens.csv)")
print(f"  v2 lite OOF MAE   : {lgbm_mae:.4f}")
print(f"  v1 OOF MAE        : 8.8703")
print("="*60)
print("  ▶ sub_v2_lite.csv 먼저, 다음 sub_v2_blend.csv 제출 권장")
