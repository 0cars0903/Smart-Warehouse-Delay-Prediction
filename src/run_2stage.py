"""
run_2stage.py
=============
P90 극값 2-Stage 모델링 실험

배경 (추가 EDA Part B)
----------------------
P90 임계값: 45.2분 (전체 데이터의 10%)
11개 피처가 P90 구간에서 상관 부호 역전:
  low_battery_ratio : +0.366(전체) → −0.298(극단)  ← 가장 강한 역전
  battery_mean      : −0.359      → +0.254
  pack_utilization  : +0.080      → +0.377  ← 극단 구간 지배 피처
  → 단일 모델이 두 체제를 동시에 표현할 수 없는 구조적 한계

2-Stage 설계
------------
  Stage1  : LGBM 이진분류 → P(delay > 45.2min) OOF 예측
  Stage2a : LGBM 회귀 (정상 구간 전용, y ≤ P90)
  Stage2b : LGBM 회귀 (극값 구간 전용, y > P90)
  최종예측 : P_ext × pred_extreme + (1 − P_ext) × pred_normal

비교 기준
---------
  단일 LGBM (log1p)  : CV ~8.88
  앙상블 현재 최고   : CV 8.8674 / Public 10.3347

실행
----
  python src/run_2stage.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

import lightgbm as lgb
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

# P90 임계값 (EDA 확인: 45.2분)
P90_THRESHOLD = 45.2

# ─────────────────────────────────────────────
# 기존 LGBM 최적 파라미터
# ─────────────────────────────────────────────
BASE_LGBM_PARAMS = {
    'num_leaves'       : 181,
    'learning_rate'    : 0.020616,
    'feature_fraction' : 0.5122,
    'bagging_fraction' : 0.9049,
    'min_child_samples': 26,
    'reg_alpha'        : 0.3805,
    'reg_lambda'       : 0.3630,
    'objective'        : 'regression_l1',
    'n_estimators'     : 3000,
    'bagging_freq'     : 1,
    'random_state'     : SEED,
    'verbosity'        : -1,
    'n_jobs'           : -1,
}

# 극값 전용 회귀 파라미터
# - 데이터 ~25k행 (전체의 10%) → 과적합 방지: 규제 강화, leaf 축소
EXTREME_LGBM_PARAMS = {
    'num_leaves'       : 63,         # 단순화
    'learning_rate'    : 0.030,
    'feature_fraction' : 0.6,
    'bagging_fraction' : 0.8,
    'min_child_samples': 50,         # 극값 소수 데이터 과적합 방지
    'reg_alpha'        : 1.0,        # 강한 L1 규제
    'reg_lambda'       : 1.0,        # 강한 L2 규제
    'objective'        : 'regression_l1',
    'n_estimators'     : 2000,
    'bagging_freq'     : 1,
    'random_state'     : SEED,
    'verbosity'        : -1,
    'n_jobs'           : -1,
}

# 분류기 파라미터
CLF_LGBM_PARAMS = {
    'num_leaves'       : 63,
    'learning_rate'    : 0.05,
    'feature_fraction' : 0.7,
    'bagging_fraction' : 0.8,
    'bagging_freq'     : 1,
    'min_child_samples': 30,
    'reg_alpha'        : 0.5,
    'reg_lambda'       : 0.5,
    'objective'        : 'binary',
    'metric'           : 'auc',
    'n_estimators'     : 1000,
    'random_state'     : SEED,
    'verbosity'        : -1,
    'n_jobs'           : -1,
}


# ─────────────────────────────────────────────
# 1. 데이터 로드 & 피처 엔지니어링
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 1: 데이터 로드 & 피처 엔지니어링")
print("="*60)

train_raw = pd.read_csv(DATA_PATH + 'train.csv')
test_raw  = pd.read_csv(DATA_PATH + 'test.csv')
layout    = pd.read_csv(DATA_PATH + 'layout_info.csv')
test_orig_ids = test_raw['ID'].values.copy()
print(f"Train: {train_raw.shape}, Test: {test_raw.shape}")

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
feat_cols = get_feature_cols(train, TARGET)
print(f"✅ 피처 수: {len(feat_cols)}")

X      = train[feat_cols].values.astype(np.float32)
y      = train[TARGET].values.astype(np.float32)
y_log  = np.log1p(y)
y_bin  = (y > P90_THRESHOLD).astype(np.int8)   # Stage1 레이블
X_test = test[feat_cols].values.astype(np.float32)
groups = train['scenario_id'].values
gkf    = GroupKFold(n_splits=N_SPLITS)

n_extreme = y_bin.sum()
print(f"\nP90 임계값 : {P90_THRESHOLD}분")
print(f"극값 비율   : {n_extreme}/{len(y)} = {n_extreme/len(y)*100:.1f}%")
print(f"y 실제 P90  : {np.percentile(y, 90):.2f}분")
print(f"y: mean={y.mean():.2f}, median={np.median(y):.2f}, std={y.std():.2f}")


# ─────────────────────────────────────────────
# 2. 기준 모델: 단일 LGBM (log1p) OOF
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 2: 기준 모델 (단일 LGBM + log1p)")
print("="*60)

oof_base  = np.zeros(len(y))
test_base = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()
    m = lgb.LGBMRegressor(**BASE_LGBM_PARAMS)
    m.fit(X[tr_idx], y_log[tr_idx],
          eval_set=[(X[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False),
                     lgb.log_evaluation(-1)])
    oof_base[val_idx]  = np.expm1(m.predict(X[val_idx])).clip(0)
    test_base         += np.expm1(m.predict(X_test)).clip(0) / N_SPLITS
    mae_f = mean_absolute_error(y[val_idx], oof_base[val_idx])
    print(f"  Fold {fold+1}: MAE={mae_f:.4f} ({time.time()-t0:.0f}s)")

base_mae = mean_absolute_error(y, oof_base)
print(f"✅ 기준 OOF MAE: {base_mae:.4f}")

# 극값/정상 구간별 기준 MAE
base_mae_normal  = mean_absolute_error(y[~y_bin.astype(bool)], oof_base[~y_bin.astype(bool)])
base_mae_extreme = mean_absolute_error(y[y_bin.astype(bool)],  oof_base[y_bin.astype(bool)])
print(f"  정상 구간 MAE (≤P90): {base_mae_normal:.4f}")
print(f"  극값 구간 MAE (>P90): {base_mae_extreme:.4f}")


# ─────────────────────────────────────────────
# 3. Stage 1: 이진 분류 (P(delay > P90))
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 3: Stage1 분류기 OOF (P(extreme))")
print("="*60)

oof_p_extreme  = np.zeros(len(y))
test_p_extreme = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_bin, groups=groups)):
    t0 = time.time()
    clf = lgb.LGBMClassifier(**CLF_LGBM_PARAMS)
    clf.fit(X[tr_idx], y_bin[tr_idx],
            eval_set=[(X[val_idx], y_bin[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)])
    oof_p_extreme[val_idx]  = clf.predict_proba(X[val_idx])[:, 1]
    test_p_extreme          += clf.predict_proba(X_test)[:, 1] / N_SPLITS
    auc = roc_auc_score(y_bin[val_idx], oof_p_extreme[val_idx])
    print(f"  Fold {fold+1}: AUC={auc:.4f} ({time.time()-t0:.0f}s)")

clf_auc = roc_auc_score(y_bin, oof_p_extreme)
print(f"✅ 분류기 OOF AUC: {clf_auc:.4f}")
print(f"  P_extreme 분포: mean={oof_p_extreme.mean():.3f}, "
      f"median={np.median(oof_p_extreme):.3f}, "
      f"P(>0.5)={np.mean(oof_p_extreme>0.5)*100:.1f}%")


# ─────────────────────────────────────────────
# 4. Stage 2: 체제별 회귀
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 4: Stage2 체제별 회귀 (정상/극값 분리)")
print("="*60)

oof_normal  = np.zeros(len(y))
oof_extreme = np.zeros(len(y))
test_normal  = np.zeros(len(X_test))
test_extreme = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
    t0 = time.time()

    # 훈련 폴드 내 체제 분할
    normal_mask  = ~y_bin[tr_idx].astype(bool)
    extreme_mask =  y_bin[tr_idx].astype(bool)
    n_norm_fold = normal_mask.sum()
    n_ext_fold  = extreme_mask.sum()

    # --- 정상 회귀 ---
    m_normal = lgb.LGBMRegressor(**BASE_LGBM_PARAMS)
    m_normal.fit(
        X[tr_idx][normal_mask], y_log[tr_idx][normal_mask],
        eval_set=[(X[val_idx], y_log[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    oof_normal[val_idx] = np.expm1(m_normal.predict(X[val_idx])).clip(0)
    test_normal         += np.expm1(m_normal.predict(X_test)).clip(0) / N_SPLITS

    # --- 극값 회귀 ---
    m_extreme = lgb.LGBMRegressor(**EXTREME_LGBM_PARAMS)
    m_extreme.fit(
        X[tr_idx][extreme_mask], y_log[tr_idx][extreme_mask],
        eval_set=[(X[val_idx][y_bin[val_idx].astype(bool)],
                   y_log[val_idx][y_bin[val_idx].astype(bool)])],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    oof_extreme[val_idx] = np.expm1(m_extreme.predict(X[val_idx])).clip(0)
    test_extreme          += np.expm1(m_extreme.predict(X_test)).clip(0) / N_SPLITS

    elapsed = time.time() - t0
    print(f"  Fold {fold+1}: normal_tr={n_norm_fold}, extreme_tr={n_ext_fold} ({elapsed:.0f}s)")

# Stage2 단독 MAE (정보용)
mae_normal_only  = mean_absolute_error(y[~y_bin.astype(bool)],
                                        oof_normal[~y_bin.astype(bool)])
mae_extreme_only = mean_absolute_error(y[y_bin.astype(bool)],
                                        oof_extreme[y_bin.astype(bool)])
print(f"✅ Stage2 정상 모델 OOF MAE (정상 구간): {mae_normal_only:.4f}")
print(f"✅ Stage2 극값 모델 OOF MAE (극값 구간): {mae_extreme_only:.4f}")


# ─────────────────────────────────────────────
# 5. Blend 최적화
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 5: 2-Stage 블렌드 & 기준 대비 비교")
print("="*60)

p = oof_p_extreme  # P(extreme)

def blend_mae(alpha):
    """alpha: 극값 모델 신뢰 강도 조절 (1.0 = P 그대로, >1.0 = 극값 쪽 강화)"""
    w_ext = np.clip(p * alpha, 0, 1)
    pred  = (1 - w_ext) * oof_normal + w_ext * oof_extreme
    return mean_absolute_error(y, pred)

# alpha 탐색
alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
print(f"\n  alpha 탐색 (blend weight 스케일):")
print(f"  {'alpha':>6}  {'MAE':>8}  {'vs Base':>10}")
for a in alphas:
    m = blend_mae(a)
    print(f"  {a:>6.1f}  {m:>8.4f}  {m - base_mae:>+10.4f}")

# scipy로 최적 alpha 탐색
res = minimize(blend_mae, x0=[1.0], method='Nelder-Mead',
               bounds=[(0.1, 5.0)], options={'xatol': 1e-4, 'maxiter': 200})
best_alpha = float(res.x[0])
best_2stage_mae = res.fun

w_ext_oof = np.clip(p * best_alpha, 0, 1)
oof_2stage = (1 - w_ext_oof) * oof_normal + w_ext_oof * oof_extreme

# 구간별 최종 MAE
final_mae_normal  = mean_absolute_error(y[~y_bin.astype(bool)],
                                         oof_2stage[~y_bin.astype(bool)])
final_mae_extreme = mean_absolute_error(y[y_bin.astype(bool)],
                                         oof_2stage[y_bin.astype(bool)])

print(f"\n  최적 alpha = {best_alpha:.3f}")
print(f"  {'':30} {'기준 LGBM':>12} {'2-Stage':>12} {'개선':>8}")
print(f"  {'-'*64}")
print(f"  {'전체 OOF MAE':30} {base_mae:>12.4f} {best_2stage_mae:>12.4f} "
      f"{best_2stage_mae - base_mae:>+8.4f}")
print(f"  {'정상 구간 MAE (≤P90)':30} {base_mae_normal:>12.4f} {final_mae_normal:>12.4f} "
      f"{final_mae_normal - base_mae_normal:>+8.4f}")
print(f"  {'극값 구간 MAE (>P90)':30} {base_mae_extreme:>12.4f} {final_mae_extreme:>12.4f} "
      f"{final_mae_extreme - base_mae_extreme:>+8.4f}")

delta = best_2stage_mae - base_mae
if delta < 0:
    print(f"\n✅ 2-Stage 모델이 기준 대비 {abs(delta):.4f} 개선!")
else:
    print(f"\n⚠️  2-Stage 모델이 기준 대비 {delta:.4f} 악화. 제출 파일 생략.")


# ─────────────────────────────────────────────
# 6. 앙상블 기준 모델과 추가 결합 (선택)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 6: 기준 LGBM + 2-Stage 혼합 탐색")
print("="*60)

def combo_mae(w):
    """기준 LGBM과 2-stage를 가중 평균"""
    w = np.clip(w[0], 0, 1)
    pred = (1 - w) * oof_base + w * oof_2stage
    return mean_absolute_error(y, pred)

res2 = minimize(combo_mae, x0=[0.5], method='Nelder-Mead',
                options={'xatol': 1e-4, 'maxiter': 200})
best_combo_w = float(np.clip(res2.x[0], 0, 1))
best_combo_mae = res2.fun

print(f"  최적 혼합 비율: base×{1-best_combo_w:.2f} + 2stage×{best_combo_w:.2f}")
print(f"  혼합 OOF MAE : {best_combo_mae:.4f}  (vs 기준 {base_mae:.4f})")


# ─────────────────────────────────────────────
# 7. 제출 파일 생성
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 7: 제출 파일 생성")
print("="*60)

os.makedirs(SUB_PATH, exist_ok=True)
sample = pd.read_csv(DATA_PATH + 'sample_submission.csv')

# 최종 테스트 예측
w_ext_test = np.clip(test_p_extreme * best_alpha, 0, 1)
test_2stage = (1 - w_ext_test) * test_normal + w_ext_test * test_extreme

# 기준+2stage 혼합
test_combo = (1 - best_combo_w) * test_base + best_combo_w * test_2stage

# --- 2-Stage 단독 ---
sub1 = sample.copy()
sub1[TARGET] = test_2stage.clip(0)
out1 = SUB_PATH + '2stage_p90.csv'
sub1.to_csv(out1, index=False)
print(f"✅ 2-Stage 단독: {out1}")
print(f"   예측: mean={test_2stage.mean():.2f}, std={test_2stage.std():.2f}")

# --- 기준+2Stage 혼합 ---
sub2 = sample.copy()
sub2[TARGET] = test_combo.clip(0)
out2 = SUB_PATH + '2stage_combo.csv'
sub2.to_csv(out2, index=False)
print(f"✅ 기준+2Stage 혼합: {out2}")
print(f"   예측: mean={test_combo.mean():.2f}, std={test_combo.std():.2f}")

print("\n" + "="*60)
print(f"  이전 최고 Public Score  : 10.3347")
print(f"  기준 LGBM CV MAE       : {base_mae:.4f}")
print(f"  2-Stage CV MAE         : {best_2stage_mae:.4f}  (Δ{best_2stage_mae-base_mae:+.4f})")
print(f"  혼합 CV MAE            : {best_combo_mae:.4f}  (Δ{best_combo_mae-base_mae:+.4f})")
print(f"  분류기 AUC             : {clf_auc:.4f}")
print("="*60)
print("\n제출 우선순위:")
scores = [
    ("2stage_combo.csv",  best_combo_mae),
    ("2stage_p90.csv",    best_2stage_mae),
]
scores.sort(key=lambda x: x[1])
for rank, (name, mae) in enumerate(scores, 1):
    print(f"  {rank}. {name}  (CV {mae:.4f})")
