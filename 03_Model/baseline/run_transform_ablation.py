"""
run_transform_ablation.py
=========================
타겟 변환 방식별 CV MAE 비교 + 예측 분포 압축 해소 실험

문제: 현재 log1p 변환 → 예측 std=13.5 vs 실제 std=27.4 → 극값 과소예측
목표: 더 적합한 변환 or 후처리로 Public < 10.0

실험 목록:
  T0. log1p (현재 기준)
  T1. sqrt 변환 (y^0.5 / pred^2)
  T2. 변환 없음 (direct MAE)
  T3. log1p + std stretch 후처리  ← 계산 비용 0, 즉시 적용 가능
  T4. sqrt + std stretch 후처리

실행: python src/run_transform_ablation.py
예상 시간: 약 10~15분 (LightGBM 5-fold × 5 실험)
"""

import pandas as pd
import numpy as np
import sys, os, warnings, time
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

# ─── 설정 ────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'
SUB_PATH  = os.path.join(os.path.dirname(__file__), '..', 'submissions') + '/'
TARGET    = 'avg_delay_minutes_next_30m'
SEED      = 42
N_SPLITS  = 5

# 현재 최적 LGBM 파라미터 (CLAUDE.md)
LGBM_PARAMS = {
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

KEY_COLS_EXT = [
    'low_battery_ratio', 'battery_mean', 'charge_queue_length',
    'robot_idle', 'order_inflow_15m', 'congestion_score',
    'max_zone_density', 'avg_trip_distance',
    'robot_utilization', 'task_reassign_15m', 'blocked_path_15m',
    'urgent_order_ratio', 'fault_count_15m', 'avg_recovery_time',
]


# ─── 변환 함수 ───────────────────────────────────────────────

class TargetTransform:
    """타겟 변환 인터페이스"""

    class Log1p:
        name = 'log1p'
        @staticmethod
        def forward(y):  return np.log1p(y)
        @staticmethod
        def inverse(y):  return np.expm1(y).clip(0)

    class Sqrt:
        name = 'sqrt'
        @staticmethod
        def forward(y):  return np.sqrt(y.clip(0))
        @staticmethod
        def inverse(y):  return np.square(y).clip(0)

    class Identity:
        name = 'identity (no transform)'
        @staticmethod
        def forward(y):  return y.copy()
        @staticmethod
        def inverse(y):  return y.clip(0)


def std_stretch(oof_pred: np.ndarray, y_true: np.ndarray,
                test_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    예측 분포 표준편차를 실제 분포에 맞춰 선형 스케일링

    pred_stretched = mean_true + (pred - mean_pred) * (std_true / std_pred)

    주의: OOF 기준으로 scale factor 계산 → test에 동일 factor 적용
    """
    mean_pred = oof_pred.mean()
    std_pred  = oof_pred.std()
    mean_true = y_true.mean()
    std_true  = y_true.std()

    scale = std_true / (std_pred + 1e-8)

    oof_stretched  = (mean_true + (oof_pred - mean_pred) * scale).clip(0)
    test_stretched = (mean_true + (test_pred - mean_pred) * scale).clip(0)

    return oof_stretched, test_stretched, scale


# ─── CV 함수 ─────────────────────────────────────────────────

def run_cv(X, y, X_test, transform, label=''):
    """5-fold GroupKFold CV with target transform → (oof_mae, oof_pred, test_pred)"""
    y_t    = transform.forward(y)
    groups = train_df['scenario_id'].values
    gkf    = GroupKFold(n_splits=N_SPLITS)
    oof    = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_t, groups=groups)):
        t0 = time.time()
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(X[tr_idx], y_t[tr_idx],
              eval_set=[(X[val_idx], y_t[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[val_idx]  = transform.inverse(m.predict(X[val_idx]))
        test_preds   += transform.inverse(m.predict(X_test)) / N_SPLITS
        fold_mae = mean_absolute_error(y[val_idx], oof[val_idx])
        print(f"    Fold {fold+1}: MAE={fold_mae:.4f}  ({time.time()-t0:.0f}s)")

    oof_mae = mean_absolute_error(y, oof)
    return oof_mae, oof, test_preds


# ─── 공통 전처리 ─────────────────────────────────────────────

print("\n" + "="*60)
print(" 공통 전처리")
print("="*60)
t0 = time.time()

train_raw     = pd.read_csv(DATA_PATH + 'train.csv')
test_raw      = pd.read_csv(DATA_PATH + 'test.csv')
layout        = pd.read_csv(DATA_PATH + 'layout_info.csv')
test_orig_ids = test_raw['ID'].values.copy()

train_df, test_df = merge_layout(train_raw.copy(), test_raw.copy(), layout)
train_df, test_df = encode_categoricals(train_df, test_df, TARGET)
train_df = add_ts_features(train_df)
test_df  = add_ts_features(test_df)
train_df, test_df = add_lag_features(train_df, test_df,
                                      key_cols=KEY_COLS_EXT, lags=[1,2,3,4,5,6])
train_df, test_df = add_rolling_features(train_df, test_df,
                                          key_cols=KEY_COLS_EXT, windows=[3,5,10])
train_df = add_domain_features(train_df)
test_df  = add_domain_features(test_df)

assert (test_df['ID'].values == test_orig_ids).all(), "❌ ID 순서 오류!"

feat_cols = get_feature_cols(train_df, TARGET)
X      = train_df[feat_cols].values.astype(np.float32)
y      = train_df[TARGET].values.astype(np.float32)
X_test = test_df[feat_cols].values.astype(np.float32)

print(f"전처리 완료 ({time.time()-t0:.0f}s) | 피처={len(feat_cols)}")
print(f"y: mean={y.mean():.2f}, std={y.std():.2f}, "
      f"p90={np.percentile(y,90):.1f}, p99={np.percentile(y,99):.1f}")


# ─── 실험 실행 ───────────────────────────────────────────────

results = []
submissions = {}
transforms = [
    TargetTransform.Log1p,
    TargetTransform.Sqrt,
    TargetTransform.Identity,
]

for tf in transforms:
    print(f"\n{'='*60}")
    print(f" 실험: {tf.name}")
    print(f"{'='*60}")

    mae, oof_pred, test_pred = run_cv(X, y, X_test, tf)

    oof_std = oof_pred.std()
    stretch_ratio = y.std() / (oof_std + 1e-8)

    print(f"  ✅ OOF MAE : {mae:.4f}")
    print(f"  예측 통계  : mean={oof_pred.mean():.2f}, std={oof_std:.2f}  "
          f"(실제 std={y.std():.2f}, 압축비={stretch_ratio:.2f}x)")

    # std stretch 후처리 적용
    oof_s, test_s, scale = std_stretch(oof_pred, y, test_pred)
    mae_s = mean_absolute_error(y, oof_s)
    print(f"  + std stretch (×{scale:.3f}): OOF MAE={mae_s:.4f}  "
          f"({'↓' if mae_s < mae else '↑'}{abs(mae_s - mae):.4f})")

    results.append({
        '변환'        : tf.name,
        'OOF_MAE'    : mae,
        'OOF_std'    : oof_std,
        'stretch비'  : stretch_ratio,
        'MAE_after_stretch': mae_s,
        'scale'      : scale,
    })
    submissions[tf.name] = {
        'raw': test_pred,
        'stretched': test_s,
    }


# ─── 결과 요약 ───────────────────────────────────────────────

print(f"\n{'='*60}")
print(" 최종 결과 요약")
print(f"{'='*60}")
base_mae = results[0]['OOF_MAE']  # log1p 기준

print(f"\n{'변환':<22} {'OOF MAE':>9}  {'Δ(vs log1p)':>12}  "
      f"{'예측std':>8}  {'압축비':>7}  {'stretch후 MAE':>14}")
print("-" * 80)
for r in results:
    delta = base_mae - r['OOF_MAE']
    delta_s = base_mae - r['MAE_after_stretch']
    print(f"  {r['변환']:<20} {r['OOF_MAE']:>9.4f}  "
          f"{'→':>3}{'-' if delta>=0 else '+'}{abs(delta):>8.4f}  "
          f"{r['OOF_std']:>8.2f}  "
          f"{r['stretch비']:>7.2f}x  "
          f"{r['MAE_after_stretch']:>14.4f}  "
          f"({'↓' if delta_s >= 0 else '↑'}{abs(delta_s):.4f})")
print("-" * 80)
print(f"  실제 std(y): {y.std():.2f}")


# ─── 최고 조합으로 제출 파일 생성 ────────────────────────────

print(f"\n{'='*60}")
print(" 제출 파일 생성 (모든 조합)")
print(f"{'='*60}")

sample_sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')

for r in results:
    tf_name = r['변환'].split()[0]  # 'log1p', 'sqrt', 'identity'
    sub     = sample_sub.copy()

    # raw
    sub[TARGET] = submissions[r['변환']]['raw'].clip(0)
    fname = f"transform_{tf_name}_raw.csv"
    sub.to_csv(SUB_PATH + fname, index=False)
    print(f"  저장: {fname}  (MAE={r['OOF_MAE']:.4f})")

    # stretched
    sub[TARGET] = submissions[r['변환']]['stretched'].clip(0)
    fname_s = f"transform_{tf_name}_stretched_x{r['scale']:.3f}.csv"
    sub.to_csv(SUB_PATH + fname_s, index=False)
    print(f"  저장: {fname_s}  (MAE={r['MAE_after_stretch']:.4f})")


# ─── CSV 저장 ────────────────────────────────────────────────

import datetime
ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M')
out_csv = os.path.join(os.path.dirname(__file__), '..', 'docs',
                        f'transform_ablation_{ts}.csv')
pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n결과 저장: {out_csv}")
print("완료!")
