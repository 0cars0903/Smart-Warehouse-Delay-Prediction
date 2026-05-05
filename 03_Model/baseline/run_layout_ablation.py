"""
run_layout_ablation.py
======================
layout_info 활용 전략 단계별 CV MAE 검증

실험 순서 (누적 적용):
  Baseline  : merge_layout + ordinal enc + lag/rolling/domain (현재 파이프라인)
  Exp A     : layout_type one-hot 인코딩
  Exp A+B   : + 파생 비율 피처 6종
  Exp A+B+D : + layout_id Target Encoding (OOF, fold 내부 적용)
  Exp A+B+D+C: + layout_type × 운영 피처 교호작용

실행: python src/run_layout_ablation.py
예상 시간: 약 15~20분 (LightGBM 5-fold × 5 실험)
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
    merge_layout, add_ts_features,
    add_lag_features, add_rolling_features, add_domain_features,
)

# ─── 설정 ────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'
TARGET    = 'avg_delay_minutes_next_30m'
SEED      = 42
N_SPLITS  = 5

# Best LGBM params (CLAUDE.md, n_estimators를 1500으로 제한해 속도 최적화)
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
    'n_estimators'     : 1500,
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


# ─── 유틸 함수 ──────────────────────────────────────────────

def get_feat_cols(df, target=TARGET):
    """모델 입력 피처 목록 반환 (비수치·메타 컬럼 제외)"""
    exclude = {'ID', 'layout_id', 'scenario_id', target}
    return [c for c in df.columns
            if c not in exclude and df[c].dtype.name not in ('object', 'category')]


def run_cv_basic(train_df, feat_cols, label=''):
    """기본 5-fold GroupKFold CV → (OOF MAE, oof 배열)"""
    X      = train_df[feat_cols].values.astype(np.float32)
    y      = train_df[TARGET].values.astype(np.float32)
    y_log  = np.log1p(y)
    groups = train_df['scenario_id'].values
    gkf    = GroupKFold(n_splits=N_SPLITS)
    oof    = np.zeros(len(X))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
        t0 = time.time()
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(X[tr_idx], y_log[tr_idx],
              eval_set=[(X[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[val_idx] = np.expm1(m.predict(X[val_idx])).clip(0)
        print(f"    Fold {fold+1}: MAE={mean_absolute_error(y[val_idx], oof[val_idx]):.4f}"
              f"  ({time.time()-t0:.0f}s)")

    mae = mean_absolute_error(y, oof)
    return mae, oof


def run_cv_with_target_enc(train_df, feat_cols_base, label=''):
    """
    Target Encoding을 OOF 방식으로 fold 내부에서 생성하여 적용.
    feat_cols_base에 target enc 컬럼은 미포함 → fold마다 추가
    """
    y      = train_df[TARGET].values.astype(np.float32)
    y_log  = np.log1p(y)
    groups = train_df['scenario_id'].values
    gkf    = GroupKFold(n_splits=N_SPLITS)
    oof    = np.zeros(len(train_df))

    te_cols   = ['layout_target_mean', 'layout_target_std']
    feat_cols = feat_cols_base + te_cols

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_df, y_log, groups=groups)):
        t0 = time.time()

        tr_df  = train_df.iloc[tr_idx].copy()
        val_df = train_df.iloc[val_idx].copy()

        # fold 내 train 기준 layout_id 통계
        agg = (tr_df.groupby('layout_id')[TARGET]
                    .agg(['mean', 'std'])
                    .rename(columns={'mean': 'layout_target_mean',
                                     'std':  'layout_target_std'}))
        global_mean = tr_df[TARGET].mean()
        global_std  = tr_df[TARGET].std()

        # join → unseen layout_id는 global 통계로 대체
        tr_df  = tr_df.join(agg,  on='layout_id')
        val_df = val_df.join(agg, on='layout_id')
        for col, fill in [('layout_target_mean', global_mean),
                           ('layout_target_std',  global_std)]:
            tr_df[col].fillna(fill, inplace=True)
            val_df[col].fillna(fill, inplace=True)

        X_tr  = tr_df[feat_cols].values.astype(np.float32)
        X_val = val_df[feat_cols].values.astype(np.float32)

        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr, y_log[tr_idx],
              eval_set=[(X_val, y_log[val_idx])],
              callbacks=[lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[val_idx] = np.expm1(m.predict(X_val)).clip(0)
        print(f"    Fold {fold+1}: MAE={mean_absolute_error(y[val_idx], oof[val_idx]):.4f}"
              f"  ({time.time()-t0:.0f}s)")

    mae = mean_absolute_error(y, oof)
    return mae, oof


# ─── 피처 생성 함수 ─────────────────────────────────────────

def add_layout_ratio_features(df):
    """전략 B: 파생 비율 피처 6종"""
    df = df.copy()
    df['robot_per_1000sqm']       = df['robot_total'] / (df['floor_area_sqm'] / 1000)
    df['charger_per_robot']        = df['charger_count'] / df['robot_total']
    df['pack_per_robot']           = df['pack_station_count'] / df['robot_total']
    df['intersection_per_1000sqm'] = df['intersection_count'] / (df['floor_area_sqm'] / 1000)
    df['aisle_x_compactness']      = df['aisle_width_avg'] * df['layout_compactness']
    df['sqm_per_exit']             = df['floor_area_sqm'] / df['emergency_exit_count']
    return df


def add_interaction_features(df):
    """전략 C: layout_type × 핵심 운영 피처 교호작용"""
    df = df.copy()
    lt_cols = [c for c in df.columns if c.startswith('lt_')]
    key_ops = ['congestion_score', 'robot_utilization', 'pack_utilization',
               'order_inflow_15m', 'low_battery_ratio']

    for lt_col in lt_cols:
        for op in key_ops:
            if op in df.columns:
                df[f'{lt_col}_x_{op}'] = df[lt_col] * df[op]

    # 물리 × 운영 교호작용 (3종 추가)
    if 'congestion_score' in df.columns:
        df['aisle_x_congestion'] = df['aisle_width_avg'] * df['congestion_score']
    if 'max_zone_density' in df.columns:
        df['compact_x_zone_density'] = df['layout_compactness'] * df['max_zone_density']
    if 'avg_charge_wait' in df.columns and 'charger_per_robot' in df.columns:
        df['charger_ratio_x_charge_wait'] = df['charger_per_robot'] * df['avg_charge_wait']

    return df


def encode_ordinal(train, test, target=TARGET):
    """기존 방식: layout_type 순서형 인코딩"""
    tr, te = train.copy(), test.copy()
    exclude = {'ID', 'layout_id', 'scenario_id', target}
    cat_cols = [c for c in tr.select_dtypes(include='object').columns if c not in exclude]
    for col in cat_cols:
        combined = pd.concat([tr[col], te[col]], axis=0)
        mapping  = {v: i for i, v in enumerate(combined.dropna().unique())}
        tr[col] = tr[col].map(mapping)
        te[col] = te[col].map(mapping)
    return tr, te


def encode_onehot(train, test):
    """전략 A: layout_type one-hot 인코딩 (lt_ 접두사)"""
    tr = train.copy()
    te = test.copy()
    # layout_type → get_dummies (train+test 통합해 카테고리 통일)
    combined = pd.concat([tr, te], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=['layout_type'], prefix='lt')
    tr = combined.iloc[:len(tr)].copy()
    te = combined.iloc[len(tr):].reset_index(drop=True)
    return tr, te


# ─── 메인: 공통 전처리 1회 실행 ─────────────────────────────

print("\n" + "="*60)
print(" 공통 전처리 (merge / ts / lag / rolling / domain)")
print("="*60)
t_prep = time.time()

train_raw = pd.read_csv(DATA_PATH + 'train.csv')
test_raw  = pd.read_csv(DATA_PATH + 'test.csv')
layout    = pd.read_csv(DATA_PATH + 'layout_info.csv')
print(f"Train: {train_raw.shape}, Test: {test_raw.shape}")

# Step 1: merge layout (layout_type은 아직 object)
base_tr, base_te = merge_layout(train_raw.copy(), test_raw.copy(), layout)

# Step 2: ts 피처
base_tr = add_ts_features(base_tr)
base_te = add_ts_features(base_te)

# Step 3: lag + rolling (가장 오래 걸림 → 1회만)
base_tr, base_te = add_lag_features(base_tr, base_te,
                                     key_cols=KEY_COLS_EXT, lags=[1,2,3,4,5,6])
base_tr, base_te = add_rolling_features(base_tr, base_te,
                                         key_cols=KEY_COLS_EXT, windows=[3,5,10])

# Step 4: domain 피처
base_tr = add_domain_features(base_tr)
base_te = add_domain_features(base_te)

print(f"공통 전처리 완료 ({time.time()-t_prep:.0f}s) | cols={base_tr.shape[1]}")


# ─── 결과 수집 ───────────────────────────────────────────────
results = []
t_total = time.time()


# ────────────────────────────────────────────────────────────
# Exp 0 — Baseline (ordinal encoding, 현재 파이프라인)
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" Exp 0: Baseline (ordinal layout_type)")
print("="*60)
tr0, te0 = encode_ordinal(base_tr, base_te)
feat0    = get_feat_cols(tr0)
print(f"  피처 수: {len(feat0)}")
mae0, _ = run_cv_basic(tr0, feat0, label='Baseline')
print(f"  ✅ OOF MAE: {mae0:.4f}")
results.append({'실험': 'Baseline (ordinal)', '피처수': len(feat0), 'OOF_MAE': mae0, '개선': 0.0})


# ────────────────────────────────────────────────────────────
# Exp 1 — 전략 A: one-hot layout_type
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" Exp 1: 전략 A — one-hot layout_type")
print("="*60)
tr1, te1 = encode_onehot(base_tr, base_te)
feat1    = get_feat_cols(tr1)
print(f"  피처 수: {len(feat1)} (+{len(feat1)-len(feat0)})")
mae1, _ = run_cv_basic(tr1, feat1, label='A')
delta1   = mae0 - mae1
print(f"  ✅ OOF MAE: {mae1:.4f}  ({'↑+' if delta1<0 else '↓-'}{abs(delta1):.4f} vs Baseline)")
results.append({'실험': 'A: one-hot', '피처수': len(feat1), 'OOF_MAE': mae1, '개선': delta1})


# ────────────────────────────────────────────────────────────
# Exp 2 — 전략 A+B: + 파생 비율 피처
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" Exp 2: 전략 A+B — + 파생 비율 피처 6종")
print("="*60)
tr2 = add_layout_ratio_features(tr1)
te2 = add_layout_ratio_features(te1)
feat2 = get_feat_cols(tr2)
print(f"  피처 수: {len(feat2)} (+{len(feat2)-len(feat1)})")
mae2, _ = run_cv_basic(tr2, feat2, label='A+B')
delta2   = mae0 - mae2
print(f"  ✅ OOF MAE: {mae2:.4f}  ({'↑+' if delta2<0 else '↓-'}{abs(delta2):.4f} vs Baseline)")
results.append({'실험': 'A+B: +비율피처', '피처수': len(feat2), 'OOF_MAE': mae2, '개선': delta2})


# ────────────────────────────────────────────────────────────
# Exp 3 — 전략 A+B+D: + Target Encoding (OOF)
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" Exp 3: 전략 A+B+D — + layout_id Target Encoding (OOF)")
print("="*60)
feat2_base = get_feat_cols(tr2)  # te_cols는 CV 내부에서 생성
print(f"  피처 수: {len(feat2_base)+2} (+2 target enc)")
mae3, _ = run_cv_with_target_enc(tr2, feat2_base, label='A+B+D')
delta3   = mae0 - mae3
print(f"  ✅ OOF MAE: {mae3:.4f}  ({'↑+' if delta3<0 else '↓-'}{abs(delta3):.4f} vs Baseline)")
results.append({'실험': 'A+B+D: +TargetEnc', '피처수': len(feat2_base)+2, 'OOF_MAE': mae3, '개선': delta3})


# ────────────────────────────────────────────────────────────
# Exp 4 — 전략 A+B+D+C: + 교호작용 피처
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" Exp 4: 전략 A+B+D+C — + 교호작용 피처")
print("="*60)
tr4 = add_interaction_features(tr2)
te4 = add_interaction_features(te2)
feat4_base = get_feat_cols(tr4)
print(f"  피처 수: {len(feat4_base)+2} (+{len(feat4_base)-len(feat2_base)} interaction, +2 target enc)")
mae4, _ = run_cv_with_target_enc(tr4, feat4_base, label='A+B+D+C')
delta4   = mae0 - mae4
print(f"  ✅ OOF MAE: {mae4:.4f}  ({'↑+' if delta4<0 else '↓-'}{abs(delta4):.4f} vs Baseline)")
results.append({'실험': 'A+B+D+C: +교호작용', '피처수': len(feat4_base)+2, 'OOF_MAE': mae4, '개선': delta4})


# ─── 최종 결과 출력 ─────────────────────────────────────────

print("\n" + "="*60)
print(" 최종 결과 요약")
print("="*60)
print(f"\n{'실험':<25} {'피처수':>6}  {'OOF MAE':>9}  {'vs Baseline':>12}")
print("-" * 58)
for r in results:
    sign = '↓' if r['개선'] >= 0 else '↑'
    print(f"  {r['실험']:<23} {r['피처수']:>6}  {r['OOF_MAE']:>9.4f}  "
          f"  {sign}{abs(r['개선']):>8.4f}")
print("-" * 58)

best = min(results, key=lambda x: x['OOF_MAE'])
print(f"\n  🏆 최고 실험: {best['실험']} (MAE={best['OOF_MAE']:.4f})")
print(f"  ⏱  총 소요 시간: {(time.time()-t_total)/60:.1f}분")
print("="*60)

# CSV 저장
import datetime
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
out_path = os.path.join(os.path.dirname(__file__), '..', 'docs', f'ablation_results_{ts}.csv')
pd.DataFrame(results).to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n결과 저장: {out_path}")
