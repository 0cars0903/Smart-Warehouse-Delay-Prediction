"""
run_exp_fe_v2.py  —  FE v2 실험
===============================================================
핵심 변경:
  1. KEY_COLS 개선
     - 제거: avg_trip_distance (r=0.021, 사실상 무신호)
     - 추가: robot_charging  (r=0.320, charge_queue_length보다 강함)
             battery_std     (r=0.308)
             sku_concentration (r=0.292)
             urgent_order_ratio (r=0.271)
     → 8→11 KEY_COLS (lag×6 + rolling×3×2 적용 대상 확대)

  2. Delta 피처 추가
     - 각 KEY_COL에 대해 col_diff1 = col - col_lag1
     - 변화율(악화/개선 방향) 포착

  3. Layout 용량 비율 피처 추가
     - robot_active_ratio     = robot_active / robot_total
     - charging_saturation    = robot_charging / charger_count
     - charger_per_robot      = charger_count / robot_total
     - orders_per_robot_total = order_inflow_15m / robot_total

  4. 모델: RF 5모델 스태킹 (현재 Public 최고: 10.2213)
     - 피처 변경으로 체크포인트 재사용 불가 → 전체 재학습
     - 예상 시간: ~80~90분 (LGBM+CB+TW+ET+RF 5-fold)

비교 기준:
  현재 최고 — CV 8.7911 / Public 10.2213 (RF 5모델, 212피처)
  목표      — CV < 8.78 / Public < 10.20
===============================================================
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

import sys
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import (
    merge_layout, encode_categoricals, add_ts_features,
    add_lag_features, add_rolling_features, add_domain_features,
)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_PATH   = 'data/'
CKPT_DIR    = 'docs/fe_v2_ckpt'
SUBMIT_PATH = 'submissions/stacking_fe_v2_rf_lgbm_meta.csv'
TARGET      = 'avg_delay_minutes_next_30m'
N_FOLDS     = 5
RANDOM_SEED = 42

os.makedirs(CKPT_DIR, exist_ok=True)

# ── 확장 KEY_COLS (avg_trip_distance 제거, 4종 추가) ──
KEY_COLS_V2 = [
    'low_battery_ratio',      # r=0.366  ← 유지
    'battery_mean',           # r=0.359  ← 유지
    'battery_std',            # r=0.308  ← NEW
    'robot_idle',             # r=0.349  ← 유지
    'robot_charging',         # r=0.320  ← NEW (charge_queue보다 강함)
    'order_inflow_15m',       # r=0.342  ← 유지
    'congestion_score',       # r=0.300  ← 유지
    'max_zone_density',       # r=0.311  ← 유지
    'charge_queue_length',    # r=0.261  ← 유지
    'sku_concentration',      # r=0.292  ← NEW
    'urgent_order_ratio',     # r=0.271  ← NEW
    # avg_trip_distance 제거 (r=0.021)
]

# ──────────────────────────────────────────────
# 모델 파라미터
# ──────────────────────────────────────────────
BEST_LGBM_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_SEED,
    'verbose': -1, 'n_jobs': -1,
}
CB_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3,
    'loss_function': 'MAE', 'eval_metric': 'MAE',
    'random_seed': RANDOM_SEED, 'verbose': 0,
    'early_stopping_rounds': 100,
}
TW_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'tweedie', 'tweedie_variance_power': 1.8,
    'metric': 'mae', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_SEED,
    'verbose': -1, 'n_jobs': -1,
}
ET_PARAMS = {
    'n_estimators': 500, 'n_jobs': -1,
    'random_state': RANDOM_SEED, 'min_samples_leaf': 26,
}
RF_PARAMS = {
    'n_estimators': 500, 'max_features': 0.33,
    'min_samples_leaf': 26, 'n_jobs': -1,
    'random_state': RANDOM_SEED,
}
META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'n_estimators': 1000, 'objective': 'regression_l1',
    'random_state': RANDOM_SEED, 'verbose': -1, 'n_jobs': -1,
}


# ──────────────────────────────────────────────
# FE v2 파이프라인
# ──────────────────────────────────────────────
def build_features_v2(train, test, layout, verbose=True):
    """FE v2: 확장 KEY_COLS + Delta 피처 + Layout 용량 비율"""
    if verbose:
        print(f'[build_features_v2] 시작: train={train.shape}, test={test.shape}')

    # 1. Layout merge
    train, test = merge_layout(train, test, layout)
    if verbose:
        print(f'  1. layout merge → {train.shape[1]} cols')

    # 2. 범주형 인코딩
    train, test = encode_categoricals(train, test, TARGET)

    # 3. TS 피처
    train = add_ts_features(train)
    test  = add_ts_features(test)
    if verbose:
        print(f'  3. ts 피처 → {train.shape[1]} cols')

    # 4. Layout 용량 비율 피처 (static)
    for df in [train, test]:
        df['robot_active_ratio']   = df['robot_active']   / (df['robot_total'] + 1)
        df['robot_idle_ratio']     = df['robot_idle']     / (df['robot_total'] + 1)
        df['charging_saturation']  = df['robot_charging'] / (df['charger_count'] + 1)
        df['charger_per_robot']    = df['charger_count']  / (df['robot_total'] + 1)
        df['orders_per_robot_total'] = df['order_inflow_15m'] / (df['robot_total'] + 1)
    if verbose:
        print(f'  4. layout 용량 비율 피처 (5종) → {train.shape[1]} cols')

    # 5. Lag 피처 (확장 KEY_COLS, lags=[1..6])
    train, test = add_lag_features(train, test, key_cols=KEY_COLS_V2, lags=[1,2,3,4,5,6])
    if verbose:
        lag_cols = [c for c in train.columns if '_lag' in c]
        print(f'  5. Lag 피처 ({len(lag_cols)}종) → {train.shape[1]} cols')

    # 6. Rolling 피처 (확장 KEY_COLS, windows=[3,5,10])
    train, test = add_rolling_features(train, test, key_cols=KEY_COLS_V2, windows=[3,5,10])
    if verbose:
        roll_cols = [c for c in train.columns if '_roll' in c]
        print(f'  6. Rolling 피처 ({len(roll_cols)}종) → {train.shape[1]} cols')

    # 7. Delta 피처 (변화율: col - col_lag1)
    for df in [train, test]:
        for col in KEY_COLS_V2:
            lag1_col = f'{col}_lag1'
            if lag1_col in df.columns and col in df.columns:
                df[f'{col}_diff1'] = df[col] - df[lag1_col]
    if verbose:
        diff_cols = [c for c in train.columns if '_diff1' in c]
        print(f'  7. Delta 피처 ({len(diff_cols)}종) → {train.shape[1]} cols')

    # 8. Domain 피처
    train = add_domain_features(train)
    test  = add_domain_features(test)
    if verbose:
        print(f'  8. Domain 피처 → {train.shape[1]} cols')

    # 최종 피처 수
    excl = ['ID', 'layout_id', 'scenario_id', TARGET]
    feat_cols = [c for c in train.columns if c not in excl
                 and train[c].dtype.name not in ['object', 'category']]
    if verbose:
        print(f'[build_features_v2] 완료: 최종 피처 수 = {len(feat_cols)}')

    return train, test


def get_feature_cols(df):
    excl = ['ID', 'layout_id', 'scenario_id', TARGET]
    return [c for c in df.columns if c not in excl
            and df[c].dtype.name not in ['object', 'category']]


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    print('비교 기준: v3 4모델 CV 8.7929 / RF 5모델 CV 8.7911 / Public 10.2213')
    print('='*60)

    # 데이터 로드
    train  = pd.read_csv(f'{DATA_PATH}train.csv')
    test   = pd.read_csv(f'{DATA_PATH}test.csv')
    layout = pd.read_csv(f'{DATA_PATH}layout_info.csv')
    sub    = pd.read_csv(f'{DATA_PATH}sample_submission.csv')

    # FE v2
    train_fe, test_fe = build_features_v2(train, test, layout)

    feat_cols    = get_feature_cols(train_fe)
    X            = train_fe[feat_cols].values
    y            = train_fe[TARGET].values
    X_test       = test_fe[feat_cols].values
    groups       = train_fe['scenario_id'].values
    gkf          = GroupKFold(n_splits=N_FOLDS)

    print(f'\n피처 수: {len(feat_cols)} (기존 212 → 신규 ?)')
    print()

    # ── Layer 1: 5 Base Models ──
    oof_lg = np.zeros(len(X));  test_lg = np.zeros(len(X_test))
    oof_cb = np.zeros(len(X));  test_cb = np.zeros(len(X_test))
    oof_tw = np.zeros(len(X));  test_tw = np.zeros(len(X_test))
    oof_et = np.zeros(len(X));  test_et = np.zeros(len(X_test))
    oof_rf = np.zeros(len(X));  test_rf = np.zeros(len(X_test))

    print('[Layer 1] Base Model 학습 (전체 재학습, 피처 변경으로 체크포인트 불가)')
    print()

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # --- LGBM ---
        ckpt = f'{CKPT_DIR}/lgbm_fold{fold}.npy'
        ckpt_test = f'{CKPT_DIR}/lgbm_fold{fold}_test.npy'
        if os.path.exists(ckpt):
            oof_lg[va_idx] = np.load(ckpt)
            test_lg += np.load(ckpt_test) / N_FOLDS
        else:
            m = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                              lgb.log_evaluation(-1)])
            p_va = m.predict(X_va); p_te = m.predict(X_test)
            oof_lg[va_idx] = p_va; test_lg += p_te / N_FOLDS
            np.save(ckpt, p_va); np.save(ckpt_test, p_te)
            mae = mean_absolute_error(y_va, p_va)
            print(f'  [LGBM] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')

    lgbm_oof_mae = mean_absolute_error(y, oof_lg)
    print(f'  LGBM OOF MAE: {lgbm_oof_mae:.4f}')
    print()

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # --- CatBoost ---
        ckpt = f'{CKPT_DIR}/cb_fold{fold}.npy'
        ckpt_test = f'{CKPT_DIR}/cb_fold{fold}_test.npy'
        if os.path.exists(ckpt):
            oof_cb[va_idx] = np.load(ckpt)
            test_cb += np.load(ckpt_test) / N_FOLDS
        else:
            m = cb.CatBoostRegressor(**CB_PARAMS)
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
            p_va = m.predict(X_va); p_te = m.predict(X_test)
            oof_cb[va_idx] = p_va; test_cb += p_te / N_FOLDS
            np.save(ckpt, p_va); np.save(ckpt_test, p_te)
            mae = mean_absolute_error(y_va, p_va)
            print(f'  [CB]   Fold {fold+1}  MAE={mae:.4f}')

    cb_oof_mae = mean_absolute_error(y, oof_cb)
    print(f'  CB OOF MAE: {cb_oof_mae:.4f}')
    print()

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # --- Tweedie(1.8) ---
        ckpt = f'{CKPT_DIR}/tw_fold{fold}.npy'
        ckpt_test = f'{CKPT_DIR}/tw_fold{fold}_test.npy'
        if os.path.exists(ckpt):
            oof_tw[va_idx] = np.load(ckpt)
            test_tw += np.load(ckpt_test) / N_FOLDS
        else:
            m = lgb.LGBMRegressor(**TW_PARAMS)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                              lgb.log_evaluation(-1)])
            p_va = m.predict(X_va); p_te = m.predict(X_test)
            p_va = np.maximum(p_va, 0); p_te = np.maximum(p_te, 0)
            oof_tw[va_idx] = p_va; test_tw += p_te / N_FOLDS
            np.save(ckpt, p_va); np.save(ckpt_test, p_te)
            mae = mean_absolute_error(y_va, p_va)
            print(f'  [TW]   Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')

    tw_oof_mae = mean_absolute_error(y, oof_tw)
    print(f'  TW OOF MAE: {tw_oof_mae:.4f}')
    print()

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # --- ExtraTrees ---
        ckpt = f'{CKPT_DIR}/et_fold{fold}.npy'
        ckpt_test = f'{CKPT_DIR}/et_fold{fold}_test.npy'
        if os.path.exists(ckpt):
            oof_et[va_idx] = np.load(ckpt)
            test_et += np.load(ckpt_test) / N_FOLDS
        else:
            X_tr_et = np.nan_to_num(X_tr, nan=0.0)
            X_va_et = np.nan_to_num(X_va, nan=0.0)
            X_te_et = np.nan_to_num(X_test, nan=0.0)
            m = ExtraTreesRegressor(**ET_PARAMS)
            m.fit(X_tr_et, y_tr)
            p_va = m.predict(X_va_et); p_te = m.predict(X_te_et)
            oof_et[va_idx] = p_va; test_et += p_te / N_FOLDS
            np.save(ckpt, p_va); np.save(ckpt_test, p_te)
            mae = mean_absolute_error(y_va, p_va)
            print(f'  [ET]   Fold {fold+1}  MAE={mae:.4f}')

    et_oof_mae = mean_absolute_error(y, oof_et)
    print(f'  ET OOF MAE: {et_oof_mae:.4f}')
    print()

    print('  [RF] Fold 학습 시작 (시간 소요 ~40분)')
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # --- RandomForest ---
        ckpt = f'{CKPT_DIR}/rf_fold{fold}.npy'
        ckpt_test = f'{CKPT_DIR}/rf_fold{fold}_test.npy'
        if os.path.exists(ckpt):
            oof_rf[va_idx] = np.load(ckpt)
            test_rf += np.load(ckpt_test) / N_FOLDS
        else:
            X_tr_rf = np.nan_to_num(X_tr, nan=0.0)
            X_va_rf = np.nan_to_num(X_va, nan=0.0)
            X_te_rf = np.nan_to_num(X_test, nan=0.0)
            m = RandomForestRegressor(**RF_PARAMS)
            m.fit(X_tr_rf, y_tr)
            p_va = m.predict(X_va_rf); p_te = m.predict(X_te_rf)
            oof_rf[va_idx] = p_va; test_rf += p_te / N_FOLDS
            np.save(ckpt, p_va); np.save(ckpt_test, p_te)
            mae = mean_absolute_error(y_va, p_va)
            print(f'  [RF]   Fold {fold+1}  MAE={mae:.4f}')

    rf_oof_mae = mean_absolute_error(y, oof_rf)
    print(f'  RF OOF MAE: {rf_oof_mae:.4f}')
    print()

    # 상관관계 출력
    from scipy.stats import pearsonr
    pairs = [('LGBM','CB',oof_lg,oof_cb), ('LGBM','TW',oof_lg,oof_tw),
             ('LGBM','ET',oof_lg,oof_et), ('LGBM','RF',oof_lg,oof_rf),
             ('CB','TW',oof_cb,oof_tw),   ('CB','ET',oof_cb,oof_et),
             ('CB','RF',oof_cb,oof_rf),   ('TW','ET',oof_tw,oof_et),
             ('TW','RF',oof_tw,oof_rf),   ('ET','RF',oof_et,oof_rf)]
    print('  OOF 상관관계:')
    for a, b, pa, pb in pairs:
        r, _ = pearsonr(pa, pb)
        print(f'    {a}-{b}: {r:.4f}')
    print()

    # 가중치 앙상블
    from scipy.optimize import minimize
    def neg_mae(w):
        w = np.array(w); w = np.maximum(w, 0); w /= w.sum()
        pred = (w[0]*oof_lg + w[1]*oof_cb + w[2]*np.log1p(np.maximum(oof_tw,0))
                + w[3]*oof_et + w[4]*oof_rf)
        return mean_absolute_error(y, pred)

    # log1p space로 통일하기 위해 simple weight ensemble
    def neg_mae_simple(w):
        w = np.array(w); w = np.maximum(w, 0); w /= w.sum()
        pred = w[0]*oof_lg + w[1]*oof_cb + w[2]*oof_tw + w[3]*oof_et + w[4]*oof_rf
        return mean_absolute_error(y, pred)

    res = minimize(neg_mae_simple, [0.4,0.2,0.2,0.1,0.1],
                   method='Nelder-Mead', options={'maxiter':1000})
    w_opt = np.maximum(res.x, 0); w_opt /= w_opt.sum()
    ensemble_mae = neg_mae_simple(w_opt)
    print(f'  5모델 가중 앙상블 CV MAE: {ensemble_mae:.4f}')
    print(f'    LGBM={w_opt[0]:.3f}, CB={w_opt[1]:.3f}, TW={w_opt[2]:.3f}, '
          f'ET={w_opt[3]:.3f}, RF={w_opt[4]:.3f}')
    print()

    # ── Layer 2: LGBM Meta ──
    print('[Layer 2] LGBM 메타 학습기')
    oof_lg_log = np.log1p(np.maximum(oof_lg, 0))
    oof_cb_log = np.log1p(np.maximum(oof_cb, 0))
    oof_tw_log = np.log1p(np.maximum(oof_tw, 0))
    oof_et_log = np.log1p(np.maximum(oof_et, 0))
    oof_rf_log = np.log1p(np.maximum(oof_rf, 0))

    test_lg_log = np.log1p(np.maximum(test_lg, 0))
    test_cb_log = np.log1p(np.maximum(test_cb, 0))
    test_tw_log = np.log1p(np.maximum(np.clip(test_tw, 0, None), 0))
    test_et_log = np.log1p(np.maximum(test_et, 0))
    test_rf_log = np.log1p(np.maximum(test_rf, 0))

    meta_X = np.column_stack([oof_lg_log, oof_cb_log, oof_tw_log, oof_et_log, oof_rf_log])
    meta_X_test = np.column_stack([test_lg_log, test_cb_log, test_tw_log, test_et_log, test_rf_log])

    meta_oof  = np.zeros(len(X))
    meta_test = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr_m, X_va_m = meta_X[tr_idx], meta_X[va_idx]
        y_tr, y_va     = y[tr_idx], y[va_idx]

        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(X_tr_m, y_tr, eval_set=[(X_va_m, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                          lgb.log_evaluation(-1)])
        p_va = m.predict(X_va_m)
        meta_oof[va_idx] = p_va
        meta_test += m.predict(meta_X_test) / N_FOLDS

        mae = mean_absolute_error(y_va, p_va)
        print(f'  [LGBM-meta] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')

    meta_mae = mean_absolute_error(y, meta_oof)
    meta_std = np.std(meta_oof)
    print(f'  [LGBM-meta] OOF MAE={meta_mae:.4f} | std={meta_std:.2f}')

    # 제출 저장
    sub[TARGET] = np.maximum(meta_test, 0)
    sub.to_csv(SUBMIT_PATH, index=False)
    print(f'\n→ 저장: {SUBMIT_PATH}')

    print()
    print('='*60)
    print('FE v2 결과 요약')
    print('='*60)
    print(f'  [비교] RF 5모델 (212피처): CV 8.7911 / Public 10.2213')
    print(f'  [결과] FE v2 RF 5모델   : CV {meta_mae:.4f}')
    print(f'  LGBM OOF: {lgbm_oof_mae:.4f}  CB OOF: {cb_oof_mae:.4f}')
    print(f'  TW OOF  : {tw_oof_mae:.4f}  ET OOF: {et_oof_mae:.4f}')
    print(f'  RF OOF  : {rf_oof_mae:.4f}')
    print(f'  5모델 가중 앙상블: {ensemble_mae:.4f}')
    delta = meta_mae - 8.7911
    sign = '+' if delta > 0 else ''
    print(f'  vs 기준: {sign}{delta:.4f}  {"✅ 개선" if delta < 0 else "⚠️ 악화"}')


if __name__ == '__main__':
    main()
