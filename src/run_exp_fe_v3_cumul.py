"""
run_exp_fe_v3_cumul.py  —  방향 B: Cumulative 피처 추가
===============================================================
아이디어: lag/rolling은 "과거 k 슬롯" 만 본다.
하지만 "이 시나리오에서 지금까지 얼마나 나빴나?" 를 모른다.
Cumulative 피처는 시나리오 내 누적 최댓값/최솟값으로 이를 포착한다.

예:
  battery_mean_cummin  → 배터리가 한 번이라도 얼마나 낮았나 (누적 최저)
  congestion_cummax    → 혼잡도가 최대 얼마까지 올라갔나 (누적 최고)
  fault_count_cumsum   → 이 시나리오에서 총 고장 횟수

FE 구조:
  FE v2 264피처 + Cumulative 피처 (KEY_COLS_V2 대상, 3종×11 = 33피처 추가)
  → 최종 ~297 피처

Cumulative 피처 종류 (per KEY_COL):
  - {col}_cummin: 현재까지의 시나리오 내 최솟값 (expanding min)
  - {col}_cummax: 현재까지의 시나리오 내 최댓값 (expanding max)
  - {col}_cummean: 현재까지의 시나리오 내 평균 (expanding mean)
  ※ 모두 shift(1) 적용 → 현재 슬롯 리크 방지

체크포인트: docs/fe_v3_cumul_ckpt/ (전체 재학습)
예상 시간: ~90분

비교 기준:
  FE v2: CV 8.7842 / RF 5모델: CV 8.7911
===============================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import (
    merge_layout, encode_categoricals, add_ts_features,
    add_lag_features, add_rolling_features, add_domain_features,
)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_PATH   = 'data/'
CKPT_DIR    = 'docs/fe_v3_cumul_ckpt'
SUBMIT_PATH = 'submissions/stacking_fe_v3_cumul_rf_lgbm_meta.csv'
TARGET      = 'avg_delay_minutes_next_30m'
N_FOLDS     = 5
RANDOM_SEED = 42

os.makedirs(CKPT_DIR, exist_ok=True)

KEY_COLS_V2 = [
    'low_battery_ratio', 'battery_mean', 'battery_std',
    'robot_idle', 'robot_charging', 'order_inflow_15m',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'sku_concentration', 'urgent_order_ratio',
]

# Cumulative 피처를 적용할 컬럼 (신호 강한 상위 + 이벤트 컬럼)
CUMUL_COLS = [
    'battery_mean',        # 누적 최솟값 → 한 번이라도 배터리 위기?
    'low_battery_ratio',   # 누적 최댓값 → 가장 심했던 배터리 부족
    'congestion_score',    # 누적 최댓값 → 최악의 혼잡도
    'robot_idle',          # 누적 최댓값 → 로봇이 최대 몇 대까지 놀았나?
    'order_inflow_15m',    # 누적 최댓값 → 최대 주문 폭주 시점
    'fault_count_15m',     # 누적 합계 → 총 고장 횟수
    'blocked_path_15m',    # 누적 합계 → 총 경로 차단 횟수
    'avg_charge_wait',     # 누적 최댓값 → 최악의 충전 대기
]

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
ET_PARAMS = {'n_estimators': 500, 'n_jobs': -1, 'random_state': RANDOM_SEED, 'min_samples_leaf': 26}
RF_PARAMS = {'n_estimators': 500, 'max_features': 0.33, 'min_samples_leaf': 26,
             'n_jobs': -1, 'random_state': RANDOM_SEED}
META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'n_estimators': 1000, 'objective': 'regression_l1',
    'random_state': RANDOM_SEED, 'verbose': -1, 'n_jobs': -1,
}


# ──────────────────────────────────────────────
# Cumulative 피처 추가
# ──────────────────────────────────────────────
def add_cumulative_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    시나리오 내 누적 통계 피처 (shift(1) 적용 → 현재 슬롯 리크 방지)

    - {col}_cummin:  누적 최솟값 → 지금까지 가장 나빴던 상태
    - {col}_cummax:  누적 최댓값 → 지금까지 가장 높았던 부하
    - {col}_cumsum:  누적 합계 (fault_count, blocked_path 등 이벤트 카운터용)
    """
    train_c = train.copy(); train_c['_split'] = 0
    test_c  = test.copy();  test_c['_split']  = 1
    test_c['_orig_order'] = np.arange(len(test_c))

    combined = (pd.concat([train_c, test_c], axis=0, ignore_index=True)
                  .sort_values(['scenario_id', 'ts_idx'])
                  .reset_index(drop=True))

    for col in CUMUL_COLS:
        if col not in combined.columns:
            continue
        # shift(1) 후 expanding → 현재 슬롯 미포함
        shifted = combined.groupby('scenario_id')[col].shift(1)

        combined[f'{col}_cummin'] = (
            shifted.groupby(combined['scenario_id'])
                   .transform(lambda x: x.expanding().min()))

        combined[f'{col}_cummax'] = (
            shifted.groupby(combined['scenario_id'])
                   .transform(lambda x: x.expanding().max()))

        # 이벤트 카운터 컬럼에만 cumsum 추가 (정수형 또는 카운트 성격)
        if col in ['fault_count_15m', 'blocked_path_15m']:
            combined[f'{col}_cumsum'] = (
                shifted.groupby(combined['scenario_id'])
                       .transform(lambda x: x.expanding().sum()))

    tr_out = combined[combined['_split'] == 0].drop(columns=['_split', '_orig_order'], errors='ignore')
    te_out = combined[combined['_split'] == 1].sort_values('_orig_order').drop(
        columns=['_split', '_orig_order'])
    return tr_out, te_out


# ──────────────────────────────────────────────
# FE v3 파이프라인 (v2 + Cumulative)
# ──────────────────────────────────────────────
def build_features_v3(train, test, layout, verbose=True):
    if verbose:
        print(f'[build_features_v3] 시작: train={train.shape}, test={test.shape}')

    train, test = merge_layout(train, test, layout)
    train, test = encode_categoricals(train, test, TARGET)
    train = add_ts_features(train); test = add_ts_features(test)

    for df in [train, test]:
        df['robot_active_ratio']     = df['robot_active']   / (df['robot_total'] + 1)
        df['charging_saturation']    = df['robot_charging'] / (df['charger_count'] + 1)
        df['charger_per_robot']      = df['charger_count']  / (df['robot_total'] + 1)
        df['orders_per_robot_total'] = df['order_inflow_15m'] / (df['robot_total'] + 1)

    train, test = add_lag_features(train, test, key_cols=KEY_COLS_V2, lags=[1,2,3,4,5,6])
    train, test = add_rolling_features(train, test, key_cols=KEY_COLS_V2, windows=[3,5,10])

    for df in [train, test]:
        for col in KEY_COLS_V2:
            if f'{col}_lag1' in df.columns:
                df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']

    # Cumulative 피처 추가 (v3 핵심)
    train, test = add_cumulative_features(train, test)
    if verbose:
        cumul_cols = [c for c in train.columns if '_cummin' in c or '_cummax' in c or '_cumsum' in c]
        print(f'  Cumulative 피처 추가: {len(cumul_cols)}종')

    train = add_domain_features(train); test = add_domain_features(test)

    excl = ['ID', 'layout_id', 'scenario_id', TARGET]
    feat_cols = [c for c in train.columns if c not in excl
                 and train[c].dtype.name not in ['object', 'category']]
    if verbose:
        print(f'[build_features_v3] 완료: 최종 피처 수 = {len(feat_cols)}')

    return train, test


def get_feature_cols(df):
    excl = ['ID', 'layout_id', 'scenario_id', TARGET]
    return [c for c in df.columns if c not in excl
            and df[c].dtype.name not in ['object', 'category']]


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    print('비교 기준: FE v2 CV 8.7842 / RF 5모델 CV 8.7911')
    print('='*60)

    train  = pd.read_csv(f'{DATA_PATH}train.csv')
    test   = pd.read_csv(f'{DATA_PATH}test.csv')
    layout = pd.read_csv(f'{DATA_PATH}layout_info.csv')
    sub    = pd.read_csv(f'{DATA_PATH}sample_submission.csv')

    train_fe, test_fe = build_features_v3(train, test, layout, verbose=True)
    feat_cols = get_feature_cols(train_fe)
    X         = train_fe[feat_cols].values
    y         = train_fe[TARGET].values
    X_test    = test_fe[feat_cols].values
    groups    = train_fe['scenario_id'].values
    gkf       = GroupKFold(n_splits=N_FOLDS)

    print(f'\n피처 수: {len(feat_cols)} (FE v2: 264)')
    print()

    # ── Layer 1: 5 Base Models ──
    oof_lg = np.zeros(len(X));  test_lg = np.zeros(len(X_test))
    oof_cb = np.zeros(len(X));  test_cb = np.zeros(len(X_test))
    oof_tw = np.zeros(len(X));  test_tw = np.zeros(len(X_test))
    oof_et = np.zeros(len(X));  test_et = np.zeros(len(X_test))
    oof_rf = np.zeros(len(X));  test_rf = np.zeros(len(X_test))

    print('[Layer 1] Base Model 학습')

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        for tag, oof_arr, test_acc, params_or_cls in [
            ('lgbm', oof_lg, None, 'lgbm'),
            ('cb',   oof_cb, None, 'cb'),
            ('tw',   oof_tw, None, 'tw'),
            ('et',   oof_et, None, 'et'),
            ('rf',   oof_rf, None, 'rf'),
        ]:
            ckpt      = f'{CKPT_DIR}/{tag}_fold{fold}.npy'
            ckpt_test = f'{CKPT_DIR}/{tag}_fold{fold}_test.npy'

            if os.path.exists(ckpt):
                oof_arr[va_idx] = np.load(ckpt)
                if   tag == 'lgbm': test_lg += np.load(ckpt_test) / N_FOLDS
                elif tag == 'cb':   test_cb += np.load(ckpt_test) / N_FOLDS
                elif tag == 'tw':   test_tw += np.load(ckpt_test) / N_FOLDS
                elif tag == 'et':   test_et += np.load(ckpt_test) / N_FOLDS
                elif tag == 'rf':   test_rf += np.load(ckpt_test) / N_FOLDS
                continue

            # 학습
            if tag == 'lgbm':
                m = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
                p_va = m.predict(X_va); p_te = m.predict(X_test)
                extra = f'  iter={m.best_iteration_}'
            elif tag == 'cb':
                m = cb.CatBoostRegressor(**CB_PARAMS)
                m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
                p_va = m.predict(X_va); p_te = m.predict(X_test)
                extra = ''
            elif tag == 'tw':
                m = lgb.LGBMRegressor(**TW_PARAMS)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
                p_va = np.maximum(m.predict(X_va), 0)
                p_te = np.maximum(m.predict(X_test), 0)
                extra = f'  iter={m.best_iteration_}'
            elif tag == 'et':
                X_tr_c = np.nan_to_num(X_tr, nan=0.0)
                X_va_c = np.nan_to_num(X_va, nan=0.0)
                X_te_c = np.nan_to_num(X_test, nan=0.0)
                m = ExtraTreesRegressor(**ET_PARAMS)
                m.fit(X_tr_c, y_tr)
                p_va = m.predict(np.nan_to_num(X_va, nan=0.0))
                p_te = m.predict(X_te_c)
                extra = ''
            elif tag == 'rf':
                X_tr_c = np.nan_to_num(X_tr, nan=0.0)
                X_te_c = np.nan_to_num(X_test, nan=0.0)
                m = RandomForestRegressor(**RF_PARAMS)
                m.fit(X_tr_c, y_tr)
                p_va = m.predict(np.nan_to_num(X_va, nan=0.0))
                p_te = m.predict(X_te_c)
                extra = ''

            oof_arr[va_idx] = p_va
            np.save(ckpt, p_va); np.save(ckpt_test, p_te)

            if   tag == 'lgbm': test_lg += p_te / N_FOLDS
            elif tag == 'cb':   test_cb += p_te / N_FOLDS
            elif tag == 'tw':   test_tw += p_te / N_FOLDS
            elif tag == 'et':   test_et += p_te / N_FOLDS
            elif tag == 'rf':   test_rf += p_te / N_FOLDS

            mae = mean_absolute_error(y_va, p_va)
            print(f'  [{tag.upper():4s}] Fold {fold+1}  MAE={mae:.4f}{extra}')

    print()
    lgbm_mae = mean_absolute_error(y, oof_lg)
    cb_mae   = mean_absolute_error(y, oof_cb)
    tw_mae   = mean_absolute_error(y, oof_tw)
    et_mae   = mean_absolute_error(y, oof_et)
    rf_mae   = mean_absolute_error(y, oof_rf)
    print(f'  LGBM OOF: {lgbm_mae:.4f} (FE v2: 8.9308)')
    print(f'  CB OOF  : {cb_mae:.4f}')
    print(f'  TW OOF  : {tw_mae:.4f}')
    print(f'  ET OOF  : {et_mae:.4f}')
    print(f'  RF OOF  : {rf_mae:.4f}')

    # 상관관계
    from scipy.stats import pearsonr
    print('\n  OOF 상관관계:')
    for a, b, pa, pb in [('LGBM','ET',oof_lg,oof_et),('LGBM','RF',oof_lg,oof_rf),
                          ('LGBM','CB',oof_lg,oof_cb),('ET','RF',oof_et,oof_rf)]:
        r, _ = pearsonr(pa, pb)
        print(f'    {a}-{b}: {r:.4f}')

    # 가중 앙상블
    from scipy.optimize import minimize
    def neg_mae(w):
        w = np.maximum(w, 0); w /= w.sum()
        return mean_absolute_error(y, w[0]*oof_lg+w[1]*oof_cb+w[2]*oof_tw+w[3]*oof_et+w[4]*oof_rf)
    res = minimize(neg_mae, [0.4,0.2,0.2,0.1,0.1], method='Nelder-Mead')
    w = np.maximum(res.x, 0); w /= w.sum()
    print(f'\n  가중 앙상블: {neg_mae(w):.4f}  '
          f'(LGBM={w[0]:.3f}, CB={w[1]:.3f}, TW={w[2]:.3f}, ET={w[3]:.3f}, RF={w[4]:.3f})')

    # ── Layer 2: Meta LGBM ──
    print('\n[Layer 2] LGBM 메타 학습기')
    def safe_log(arr): return np.log1p(np.maximum(arr, 0))
    meta_X      = np.column_stack([safe_log(oof_lg), safe_log(oof_cb),
                                   safe_log(oof_tw), safe_log(oof_et), safe_log(oof_rf)])
    meta_X_test = np.column_stack([safe_log(test_lg), safe_log(test_cb),
                                   safe_log(test_tw), safe_log(test_et), safe_log(test_rf)])

    meta_oof = np.zeros(len(X)); meta_test = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(meta_X[tr_idx], y[tr_idx],
              eval_set=[(meta_X[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        meta_oof[va_idx] = m.predict(meta_X[va_idx])
        meta_test += m.predict(meta_X_test) / N_FOLDS
        print(f'  Fold {fold+1}  MAE={mean_absolute_error(y[va_idx], meta_oof[va_idx]):.4f}  iter={m.best_iteration_}')

    meta_mae = mean_absolute_error(y, meta_oof)
    print(f'  [LGBM-meta] OOF MAE={meta_mae:.4f} | std={np.std(meta_oof):.2f}')

    sub[TARGET] = np.maximum(meta_test, 0)
    sub.to_csv(SUBMIT_PATH, index=False)
    print(f'\n→ 저장: {SUBMIT_PATH}')

    print('\n' + '='*60)
    print('방향 B 결과 요약')
    print('='*60)
    print(f'  [비교] FE v2 (264피처): CV 8.7842')
    print(f'  [결과] FE v3 Cumulative: CV {meta_mae:.4f}')
    delta_v2 = meta_mae - 8.7842
    delta_rf = meta_mae - 8.7911
    sign = lambda x: '+' if x > 0 else ''
    print(f'  vs FE v2: {sign(delta_v2)}{delta_v2:.4f}  '
          f'{"✅ 개선" if delta_v2 < 0 else "⚠️ 악화"}')
    print(f'  vs RF 5모델 (212피처): {sign(delta_rf)}{delta_rf:.4f}')


if __name__ == '__main__':
    main()
