"""
run_exp_fe_v2_optuna.py  —  방향 A: FE v2 기반 LGBM 파라미터 재튜닝
===============================================================
FE v2(264피처)에서 LGBM OOF MAE가 8.9308로 저하됨.
기존 BEST_LGBM_PARAMS는 212피처에서 최적화된 파라미터 → 264피처에 맞게 재튜닝 필요.

전략:
  1. FE v2 파이프라인으로 264피처 생성
  2. Optuna로 LGBM 베이스 파라미터 탐색 (N_TRIALS=50, 2-fold 빠른 검증)
  3. 최적 파라미터로 5-fold 전체 학습 후 RF 5모델 스태킹
  4. FE v2 체크포인트 재사용 (CB/TW/ET/RF는 변경 없음)

예상 시간:
  - Optuna 50 trials × 2-fold: ~20분
  - LGBM 5-fold 재학습: ~10분 (기존 체크포인트 없으므로)
  - 총: ~30분

비교 기준:
  FE v2 (기존 LGBM params): CV 8.7842 / 단독 LGBM OOF 8.9308
  목표: LGBM OOF 8.89 이하 → 메타 CV < 8.78
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

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
DATA_PATH    = 'data/'
CKPT_DIR     = 'docs/fe_v2_ckpt'         # CB/TW/ET/RF 체크포인트 재사용
CKPT_DIR_A   = 'docs/fe_v2_optuna_ckpt'  # LGBM 재튜닝용 신규 ckpt
SUBMIT_PATH  = 'submissions/stacking_fe_v2_optuna_lgbm_meta.csv'
TARGET       = 'avg_delay_minutes_next_30m'
N_FOLDS      = 5
N_TRIALS     = 50
RANDOM_SEED  = 42

os.makedirs(CKPT_DIR_A, exist_ok=True)

KEY_COLS_V2 = [
    'low_battery_ratio', 'battery_mean', 'battery_std',
    'robot_idle', 'robot_charging', 'order_inflow_15m',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'sku_concentration', 'urgent_order_ratio',
]

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
ET_PARAMS  = {'n_estimators': 500, 'n_jobs': -1, 'random_state': RANDOM_SEED, 'min_samples_leaf': 26}
RF_PARAMS  = {'n_estimators': 500, 'max_features': 0.33, 'min_samples_leaf': 26,
              'n_jobs': -1, 'random_state': RANDOM_SEED}
META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'n_estimators': 1000, 'objective': 'regression_l1',
    'random_state': RANDOM_SEED, 'verbose': -1, 'n_jobs': -1,
}


# ──────────────────────────────────────────────
# FE v2 파이프라인 (run_exp_fe_v2.py와 동일)
# ──────────────────────────────────────────────
def build_features_v2(train, test, layout, verbose=True):
    if verbose:
        print(f'[build_features_v2] 시작: train={train.shape}, test={test.shape}')
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
    train = add_domain_features(train); test = add_domain_features(test)
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
# Optuna: LGBM 파라미터 탐색 (2-fold 빠른 검증)
# ──────────────────────────────────────────────
def tune_lgbm(X, y, groups):
    print(f'[Optuna] LGBM 파라미터 탐색 시작 (N_TRIALS={N_TRIALS}, 2-fold)')
    gkf2 = GroupKFold(n_splits=2)

    def objective(trial):
        params = {
            'num_leaves':       trial.suggest_int('num_leaves', 50, 300),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.8),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
            'min_child_samples':trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
            'objective': 'regression_l1', 'n_estimators': 3000,
            'bagging_freq': 1, 'random_state': RANDOM_SEED,
            'verbose': -1, 'n_jobs': -1,
        }
        maes = []
        for tr_idx, va_idx in gkf2.split(X, y, groups):
            m = lgb.LGBMRegressor(**params)
            m.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
            maes.append(mean_absolute_error(y[va_idx], m.predict(X[va_idx])))
        return np.mean(maes)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    best.update({'objective': 'regression_l1', 'n_estimators': 3000,
                 'bagging_freq': 1, 'random_state': RANDOM_SEED,
                 'verbose': -1, 'n_jobs': -1})
    print(f'  최적 2-fold CV: {study.best_value:.4f}')
    print(f'  최적 파라미터: num_leaves={best["num_leaves"]}, lr={best["learning_rate"]:.5f}, '
          f'feat_frac={best["feature_fraction"]:.4f}')
    return best


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

    train_fe, test_fe = build_features_v2(train, test, layout)
    feat_cols = get_feature_cols(train_fe)
    X         = train_fe[feat_cols].values
    y         = train_fe[TARGET].values
    X_test    = test_fe[feat_cols].values
    groups    = train_fe['scenario_id'].values
    gkf       = GroupKFold(n_splits=N_FOLDS)

    print(f'피처 수: {len(feat_cols)}')
    print()

    # ── Optuna LGBM 튜닝 ──
    best_lgbm_params = tune_lgbm(X, y, groups)
    print()

    # ── Layer 1: 5 Base Models ──
    oof_lg = np.zeros(len(X));  test_lg = np.zeros(len(X_test))
    oof_cb = np.zeros(len(X));  test_cb = np.zeros(len(X_test))
    oof_tw = np.zeros(len(X));  test_tw = np.zeros(len(X_test))
    oof_et = np.zeros(len(X));  test_et = np.zeros(len(X_test))
    oof_rf = np.zeros(len(X));  test_rf = np.zeros(len(X_test))

    print('[Layer 1] Base Models')

    # LGBM — 재튜닝 파라미터로 재학습 (기존 ckpt 무시)
    print('  [LGBM] Optuna 튜닝 파라미터로 재학습')
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        ckpt = f'{CKPT_DIR_A}/lgbm_fold{fold}.npy'
        ckpt_test = f'{CKPT_DIR_A}/lgbm_fold{fold}_test.npy'
        if os.path.exists(ckpt):
            oof_lg[va_idx] = np.load(ckpt)
            test_lg += np.load(ckpt_test) / N_FOLDS
        else:
            m = lgb.LGBMRegressor(**best_lgbm_params)
            m.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                              lgb.log_evaluation(-1)])
            p_va = m.predict(X[va_idx]); p_te = m.predict(X_test)
            oof_lg[va_idx] = p_va; test_lg += p_te / N_FOLDS
            np.save(ckpt, p_va); np.save(ckpt_test, p_te)
            print(f'    Fold {fold+1}  MAE={mean_absolute_error(y[va_idx], p_va):.4f}  iter={m.best_iteration_}')
    print(f'  LGBM OOF MAE: {mean_absolute_error(y, oof_lg):.4f}  (FE v2 기준: 8.9308)')

    # CB / TW / ET / RF — FE v2 체크포인트 재사용
    print('\n  [CB/TW/ET/RF] FE v2 체크포인트 재사용')
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        for tag, oof_arr, test_arr, params, model_cls in [
            ('cb',  oof_cb, None, CB_PARAMS, 'cb'),
            ('tw',  oof_tw, None, TW_PARAMS, 'tw'),
            ('et',  oof_et, None, ET_PARAMS, 'et'),
            ('rf',  oof_rf, None, RF_PARAMS, 'rf'),
        ]:
            ckpt      = f'{CKPT_DIR}/{tag}_fold{fold}.npy'
            ckpt_test = f'{CKPT_DIR}/{tag}_fold{fold}_test.npy'
            if os.path.exists(ckpt) and os.path.exists(ckpt_test):
                oof_arr[va_idx] = np.load(ckpt)
                if tag == 'cb':  test_cb += np.load(ckpt_test) / N_FOLDS
                if tag == 'tw':  test_tw += np.load(ckpt_test) / N_FOLDS
                if tag == 'et':  test_et += np.load(ckpt_test) / N_FOLDS
                if tag == 'rf':  test_rf += np.load(ckpt_test) / N_FOLDS
            else:
                print(f'  ⚠️  체크포인트 없음: {ckpt} — FE v2 먼저 실행 필요')

    for tag, oof_arr in [('CB', oof_cb), ('TW', oof_tw), ('ET', oof_et), ('RF', oof_rf)]:
        if oof_arr.sum() != 0:
            print(f'  {tag} OOF MAE: {mean_absolute_error(y, oof_arr):.4f}')

    # 상관관계
    print('\n  OOF 상관관계:')
    from scipy.stats import pearsonr
    for a, b, pa, pb in [('LGBM','ET',oof_lg,oof_et),('LGBM','RF',oof_lg,oof_rf),
                          ('LGBM','CB',oof_lg,oof_cb),('LGBM','TW',oof_lg,oof_tw),
                          ('ET','RF',oof_et,oof_rf)]:
        if pa.sum() != 0 and pb.sum() != 0:
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
    print('방향 A 결과 요약')
    print('='*60)
    print(f'  [비교] FE v2 (기존 params): CV 8.7842')
    print(f'  [결과] FE v2 + Optuna LGBM: CV {meta_mae:.4f}')
    delta = meta_mae - 8.7842
    sign = '+' if delta > 0 else ''
    print(f'  vs FE v2: {sign}{delta:.4f}  {"✅ 개선" if delta < 0 else "⚠️ 악화 또는 동일"}')


if __name__ == '__main__':
    main()
