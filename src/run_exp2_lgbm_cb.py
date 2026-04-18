"""
실험 2: XGBoost 제외 LGBM + CatBoost 2모델 앙상블 (일반화 개선)
=================================================================
목적 : XGBoost(정규화 미작동, 기여도 0.04~0.10)를 완전 제외하고
       LGBM + CB 2모델 최적 블렌딩으로 노이즈 제거 → 갭 개선

기반  : ensemble_ts0 파이프라인 (CV 8.8649, 현재 CV 최고)
변경  : XGBoost 학습·블렌딩 제거 / 2모델 가중치 재최적화
출력  : submissions/ensemble_lgbm_cb_clean.csv

예상 실행 시간: ~20분 (5-fold × 2모델)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
import warnings, gc, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import build_features

warnings.filterwarnings('ignore')

# ─── 상수 ─────────────────────────────────────────────────────
_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
N_SPLITS     = 5
RANDOM_STATE = 42

# ensemble_ts0 기준 LGBM 파라미터 (신규 Optuna, num_leaves=183)
LGBM_PARAMS = {
    'num_leaves': 183, 'learning_rate': 0.020703,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# run_optuna_full 기준 CB 파라미터 (Optuna 20 trials 최적)
CB_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3.0,
    'bagging_temperature': 0.5,
    'loss_function': 'MAE', 'eval_metric': 'MAE',
    'random_seed': RANDOM_STATE, 'verbose': 0,
    'early_stopping_rounds': 100,
}


# ─── 데이터 로드 ──────────────────────────────────────────────
def load_data():
    train  = pd.read_csv(f'{DATA_DIR}/train.csv')
    test   = pd.read_csv(f'{DATA_DIR}/test.csv')
    layout = pd.read_csv(f'{DATA_DIR}/layout_info.csv')
    # 284피처 기준선: lag(1~6) + rolling(3,5,10)
    train, test = build_features(
        train, test, layout,
        lag_lags=[1, 2, 3, 4, 5, 6],
        rolling_windows=[3, 5, 10],
    )
    return train, test


# ─── 타깃 변환 ────────────────────────────────────────────────
def t(y):  return np.log1p(y)
def it(y): return np.expm1(y)


# ─── 최적 가중치 (2모델) ──────────────────────────────────────
def find_weights_2(oof_a, oof_b, y_true):
    def loss(w):
        w = np.clip(w, 0, 1)
        blend = w * oof_a + (1 - w) * oof_b
        return np.mean(np.abs(blend - y_true))
    best_loss, best_w = np.inf, 0.5
    for alpha in np.linspace(0, 1, 101):
        l = loss(alpha)
        if l < best_loss:
            best_loss = l
            best_w = alpha
    return best_w, best_loss   # w_lgbm, best_mae


# ─── 메인 ─────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('실험 2 : LGBM + CatBoost 2모델 앙상블 (XGBoost 제외)')
    print('=' * 60)

    train, test = load_data()
    drop_cols   = ['ID', 'scenario_id', 'ts_idx', 'avg_delay_minutes_next_30m']
    feat_cols   = [c for c in train.columns
                   if c not in drop_cols and train[c].dtype != object]
    obj_cols = [c for c in train.columns
                if c not in drop_cols and train[c].dtype == object]
    if obj_cols:
        print(f'  ※ object 타입 컬럼 제외: {obj_cols}')
    print(f'피처 수: {len(feat_cols)}개')

    y_raw  = train['avg_delay_minutes_next_30m']
    y      = t(y_raw)
    groups = train['scenario_id']
    gkf    = GroupKFold(n_splits=N_SPLITS)

    oof_lgbm_log = np.zeros(len(train))
    oof_cb_log   = np.zeros(len(train))
    test_lgbm    = np.zeros(len(test))
    test_cb      = np.zeros(len(test))

    # ── 5-fold 학습 ──────────────────────────────────────────
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups)):
        X_tr = train.iloc[tr_idx][feat_cols]
        X_va = train.iloc[va_idx][feat_cols]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        print(f'\n── Fold {fold+1}/{N_SPLITS} ──')

        # LightGBM
        m_lg = lgb.LGBMRegressor(**LGBM_PARAMS)
        m_lg.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        oof_lgbm_log[va_idx] = m_lg.predict(X_va)
        test_lgbm += it(m_lg.predict(test[feat_cols])) / N_SPLITS
        mae_lg = np.mean(np.abs(it(oof_lgbm_log[va_idx]) - y_raw.iloc[va_idx]))
        print(f'  LGBM MAE: {mae_lg:.4f}  (best_iter={m_lg.best_iteration_})')

        # CatBoost
        m_cb = cb.CatBoostRegressor(**CB_PARAMS)
        m_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        oof_cb_log[va_idx] = m_cb.predict(X_va)
        test_cb += it(m_cb.predict(test[feat_cols])) / N_SPLITS
        mae_cb = np.mean(np.abs(it(oof_cb_log[va_idx]) - y_raw.iloc[va_idx]))
        print(f'  CB   MAE: {mae_cb:.4f}  (best_iter={m_cb.best_iteration_})')

        gc.collect()

    # ── OOF 성능 ─────────────────────────────────────────────
    oof_lgbm_raw = it(oof_lgbm_log)
    oof_cb_raw   = it(oof_cb_log)

    mae_lgbm_oof = np.mean(np.abs(oof_lgbm_raw - y_raw))
    mae_cb_oof   = np.mean(np.abs(oof_cb_raw   - y_raw))
    mae_equal    = np.mean(np.abs(0.5*oof_lgbm_raw + 0.5*oof_cb_raw - y_raw))
    w_lgbm, mae_opt = find_weights_2(oof_lgbm_raw, oof_cb_raw, y_raw.values)
    w_cb = 1 - w_lgbm

    print(f'\n{"="*60}')
    print('OOF 성능 요약')
    print(f'{"="*60}')
    print(f'  LGBM 단독 OOF MAE : {mae_lgbm_oof:.4f}')
    print(f'  CB   단독 OOF MAE : {mae_cb_oof:.4f}')
    print(f'  균등 앙상블 MAE   : {mae_equal:.4f}')
    print(f'  최적 가중치 MAE   : {mae_opt:.4f}  (LGBM={w_lgbm:.3f}, CB={w_cb:.3f})')

    # 예측 분포 확인
    test_blend = w_lgbm * test_lgbm + w_cb * test_cb
    test_blend = np.maximum(test_blend, 0)
    print(f'\n  예측 분포: mean={test_blend.mean():.2f}, '
          f'std={test_blend.std():.2f}, max={test_blend.max():.2f}')

    # ── 제출 파일 ─────────────────────────────────────────────
    sample_sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
    sample_sub['avg_delay_minutes_next_30m'] = test_blend

    sub_name = 'ensemble_lgbm_cb_clean.csv'
    sub_path = f'{SUB_DIR}/{sub_name}'
    sample_sub.to_csv(sub_path, index=False)
    print(f'\n  → 제출 파일 저장: submissions/{sub_name}')
    print(f'     CV MAE = {mae_opt:.4f}  |  (참고) 이전 3모델 CV = 8.8649')


if __name__ == '__main__':
    main()
