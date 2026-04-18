"""
실험 1: sqrt 타깃 변환 + LGBM+CB 앙상블 (일반화 개선)
=======================================================
목적 : log1p 대신 sqrt 변환을 사용해 극값 분포 표현력 개선
       → 예측 std 압축 문제(13.8 vs 실제 27.4) 완화

배경 (Transform Ablation 04.11):
  - log1p CV 8.8836 / sqrt CV 8.8956 (Δ0.0120, log1p 우위)
  - 단, 그 실험은 단일 LGBM 기준. LGBM+CB 앙상블에서는 다를 수 있음
  - sqrt는 분포를 덜 압축 → 극값(>45분) 과소예측 개선 가능성
  - Public 갭이 극값 구간 오차에서 기인한다면 sqrt가 실질 우위일 수 있음

전략:
  1. sqrt 변환 LGBM+CB 앙상블 (XGBoost 제외, 실험 2와 동일 구조)
  2. 예측 분포 비교 (log1p vs sqrt): std, max 변화 확인
  3. (옵션) log1p와 sqrt 앙상블 블렌딩 → 최적 비율 탐색

출력:
  - submissions/ensemble_sqrt_lgbm_cb.csv
  - submissions/ensemble_sqrt_log1p_blend.csv  (블렌딩 버전)
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

LGBM_PARAMS = {
    'num_leaves': 183, 'learning_rate': 0.020703,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

CB_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3.0,
    'bagging_temperature': 0.5,
    'loss_function': 'MAE', 'eval_metric': 'MAE',
    'random_seed': RANDOM_STATE, 'verbose': 0,
    'early_stopping_rounds': 100,
}

# ─── 타깃 변환 ────────────────────────────────────────────────
def t_sqrt(y):  return np.sqrt(np.maximum(y, 0))
def it_sqrt(y): return np.square(np.maximum(y, 0))

def t_log(y):   return np.log1p(np.maximum(y, 0))
def it_log(y):  return np.expm1(y)


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


# ─── 2모델 최적 가중치 ────────────────────────────────────────
def find_weights_2(oof_a, oof_b, y_true):
    best_loss, best_w = np.inf, 0.5
    for alpha in np.linspace(0, 1, 101):
        blend = alpha * oof_a + (1 - alpha) * oof_b
        l = np.mean(np.abs(blend - y_true))
        if l < best_loss:
            best_loss = l
            best_w = alpha
    return best_w, best_loss


# ─── 5-fold 학습 공통 함수 ────────────────────────────────────
def run_fold(train, test, feat_cols, y_transformed, y_raw, groups,
             transform_fn_inv, label):
    gkf  = GroupKFold(n_splits=N_SPLITS)
    oof_lgbm_t = np.zeros(len(train))
    oof_cb_t   = np.zeros(len(train))
    test_lgbm  = np.zeros(len(test))
    test_cb    = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_transformed, groups)):
        X_tr = train.iloc[tr_idx][feat_cols]
        X_va = train.iloc[va_idx][feat_cols]
        y_tr = y_transformed.iloc[tr_idx]
        y_va = y_transformed.iloc[va_idx]

        # LGBM
        m_lg = lgb.LGBMRegressor(**LGBM_PARAMS)
        m_lg.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        oof_lgbm_t[va_idx] = m_lg.predict(X_va)
        test_lgbm += transform_fn_inv(m_lg.predict(test[feat_cols])) / N_SPLITS
        mae_lg = np.mean(np.abs(transform_fn_inv(oof_lgbm_t[va_idx]) - y_raw.iloc[va_idx]))

        # CB
        m_cb = cb.CatBoostRegressor(**CB_PARAMS)
        m_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        oof_cb_t[va_idx] = m_cb.predict(X_va)
        test_cb += transform_fn_inv(m_cb.predict(test[feat_cols])) / N_SPLITS
        mae_cb = np.mean(np.abs(transform_fn_inv(oof_cb_t[va_idx]) - y_raw.iloc[va_idx]))

        print(f'  [{label}] Fold {fold+1}  LGBM={mae_lg:.4f}  CB={mae_cb:.4f}')
        gc.collect()

    oof_lgbm_raw = transform_fn_inv(oof_lgbm_t)
    oof_cb_raw   = transform_fn_inv(oof_cb_t)
    return oof_lgbm_raw, oof_cb_raw, test_lgbm, test_cb


# ─── 메인 ─────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('실험 1 : sqrt 타깃 변환 + LGBM+CB 앙상블')
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
    groups = train['scenario_id']

    # ── sqrt 변환 앙상블 ──────────────────────────────────────
    print('\n[sqrt 변환] 5-fold 학습 시작')
    y_sqrt = y_raw.pipe(t_sqrt)

    oof_lg_sqrt, oof_cb_sqrt, test_lg_sqrt, test_cb_sqrt = run_fold(
        train, test, feat_cols, y_sqrt, y_raw, groups, it_sqrt, 'sqrt'
    )

    w_sqrt, mae_sqrt = find_weights_2(oof_lg_sqrt, oof_cb_sqrt, y_raw.values)
    print(f'\n  sqrt 앙상블: LGBM={w_sqrt:.3f}, CB={1-w_sqrt:.3f}  CV MAE={mae_sqrt:.4f}')

    test_sqrt_blend = w_sqrt * test_lg_sqrt + (1 - w_sqrt) * test_cb_sqrt
    test_sqrt_blend = np.maximum(test_sqrt_blend, 0)
    print(f'  예측 분포 (sqrt): mean={test_sqrt_blend.mean():.2f}, '
          f'std={test_sqrt_blend.std():.2f}, max={test_sqrt_blend.max():.2f}')

    # ── log1p 변환 앙상블 (비교 기준) ─────────────────────────
    print('\n[log1p 변환] 5-fold 학습 시작 (비교 기준)')
    y_log = y_raw.pipe(t_log)

    oof_lg_log, oof_cb_log, test_lg_log, test_cb_log = run_fold(
        train, test, feat_cols, y_log, y_raw, groups, it_log, 'log1p'
    )

    w_log, mae_log = find_weights_2(oof_lg_log, oof_cb_log, y_raw.values)
    print(f'\n  log1p 앙상블: LGBM={w_log:.3f}, CB={1-w_log:.3f}  CV MAE={mae_log:.4f}')

    test_log_blend = w_log * test_lg_log + (1 - w_log) * test_cb_log
    test_log_blend = np.maximum(test_log_blend, 0)
    print(f'  예측 분포 (log1p): mean={test_log_blend.mean():.2f}, '
          f'std={test_log_blend.std():.2f}, max={test_log_blend.max():.2f}')

    # ── sqrt×log1p 블렌딩 ─────────────────────────────────────
    print('\n[sqrt+log1p 블렌딩] 최적 비율 탐색')
    best_blend_loss, best_alpha = np.inf, 0.5
    for alpha in np.linspace(0, 1, 101):
        oof_blend = (alpha * (w_sqrt*oof_lg_sqrt + (1-w_sqrt)*oof_cb_sqrt)
                     + (1-alpha) * (w_log*oof_lg_log + (1-w_log)*oof_cb_log))
        l = np.mean(np.abs(oof_blend - y_raw))
        if l < best_blend_loss:
            best_blend_loss = l
            best_alpha = alpha

    print(f'  최적 비율: sqrt×{best_alpha:.2f} + log1p×{1-best_alpha:.2f}  '
          f'CV MAE={best_blend_loss:.4f}')

    test_final_blend = (best_alpha * test_sqrt_blend
                        + (1 - best_alpha) * test_log_blend)
    test_final_blend = np.maximum(test_final_blend, 0)
    print(f'  예측 분포 (블렌딩): mean={test_final_blend.mean():.2f}, '
          f'std={test_final_blend.std():.2f}, max={test_final_blend.max():.2f}')

    # ── 결과 요약 ─────────────────────────────────────────────
    sample_sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
    print(f'\n{"="*60}')
    print('실험 1 결과 요약')
    print(f'{"="*60}')
    print(f'  log1p 앙상블  CV MAE: {mae_log:.4f}  (std={test_log_blend.std():.2f})')
    print(f'  sqrt  앙상블  CV MAE: {mae_sqrt:.4f}  (std={test_sqrt_blend.std():.2f})')
    print(f'  sqrt+log 블렌드 MAE : {best_blend_loss:.4f}  (std={test_final_blend.std():.2f})')
    print(f'  실제 타깃 std       : {y_raw.std():.2f}')

    # sqrt 단독 제출
    sub1 = sample_sub.copy()
    sub1['avg_delay_minutes_next_30m'] = test_sqrt_blend
    sub1.to_csv(f'{SUB_DIR}/ensemble_sqrt_lgbm_cb.csv', index=False)
    print(f'\n  → 저장: submissions/ensemble_sqrt_lgbm_cb.csv')

    # sqrt+log 블렌딩 제출
    sub2 = sample_sub.copy()
    sub2['avg_delay_minutes_next_30m'] = test_final_blend
    sub2.to_csv(f'{SUB_DIR}/ensemble_sqrt_log1p_blend.csv', index=False)
    print(f'  → 저장: submissions/ensemble_sqrt_log1p_blend.csv')

    # ── 제출 우선순위 권장 ─────────────────────────────────────
    results = [
        ('log1p 앙상블',     mae_log,         'ensemble_lgbm_cb_clean.csv (실험2 결과)'),
        ('sqrt 앙상블',      mae_sqrt,         'ensemble_sqrt_lgbm_cb.csv'),
        ('sqrt+log 블렌드',  best_blend_loss,  'ensemble_sqrt_log1p_blend.csv'),
    ]
    results.sort(key=lambda x: x[1])
    print(f'\n  [제출 우선순위 권장]')
    for rank, (name, mae, fname) in enumerate(results, 1):
        print(f'  {rank}위: {name}  CV={mae:.4f}  →  {fname}')


if __name__ == '__main__':
    main()
