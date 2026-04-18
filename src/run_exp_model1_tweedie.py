"""
모델 실험 1: LightGBM Tweedie 목적함수
========================================
목적 : MAE(L1) 대신 Tweedie loss로 right-skewed 분포 직접 모델링
       → 예측 std 압축(13.5 vs 실제 27.4) 문제 공략

배경 :
  - 지연 시간 분포: right-skewed (왜도 5.68), 비음수
  - Tweedie: 복합 포아송-감마 분포, 극값을 자연스럽게 표현
  - variance_power p: 1→Poisson-like, 2→Gamma-like, 1.5=균형점
  - Tweedie는 log link 내장 → raw 타깃 직접 사용 (log1p 변환 불필요)

비교 실험:
  A. Tweedie(p=1.5) LGBM + CB(MAE, log1p) 앙상블
  B. Tweedie(p=1.2) + Tweedie(p=1.8) 블렌딩 (분포 다변화)
  C. Tweedie 단독 vs 현재 최고 L1 기준선 비교

출력 :
  submissions/ensemble_tweedie15_cb.csv
  submissions/ensemble_tweedie_blend.csv

예상 실행 시간: ~25분
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
import warnings, gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features

warnings.filterwarnings('ignore')

# ─── 상수 ────────────────────────────────────────────────────
_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
N_SPLITS     = 5
RANDOM_STATE = 42

# 현재 최고 LGBM 파라미터 (L1 기준, 비교용)
BASE_LGBM_PARAMS = {
    'num_leaves': 183, 'learning_rate': 0.020703,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# Tweedie LGBM 파라미터 (L1과 동일 구조, objective만 변경)
def make_tweedie_params(power: float) -> dict:
    p = BASE_LGBM_PARAMS.copy()
    p['objective'] = 'tweedie'
    p['tweedie_variance_power'] = power
    p['metric'] = 'tweedie'   # eval metric 변경
    return p

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
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(
        train, test, layout,
        lag_lags=[1,2,3,4,5,6],
        rolling_windows=[3,5,10],
    )
    return train, test


# ─── 공통 유틸 ────────────────────────────────────────────────
def get_feat_cols(df):
    drop = {'ID', 'scenario_id', 'ts_idx', 'avg_delay_minutes_next_30m'}
    return [c for c in df.columns if c not in drop and df[c].dtype != object]

def find_best_weights(oofs: list, y_true: np.ndarray) -> tuple:
    """n개 OOF → MAE 최소화 가중치 탐색"""
    n = len(oofs)
    def loss(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        return np.mean(np.abs(sum(wi*o for wi, o in zip(w, oofs)) - y_true))
    best_loss, best_w = np.inf, np.ones(n) / n
    for _ in range(300):
        w0 = np.random.dirichlet(np.ones(n))
        res = minimize(loss, w0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-6})
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = np.abs(res.x) / (np.abs(res.x).sum() + 1e-9)
    return best_w, best_loss


# ─── 5-fold 학습 (Tweedie LGBM) ───────────────────────────────
def run_lgbm_tweedie(train, test, feat_cols, y_raw, groups, power, label):
    """Tweedie: raw positive target, log link 내장 → 변환 불필요"""
    gkf  = GroupKFold(n_splits=N_SPLITS)
    oof  = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    params = make_tweedie_params(power)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_raw, groups)):
        X_tr, X_va = train.iloc[tr_idx][feat_cols], train.iloc[va_idx][feat_cols]
        y_tr, y_va = y_raw.iloc[tr_idx], y_raw.iloc[va_idx]

        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(150, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        oof[va_idx] = np.maximum(m.predict(X_va), 0)
        test_pred  += np.maximum(m.predict(test[feat_cols]), 0) / N_SPLITS
        mae = np.mean(np.abs(oof[va_idx] - y_va.values))
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        gc.collect()

    oof_mae = np.mean(np.abs(oof - y_raw.values))
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | '
          f'pred std={oof.std():.2f} / max={oof.max():.2f}')
    return oof, test_pred, oof_mae


# ─── 5-fold 학습 (CatBoost, log1p) ────────────────────────────
def run_cb_log1p(train, test, feat_cols, y_raw, groups):
    gkf  = GroupKFold(n_splits=N_SPLITS)
    oof  = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    y_log = np.log1p(y_raw)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_log, groups)):
        X_tr, X_va = train.iloc[tr_idx][feat_cols], train.iloc[va_idx][feat_cols]
        y_tr, y_va = y_log.iloc[tr_idx], y_log.iloc[va_idx]

        m = cb.CatBoostRegressor(**CB_PARAMS)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        oof[va_idx] = np.expm1(np.maximum(m.predict(X_va), 0))
        test_pred  += np.expm1(np.maximum(m.predict(test[feat_cols]), 0)) / N_SPLITS
        mae = np.mean(np.abs(oof[va_idx] - y_raw.iloc[va_idx].values))
        print(f'  [CB-log1p] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        gc.collect()

    oof_mae = np.mean(np.abs(oof - y_raw.values))
    print(f'  [CB-log1p] OOF MAE={oof_mae:.4f}')
    return oof, test_pred, oof_mae


# ─── 제출 저장 ────────────────────────────────────────────────
def save_sub(preds, filename):
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(preds, 0)
    path = os.path.join(SUB_DIR, filename)
    sample.to_csv(path, index=False)
    print(f'  → 저장: submissions/{filename}')


# ─── 메인 ─────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('모델 실험 1: LightGBM Tweedie 목적함수')
    print('=' * 60)

    train, test = load_data()
    feat_cols = get_feat_cols(train)
    print(f'피처 수: {len(feat_cols)}개')

    y_raw  = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']
    print(f'타깃 분포: mean={y_raw.mean():.2f}, std={y_raw.std():.2f}, '
          f'skew={y_raw.skew():.2f}, max={y_raw.max():.1f}')

    # ── A. Tweedie(p=1.5) + CB 앙상블 ─────────────────────────
    print('\n[A] Tweedie(p=1.5) LGBM 학습')
    oof_tw15, test_tw15, mae_tw15 = run_lgbm_tweedie(
        train, test, feat_cols, y_raw, groups, power=1.5, label='Tweedie-1.5')

    print('\n[A] CatBoost(log1p) 학습')
    oof_cb, test_cb, mae_cb = run_cb_log1p(train, test, feat_cols, y_raw, groups)

    w_a, mae_a = find_best_weights([oof_tw15, oof_cb], y_raw.values)
    test_a = w_a[0]*test_tw15 + w_a[1]*test_cb
    print(f'\n[A] 최적 앙상블: Tweedie={w_a[0]:.3f}, CB={w_a[1]:.3f}  '
          f'CV MAE={mae_a:.4f}')
    print(f'    예측 분포: mean={test_a.mean():.2f}, '
          f'std={test_a.std():.2f}, max={test_a.max():.2f}')
    save_sub(test_a, 'ensemble_tweedie15_cb.csv')

    # ── B. Tweedie(p=1.2) + Tweedie(p=1.8) 블렌딩 ────────────
    print('\n[B] Tweedie(p=1.2) 학습')
    oof_tw12, test_tw12, mae_tw12 = run_lgbm_tweedie(
        train, test, feat_cols, y_raw, groups, power=1.2, label='Tweedie-1.2')

    print('\n[B] Tweedie(p=1.8) 학습')
    oof_tw18, test_tw18, mae_tw18 = run_lgbm_tweedie(
        train, test, feat_cols, y_raw, groups, power=1.8, label='Tweedie-1.8')

    w_b, mae_b = find_best_weights(
        [oof_tw12, oof_tw15, oof_tw18, oof_cb], y_raw.values)
    test_b = (w_b[0]*test_tw12 + w_b[1]*test_tw15
              + w_b[2]*test_tw18 + w_b[3]*test_cb)
    print(f'\n[B] 4모델 앙상블: '
          f'Tw1.2={w_b[0]:.3f}, Tw1.5={w_b[1]:.3f}, '
          f'Tw1.8={w_b[2]:.3f}, CB={w_b[3]:.3f}  CV MAE={mae_b:.4f}')
    print(f'    예측 분포: mean={test_b.mean():.2f}, '
          f'std={test_b.std():.2f}, max={test_b.max():.2f}')
    save_sub(test_b, 'ensemble_tweedie_blend.csv')

    # ── 최종 요약 ─────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('실험 결과 요약')
    print(f'{"="*60}')
    print(f'  Tweedie(1.2) 단독    OOF MAE: {mae_tw12:.4f}')
    print(f'  Tweedie(1.5) 단독    OOF MAE: {mae_tw15:.4f}')
    print(f'  Tweedie(1.8) 단독    OOF MAE: {mae_tw18:.4f}')
    print(f'  CatBoost(log1p) 단독 OOF MAE: {mae_cb:.4f}')
    print(f'  [A] Tw1.5 + CB 앙상블 CV MAE: {mae_a:.4f}'
          f'  → submissions/ensemble_tweedie15_cb.csv')
    print(f'  [B] 4모델 블렌드     CV MAE: {mae_b:.4f}'
          f'  → submissions/ensemble_tweedie_blend.csv')
    print(f'\n  (참고) 현재 Public 최고: 8.8674 / 10.3347')
    print(f'  예측 std 개선 여부를 std 수치로 확인 (목표: 현재 13.5 → 상승)')


if __name__ == '__main__':
    main()
