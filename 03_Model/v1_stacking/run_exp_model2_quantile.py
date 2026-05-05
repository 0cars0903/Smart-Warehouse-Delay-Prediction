"""
모델 실험 2: Quantile 블렌딩
==============================
목적 : q=0.3 / 0.5 / 0.7 분위수 모델을 블렌딩해 예측 범위 확장
       → 현재 예측 std 압축(13.5 vs 실제 27.4) 우회

아이디어 :
  - MAE 최적화 = q=0.5 quantile (중앙값) 최적화
  - 중앙값만 최적화하면 모든 모델이 중간값으로 수렴 → std 압축
  - 낮은 분위수(q=0.3) + 높은 분위수(q=0.7)를 함께 블렌딩하면
    예측 분포의 tail이 살아남 → MAE 자체도 개선 가능

설계 :
  - LGBM q=0.3, 0.5, 0.7 각각 5-fold 학습 (raw target, 변환 없음)
  - CB MAE (log1p) 5-fold 학습
  - 최적 가중치로 4모델 블렌딩
  - 추가로: q=0.3/0.5/0.7만 쓴 3모델 블렌드도 비교

출력 :
  submissions/ensemble_quantile_4model.csv   (LGBM q×3 + CB)
  submissions/ensemble_quantile_lgbm3.csv    (LGBM q×3 only)

예상 실행 시간: ~35분 (LGBM 3종 × 5-fold + CB 5-fold)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
import warnings, gc, os, sys
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

# ─── 상수 ────────────────────────────────────────────────────
_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
N_SPLITS     = 5
RANDOM_STATE = 42

QUANTILES = [0.3, 0.5, 0.7]   # 블렌딩할 분위수 목록

BASE_LGBM_PARAMS = {
    'num_leaves': 183, 'learning_rate': 0.020703,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'n_estimators': 3000, 'bagging_freq': 1,
    'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
}

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


def get_feat_cols(df):
    drop = {'ID', 'scenario_id', 'ts_idx', 'avg_delay_minutes_next_30m'}
    return [c for c in df.columns if c not in drop and df[c].dtype != object]


def find_best_weights(oofs: list, y_true: np.ndarray) -> tuple:
    n = len(oofs)
    def loss(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        return np.mean(np.abs(sum(wi*o for wi, o in zip(w, oofs)) - y_true))
    best_loss, best_w = np.inf, np.ones(n)/n
    for _ in range(300):
        w0 = np.random.dirichlet(np.ones(n))
        res = minimize(loss, w0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-6})
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = np.abs(res.x) / (np.abs(res.x).sum() + 1e-9)
    return best_w, best_loss


# ─── LGBM Quantile 학습 ───────────────────────────────────────
def run_lgbm_quantile(train, test, feat_cols, y_raw, groups, alpha):
    """
    Quantile regression: raw target 사용 (monotonic 변환 불필요)
    단, 예측값이 음수가 될 수 있으므로 clip(0) 처리
    """
    params = BASE_LGBM_PARAMS.copy()
    params['objective'] = 'quantile'
    params['alpha']     = alpha
    params['metric']    = 'quantile'

    gkf  = GroupKFold(n_splits=N_SPLITS)
    oof  = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    label = f'LGBM-q{alpha:.1f}'

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
          f'std={oof.std():.2f}, max={oof.max():.2f}')
    return oof, test_pred, oof_mae


# ─── CatBoost MAE (log1p) 학습 ────────────────────────────────
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


def save_sub(preds, filename):
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(preds, 0)
    sample.to_csv(os.path.join(SUB_DIR, filename), index=False)
    print(f'  → 저장: submissions/{filename}')


# ─── 메인 ─────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('모델 실험 2: Quantile 블렌딩')
    print('=' * 60)

    train, test = load_data()
    feat_cols = get_feat_cols(train)
    print(f'피처 수: {len(feat_cols)}개')

    y_raw  = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']

    # ── Quantile LGBM 3종 학습 ────────────────────────────────
    q_oofs, q_tests, q_maes = [], [], []
    for q in QUANTILES:
        print(f'\n[LGBM Quantile q={q:.1f}] 5-fold 학습')
        oof_q, test_q, mae_q = run_lgbm_quantile(
            train, test, feat_cols, y_raw, groups, alpha=q)
        q_oofs.append(oof_q)
        q_tests.append(test_q)
        q_maes.append(mae_q)

    # ── CatBoost(log1p) 학습 ──────────────────────────────────
    print('\n[CatBoost log1p] 5-fold 학습')
    oof_cb, test_cb, mae_cb = run_cb_log1p(
        train, test, feat_cols, y_raw, groups)

    # ── LGBM 3-quantile 블렌드 ────────────────────────────────
    w3, mae_q3 = find_best_weights(q_oofs, y_raw.values)
    test_q3 = sum(w*t for w, t in zip(w3, q_tests))
    print(f'\n[LGBM 3-quantile 블렌드]')
    print(f'  가중치: q0.3={w3[0]:.3f}, q0.5={w3[1]:.3f}, q0.7={w3[2]:.3f}')
    print(f'  CV MAE: {mae_q3:.4f} | std={test_q3.std():.2f}, max={test_q3.max():.2f}')
    save_sub(test_q3, 'ensemble_quantile_lgbm3.csv')

    # ── 4모델 블렌드 (LGBM 3종 + CB) ─────────────────────────
    all_oofs  = q_oofs + [oof_cb]
    all_tests = q_tests + [test_cb]
    w4, mae_q4 = find_best_weights(all_oofs, y_raw.values)
    test_q4 = sum(w*t for w, t in zip(w4, all_tests))
    print(f'\n[4모델 블렌드 (q×3 + CB)]')
    print(f'  가중치: q0.3={w4[0]:.3f}, q0.5={w4[1]:.3f}, '
          f'q0.7={w4[2]:.3f}, CB={w4[3]:.3f}')
    print(f'  CV MAE: {mae_q4:.4f} | std={test_q4.std():.2f}, max={test_q4.max():.2f}')
    save_sub(test_q4, 'ensemble_quantile_4model.csv')

    # ── 최종 요약 ─────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('실험 2 결과 요약')
    print(f'{"="*60}')
    print(f'  LGBM q=0.3  단독  OOF MAE: {q_maes[0]:.4f}  '
          f'(std={q_oofs[0].std():.2f})')
    print(f'  LGBM q=0.5  단독  OOF MAE: {q_maes[1]:.4f}  '
          f'(std={q_oofs[1].std():.2f})')
    print(f'  LGBM q=0.7  단독  OOF MAE: {q_maes[2]:.4f}  '
          f'(std={q_oofs[2].std():.2f})')
    print(f'  CB log1p    단독  OOF MAE: {mae_cb:.4f}')
    print(f'  LGBM q×3 블렌드   CV MAE: {mae_q3:.4f}  '
          f'(std={test_q3.std():.2f})')
    print(f'  4모델 블렌드      CV MAE: {mae_q4:.4f}  '
          f'(std={test_q4.std():.2f})')
    print(f'  실제 타깃 std: {y_raw.std():.2f}')
    print(f'\n  (참고) 현재 Public 최고: 8.8674 / 10.3347')
    print(f'  핵심 지표: q0.7이 q0.5보다 높은 std를 가지면 긍정적 신호')


if __name__ == '__main__':
    main()
