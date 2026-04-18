"""
옵션 C: Ridge + LGBM 메타 앙상블 블렌딩
=========================================
목적 : 동일한 4모델 OOF(v3)에 대해 Ridge-meta와 LGBM-meta를 각각 학습하고
       OOF 기준 최적 가중치로 블렌딩 → 두 메타 학습기의 보완 효과 기대

근거:
  - LGBM-meta: 비선형 결합 가능, 과적합 리스크 있음
  - Ridge-meta: 선형 결합, 정규화 강해서 안정적
  - 블렌딩으로 두 메타의 오차를 상쇄 가능

비교 기준: v3 LGBM-meta CV 8.7929 / Public 10.2264 🏆

체크포인트 재활용: docs/stacking_ckpt/ + docs/stacking_v2_ckpt/

예상 시간: ~7분 (Ridge 빠름, LGBM ~5분 재실행)
출력: submissions/stacking_4model_ridge_lgbm_blend.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import warnings, gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_V1  = os.path.join(_BASE, '..', 'docs', 'stacking_ckpt')
CKPT_V2  = os.path.join(_BASE, '..', 'docs', 'stacking_v2_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}


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


def save_sub(preds, filename):
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(preds, 0)
    sample.to_csv(os.path.join(SUB_DIR, filename), index=False)
    print(f'  → 저장: submissions/{filename}')


def load_ckpts():
    oof_lg, test_lg = (np.load(os.path.join(CKPT_V1, f'lgbm_{s}.npy')) for s in ['oof','test'])
    oof_cb, test_cb = (np.load(os.path.join(CKPT_V1, f'cb_{s}.npy'))   for s in ['oof','test'])
    oof_et, test_et = (np.load(os.path.join(CKPT_V1, f'et_{s}.npy'))   for s in ['oof','test'])
    oof_tw, test_tw = (np.load(os.path.join(CKPT_V2, f'tw18_{s}.npy')) for s in ['oof','test'])
    print('  체크포인트 로드: LGBM / CB / ET / TW1.8')
    return (oof_lg, test_lg, oof_cb, test_cb, oof_et, test_et, oof_tw, test_tw)


def run_ridge_meta(meta_train, meta_test, y_raw, groups):
    """Ridge 메타: log 공간 학습, 다양한 alpha 탐색"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_meta  = np.zeros(len(y_raw))
    test_meta = np.zeros(meta_test.shape[0])
    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
        y_tr = np.log1p(y_raw.iloc[tr_idx].values)

        # RidgeCV로 fold 내 alpha 자동 선택
        m = RidgeCV(alphas=alphas, fit_intercept=True, cv=3)
        m.fit(X_tr, y_tr)
        oof_meta[va_idx] = np.expm1(m.predict(X_va))
        test_meta       += np.expm1(m.predict(meta_test)) / N_SPLITS

        mae  = np.mean(np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values))
        coef = {f'f{i}': f'{c:.3f}' for i, c in enumerate(m.coef_)}
        print(f'  [Ridge-meta] Fold {fold+1}  MAE={mae:.4f}  alpha={m.alpha_:.3f}  coef={coef}')

    oof_mae = np.mean(np.abs(oof_meta - y_raw.values))
    print(f'  [Ridge-meta] OOF MAE={oof_mae:.4f} | std={oof_meta.std():.2f}')
    return oof_meta, test_meta, oof_mae


def run_lgbm_meta(meta_train, meta_test, y_raw, groups):
    """LGBM 메타 (v3와 동일 파라미터)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_meta  = np.zeros(len(y_raw))
    test_meta = np.zeros(meta_test.shape[0])

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
        y_tr_log = np.log1p(y_raw.iloc[tr_idx].values)
        y_va_log = np.log1p(y_raw.iloc[va_idx].values)

        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(X_tr, y_tr_log,
              eval_set=[(X_va, y_va_log)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        oof_meta[va_idx] = np.expm1(m.predict(X_va))
        test_meta       += np.expm1(m.predict(meta_test)) / N_SPLITS

        mae = np.mean(np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values))
        print(f'  [LGBM-meta]  Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_mae = np.mean(np.abs(oof_meta - y_raw.values))
    print(f'  [LGBM-meta]  OOF MAE={oof_mae:.4f} | std={oof_meta.std():.2f}')
    return oof_meta, test_meta, oof_mae


def main():
    print('=' * 60)
    print('옵션 C: 4모델 OOF → Ridge + LGBM 메타 블렌딩')
    print('비교 기준: v3 LGBM-meta CV 8.7929 / Public 10.2264 🏆')
    print('=' * 60)

    train, test = load_data()
    y_raw  = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']

    print('\n[체크포인트 로드]')
    oof_lg, test_lg, oof_cb, test_cb, oof_et, test_et, oof_tw, test_tw = load_ckpts()

    # 메타 피처 (log 공간 통일, TW → log1p 변환)
    test_tw_clipped = np.maximum(test_tw, 0)
    meta_train = np.column_stack([oof_lg, oof_cb, np.log1p(oof_tw), oof_et])
    meta_test  = np.column_stack([test_lg, test_cb, np.log1p(test_tw_clipped), test_et])

    print('\n[메타 학습기 1] Ridge (RidgeCV)')
    oof_ridge, test_ridge, mae_ridge = run_ridge_meta(meta_train, meta_test, y_raw, groups)

    print('\n[메타 학습기 2] LightGBM (v3 파라미터)')
    oof_lgbm, test_lgbm, mae_lgbm = run_lgbm_meta(meta_train, meta_test, y_raw, groups)

    # ── OOF 기반 최적 블렌딩 가중치 탐색 ────────────────────
    print('\n[최적 블렌딩]')
    corr = np.corrcoef(oof_ridge, oof_lgbm)[0, 1]
    print(f'  Ridge-LGBM 메타 OOF 상관: {corr:.4f}')

    def blend_loss(w):
        w = np.clip(w, 0, 1)
        pred = w * oof_ridge + (1 - w) * oof_lgbm
        return np.mean(np.abs(pred - y_raw.values))

    best_loss, best_w = np.inf, 0.5
    for w in np.linspace(0, 1, 101):
        loss = blend_loss(w)
        if loss < best_loss:
            best_loss = loss
            best_w = w

    print(f'  최적 가중치: Ridge={best_w:.2f}, LGBM={1-best_w:.2f}')
    print(f'  블렌딩 CV MAE: {best_loss:.4f}')
    print(f'    (Ridge 단독: {mae_ridge:.4f} / LGBM 단독: {mae_lgbm:.4f})')

    # 최적 가중치로 테스트 예측
    test_blend = best_w * test_ridge + (1 - best_w) * test_lgbm
    save_sub(test_blend, 'stacking_4model_ridge_lgbm_blend.csv')

    # 단순 0.5:0.5도 저장
    test_half = 0.5 * test_ridge + 0.5 * test_lgbm
    half_oof  = 0.5 * oof_ridge  + 0.5 * oof_lgbm
    mae_half  = np.mean(np.abs(half_oof - y_raw.values))
    save_sub(test_half, 'stacking_4model_half_blend.csv')

    print(f'\n{"="*60}')
    print('옵션 C 결과 요약')
    print(f'{"="*60}')
    print(f'  [비교] v3 LGBM-meta : CV 8.7929 / Public 10.2264 🏆')
    print(f'  Ridge-meta          : CV {mae_ridge:.4f}')
    print(f'  LGBM-meta (재실행)  : CV {mae_lgbm:.4f}')
    print(f'  최적 블렌드         : CV {best_loss:.4f}  (Ridge={best_w:.2f})')
    print(f'  0.5:0.5 블렌드      : CV {mae_half:.4f}')
    print(f'\n  제출 파일:')
    print(f'    stacking_4model_ridge_lgbm_blend.csv  (최적 블렌드)')
    print(f'    stacking_4model_half_blend.csv         (0.5:0.5)')


if __name__ == '__main__':
    main()
