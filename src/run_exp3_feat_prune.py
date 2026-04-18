"""
실험 3: 피처 중요도 하위 컷 (일반화 개선)
============================================
목적 : 296개 피처에서 LGBM importance 하위 k%를 제거해
       과적합 유발 피처를 제거 → CV→Public 갭 개선

전략 :
  1. 현재 최고 파이프라인(ensemble_ts0 기반) 그대로 5-fold 학습
  2. 5 fold LGBM importance 평균 계산
  3. 하위 threshold(%)별 제거 후 5-fold CV MAE 비교
     - 5%, 10%, 15% 세 가지 cutoff 순차 실험
  4. 최적 cutoff로 최종 제출 CSV 생성

결과 파일 : submissions/feat_prune_top{pct}pct_LGBM_CB.csv
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

# ─── 공통 상수 ───────────────────────────────────────────────
_BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_BASE, '..', 'data')
SUB_DIR    = os.path.join(_BASE, '..', 'submissions')
DOCS_DIR   = os.path.join(_BASE, '..', 'docs')
N_SPLITS   = 5
RANDOM_STATE = 42

# 현재 최고 파라미터 (CLAUDE.md 6절 + run_ensemble_ts0 기준)
BEST_LGBM_PARAMS = {
    'num_leaves': 183, 'learning_rate': 0.020703,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

BEST_CB_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3.0,
    'bagging_temperature': 0.5,
    'loss_function': 'MAE', 'eval_metric': 'MAE',
    'random_seed': RANDOM_STATE, 'verbose': 0,
    'early_stopping_rounds': 100,
}

PRUNE_THRESHOLDS = [5, 10, 15]   # 하위 제거 비율(%)


# ─── 데이터 로드 & 피처 엔지니어링 ───────────────────────────
def load_data():
    train = pd.read_csv(f'{DATA_DIR}/train.csv')
    test  = pd.read_csv(f'{DATA_DIR}/test.csv')
    layout = pd.read_csv(f'{DATA_DIR}/layout_info.csv')
    # 284피처 기준선: lag(1~6) + rolling(3,5,10)
    train, test = build_features(
        train, test, layout,
        lag_lags=[1, 2, 3, 4, 5, 6],
        rolling_windows=[3, 5, 10],
    )
    return train, test


# ─── 타깃 변환 (log1p) ───────────────────────────────────────
def transform_target(y):   return np.log1p(y)
def inverse_target(y):     return np.expm1(y)


# ─── OOF 예측 함수 ───────────────────────────────────────────
def oof_lgbm(X, y, groups, feat_cols, params):
    gkf   = GroupKFold(n_splits=N_SPLITS)
    oof   = np.zeros(len(y))
    imps  = np.zeros(len(feat_cols))
    models = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr_idx][feat_cols], X.iloc[va_idx][feat_cols]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        oof[va_idx] = m.predict(X_va)
        imps += m.feature_importances_
        models.append(m)
        mae = np.mean(np.abs(inverse_target(oof[va_idx]) - inverse_target(y_va)))
        print(f'  LGBM fold {fold+1} MAE: {mae:.4f}')

    imps /= N_SPLITS
    return oof, imps, models


def oof_cb(X, y, groups, feat_cols, params):
    gkf  = GroupKFold(n_splits=N_SPLITS)
    oof  = np.zeros(len(y))
    models = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr_idx][feat_cols], X.iloc[va_idx][feat_cols]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        m = cb.CatBoostRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        oof[va_idx] = m.predict(X_va)
        models.append(m)
        mae = np.mean(np.abs(inverse_target(oof[va_idx]) - inverse_target(y_va)))
        print(f'  CB   fold {fold+1} MAE: {mae:.4f}')

    return oof, models


# ─── 최적 가중치 탐색 ─────────────────────────────────────────
def find_weights(oofs, y_true, n=2):
    """oofs: list of n OOF arrays (원공간), y_true: 원공간"""
    def loss(w):
        w = np.array(w)
        w = np.abs(w) / np.abs(w).sum()
        blend = sum(wi * o for wi, o in zip(w, oofs))
        return np.mean(np.abs(blend - y_true))

    best_loss, best_w = np.inf, None
    for _ in range(200):
        w0 = np.random.dirichlet(np.ones(n))
        res = minimize(loss, w0, method='Nelder-Mead')
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = np.abs(res.x) / np.abs(res.x).sum()
    return best_w, best_loss


# ─── 메인 ─────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('실험 3 : 피처 중요도 하위 컷 (일반화 개선)')
    print('=' * 60)

    train, test = load_data()
    drop_cols = ['ID', 'scenario_id', 'ts_idx', 'avg_delay_minutes_next_30m']
    feat_all   = [c for c in train.columns
                  if c not in drop_cols and train[c].dtype != object]
    obj_cols = [c for c in train.columns
                if c not in drop_cols and train[c].dtype == object]
    if obj_cols:
        print(f'  ※ object 타입 컬럼 제외: {obj_cols}')

    y_raw    = train['avg_delay_minutes_next_30m']
    y        = transform_target(y_raw)
    groups   = train['scenario_id']

    # ── Step 1: 전체 피처로 LGBM 학습 → importance 수집 ──────
    print(f'\n[Step 1] 전체 피처({len(feat_all)}개) 기준 LGBM importance 계산')
    oof_lgbm_full, importances, _ = oof_lgbm(train, y, groups, feat_all, BEST_LGBM_PARAMS)
    mae_full = np.mean(np.abs(inverse_target(oof_lgbm_full) - y_raw))
    print(f'  전체 피처 LGBM CV MAE: {mae_full:.4f}')

    # Importance 분포 출력
    imp_df = pd.DataFrame({'feature': feat_all, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    print(f'\n  상위 10개 피처:')
    print(imp_df.head(10).to_string(index=False))
    print(f'\n  하위 10개 피처:')
    print(imp_df.tail(10).to_string(index=False))
    imp_df.to_csv(os.path.join(DOCS_DIR, 'feature_importance_exp3.csv'), index=False)
    print(f'  → 전체 importance 저장: docs/feature_importance_exp3.csv')

    # ── Step 2: 각 cutoff별 LGBM+CB 앙상블 실험 ─────────────
    results = []

    for pct in PRUNE_THRESHOLDS:
        n_keep   = int(len(feat_all) * (1 - pct / 100))
        feat_sel = imp_df['feature'].iloc[:n_keep].tolist()
        print(f'\n{"─"*50}')
        print(f'[Cutoff {pct}%] 제거 피처 {len(feat_all)-n_keep}개 → 남은 피처 {n_keep}개')

        # LGBM
        print('  LGBM 학습...')
        oof_lg, _, _ = oof_lgbm(train, y, groups, feat_sel, BEST_LGBM_PARAMS)
        # CB
        print('  CatBoost 학습...')
        oof_cb_arr, _ = oof_cb(train, y, groups, feat_sel, BEST_CB_PARAMS)

        # 원공간 변환
        oof_lg_raw = inverse_target(oof_lg)
        oof_cb_raw = inverse_target(oof_cb_arr)

        # 최적 가중치
        w, blend_mae = find_weights([oof_lg_raw, oof_cb_raw], y_raw.values)
        print(f'  최적 가중치: LGBM={w[0]:.3f}, CB={w[1]:.3f}')
        print(f'  블렌딩 CV MAE: {blend_mae:.4f}')

        # 테스트 예측
        lgbm_models_sel = []
        cb_models_sel   = []

        gkf = GroupKFold(n_splits=N_SPLITS)
        lgbm_test_preds = np.zeros(len(test))
        cb_test_preds   = np.zeros(len(test))

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups)):
            # LGBM
            m_lg = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
            m_lg.fit(
                train.iloc[tr_idx][feat_sel], y.iloc[tr_idx],
                eval_set=[(train.iloc[va_idx][feat_sel], y.iloc[va_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            lgbm_test_preds += inverse_target(m_lg.predict(test[feat_sel])) / N_SPLITS

            # CB
            m_cb = cb.CatBoostRegressor(**BEST_CB_PARAMS)
            m_cb.fit(
                train.iloc[tr_idx][feat_sel], y.iloc[tr_idx],
                eval_set=(train.iloc[va_idx][feat_sel], y.iloc[va_idx]),
            )
            cb_test_preds += inverse_target(m_cb.predict(test[feat_sel])) / N_SPLITS

        test_preds = w[0] * lgbm_test_preds + w[1] * cb_test_preds
        test_preds = np.maximum(test_preds, 0)

        # 제출 파일 저장
        sub_name = f'feat_prune_bot{pct}pct_LGBM_CB.csv'
        sub_path = f'{SUB_DIR}/{sub_name}'
        sample_sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
        sample_sub['avg_delay_minutes_next_30m'] = test_preds
        sample_sub.to_csv(sub_path, index=False)
        print(f'  → 제출 파일 저장: submissions/{sub_name}')

        results.append({
            'cutoff_pct': pct,
            'n_features': n_keep,
            'cv_mae': blend_mae,
            'w_lgbm': w[0],
            'w_cb': w[1],
            'submission': sub_name,
        })
        gc.collect()

    # ── 최종 결과 요약 ────────────────────────────────────────
    print(f'\n{"="*60}')
    print('실험 3 결과 요약')
    print(f'{"="*60}')
    print(f'  기준(전체 {len(feat_all)}개 LGBM 단독) CV MAE: {mae_full:.4f}')
    print()
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    best = res_df.loc[res_df['cv_mae'].idxmin()]
    print(f'\n  ★ 최적 cutoff: 하위 {best.cutoff_pct}% 제거')
    print(f'    피처 수  : {best.n_features}개')
    print(f'    CV MAE   : {best.cv_mae:.4f}')
    print(f'    제출 파일: {best.submission}')


if __name__ == '__main__':
    main()
