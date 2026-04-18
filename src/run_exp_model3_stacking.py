"""
모델 실험 3: ExtraTrees + 메타 스태킹
========================================
목적 : GBDT와 오차 패턴이 다른 ExtraTreesRegressor를 추가하고
       OOF 예측을 메타 피처로 활용하는 2-layer 스태킹

구조 :
  Layer 1 (Base Models, 각각 GroupKFold 5-fold OOF 생성)
    ├─ LightGBM    (log1p, 현재 최고 파라미터)
    ├─ CatBoost    (log1p, Optuna 파라미터)
    └─ ExtraTrees  (log1p, sklearn — 무작위 split으로 GBDT와 오차 독립)

  Layer 2 (Meta Learner, 5-fold OOF 다시 학습)
    ├─ Ridge Regression  (빠른 선형 보정)
    └─ LightGBM          (비선형 메타 학습)
    → 더 좋은 쪽 선택

왜 ExtraTrees?
  - 완전 무작위 split (feature + threshold 모두 무작위)
  - LGBM/CB의 greedy split과 오차 상관이 낮음 → 앙상블 다양성 ↑
  - sklearn 구현이라 Tweedie/Quantile 없이도 다른 패턴 학습 가능
  - 단독 성능은 GBDT보다 낮지만 스태킹 기여도가 높음

출력 :
  submissions/stacking_ridge_meta.csv    (Ridge 메타)
  submissions/stacking_lgbm_meta.csv     (LightGBM 메타)

예상 실행 시간: ~60분 (ExtraTrees 5-fold가 가장 오래 걸림)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import warnings, gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features

warnings.filterwarnings('ignore')

# ─── 상수 ────────────────────────────────────────────────────
_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
N_SPLITS     = 5
N_SPLITS_META = 5
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

ET_PARAMS = {
    'n_estimators': 500,
    'max_features': 0.5,        # LGBM feature_fraction과 유사
    'min_samples_leaf': 26,     # LGBM min_child_samples와 동일
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
}

META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
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


def save_sub(preds, filename):
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(preds, 0)
    sample.to_csv(os.path.join(SUB_DIR, filename), index=False)
    print(f'  → 저장: submissions/{filename}')


# ─── Layer 1: Base Model OOF 생성 ─────────────────────────────
def get_lgbm_oof(train, test, feat_cols, y_log, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_log, groups)):
        X_tr, X_va = train.iloc[tr_idx][feat_cols], train.iloc[va_idx][feat_cols]
        y_tr, y_va = y_log.iloc[tr_idx], y_log.iloc[va_idx]

        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[va_idx]  = m.predict(X_va)
        test_pred   += m.predict(test[feat_cols]) / N_SPLITS
        print(f'  [LGBM] Fold {fold+1}  '
              f'MAE={np.mean(np.abs(np.expm1(oof[va_idx]) - np.expm1(y_va))):.4f}')
        gc.collect()

    return oof, test_pred   # log 공간 그대로 반환


def get_cb_oof(train, test, feat_cols, y_log, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_log, groups)):
        X_tr, X_va = train.iloc[tr_idx][feat_cols], train.iloc[va_idx][feat_cols]
        y_tr, y_va = y_log.iloc[tr_idx], y_log.iloc[va_idx]

        m = cb.CatBoostRegressor(**CB_PARAMS)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        oof[va_idx]  = m.predict(X_va)
        test_pred   += m.predict(test[feat_cols]) / N_SPLITS
        print(f'  [CB]   Fold {fold+1}  '
              f'MAE={np.mean(np.abs(np.expm1(oof[va_idx]) - np.expm1(y_va))):.4f}')
        gc.collect()

    return oof, test_pred   # log 공간


def get_et_oof(train, test, feat_cols, y_log, groups):
    """ExtraTrees: 무작위 split으로 GBDT와 다른 오차 패턴 형성
    NaN 처리: lag/rolling 피처의 초반 슬롯 NaN → 0으로 대체
    (시나리오 초반 = 이전 이력 없음 → 0이 자연스러운 대체값)
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    # NaN → 0 (lag/rolling 피처: 이전 이력 없음을 0으로 표현)
    X_all_train = train[feat_cols].fillna(0).values
    X_all_test  = test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_log, groups)):
        X_tr = X_all_train[tr_idx]
        X_va = X_all_train[va_idx]
        y_tr = y_log.iloc[tr_idx].values
        y_va = y_log.iloc[va_idx].values

        m = ExtraTreesRegressor(**ET_PARAMS)
        m.fit(X_tr, y_tr)
        oof[va_idx]  = m.predict(X_va)
        test_pred   += m.predict(X_all_test) / N_SPLITS
        mae = np.mean(np.abs(np.expm1(oof[va_idx]) - np.expm1(y_va)))
        print(f'  [ET]   Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()

    return oof, test_pred   # log 공간


# ─── Layer 2: 메타 학습기 ──────────────────────────────────────
def run_meta_ridge(meta_train, meta_test, y_raw, groups, base_names):
    """Ridge: OOF 3개를 피처로 선형 결합 학습"""
    gkf = GroupKFold(n_splits=N_SPLITS_META)
    oof_meta = np.zeros(len(y_raw))
    test_meta = np.zeros(meta_test.shape[0])

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
        y_tr = y_raw.iloc[tr_idx].values

        # Ridge는 log 공간에서 학습, expm1 후 MAE 평가
        m = Ridge(alpha=1.0, fit_intercept=True)
        m.fit(X_tr, np.log1p(y_tr))
        oof_meta[va_idx] = np.expm1(m.predict(X_va))
        test_meta       += np.expm1(m.predict(meta_test)) / N_SPLITS_META

        mae = np.mean(np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values))
        coefs = {n: f'{c:.3f}' for n, c in zip(base_names, m.coef_)}
        print(f'  [Ridge-meta] Fold {fold+1}  MAE={mae:.4f}  coef={coefs}')

    oof_mae = np.mean(np.abs(oof_meta - y_raw.values))
    print(f'  [Ridge-meta] OOF MAE={oof_mae:.4f} | '
          f'std={oof_meta.std():.2f}, max={oof_meta.max():.2f}')
    return oof_meta, test_meta, oof_mae


def run_meta_lgbm(meta_train, meta_test, y_raw, groups):
    """LightGBM 메타: 비선형 보정 허용"""
    gkf = GroupKFold(n_splits=N_SPLITS_META)
    oof_meta = np.zeros(len(y_raw))
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
        test_meta       += np.expm1(m.predict(meta_test)) / N_SPLITS_META

        mae = np.mean(np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values))
        print(f'  [LGBM-meta] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_mae = np.mean(np.abs(oof_meta - y_raw.values))
    print(f'  [LGBM-meta] OOF MAE={oof_mae:.4f} | '
          f'std={oof_meta.std():.2f}, max={oof_meta.max():.2f}')
    return oof_meta, test_meta, oof_mae


# ─── 메인 ─────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('모델 실험 3: ExtraTrees + 메타 스태킹')
    print('=' * 60)

    train, test = load_data()
    feat_cols = get_feat_cols(train)
    print(f'피처 수: {len(feat_cols)}개')

    y_raw  = train['avg_delay_minutes_next_30m']
    y_log  = np.log1p(y_raw)
    groups = train['scenario_id']

    # ── Layer 1: Base Model OOF (체크포인트 활용) ─────────────
    ckpt_dir = os.path.join(_BASE, '..', 'docs', 'stacking_ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    def load_or_run(name, run_fn):
        oof_path  = os.path.join(ckpt_dir, f'{name}_oof.npy')
        test_path = os.path.join(ckpt_dir, f'{name}_test.npy')
        if os.path.exists(oof_path) and os.path.exists(test_path):
            print(f'  [{name}] 체크포인트 로드 (재학습 생략)')
            return np.load(oof_path), np.load(test_path)
        oof, test_pred = run_fn()
        np.save(oof_path, oof)
        np.save(test_path, test_pred)
        return oof, test_pred

    print('\n[Layer 1] LightGBM OOF 생성')
    oof_lg, test_lg = load_or_run(
        'lgbm', lambda: get_lgbm_oof(train, test, feat_cols, y_log, groups))
    mae_lg = np.mean(np.abs(np.expm1(oof_lg) - y_raw.values))
    print(f'  LGBM OOF MAE: {mae_lg:.4f}  std={np.expm1(oof_lg).std():.2f}')

    print('\n[Layer 1] CatBoost OOF 생성')
    oof_cb, test_cb = load_or_run(
        'cb', lambda: get_cb_oof(train, test, feat_cols, y_log, groups))
    mae_cb = np.mean(np.abs(np.expm1(oof_cb) - y_raw.values))
    print(f'  CB   OOF MAE: {mae_cb:.4f}  std={np.expm1(oof_cb).std():.2f}')

    print('\n[Layer 1] ExtraTrees OOF 생성 (시간 소요)')
    oof_et, test_et = load_or_run(
        'et', lambda: get_et_oof(train, test, feat_cols, y_log, groups))
    mae_et = np.mean(np.abs(np.expm1(oof_et) - y_raw.values))
    print(f'  ET   OOF MAE: {mae_et:.4f}  std={np.expm1(oof_et).std():.2f}')

    # OOF 상관관계 확인 (낮을수록 스태킹 효과 큼)
    oof_raw_lg = np.expm1(oof_lg)
    oof_raw_cb = np.expm1(oof_cb)
    oof_raw_et = np.expm1(oof_et)
    corr_lg_cb = np.corrcoef(oof_raw_lg, oof_raw_cb)[0,1]
    corr_lg_et = np.corrcoef(oof_raw_lg, oof_raw_et)[0,1]
    corr_cb_et = np.corrcoef(oof_raw_cb, oof_raw_et)[0,1]
    print(f'\n  OOF 상관관계 (낮을수록 스태킹 효과 ↑):')
    print(f'    LGBM-CB: {corr_lg_cb:.4f}')
    print(f'    LGBM-ET: {corr_lg_et:.4f}')
    print(f'    CB-ET  : {corr_cb_et:.4f}')

    # ── 단순 최적 가중치 앙상블 (Layer 1만) ──────────────────
    def loss3(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        blend = w[0]*oof_raw_lg + w[1]*oof_raw_cb + w[2]*oof_raw_et
        return np.mean(np.abs(blend - y_raw.values))

    best_loss1, best_w1 = np.inf, np.array([1/3, 1/3, 1/3])
    for _ in range(300):
        w0 = np.random.dirichlet(np.ones(3))
        res = minimize(loss3, w0, method='Nelder-Mead')
        if res.fun < best_loss1:
            best_loss1 = res.fun
            best_w1 = np.abs(res.x) / np.abs(res.x).sum()
    print(f'\n  단순 3모델 가중치 앙상블 CV MAE: {best_loss1:.4f}')
    print(f'    가중치: LGBM={best_w1[0]:.3f}, CB={best_w1[1]:.3f}, ET={best_w1[2]:.3f}')

    # ── Layer 2: 메타 학습기 ─────────────────────────────────
    # 메타 피처: [LGBM_OOF, CB_OOF, ET_OOF] → log 공간 그대로 사용
    meta_train_feat = np.column_stack([oof_lg, oof_cb, oof_et])
    meta_test_feat  = np.column_stack([test_lg, test_cb, test_et])
    base_names = ['LGBM', 'CB', 'ET']

    print('\n[Layer 2] Ridge 메타 학습기')
    _, test_ridge, mae_ridge = run_meta_ridge(
        meta_train_feat, meta_test_feat, y_raw, groups, base_names)

    print('\n[Layer 2] LightGBM 메타 학습기')
    _, test_lgbm_meta, mae_lgbm_meta = run_meta_lgbm(
        meta_train_feat, meta_test_feat, y_raw, groups)

    # ── 제출 파일 저장 ────────────────────────────────────────
    save_sub(test_ridge,     'stacking_ridge_meta.csv')
    save_sub(test_lgbm_meta, 'stacking_lgbm_meta.csv')

    # ── 최종 요약 ─────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('실험 3 결과 요약')
    print(f'{"="*60}')
    print(f'  [Layer 1] LGBM  단독   OOF MAE: {mae_lg:.4f}')
    print(f'  [Layer 1] CB    단독   OOF MAE: {mae_cb:.4f}')
    print(f'  [Layer 1] ET    단독   OOF MAE: {mae_et:.4f}  ← GBDT 대비 차이 확인')
    print(f'  [Layer 1] 3모델 가중치 CV MAE: {best_loss1:.4f}')
    print(f'  [Layer 2] Ridge 메타   CV MAE: {mae_ridge:.4f}')
    print(f'  [Layer 2] LGBM 메타    CV MAE: {mae_lgbm_meta:.4f}')
    print(f'\n  (참고) 현재 Public 최고: 8.8674 / 10.3347')
    print(f'  핵심 지표: LGBM-ET 상관계수({corr_lg_et:.4f})가 낮을수록 스태킹 효과 ↑')
    print(f'  판단 기준: 메타 MAE < {min(mae_lg, mae_cb):.4f}이면 스태킹 유효')


if __name__ == '__main__':
    main()
