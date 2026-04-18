"""
옵션 A-2: 5모델 스태킹 — Quantile(q=0.5) LGBM 추가
====================================================
목적 : v3(LGBM+TW1.8+CB+ET)에 LGBM Quantile(q=0.5)을 추가하여
       Layer 1 다양성 강화

Quantile(q=0.5) vs L1 LGBM 차이:
  - L1(MAE) LGBM: 평균으로의 수렴 → 현재 베이스라인
  - Quantile(q=0.5, 중앙값 회귀): L1과 비슷하지만 분위수 기반
    → 미묘하게 다른 수렴점 → 다른 오차 패턴 기대
  - q=0.5 단독 OOF MAE 8.9084 (실험 2에서 확인)

체크포인트 재활용:
  docs/stacking_ckpt/   → lgbm, cb, et
  docs/stacking_v2_ckpt/ → tw18
  docs/stacking_5model_q05_ckpt/ → q05 (신규)

예상 시간: ~25분 (Q05 5-fold ~20분 + 메타 ~5분)
출력: submissions/stacking_5model_q05_lgbm_meta.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
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
CKPT_Q05 = os.path.join(_BASE, '..', 'docs', 'stacking_5model_q05_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# Quantile q=0.5 (중앙값 회귀) — raw 타깃 사용
Q05_PARAMS = {
    'num_leaves'       : 183,
    'learning_rate'    : 0.020703,
    'feature_fraction' : 0.5122,
    'bagging_fraction' : 0.9049,
    'min_child_samples': 26,
    'reg_alpha'        : 0.3805,
    'reg_lambda'       : 0.3630,
    'objective'        : 'quantile',
    'alpha'            : 0.5,
    'metric'           : 'quantile',
    'n_estimators'     : 3000,
    'bagging_freq'     : 1,
    'random_state'     : RANDOM_STATE,
    'verbosity'        : -1,
    'n_jobs'           : -1,
}

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


def load_ckpt(name, ckpt_dir):
    oof  = np.load(os.path.join(ckpt_dir, f'{name}_oof.npy'))
    test = np.load(os.path.join(ckpt_dir, f'{name}_test.npy'))
    return oof, test


def save_ckpt(name, oof, test_pred, ckpt_dir):
    np.save(os.path.join(ckpt_dir, f'{name}_oof.npy'),  oof)
    np.save(os.path.join(ckpt_dir, f'{name}_test.npy'), test_pred)


def get_q05_oof(train, test, feat_cols, y_raw, groups):
    """Quantile(q=0.5) LGBM OOF — raw 타깃, early stopping 없음(quantile 불안정)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_raw, groups)):
        X_tr = train.iloc[tr_idx][feat_cols]
        X_va = train.iloc[va_idx][feat_cols]
        y_tr = y_raw.iloc[tr_idx]
        y_va = y_raw.iloc[va_idx]

        m = lgb.LGBMRegressor(**Q05_PARAMS)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[va_idx]  = np.maximum(m.predict(X_va), 0)
        test_pred   += np.maximum(m.predict(test[feat_cols]), 0) / N_SPLITS
        mae = np.mean(np.abs(oof[va_idx] - y_va.values))
        print(f'  [Q05] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}  '
              f'std={oof[va_idx].std():.2f}')
        del m; gc.collect()

    return oof, test_pred  # raw 공간


def run_meta_lgbm(meta_train, meta_test, y_raw, groups, label='LGBM-meta'):
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
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_mae = np.mean(np.abs(oof_meta - y_raw.values))
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | std={oof_meta.std():.2f}')
    return oof_meta, test_meta, oof_mae


def main():
    print('=' * 60)
    print('옵션 A-2: 5모델 스태킹 (LGBM+TW1.8+CB+ET+Q05 → LGBM-meta)')
    print('비교 기준: v3 4모델 CV 8.7929 / Public 10.2264 🏆')
    print('=' * 60)

    os.makedirs(CKPT_Q05, exist_ok=True)
    train, test = load_data()
    feat_cols = [c for c in train.columns
                 if c not in {'ID','scenario_id','ts_idx','avg_delay_minutes_next_30m'}
                 and train[c].dtype != object]
    y_raw  = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']

    # ── Layer 1: 체크포인트 로드 ─────────────────────────────
    print('\n[Layer 1] v3 체크포인트 로드')
    oof_lg, test_lg = load_ckpt('lgbm', CKPT_V1)
    oof_cb, test_cb = load_ckpt('cb',   CKPT_V1)
    oof_et, test_et = load_ckpt('et',   CKPT_V1)
    oof_tw, test_tw = load_ckpt('tw18', CKPT_V2)
    print('  LGBM / CB / ET / TW1.8 로드 완료')

    # Q05: 체크포인트 없으면 새로 계산
    q05_oof_path  = os.path.join(CKPT_Q05, 'q05_oof.npy')
    q05_test_path = os.path.join(CKPT_Q05, 'q05_test.npy')
    if os.path.exists(q05_oof_path) and os.path.exists(q05_test_path):
        print('\n[Layer 1] Q05 체크포인트 로드 (재학습 생략)')
        oof_q05, test_q05 = np.load(q05_oof_path), np.load(q05_test_path)
    else:
        print('\n[Layer 1] Quantile(q=0.5) OOF 계산 (~20분)')
        oof_q05, test_q05 = get_q05_oof(train, test, feat_cols, y_raw, groups)
        save_ckpt('q05', oof_q05, test_q05, CKPT_Q05)

    mae_q05 = np.mean(np.abs(oof_q05 - y_raw.values))
    print(f'  Q05 OOF MAE: {mae_q05:.4f}  std={oof_q05.std():.2f}')

    # ── OOF 상관관계 ─────────────────────────────────────────
    oof_raw_lg = np.expm1(oof_lg)
    oof_raw_cb = np.expm1(oof_cb)
    oof_raw_et = np.expm1(oof_et)

    print(f'\n  Q05 상관관계 (낮을수록 다양성 ↑):')
    print(f'    Q05-LGBM: {np.corrcoef(oof_q05, oof_raw_lg)[0,1]:.4f}')
    print(f'    Q05-CB  : {np.corrcoef(oof_q05, oof_raw_cb)[0,1]:.4f}')
    print(f'    Q05-TW  : {np.corrcoef(oof_q05, oof_tw)[0,1]:.4f}')
    print(f'    Q05-ET  : {np.corrcoef(oof_q05, oof_raw_et)[0,1]:.4f}')

    # ── 5모델 단순 가중치 앙상블 (비교용) ───────────────────
    def loss5(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        blend = (w[0]*oof_raw_lg + w[1]*oof_raw_cb +
                 w[2]*oof_tw + w[3]*oof_raw_et + w[4]*oof_q05)
        return np.mean(np.abs(blend - y_raw.values))

    best_loss, best_w = np.inf, np.ones(5)/5
    for _ in range(500):
        w0 = np.random.dirichlet(np.ones(5))
        res = minimize(loss5, w0, method='Nelder-Mead')
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = np.abs(res.x) / np.abs(res.x).sum()
    print(f'\n  5모델 가중치 앙상블 CV MAE: {best_loss:.4f}')
    print(f'    LGBM={best_w[0]:.3f}, CB={best_w[1]:.3f}, TW={best_w[2]:.3f}, '
          f'ET={best_w[3]:.3f}, Q05={best_w[4]:.3f}')

    # ── Layer 2: 5모델 LGBM-meta ─────────────────────────────
    # 메타 피처: 모두 log 공간 (Q05는 raw → log1p 변환)
    test_tw_clipped  = np.maximum(test_tw,  0)
    test_q05_clipped = np.maximum(test_q05, 0)
    meta_train = np.column_stack([
        oof_lg, oof_cb, np.log1p(oof_tw), oof_et, np.log1p(oof_q05)])
    meta_test  = np.column_stack([
        test_lg, test_cb, np.log1p(test_tw_clipped),
        test_et, np.log1p(test_q05_clipped)])

    print('\n[Layer 2] 5모델 LGBM 메타 학습기')
    _, test_meta, mae_meta = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    save_sub(test_meta, 'stacking_5model_q05_lgbm_meta.csv')

    print(f'\n{"="*60}')
    print('옵션 A-2 결과 요약')
    print(f'{"="*60}')
    print(f'  [비교] v3 4모델  : CV 8.7929 / Public 10.2264 🏆')
    print(f'  [결과] 5모델+Q05 : CV {mae_meta:.4f}')
    print(f'  Q05 OOF MAE      : {mae_q05:.4f}  (기존 실험2: 8.9084)')
    print(f'  5모델 가중 앙상블 : {best_loss:.4f}  (4모델: 8.8546)')


if __name__ == '__main__':
    main()
