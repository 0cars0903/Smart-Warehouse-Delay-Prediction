"""
옵션 A-1: 5모델 스태킹 — RandomForest 추가
===========================================
목적 : v3(LGBM+TW1.8+CB+ET) 4모델에 RandomForest를 추가하여
       Layer 1 다양성 강화

RandomForest vs ExtraTrees 차이:
  - ET: 분할 임계값도 완전 무작위 (완전 랜덤)
  - RF: 최적 분할 기반이지만 bootstrap + max_features 서브샘플링
    → ET보다 낮은 분산, 높은 편향 → 다른 오차 패턴 기대

체크포인트 재활용:
  docs/stacking_ckpt/   → lgbm, cb, et
  docs/stacking_v2_ckpt/ → tw18
  docs/stacking_5model_rf_ckpt/ → rf (신규)

예상 시간: ~45분 (RF 5-fold ~35분 + 메타 ~5분)
출력: submissions/stacking_5model_rf_lgbm_meta.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
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
CKPT_RF  = os.path.join(_BASE, '..', 'docs', 'stacking_5model_rf_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

RF_PARAMS = {
    'n_estimators' : 500,
    'max_features' : 0.33,       # ET(0.5)보다 작게 → 더 독립적인 트리
    'min_samples_leaf': 26,      # LGBM min_child_samples와 동일
    'n_jobs'       : -1,
    'random_state' : RANDOM_STATE,
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


def get_rf_oof(train, test, feat_cols, y_log, groups):
    """RandomForest OOF (log1p 공간, NaN→0)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    X_all_train = train[feat_cols].fillna(0).values
    X_all_test  = test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_log, groups)):
        X_tr = X_all_train[tr_idx]
        X_va = X_all_train[va_idx]
        y_tr = y_log.iloc[tr_idx].values
        y_va = y_log.iloc[va_idx].values

        m = RandomForestRegressor(**RF_PARAMS)
        m.fit(X_tr, y_tr)
        oof[va_idx]  = m.predict(X_va)
        test_pred   += m.predict(X_all_test) / N_SPLITS
        mae = np.mean(np.abs(np.expm1(oof[va_idx]) - np.expm1(y_va)))
        print(f'  [RF] Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()

    return oof, test_pred  # log 공간


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
    print('옵션 A-1: 5모델 스태킹 (LGBM+TW1.8+CB+ET+RF → LGBM-meta)')
    print('비교 기준: v3 4모델 CV 8.7929 / Public 10.2264 🏆')
    print('=' * 60)

    os.makedirs(CKPT_RF, exist_ok=True)
    train, test = load_data()
    feat_cols = [c for c in train.columns
                 if c not in {'ID','scenario_id','ts_idx','avg_delay_minutes_next_30m'}
                 and train[c].dtype != object]
    y_raw  = train['avg_delay_minutes_next_30m']
    y_log  = np.log1p(y_raw)
    groups = train['scenario_id']

    # ── Layer 1: 체크포인트 로드 ─────────────────────────────
    print('\n[Layer 1] v3 체크포인트 로드')
    oof_lg, test_lg = load_ckpt('lgbm', CKPT_V1)
    oof_cb, test_cb = load_ckpt('cb',   CKPT_V1)
    oof_et, test_et = load_ckpt('et',   CKPT_V1)
    oof_tw, test_tw = load_ckpt('tw18', CKPT_V2)
    print('  LGBM / CB / ET / TW1.8 로드 완료')

    # RF: 체크포인트 없으면 새로 계산
    rf_oof_path  = os.path.join(CKPT_RF, 'rf_oof.npy')
    rf_test_path = os.path.join(CKPT_RF, 'rf_test.npy')
    if os.path.exists(rf_oof_path) and os.path.exists(rf_test_path):
        print('\n[Layer 1] RF 체크포인트 로드 (재학습 생략)')
        oof_rf, test_rf = np.load(rf_oof_path), np.load(rf_test_path)
    else:
        print('\n[Layer 1] RandomForest OOF 계산 (시간 소요 ~35분)')
        oof_rf, test_rf = get_rf_oof(train, test, feat_cols, y_log, groups)
        save_ckpt('rf', oof_rf, test_rf, CKPT_RF)

    mae_rf = np.mean(np.abs(np.expm1(oof_rf) - y_raw.values))
    print(f'  RF OOF MAE: {mae_rf:.4f}  std={np.expm1(oof_rf).std():.2f}')

    # ── OOF 상관관계 ─────────────────────────────────────────
    oof_raw_lg = np.expm1(oof_lg)
    oof_raw_cb = np.expm1(oof_cb)
    oof_raw_et = np.expm1(oof_et)
    oof_raw_rf = np.expm1(oof_rf)

    print(f'\n  RF 상관관계 (낮을수록 다양성 ↑):')
    print(f'    RF-LGBM : {np.corrcoef(oof_raw_rf, oof_raw_lg)[0,1]:.4f}  (ET-LGBM: 0.9744)')
    print(f'    RF-CB   : {np.corrcoef(oof_raw_rf, oof_raw_cb)[0,1]:.4f}  (ET-CB  : 0.9685)')
    print(f'    RF-TW   : {np.corrcoef(oof_raw_rf, oof_tw)[0,1]:.4f}  (ET-TW  : 0.9438)')
    print(f'    RF-ET   : {np.corrcoef(oof_raw_rf, oof_raw_et)[0,1]:.4f}')

    # ── 5모델 단순 가중치 앙상블 (비교용) ───────────────────
    def loss5(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        blend = (w[0]*oof_raw_lg + w[1]*oof_raw_cb +
                 w[2]*oof_tw + w[3]*oof_raw_et + w[4]*oof_raw_rf)
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
          f'ET={best_w[3]:.3f}, RF={best_w[4]:.3f}')

    # ── Layer 2: 5모델 LGBM-meta ─────────────────────────────
    test_tw_clipped = np.maximum(test_tw, 0)
    meta_train = np.column_stack([oof_lg, oof_cb, np.log1p(oof_tw), oof_et, oof_rf])
    meta_test  = np.column_stack([test_lg, test_cb, np.log1p(test_tw_clipped), test_et, test_rf])

    print('\n[Layer 2] 5모델 LGBM 메타 학습기')
    _, test_meta, mae_meta = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    save_sub(test_meta, 'stacking_5model_rf_lgbm_meta.csv')

    print(f'\n{"="*60}')
    print('옵션 A-1 결과 요약')
    print(f'{"="*60}')
    print(f'  [비교] v3 4모델 : CV 8.7929 / Public 10.2264 🏆')
    print(f'  [결과] 5모델+RF : CV {mae_meta:.4f}')
    print(f'  RF OOF MAE     : {mae_rf:.4f}  (ET: 9.0013)')
    print(f'  5모델 가중 앙상블: {best_loss:.4f}  (4모델: 8.8546)')


if __name__ == '__main__':
    main()
