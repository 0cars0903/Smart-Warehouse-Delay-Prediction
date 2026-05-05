"""
run_exp_model10_histgb.py  —  FE v1 + HistGradientBoosting 6모델 스태킹
===============================================================
배경 (04.15 최종 결론):
  FE 확장 방향 전체 차단 (Cumulative/KEY_COLS_V2/Delta/ExtLag 모두 배율 1.170)
  현재 최강: RF 5모델 (FE v1) — CV 8.7911 / Public 10.2213 / 배율 1.1627

탐색 가설:
  sklearn HistGradientBoostingRegressor를 6번째 베이스 모델로 추가.
  HGB 특성:
    - LightGBM과 유사하지만 독립적인 구현 → 다른 오차 패턴 가능성
    - NaN 자체 처리 (ET/RF처럼 fillna 불필요)
    - 고유 파라미터 공간 (max_leaf_nodes, l2_regularization 등)

체크포인트 재활용 전략:
  model8과 동일한 형식의 OOF 로드:
    docs/stacking_ckpt/   → lgbm_oof, cb_oof, et_oof (log 공간)
    docs/stacking_v2_ckpt/ → tw18_oof (log 공간)
    docs/stacking_5model_rf_ckpt/ → rf_oof (log 공간)
  HGB만 새로 GroupKFold 학습 → docs/histgb_ckpt/ 저장

예상 시간: ~30분 (HGB만 신규)
===============================================================
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings, gc, os, sys
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_V1  = os.path.join(_BASE, '..', 'docs', 'stacking_ckpt')
CKPT_V2  = os.path.join(_BASE, '..', 'docs', 'stacking_v2_ckpt')
CKPT_RF  = os.path.join(_BASE, '..', 'docs', 'stacking_5model_rf_ckpt')
CKPT_HGB = os.path.join(_BASE, '..', 'docs', 'histgb_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# HGB 파라미터 (MAE 목적, NaN 자체 처리)
HGB_PARAMS = {
    'max_iter'         : 500,
    'max_leaf_nodes'   : 31,
    'learning_rate'    : 0.05,
    'l2_regularization': 1.0,
    'min_samples_leaf' : 26,
    'loss'             : 'absolute_error',
    'random_state'     : RANDOM_STATE,
    'early_stopping'   : False,   # GroupKFold 환경 — 내부 검증 비활성
}

META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 1000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}


# ──────────────────────────────────────────────
# 데이터 로드 (model8과 동일한 FE v1)
# ──────────────────────────────────────────────
def load_data():
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(
        train, test, layout,
        lag_lags=[1, 2, 3, 4, 5, 6],
        rolling_windows=[3, 5, 10],
    )
    return train, test


# ──────────────────────────────────────────────
# HGB GroupKFold 학습 (log1p 공간)
# ──────────────────────────────────────────────
def get_hgb_oof(train, test, feat_cols, y_log, groups):
    """HistGradientBoosting OOF — log1p 공간에서 학습"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    X_tr_all = train[feat_cols].values   # HGB는 NaN 자체 처리
    X_te_all = test[feat_cols].values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_log, groups)):
        X_tr = X_tr_all[tr_idx]
        X_va = X_tr_all[va_idx]
        y_tr = y_log.iloc[tr_idx].values
        y_va = y_log.iloc[va_idx].values

        m = HistGradientBoostingRegressor(**HGB_PARAMS)
        m.fit(X_tr, y_tr)
        oof[va_idx]  = m.predict(X_va)
        test_pred   += m.predict(X_te_all) / N_SPLITS

        mae_raw = mean_absolute_error(
            np.expm1(y_va), np.expm1(oof[va_idx])
        )
        print(f'  [HGB] Fold {fold+1}  MAE={mae_raw:.4f}')
        del m; gc.collect()

    oof_mae = mean_absolute_error(
        np.expm1(y_log.values), np.expm1(oof)
    )
    print(f'  [HGB] 전체 OOF MAE={oof_mae:.4f}  pred_std={np.expm1(oof).std():.2f}')
    return oof, test_pred   # log 공간 반환


# ──────────────────────────────────────────────
# 메타 LGBM (model8과 동일 방식 — log 입력, expm1 출력)
# ──────────────────────────────────────────────
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
        mae = mean_absolute_error(oof_meta[va_idx], y_raw.iloc[va_idx].values)
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_mae = mean_absolute_error(oof_meta, y_raw.values)
    pred_std = np.std(oof_meta)
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | pred_std={pred_std:.2f}')
    return oof_meta, test_meta, oof_mae


def main():
    print('=' * 60)
    print('FE v1 + HistGradientBoosting 6모델 스태킹')
    print('체크포인트 재활용: LGBM+CB+ET+TW1.8+RF (model8 동일)')
    print('신규 학습: HGB만 (약 30분)')
    print('=' * 60)

    os.makedirs(CKPT_HGB, exist_ok=True)

    train, test = load_data()
    feat_cols = [c for c in train.columns
                 if c not in {'ID', 'scenario_id', 'ts_idx', 'avg_delay_minutes_next_30m'}
                 and train[c].dtype != object]
    y_raw  = train['avg_delay_minutes_next_30m']
    y_log  = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'피처 수: {len(feat_cols)}')

    # ── Layer 1: 기존 5모델 체크포인트 로드 ──────────────────
    print('\n[Layer 1] 기존 5모델 체크포인트 로드...')
    oof_lg  = np.load(os.path.join(CKPT_V1,  'lgbm_oof.npy'))
    test_lg = np.load(os.path.join(CKPT_V1,  'lgbm_test.npy'))
    oof_cb  = np.load(os.path.join(CKPT_V1,  'cb_oof.npy'))
    test_cb = np.load(os.path.join(CKPT_V1,  'cb_test.npy'))
    oof_et  = np.load(os.path.join(CKPT_V1,  'et_oof.npy'))
    test_et = np.load(os.path.join(CKPT_V1,  'et_test.npy'))
    oof_tw  = np.load(os.path.join(CKPT_V2,  'tw18_oof.npy'))
    test_tw = np.load(os.path.join(CKPT_V2,  'tw18_test.npy'))
    oof_rf  = np.load(os.path.join(CKPT_RF,  'rf_oof.npy'))
    test_rf = np.load(os.path.join(CKPT_RF,  'rf_test.npy'))
    print('  LGBM / CB / ET / TW1.8 / RF 로드 완료')

    for name, oof in [('LGBM', oof_lg),('CB', oof_cb),('ET', oof_et),
                       ('TW1.8', oof_tw),('RF', oof_rf)]:
        mae_raw = mean_absolute_error(y_raw.values, np.expm1(oof))
        print(f'  {name:<6} OOF MAE={mae_raw:.4f}')

    # ── Layer 1: HGB (체크포인트 없으면 학습) ─────────────────
    hgb_oof_path  = os.path.join(CKPT_HGB, 'hgb_oof.npy')
    hgb_test_path = os.path.join(CKPT_HGB, 'hgb_test.npy')

    if os.path.exists(hgb_oof_path) and os.path.exists(hgb_test_path):
        print('\n[Layer 1] HGB 체크포인트 로드 (재학습 생략)')
        oof_hgb  = np.load(hgb_oof_path)
        test_hgb = np.load(hgb_test_path)
        mae_raw = mean_absolute_error(y_raw.values, np.expm1(oof_hgb))
        print(f'  HGB    OOF MAE={mae_raw:.4f}')
    else:
        print('\n[Layer 1] HGB OOF 학습 중 (~30분)...')
        oof_hgb, test_hgb = get_hgb_oof(train, test, feat_cols, y_log, groups)
        np.save(hgb_oof_path,  oof_hgb)
        np.save(hgb_test_path, test_hgb)

    # ── OOF 상관관계 분석 ─────────────────────────────────────
    print('\n── OOF 상관관계 (raw 공간) ──')
    oof_raw = {
        'lgbm': np.expm1(oof_lg), 'cb': np.expm1(oof_cb),
        'et':   np.expm1(oof_et), 'tw': np.expm1(oof_tw),
        'rf':   np.expm1(oof_rf), 'hgb': np.expm1(oof_hgb),
    }
    oofs_df = pd.DataFrame(oof_raw)
    corr = oofs_df.corr()
    for a, b in [('lgbm','hgb'),('hgb','et'),('hgb','rf'),('hgb','cb'),('lgbm','et')]:
        print(f'  {a.upper()}-{b.upper()}: {corr.loc[a,b]:.4f}')

    # ── Layer 2: 메타 LGBM ──────────────────────────────────
    meta_train = np.column_stack([oof_lg, oof_cb, oof_et, oof_tw, oof_rf, oof_hgb])
    meta_test  = np.column_stack([test_lg,test_cb,test_et,test_tw,test_rf,test_hgb])

    print('\n[Layer 2] 6모델 메타 LGBM 학습...')
    oof_meta, test_meta, meta_cv = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    # ── 5모델 메타 비교 (6모델과 비교용) ──────────────────────
    meta_train_5 = np.column_stack([oof_lg, oof_cb, oof_et, oof_tw, oof_rf])
    meta_test_5  = np.column_stack([test_lg,test_cb,test_et,test_tw,test_rf])
    print('\n[참고] 5모델 메타 LGBM (비교용)...')
    _, _, meta_cv_5 = run_meta_lgbm(meta_train_5, meta_test_5, y_raw, groups, label='5model-meta')

    # ── 제출 저장 ─────────────────────────────────────────────
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    out_path = os.path.join(SUB_DIR, 'stacking_6model_histgb_lgbm_meta.csv')
    sample.to_csv(out_path, index=False)
    print(f'\n제출 파일 저장: submissions/stacking_6model_histgb_lgbm_meta.csv')

    print('\n── 최종 비교 ──')
    print(f'  5모델 메타 CV: {meta_cv_5:.4f}  (기준: 8.7911 / Public 10.2213)')
    print(f'  6모델+HGB 메타 CV: {meta_cv:.4f}')
    print(f'  CV 변화: {meta_cv - meta_cv_5:+.4f}')
    print(f'  기대 Public (배율 1.1627): {meta_cv * 1.1627:.4f}')
    print('\n완료!')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
