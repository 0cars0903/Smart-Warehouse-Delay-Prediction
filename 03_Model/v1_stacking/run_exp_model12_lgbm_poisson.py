"""
run_exp_model12_lgbm_poisson.py  —  FE v1 + LGBM Poisson 6모델 스태킹
===============================================================
배경 (04.15 현재):
  모델 다양성 탐색 현황:
    HGB: LGBM-HGB 상관 0.9862 → 폐기 (Histogram-based GBDT = 동일 계열)
    MLP: v1 과소학습(OOF 9.87), v2 과적합(OOF 12.7) → 폐기 (시나리오 분리 환경 취약)
  현재 최강: RF 5모델 (FE v1) — CV 8.7911 / Public 10.2213 / 배율 1.1627

탐색 가설:
  LightGBM Poisson 목적함수를 6번째 베이스 모델로 추가.

  Poisson vs 기존 목적함수 비교:
    - LGBM L1 (regression_l1): MAE 최소화, 선형 gradient
    - Tweedie(p=1.8): 분산이 μ^1.8에 비례 (복합 분포)
    - CatBoost: 자체 gradient boosting 구현
    - **Poisson (objective='poisson')**: 로그링크 함수, 분산=μ (카운트 데이터)

  Poisson의 특성:
    - 출고 지연 시간 = 양의 왜도 연속값 → Poisson 가정 근사 가능
    - 로그링크 함수: exp(prediction) → 자연스럽게 양수 보장
    - Tweedie(p=1)과 동치지만 다른 구현 → 미세 차이 가능
    - 기존 TW1.8과 상관관계: Tweedie 계열이므로 높을 수 있음 (~0.97?)
    - 기존 LGBM-L1과 상관관계: 다른 objective → ~0.95?

  핵심 질문: Poisson이 TW1.8(p=1.8)과 얼마나 다른가?
    - p=1.0 (Poisson) vs p=1.8 (Tweedie) → 분포 가정 차이
    - 특히 극단값 처리: p=1.8이 더 heavy tail

체크포인트 재활용:
  model8 동일한 5모델 체크포인트 재사용
  Poisson LGBM만 신규 학습 → docs/lgbm_poisson_ckpt/ 저장

예상 시간: ~30분 (LGBM 5-fold × GroupKFold)
종료 기준:
  LGBM_L1-Poisson 상관 < 0.95 AND TW1.8-Poisson 상관 < 0.95: 다양성 유효 → meta 기여 기대
  둘 중 하나라도 ≥ 0.97: 다양성 미미 → 방향 종료
===============================================================
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings, gc, os, sys
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(_BASE, '..', 'data')
SUB_DIR       = os.path.join(_BASE, '..', 'submissions')
CKPT_V1       = os.path.join(_BASE, '..', 'docs', 'stacking_ckpt')
CKPT_V2       = os.path.join(_BASE, '..', 'docs', 'stacking_v2_ckpt')
CKPT_RF       = os.path.join(_BASE, '..', 'docs', 'stacking_5model_rf_ckpt')
CKPT_POISSON  = os.path.join(_BASE, '..', 'docs', 'lgbm_poisson_ckpt')
N_SPLITS      = 5
RANDOM_STATE  = 42

# LGBM Poisson 파라미터
# Optuna 기반이지만 Poisson에 맞게 조정
# - Poisson은 log-link → 예측값이 exp()으로 변환됨 (raw 공간 직접 예측)
# - log1p 타겟으로 학습하면 이중 로그 적용 → RAW 공간에서 학습이 자연스러움
# - 단, 다른 모델과 meta 입력 일관성을 위해 log1p 공간 유지 (expm1로 일관 처리)
POISSON_PARAMS = {
    'objective'        : 'poisson',
    'num_leaves'       : 181,
    'learning_rate'    : 0.020616,       # BEST_LGBM_PARAMS 재사용
    'feature_fraction' : 0.5122,
    'bagging_fraction' : 0.9049,
    'min_child_samples': 26,
    'reg_alpha'        : 0.3805,
    'reg_lambda'       : 0.3630,
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
    'objective': 'regression_l1', 'n_estimators': 1000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}


# ──────────────────────────────────────────────
# 데이터 로드 (FE v1 — model8과 동일)
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
# LGBM Poisson GroupKFold 학습
# ──────────────────────────────────────────────
def get_poisson_oof(train, test, feat_cols, y_log, y_raw, groups):
    """
    LGBM Poisson OOF
    - Poisson은 log-link를 내부적으로 적용하므로 raw 공간 타겟으로 학습
    - 단, 다른 모델과 일관성 위해 OOF는 log1p 공간으로 변환해 반환
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_raw = np.zeros(len(train))    # raw 공간 OOF
    test_pred_raw = np.zeros(len(test))

    X_tr_all = train[feat_cols].fillna(0).values
    X_te_all = test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_log, groups)):
        X_tr = X_tr_all[tr_idx]
        X_va = X_tr_all[va_idx]
        y_tr = y_raw.iloc[tr_idx].values   # Poisson: raw 공간 타겟
        y_va = y_raw.iloc[va_idx].values

        m = lgb.LGBMRegressor(**POISSON_PARAMS)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])

        oof_raw[va_idx]  = m.predict(X_va)
        test_pred_raw   += m.predict(X_te_all) / N_SPLITS

        mae = mean_absolute_error(y_va, np.maximum(oof_raw[va_idx], 0))
        print(f'  [Poisson] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    # raw 공간 클리핑 (음수 예방)
    oof_raw = np.maximum(oof_raw, 0)
    test_pred_raw = np.maximum(test_pred_raw, 0)

    oof_mae = mean_absolute_error(y_raw.values, oof_raw)
    pred_std = oof_raw.std()
    print(f'  [Poisson] 전체 OOF MAE={oof_mae:.4f}  pred_std={pred_std:.2f}')

    # log1p 공간으로 변환 (meta 입력 일관성)
    oof_log  = np.log1p(oof_raw)
    test_log = np.log1p(test_pred_raw)
    return oof_log, test_log


# ──────────────────────────────────────────────
# 메타 LGBM
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
    print('FE v1 + LGBM Poisson 6모델 스태킹')
    print('체크포인트 재활용: LGBM+CB+ET+TW1.8+RF (model8 동일)')
    print('신규 학습: Poisson LGBM만 (~30분)')
    print('=' * 60)

    os.makedirs(CKPT_POISSON, exist_ok=True)

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

    for name, oof in [('LGBM', oof_lg), ('CB', oof_cb), ('ET', oof_et), ('RF', oof_rf)]:
        mae_raw = mean_absolute_error(y_raw.values, np.expm1(oof))
        print(f'  {name:<6} OOF MAE={mae_raw:.4f}')

    # ── Layer 1: LGBM Poisson ──────────────────────────────────
    po_oof_path  = os.path.join(CKPT_POISSON, 'poisson_oof.npy')
    po_test_path = os.path.join(CKPT_POISSON, 'poisson_test.npy')

    if os.path.exists(po_oof_path) and os.path.exists(po_test_path):
        print('\n[Layer 1] Poisson 체크포인트 로드 (재학습 생략)')
        oof_po  = np.load(po_oof_path)
        test_po = np.load(po_test_path)
        mae_raw = mean_absolute_error(y_raw.values, np.expm1(oof_po))
        print(f'  Poisson OOF MAE={mae_raw:.4f}')
    else:
        print('\n[Layer 1] LGBM Poisson OOF 학습 중...')
        oof_po, test_po = get_poisson_oof(train, test, feat_cols, y_log, y_raw, groups)
        np.save(po_oof_path,  oof_po)
        np.save(po_test_path, test_po)

    # ── OOF 상관관계 분석 (TW expm1 overflow 제외) ────────────
    print('\n── OOF 상관관계 (raw 공간, TW 제외) ──')
    oof_raw_safe = {
        'lgbm'   : np.expm1(oof_lg),
        'cb'     : np.expm1(oof_cb),
        'et'     : np.expm1(oof_et),
        'rf'     : np.expm1(oof_rf),
        'poisson': np.expm1(oof_po),
    }
    oofs_df = pd.DataFrame(oof_raw_safe)
    corr = oofs_df.corr()
    print(f'  LGBM-Poisson : {corr.loc["lgbm","poisson"]:.4f}')
    print(f'  ET-Poisson   : {corr.loc["et","poisson"]:.4f}')
    print(f'  RF-Poisson   : {corr.loc["rf","poisson"]:.4f}')
    print(f'  CB-Poisson   : {corr.loc["cb","poisson"]:.4f}')
    print(f'  LGBM-ET      : {corr.loc["lgbm","et"]:.4f}  (참고: v1 기준선)')

    # TW와 Poisson의 상관 (log 공간에서 안전하게)
    tw_po_corr = np.corrcoef(oof_tw, oof_po)[0, 1]
    print(f'  TW1.8-Poisson: {tw_po_corr:.4f}  (log 공간)')

    lgbm_po_corr = corr.loc['lgbm', 'poisson']
    tw_po_corr_abs = abs(tw_po_corr)
    print('\n  판정:')
    if lgbm_po_corr < 0.95 and tw_po_corr_abs < 0.97:
        print(f'  ✅ Poisson 다양성 유효 (LGBM={lgbm_po_corr:.4f}, TW={tw_po_corr_abs:.4f})')
    elif tw_po_corr_abs >= 0.97:
        print(f'  ⚠️  TW1.8-Poisson 상관 높음 ({tw_po_corr_abs:.4f}) — Tweedie 계열 중복')
    else:
        print(f'  ⚠️  LGBM-Poisson 상관 높음 ({lgbm_po_corr:.4f}) — L1 대비 차별성 제한')

    # ── Layer 2: 메타 LGBM ──────────────────────────────────
    meta_train = np.column_stack([oof_lg, oof_cb, oof_et, oof_tw, oof_rf, oof_po])
    meta_test  = np.column_stack([test_lg,test_cb,test_et,test_tw,test_rf,test_po])

    print('\n[Layer 2] 6모델 메타 LGBM 학습...')
    oof_meta, test_meta, meta_cv = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    # ── 5모델 기준 비교 ───────────────────────────────────────
    meta_train_5 = np.column_stack([oof_lg, oof_cb, oof_et, oof_tw, oof_rf])
    meta_test_5  = np.column_stack([test_lg,test_cb,test_et,test_tw,test_rf])
    print('\n[참고] 5모델 메타 LGBM (비교용)...')
    _, _, meta_cv_5 = run_meta_lgbm(meta_train_5, meta_test_5, y_raw, groups, label='5model-meta')

    # ── 제출 파일 저장 ────────────────────────────────────────
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    out_path = os.path.join(SUB_DIR, 'stacking_6model_poisson_lgbm_meta.csv')
    sample.to_csv(out_path, index=False)
    print(f'\n제출 파일 저장: submissions/stacking_6model_poisson_lgbm_meta.csv')

    # ── 최종 요약 ──────────────────────────────────────────────
    print('\n── 최종 비교 ──')
    print(f'  5모델 기준 CV : {meta_cv_5:.4f}')
    print(f'  6모델+Poisson : {meta_cv:.4f}')
    print(f'  RF5 대비 변화 : {meta_cv - 8.7911:+.4f}  (기준: 8.7911 / Public 10.2213 / 배율 1.1627)')
    print(f'  기대 Public (배율 1.1627): {meta_cv * 1.1627:.4f}')
    print(f'  기대 Public (배율 1.1700): {meta_cv * 1.1700:.4f}')

    pred_std = np.std(np.maximum(test_meta, 0))
    print(f'  제출 예측 std : {pred_std:.2f}  (실제 27.35 기준)')
    if pred_std < 18:
        print('  ⚠️  pred_std 압축 — 배율 1.170 가능성')
    else:
        print('  ✅ pred_std 양호 — 배율 1.1627 가능성')
    print('\n완료!')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
