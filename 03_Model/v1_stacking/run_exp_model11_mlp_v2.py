"""
run_exp_model11_mlp_v2.py  —  FE v1 + MLP v2 6모델 스태킹
===============================================================
v1 문제점 분석 (04.15):
  early_stopping=True + n_iter_no_change=20 → iter=31에서 조기종료
  실제로 11번만 학습 (31 - 20 = 11 improvement iterations)
  MLP OOF MAE=9.8659 — 트리보다 훨씬 나쁨 → meta 기여 불충분
  (LGBM-MLP 상관 0.8043 ✅ 다양성은 유효하나 품질이 문제)

v2 개선:
  early_stopping=False → 300 iter 완전 학습
  learning_rate_init: 5e-4 → 1e-4 (더 안정적인 수렴)
  GroupKFold OOF로 평가 (내부 검증 불필요)
  validation_fraction 제거 → 전체 fold train으로 학습

기대:
  MLP OOF MAE: 9.8659 → ~9.2~9.5 (트리와 경쟁 가능한 수준)
  LGBM-MLP 상관 0.80 유지 + 품질 개선 → meta CV 개선 가능

주의:
  TW1.8 상관 = expm1 overflow 수치 오류 → 상관 분석에서 TW 제외
  MLP-TW 0.0028은 실제값 아님 (inf 상관 계산 버그)

체크포인트: docs/mlp_v2_ckpt/ (v1과 분리)
예상 시간: ~30-60분 (5-fold × 300 iter)
===============================================================
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings, gc, os, sys
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_BASE, '..', 'data')
SUB_DIR     = os.path.join(_BASE, '..', 'submissions')
CKPT_V1     = os.path.join(_BASE, '..', 'docs', 'stacking_ckpt')
CKPT_V2     = os.path.join(_BASE, '..', 'docs', 'stacking_v2_ckpt')
CKPT_RF     = os.path.join(_BASE, '..', 'docs', 'stacking_5model_rf_ckpt')
CKPT_MLP_V2 = os.path.join(_BASE, '..', 'docs', 'mlp_v2_ckpt')
N_SPLITS    = 5
RANDOM_STATE = 42

# ── MLP v2 파라미터 (early_stopping 제거, lr 낮춤) ──────────────
MLP_PARAMS = {
    'hidden_layer_sizes' : (256, 128, 64),
    'activation'         : 'relu',
    'solver'             : 'adam',
    'alpha'              : 0.01,       # L2 정규화 (과적합 방지)
    'batch_size'         : 512,
    'learning_rate_init' : 1e-4,       # v1(5e-4)보다 보수적 → 안정 수렴
    'max_iter'           : 300,        # 충분한 학습 보장
    'early_stopping'     : False,      # v1 문제 수정: 내부 조기종료 제거
    'random_state'       : RANDOM_STATE,
    'verbose'            : False,
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
# MLP v2 GroupKFold 학습
# ──────────────────────────────────────────────
def get_mlp_oof(train, test, feat_cols, y_log, groups):
    """MLP v2 OOF — early_stopping 없이 300 iter 완전 학습"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    X_tr_all = train[feat_cols].fillna(0).values  # NaN→0 (lag 초기값)
    X_te_all = test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_log, groups)):
        X_tr_raw = X_tr_all[tr_idx]
        X_va_raw = X_tr_all[va_idx]
        y_tr = y_log.iloc[tr_idx].values
        y_va = y_log.iloc[va_idx].values

        # fold 내 StandardScaler (train 기준 fit)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)
        X_te = scaler.transform(X_te_all)

        m = MLPRegressor(**MLP_PARAMS)
        m.fit(X_tr, y_tr)
        oof[va_idx]  = m.predict(X_va)
        test_pred   += m.predict(X_te) / N_SPLITS

        mae_raw = mean_absolute_error(
            np.expm1(y_va), np.expm1(oof[va_idx])
        )
        print(f'  [MLP-v2] Fold {fold+1}  MAE={mae_raw:.4f}  iter={m.n_iter_}')
        del m, scaler; gc.collect()

    oof_mae = mean_absolute_error(
        np.expm1(y_log.values), np.expm1(oof)
    )
    pred_std = np.expm1(oof).std()
    print(f'  [MLP-v2] 전체 OOF MAE={oof_mae:.4f}  pred_std={pred_std:.2f}')
    return oof, test_pred  # log 공간


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
    print('FE v1 + MLP v2 6모델 스태킹')
    print('개선: early_stopping=False, lr=1e-4, max_iter=300')
    print('체크포인트 재활용: LGBM+CB+ET+TW1.8+RF (model8 동일)')
    print('=' * 60)

    os.makedirs(CKPT_MLP_V2, exist_ok=True)

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
    oof_lg  = np.load(os.path.join(CKPT_V1, 'lgbm_oof.npy'))
    test_lg = np.load(os.path.join(CKPT_V1, 'lgbm_test.npy'))
    oof_cb  = np.load(os.path.join(CKPT_V1, 'cb_oof.npy'))
    test_cb = np.load(os.path.join(CKPT_V1, 'cb_test.npy'))
    oof_et  = np.load(os.path.join(CKPT_V1, 'et_oof.npy'))
    test_et = np.load(os.path.join(CKPT_V1, 'et_test.npy'))
    oof_tw  = np.load(os.path.join(CKPT_V2, 'tw18_oof.npy'))
    test_tw = np.load(os.path.join(CKPT_V2, 'tw18_test.npy'))
    oof_rf  = np.load(os.path.join(CKPT_RF, 'rf_oof.npy'))
    test_rf = np.load(os.path.join(CKPT_RF, 'rf_test.npy'))
    print('  LGBM / CB / ET / TW1.8 / RF 로드 완료')

    for name, oof in [('LGBM', oof_lg), ('CB', oof_cb), ('ET', oof_et), ('RF', oof_rf)]:
        mae_raw = mean_absolute_error(y_raw.values, np.expm1(oof))
        print(f'  {name:<6} OOF MAE={mae_raw:.4f}')

    # ── Layer 1: MLP v2 ───────────────────────────────────────
    mlp_oof_path  = os.path.join(CKPT_MLP_V2, 'mlp_v2_oof.npy')
    mlp_test_path = os.path.join(CKPT_MLP_V2, 'mlp_v2_test.npy')

    if os.path.exists(mlp_oof_path) and os.path.exists(mlp_test_path):
        print('\n[Layer 1] MLP v2 체크포인트 로드 (재학습 생략)')
        oof_mlp  = np.load(mlp_oof_path)
        test_mlp = np.load(mlp_test_path)
        mae_raw = mean_absolute_error(y_raw.values, np.expm1(oof_mlp))
        print(f'  MLP-v2 OOF MAE={mae_raw:.4f}')
    else:
        print('\n[Layer 1] MLP v2 OOF 학습 중 (~30-60분)...')
        oof_mlp, test_mlp = get_mlp_oof(train, test, feat_cols, y_log, groups)
        np.save(mlp_oof_path,  oof_mlp)
        np.save(mlp_test_path, test_mlp)

    # ── OOF 상관관계 분석 (TW expm1 overflow 방지 → 제외) ───
    print('\n── OOF 상관관계 (raw 공간, TW 제외 — expm1 overflow) ──')
    oof_raw_safe = {
        'lgbm': np.expm1(oof_lg), 'cb': np.expm1(oof_cb),
        'et':   np.expm1(oof_et), 'rf': np.expm1(oof_rf),
        'mlp':  np.expm1(oof_mlp),
    }
    oofs_df = pd.DataFrame(oof_raw_safe)
    corr = oofs_df.corr()
    print(f'  LGBM-MLP : {corr.loc["lgbm","mlp"]:.4f}')
    print(f'  ET-MLP   : {corr.loc["et","mlp"]:.4f}')
    print(f'  RF-MLP   : {corr.loc["rf","mlp"]:.4f}')
    print(f'  CB-MLP   : {corr.loc["cb","mlp"]:.4f}')
    print(f'  LGBM-ET  : {corr.loc["lgbm","et"]:.4f}  (참고: v1 기준선)')

    lgbm_mlp_corr = corr.loc['lgbm','mlp']
    if lgbm_mlp_corr < 0.92:
        print(f'\n  ✅ MLP 다양성 유효 (LGBM-MLP {lgbm_mlp_corr:.4f} < 0.92)')
    else:
        print(f'\n  ⚠️  MLP 다양성 제한적 (LGBM-MLP {lgbm_mlp_corr:.4f} ≥ 0.92)')

    # v1 대비 OOF 품질 개선 확인
    mlp_mae = mean_absolute_error(y_raw.values, np.expm1(oof_mlp))
    print(f'\n  MLP v2 OOF MAE: {mlp_mae:.4f}  (v1: 9.8659, 트리: ~9.0)')
    if mlp_mae < 9.5:
        print('  ✅ v1 대비 품질 개선 — meta 기여 기대')
    elif mlp_mae < 9.8:
        print('  ⚠️  v1 대비 개선이나 트리와 격차 여전')
    else:
        print('  ❌ v1과 유사 — MLP 방향 한계')

    # ── Layer 2: 메타 LGBM ──────────────────────────────────
    meta_train = np.column_stack([oof_lg, oof_cb, oof_et, oof_tw, oof_rf, oof_mlp])
    meta_test  = np.column_stack([test_lg,test_cb,test_et,test_tw,test_rf,test_mlp])

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
    out_path = os.path.join(SUB_DIR, 'stacking_6model_mlp_v2_lgbm_meta.csv')
    sample.to_csv(out_path, index=False)
    print(f'\n제출 파일 저장: submissions/stacking_6model_mlp_v2_lgbm_meta.csv')

    # ── 최종 요약 ──────────────────────────────────────────────
    print('\n── 최종 비교 ──')
    print(f'  5모델 기준 CV : {meta_cv_5:.4f}  (RF5 실제: 8.7911 / Public 10.2213 / 배율 1.1627)')
    print(f'  6모델+MLP v1  : 8.7919  (v1 참고, iter=31 과소학습)')
    print(f'  6모델+MLP v2  : {meta_cv:.4f}')
    print(f'  v1 대비 변화  : {meta_cv - 8.7919:+.4f}')
    print(f'  RF5 대비 변화 : {meta_cv - 8.7911:+.4f}')
    print(f'  기대 Public (배율 1.1627): {meta_cv * 1.1627:.4f}')
    print(f'  기대 Public (배율 1.1700): {meta_cv * 1.1700:.4f}')

    pred_std = np.std(np.maximum(test_meta, 0))
    print(f'  제출 예측 std : {pred_std:.2f}  (실제 27.35 기준)')
    if pred_std < 18:
        print('  ⚠️  pred_std 압축 — 배율 1.170 가능성 높음')
    else:
        print('  ✅ pred_std 양호 — 배율 1.1627 가능성 있음')

    print('\n완료!')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
