"""
run_exp_model11_mlp.py  —  FE v1 + MLP 6모델 스태킹
===============================================================
배경 (04.15 현재):
  FE 확장 방향 전체 차단 (Cumulative/ExtLag/Delta/KEY_COLS_V2 모두 배율 1.170)
  모델 다양성 탐색: HGB → LGBM-HGB 상관 0.9862 → 트리 계열 포화 → 방향 폐기
  현재 최강: RF 5모델 (FE v1) — CV 8.7911 / Public 10.2213 / 배율 1.1627

탐색 가설:
  sklearn MLPRegressor를 6번째 베이스 모델로 추가.
  MLP 특성:
    - 트리 기반 앙상블과 완전히 다른 최적화 경관 (gradient descent, 연속 피처 공간)
    - 비선형 함수 조합 → GBDT와 다른 오차 패턴 기대
    - LGBM-MLP 상관 ~0.85-0.92 기대 (HGB 0.9862 대비 낮아야 의미)
    - NaN 처리: 학습 전 fillna(0) 또는 imputer 필요 (FE v1은 lag/rolling NaN 존재)

체크포인트 재활용 전략:
  model8과 동일한 OOF 로드:
    docs/stacking_ckpt/       → lgbm_oof, cb_oof, et_oof (log 공간)
    docs/stacking_v2_ckpt/    → tw18_oof (log 공간)
    docs/stacking_5model_rf_ckpt/ → rf_oof (log 공간)
  MLP만 새로 GroupKFold 학습 → docs/mlp_ckpt/ 저장

전처리:
  MLP는 스케일 민감 → StandardScaler 필수
  NaN → fillna(0) (lag/rolling 초기 NaN)
  log1p 공간에서 학습 (다른 모델과 동일)

예상 시간: ~30-60분 (5-fold × MLP 학습, hidden_layer_sizes 규모에 따라)
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_V1  = os.path.join(_BASE, '..', 'docs', 'stacking_ckpt')
CKPT_V2  = os.path.join(_BASE, '..', 'docs', 'stacking_v2_ckpt')
CKPT_RF  = os.path.join(_BASE, '..', 'docs', 'stacking_5model_rf_ckpt')
CKPT_MLP = os.path.join(_BASE, '..', 'docs', 'mlp_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# MLP 파라미터
# 중간 규모 네트워크: 과적합 방지 위해 dropout 대신 alpha(L2) + early_stopping
MLP_PARAMS = {
    'hidden_layer_sizes': (256, 128, 64),
    'activation'        : 'relu',
    'solver'            : 'adam',
    'alpha'             : 0.01,          # L2 정규화
    'batch_size'        : 512,
    'learning_rate_init': 5e-4,
    'max_iter'          : 300,
    'early_stopping'    : True,
    'validation_fraction': 0.1,          # 내부 검증 (GroupKFold와 독립)
    'n_iter_no_change'  : 20,
    'random_state'      : RANDOM_STATE,
    'verbose'           : False,
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
# MLP GroupKFold 학습 (log1p 공간 + StandardScaler)
# ──────────────────────────────────────────────
def get_mlp_oof(train, test, feat_cols, y_log, groups):
    """MLP OOF — log1p 공간, StandardScaler, NaN→0 전처리"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    # NaN → 0 채움 (MLP는 NaN 불가)
    X_tr_all = train[feat_cols].fillna(0).values
    X_te_all = test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_log, groups)):
        X_tr_raw = X_tr_all[tr_idx]
        X_va_raw = X_tr_all[va_idx]
        y_tr = y_log.iloc[tr_idx].values
        y_va = y_log.iloc[va_idx].values

        # 폴드 내 스케일러 (train 기준으로 fit)
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
        n_iter = m.n_iter_
        print(f'  [MLP] Fold {fold+1}  MAE={mae_raw:.4f}  iter={n_iter}')
        del m, scaler; gc.collect()

    oof_mae = mean_absolute_error(
        np.expm1(y_log.values), np.expm1(oof)
    )
    pred_std = np.expm1(oof).std()
    print(f'  [MLP] 전체 OOF MAE={oof_mae:.4f}  pred_std={pred_std:.2f}')
    return oof, test_pred   # log 공간 반환


# ──────────────────────────────────────────────
# 메타 LGBM (model8 동일 방식)
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
    print('FE v1 + MLP 6모델 스태킹')
    print('체크포인트 재활용: LGBM+CB+ET+TW1.8+RF (model8 동일)')
    print('신규 학습: MLP만 (~30-60분)')
    print('=' * 60)

    os.makedirs(CKPT_MLP, exist_ok=True)

    train, test = load_data()
    feat_cols = [c for c in train.columns
                 if c not in {'ID', 'scenario_id', 'ts_idx', 'avg_delay_minutes_next_30m'}
                 and train[c].dtype != object]
    y_raw  = train['avg_delay_minutes_next_30m']
    y_log  = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'피처 수: {len(feat_cols)}')
    print(f'NaN 비율 (train): {train[feat_cols].isna().mean().mean():.4f}')

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

    # ── Layer 1: MLP (체크포인트 없으면 학습) ─────────────────
    mlp_oof_path  = os.path.join(CKPT_MLP, 'mlp_oof.npy')
    mlp_test_path = os.path.join(CKPT_MLP, 'mlp_test.npy')

    if os.path.exists(mlp_oof_path) and os.path.exists(mlp_test_path):
        print('\n[Layer 1] MLP 체크포인트 로드 (재학습 생략)')
        oof_mlp  = np.load(mlp_oof_path)
        test_mlp = np.load(mlp_test_path)
        mae_raw = mean_absolute_error(y_raw.values, np.expm1(oof_mlp))
        print(f'  MLP    OOF MAE={mae_raw:.4f}')
    else:
        print('\n[Layer 1] MLP OOF 학습 중 (~30-60분)...')
        oof_mlp, test_mlp = get_mlp_oof(train, test, feat_cols, y_log, groups)
        np.save(mlp_oof_path,  oof_mlp)
        np.save(mlp_test_path, test_mlp)

    # ── OOF 상관관계 분석 ─────────────────────────────────────
    print('\n── OOF 상관관계 (raw 공간) ──')
    oof_raw = {
        'lgbm': np.expm1(oof_lg), 'cb': np.expm1(oof_cb),
        'et':   np.expm1(oof_et), 'tw': np.expm1(oof_tw),
        'rf':   np.expm1(oof_rf), 'mlp': np.expm1(oof_mlp),
    }
    oofs_df = pd.DataFrame(oof_raw)
    corr = oofs_df.corr()
    for a, b in [('lgbm','mlp'),('mlp','et'),('mlp','rf'),('mlp','cb'),('mlp','tw')]:
        print(f'  {a.upper()}-{b.upper()}: {corr.loc[a,b]:.4f}')

    # 핵심 판정: LGBM-MLP 상관
    lgbm_mlp_corr = corr.loc['lgbm','mlp']
    if lgbm_mlp_corr < 0.92:
        print(f'\n  ✅ MLP 다양성 유효 (LGBM-MLP {lgbm_mlp_corr:.4f} < 0.92)')
    else:
        print(f'\n  ⚠️  MLP 다양성 제한적 (LGBM-MLP {lgbm_mlp_corr:.4f} ≥ 0.92)')

    # ── Layer 2: 메타 LGBM ──────────────────────────────────
    meta_train = np.column_stack([oof_lg, oof_cb, oof_et, oof_tw, oof_rf, oof_mlp])
    meta_test  = np.column_stack([test_lg,test_cb,test_et,test_tw,test_rf,test_mlp])

    print('\n[Layer 2] 6모델 메타 LGBM 학습...')
    oof_meta, test_meta, meta_cv = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    # ── 5모델 메타 비교 (기준선) ──────────────────────────────
    meta_train_5 = np.column_stack([oof_lg, oof_cb, oof_et, oof_tw, oof_rf])
    meta_test_5  = np.column_stack([test_lg,test_cb,test_et,test_tw,test_rf])
    print('\n[참고] 5모델 메타 LGBM (비교용)...')
    _, _, meta_cv_5 = run_meta_lgbm(meta_train_5, meta_test_5, y_raw, groups, label='5model-meta')

    # ── 제출 저장 ─────────────────────────────────────────────
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    out_path = os.path.join(SUB_DIR, 'stacking_6model_mlp_lgbm_meta.csv')
    sample.to_csv(out_path, index=False)
    print(f'\n제출 파일 저장: submissions/stacking_6model_mlp_lgbm_meta.csv')

    # ── 최종 요약 ──────────────────────────────────────────────
    print('\n── 최종 비교 ──')
    print(f'  5모델 기준 CV: {meta_cv_5:.4f}  (기준: 8.7911 / Public 10.2213 / 배율 1.1627)')
    print(f'  6모델+MLP  CV: {meta_cv:.4f}')
    print(f'  CV 변화: {meta_cv - meta_cv_5:+.4f}')
    print(f'  기대 Public (배율 1.1627): {meta_cv * 1.1627:.4f}')
    print(f'  기대 Public (배율 1.1700): {meta_cv * 1.1700:.4f}')

    pred_std = np.std(np.maximum(test_meta, 0))
    print(f'  제출 예측 std: {pred_std:.2f}  (실제 27.35 기준)')
    if pred_std < 18:
        print('  ⚠️  pred_std 압축 — 배율 1.170 가능성 높음')
    else:
        print('  ✅ pred_std 양호 — 배율 1.1627 가능성 있음')
    print('\n완료!')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
