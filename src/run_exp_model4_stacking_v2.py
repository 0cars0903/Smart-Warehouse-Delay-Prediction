"""
모델 실험 4: 확장 스태킹 v2 (Tweedie 대체 + LGBM-meta)
==========================================================
목적 : 기존 스태킹(LGBM+CB+ET → LGBM-meta, CV 8.8541)에서
       CB(log1p)를 Tweedie(p=1.8)로 교체하여 Layer 1 다양성 강화

근거 :
  - CB(log1p)는 LGBM(log1p)과 OOF 상관 0.9788 → 중복
  - Tweedie(1.8)는 다른 목적함수(내부 log link, power variance)
    → LGBM(L1 MAE)과 오차 패턴이 다를 것으로 기대
  - p-sweep 실험에서 p=1.8이 단독 최고(OOF 8.9823)
  - ET는 이미 독립성 확인(LGBM-ET 상관 0.9744)

구조 :
  Layer 1 (Base Models)
    ├─ LightGBM(log1p)   — 현재 최고 파라미터
    ├─ Tweedie(p=1.8)    — CB 대체, 다른 손실함수
    └─ ExtraTrees(log1p) — 완전 무작위 split

  Layer 2 (Meta Learner)
    └─ LightGBM 메타     — 비선형 결합 (v1에서 검증)

비교 :
  v1 (LGBM+CB+ET → LGBM-meta): CV 8.8541 / Public 10.3032
  v2 (LGBM+TW+ET → LGBM-meta): ???

출력 :
  submissions/stacking_v2_lgbm_tw_et.csv
  docs/stacking_v2_ckpt/  (체크포인트)

예상 실행 시간: ~60분 (LGBM/ET 체크포인트 없으면 풀 실행)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
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
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'stacking_v2_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# Layer 1: LightGBM (log1p, 현재 최고 파라미터)
LGBM_PARAMS = {
    'num_leaves': 183, 'learning_rate': 0.020703,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# Layer 1: Tweedie(p=1.8) — raw 양수 타깃, 내부 log link 사용
TWEEDIE_PARAMS = {
    'num_leaves': 183, 'learning_rate': 0.020703,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'tweedie', 'tweedie_variance_power': 1.8,
    'metric': 'tweedie',
    'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# Layer 1: ExtraTrees
ET_PARAMS = {
    'n_estimators': 500,
    'max_features': 0.5,
    'min_samples_leaf': 26,
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
}

# Layer 2: LGBM 메타
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


# ─── 체크포인트 유틸 ──────────────────────────────────────────
def load_or_run(name, run_fn, ckpt_dir):
    oof_path  = os.path.join(ckpt_dir, f'{name}_oof.npy')
    test_path = os.path.join(ckpt_dir, f'{name}_test.npy')
    if os.path.exists(oof_path) and os.path.exists(test_path):
        print(f'  [{name}] 체크포인트 로드 (재학습 생략)')
        return np.load(oof_path), np.load(test_path)
    oof, test_pred = run_fn()
    np.save(oof_path, oof)
    np.save(test_path, test_pred)
    return oof, test_pred


# ─── Layer 1: Base Model OOF ─────────────────────────────────
def get_lgbm_oof(train, test, feat_cols, y_log, groups):
    """LightGBM (log1p 공간)"""
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
    return oof, test_pred  # log 공간


def get_tweedie_oof(train, test, feat_cols, y_raw, groups):
    """Tweedie(p=1.8) — 원본(raw) 양수 타깃 사용, 내부 log link"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y_raw, groups)):
        X_tr, X_va = train.iloc[tr_idx][feat_cols], train.iloc[va_idx][feat_cols]
        y_tr, y_va = y_raw.iloc[tr_idx], y_raw.iloc[va_idx]
        m = lgb.LGBMRegressor(**TWEEDIE_PARAMS)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[va_idx]  = m.predict(X_va)
        test_pred   += m.predict(test[feat_cols]) / N_SPLITS
        mae = np.mean(np.abs(oof[va_idx] - y_va.values))
        print(f'  [TW1.8] Fold {fold+1}  MAE={mae:.4f}')
        gc.collect()
    return oof, test_pred  # raw 공간 (Tweedie는 직접 예측)


def get_et_oof(train, test, feat_cols, y_log, groups):
    """ExtraTrees (log1p 공간, NaN→0)"""
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
        m = ExtraTreesRegressor(**ET_PARAMS)
        m.fit(X_tr, y_tr)
        oof[va_idx]  = m.predict(X_va)
        test_pred   += m.predict(X_all_test) / N_SPLITS
        mae = np.mean(np.abs(np.expm1(oof[va_idx]) - np.expm1(y_va)))
        print(f'  [ET]   Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    return oof, test_pred  # log 공간


# ─── Layer 2: LGBM 메타 학습기 ───────────────────────────────
def run_meta_lgbm(meta_train, meta_test, y_raw, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
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
        test_meta       += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.mean(np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values))
        print(f'  [LGBM-meta] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    oof_mae = np.mean(np.abs(oof_meta - y_raw.values))
    print(f'  [LGBM-meta] OOF MAE={oof_mae:.4f} | std={oof_meta.std():.2f}')
    return oof_meta, test_meta, oof_mae


# ─── 메인 ─────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('모델 실험 4: 확장 스태킹 v2 (LGBM + Tweedie(1.8) + ET)')
    print('비교: v1 LGBM+CB+ET → LGBM-meta: CV 8.8541 / Public 10.3032')
    print('=' * 60)

    os.makedirs(CKPT_DIR, exist_ok=True)
    train, test = load_data()
    feat_cols = get_feat_cols(train)
    print(f'피처 수: {len(feat_cols)}개')

    y_raw  = train['avg_delay_minutes_next_30m']
    y_log  = np.log1p(y_raw)
    groups = train['scenario_id']

    # ── Layer 1: Base Model OOF ───────────────────────────────
    print('\n[Layer 1] LightGBM OOF (log1p)')
    oof_lg, test_lg = load_or_run(
        'lgbm', lambda: get_lgbm_oof(train, test, feat_cols, y_log, groups), CKPT_DIR)
    mae_lg = np.mean(np.abs(np.expm1(oof_lg) - y_raw.values))
    print(f'  LGBM   OOF MAE: {mae_lg:.4f}  std={np.expm1(oof_lg).std():.2f}')

    print('\n[Layer 1] Tweedie(p=1.8) OOF (raw 타깃)')
    oof_tw, test_tw = load_or_run(
        'tw18', lambda: get_tweedie_oof(train, test, feat_cols, y_raw, groups), CKPT_DIR)
    mae_tw = np.mean(np.abs(oof_tw - y_raw.values))
    print(f'  TW1.8  OOF MAE: {mae_tw:.4f}  std={oof_tw.std():.2f}')

    print('\n[Layer 1] ExtraTrees OOF (log1p, 시간 소요)')
    oof_et, test_et = load_or_run(
        'et', lambda: get_et_oof(train, test, feat_cols, y_log, groups), CKPT_DIR)
    mae_et = np.mean(np.abs(np.expm1(oof_et) - y_raw.values))
    print(f'  ET     OOF MAE: {mae_et:.4f}  std={np.expm1(oof_et).std():.2f}')

    # ── OOF 상관관계 확인 ──────────────────────────────────────
    oof_raw_lg = np.expm1(oof_lg)
    oof_raw_et = np.expm1(oof_et)
    # Tweedie는 이미 raw 공간
    corr_lg_tw = np.corrcoef(oof_raw_lg, oof_tw)[0, 1]
    corr_lg_et = np.corrcoef(oof_raw_lg, oof_raw_et)[0, 1]
    corr_tw_et = np.corrcoef(oof_tw, oof_raw_et)[0, 1]
    print(f'\n  OOF 상관관계:')
    print(f'    LGBM-TW1.8: {corr_lg_tw:.4f}  (v1 LGBM-CB: 0.9788)')
    print(f'    LGBM-ET   : {corr_lg_et:.4f}  (v1: 0.9744)')
    print(f'    TW1.8-ET  : {corr_tw_et:.4f}  (v1 CB-ET: 0.9685)')

    # ── 단순 가중치 앙상블 (Layer 1만, 비교용) ────────────────
    # 메타 피처 정렬: 모두 raw 공간으로 변환
    oof_raw_tw = oof_tw  # 이미 raw

    def loss3(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        blend = w[0]*oof_raw_lg + w[1]*oof_raw_tw + w[2]*oof_raw_et
        return np.mean(np.abs(blend - y_raw.values))

    best_loss1, best_w1 = np.inf, np.array([1/3, 1/3, 1/3])
    for _ in range(300):
        w0 = np.random.dirichlet(np.ones(3))
        res = minimize(loss3, w0, method='Nelder-Mead')
        if res.fun < best_loss1:
            best_loss1 = res.fun
            best_w1 = np.abs(res.x) / np.abs(res.x).sum()
    print(f'\n  단순 3모델 가중치 앙상블 CV MAE: {best_loss1:.4f}')
    print(f'    가중치: LGBM={best_w1[0]:.3f}, TW1.8={best_w1[1]:.3f}, ET={best_w1[2]:.3f}')

    # ── Layer 2: LGBM 메타 ────────────────────────────────────
    # 메타 피처: log 공간으로 통일 (Tweedie도 log1p 변환)
    meta_train_feat = np.column_stack([oof_lg, np.log1p(oof_tw), oof_et])
    test_tw_clipped = np.maximum(test_tw, 0)
    meta_test_feat  = np.column_stack([test_lg, np.log1p(test_tw_clipped), test_et])

    print('\n[Layer 2] LightGBM 메타 학습기')
    _, test_lgbm_meta, mae_lgbm_meta = run_meta_lgbm(
        meta_train_feat, meta_test_feat, y_raw, groups)

    save_sub(test_lgbm_meta, 'stacking_v2_lgbm_tw_et.csv')

    # ── 최종 요약 ─────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('실험 4 결과 요약')
    print(f'{"="*60}')
    print(f'  [Layer 1] LGBM   단독 MAE: {mae_lg:.4f}')
    print(f'  [Layer 1] TW1.8  단독 MAE: {mae_tw:.4f}')
    print(f'  [Layer 1] ET     단독 MAE: {mae_et:.4f}')
    print(f'  [Layer 1] 가중치 앙상블  : {best_loss1:.4f}')
    print(f'  [Layer 2] LGBM-meta      : {mae_lgbm_meta:.4f}')
    print(f'\n  [비교] v1 (LGBM+CB+ET): CV 8.8541 / Public 10.3032')
    print(f'  [비교] v2 (LGBM+TW+ET): CV {mae_lgbm_meta:.4f}')
    print(f'\n  OOF 상관 비교:')
    print(f'    v1 LGBM-CB=0.9788 vs v2 LGBM-TW={corr_lg_tw:.4f}')
    print(f'    v1 CB-ET=0.9685   vs v2 TW-ET={corr_tw_et:.4f}')
    print(f'  → 상관이 낮을수록 스태킹 효과 ↑')


if __name__ == '__main__':
    main()
