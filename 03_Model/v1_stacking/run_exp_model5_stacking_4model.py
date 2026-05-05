"""
모델 실험 5: 4모델 스태킹 (LGBM + TW1.8 + CB + ET → LGBM-meta)
================================================================
목적 : v1(LGBM+CB+ET)과 v2(LGBM+TW1.8+ET)의 장점을 결합
       - v1: CB가 일반화 앵커 역할 (Public 10.3032 최고)
       - v2: TW1.8이 CV 다양성 기여 (CV 8.8087 최고)
       → CB + TW1.8 동시 포함 → 최적 조합 기대

체크포인트 재활용 (재학습 없음, 예상 ~5분):
  docs/stacking_ckpt/   → lgbm, cb, et (OOF + test 예측)
  docs/stacking_v2_ckpt/ → tw18 (OOF + test 예측)

구조 :
  Layer 1 (OOF 체크포인트 로드, 재학습 없음)
    ├─ LightGBM(log1p)   ← stacking_ckpt/lgbm_*.npy
    ├─ Tweedie(p=1.8)    ← stacking_v2_ckpt/tw18_*.npy
    ├─ CatBoost(log1p)   ← stacking_ckpt/cb_*.npy
    └─ ExtraTrees(log1p) ← stacking_ckpt/et_*.npy

  Layer 2 (새로 학습, ~5분)
    └─ LightGBM 메타 (GroupKFold 5-fold)

비교 기준:
  v1 (LGBM+CB+ET):    CV 8.8541 / Public 10.3032 🏆
  v2 (LGBM+TW1.8+ET): CV 8.8087 / Public 10.3118

출력 :
  submissions/stacking_4model_lgbm_meta.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import warnings, gc, os, sys
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

# ─── 상수 ────────────────────────────────────────────────────
_BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_BASE, '..', 'data')
SUB_DIR   = os.path.join(_BASE, '..', 'submissions')
CKPT_V1   = os.path.join(_BASE, '..', 'docs', 'stacking_ckpt')     # lgbm, cb, et
CKPT_V2   = os.path.join(_BASE, '..', 'docs', 'stacking_v2_ckpt')  # tw18
N_SPLITS  = 5
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


def load_ckpt(name, ckpt_dir):
    oof  = np.load(os.path.join(ckpt_dir, f'{name}_oof.npy'))
    test = np.load(os.path.join(ckpt_dir, f'{name}_test.npy'))
    print(f'  [{name}] 체크포인트 로드 완료')
    return oof, test


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
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | std={oof_meta.std():.2f}, max={oof_meta.max():.2f}')
    return oof_meta, test_meta, oof_mae


def main():
    print('=' * 60)
    print('모델 실험 5: 4모델 스태킹 (LGBM+TW1.8+CB+ET → LGBM-meta)')
    print('모든 체크포인트 재활용 → 메타 학습기만 재실행 (~5분)')
    print('=' * 60)

    train, test = load_data()
    y_raw  = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']

    # ── Layer 1: 체크포인트 로드 (재학습 없음) ──────────────
    print('\n[Layer 1] 체크포인트 로드')
    oof_lg, test_lg = load_ckpt('lgbm', CKPT_V1)   # log 공간
    oof_cb, test_cb = load_ckpt('cb',   CKPT_V1)   # log 공간
    oof_et, test_et = load_ckpt('et',   CKPT_V1)   # log 공간
    oof_tw, test_tw = load_ckpt('tw18', CKPT_V2)   # raw 공간

    # 단일 모델 OOF MAE 확인
    mae_lg = np.mean(np.abs(np.expm1(oof_lg) - y_raw.values))
    mae_cb = np.mean(np.abs(np.expm1(oof_cb) - y_raw.values))
    mae_et = np.mean(np.abs(np.expm1(oof_et) - y_raw.values))
    mae_tw = np.mean(np.abs(oof_tw - y_raw.values))
    print(f'\n  단일 OOF MAE:')
    print(f'    LGBM : {mae_lg:.4f}  (log 공간)')
    print(f'    CB   : {mae_cb:.4f}  (log 공간)')
    print(f'    TW1.8: {mae_tw:.4f}  (raw 공간)')
    print(f'    ET   : {mae_et:.4f}  (log 공간)')

    # ── OOF 상관관계 ─────────────────────────────────────────
    oof_raw_lg = np.expm1(oof_lg)
    oof_raw_cb = np.expm1(oof_cb)
    oof_raw_et = np.expm1(oof_et)
    oof_raw_tw = oof_tw  # 이미 raw 공간

    print(f'\n  OOF 상관관계 (4모델):')
    for n1, a1 in [('LG', oof_raw_lg), ('CB', oof_raw_cb), ('TW', oof_raw_tw), ('ET', oof_raw_et)]:
        for n2, a2 in [('LG', oof_raw_lg), ('CB', oof_raw_cb), ('TW', oof_raw_tw), ('ET', oof_raw_et)]:
            if n1 < n2:
                c = np.corrcoef(a1, a2)[0, 1]
                print(f'    {n1}-{n2}: {c:.4f}')

    # ── 단순 가중치 앙상블 (4모델, 비교용) ──────────────────
    def loss4(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        blend = w[0]*oof_raw_lg + w[1]*oof_raw_cb + w[2]*oof_raw_tw + w[3]*oof_raw_et
        return np.mean(np.abs(blend - y_raw.values))

    best_loss, best_w = np.inf, np.ones(4) / 4
    for _ in range(400):
        w0 = np.random.dirichlet(np.ones(4))
        res = minimize(loss4, w0, method='Nelder-Mead')
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = np.abs(res.x) / np.abs(res.x).sum()
    print(f'\n  4모델 단순 가중치 앙상블 CV MAE: {best_loss:.4f}')
    print(f'    가중치: LGBM={best_w[0]:.3f}, CB={best_w[1]:.3f}, TW1.8={best_w[2]:.3f}, ET={best_w[3]:.3f}')

    # ── Layer 2: 4모델 LGBM-meta ─────────────────────────────
    # 메타 피처: 모두 log 공간으로 통일 (TW1.8 → log1p 변환)
    test_tw_clipped = np.maximum(test_tw, 0)
    meta_train_feat = np.column_stack([oof_lg, oof_cb, np.log1p(oof_tw), oof_et])
    meta_test_feat  = np.column_stack([test_lg, test_cb, np.log1p(test_tw_clipped), test_et])

    print('\n[Layer 2] 4모델 LGBM 메타 학습기')
    _, test_meta, mae_meta = run_meta_lgbm(
        meta_train_feat, meta_test_feat, y_raw, groups)

    save_sub(test_meta, 'stacking_4model_lgbm_meta.csv')

    # ── 최종 요약 ─────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('실험 5 결과 요약')
    print(f'{"="*60}')
    print(f'  [단일]  LGBM   OOF MAE : {mae_lg:.4f}')
    print(f'  [단일]  CB     OOF MAE : {mae_cb:.4f}')
    print(f'  [단일]  TW1.8  OOF MAE : {mae_tw:.4f}')
    print(f'  [단일]  ET     OOF MAE : {mae_et:.4f}')
    print(f'  [4모델] 가중치 앙상블  : {best_loss:.4f}')
    print(f'  [4모델] LGBM-meta      : {mae_meta:.4f}')
    print(f'\n  [비교] v1 LGBM+CB+ET    : CV 8.8541 / Public 10.3032 🏆')
    print(f'  [비교] v2 LGBM+TW+ET    : CV 8.8087 / Public 10.3118')
    print(f'  [비교] v3 LGBM+TW+CB+ET : CV {mae_meta:.4f}')


if __name__ == '__main__':
    main()
