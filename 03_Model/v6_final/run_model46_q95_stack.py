"""
model46 + q95 7모델 스태킹 (경량 버전)
================================================================
model46a/c OOF 체크포인트 + q95 OOF를 합쳐 7모델 메타 스태킹 수행.
y_tr/grp는 model46a가 저장한 docs/y_tr_fe_order.npy 재사용.

실험:
  (A) model46c + q95: CV 기준 8.4600 → q95 추가 시 기대 ↓
  (B) model46a + q95: CV 기준 8.4647 → q95 추가 시 기대 ↓
  (C) model46c + model46a 6모델 블렌드 (상관 확인 후)

현재 최고: q95 7모델 CV=8.4684 / Public=9.7931

실행: python src/run_model46_q95_stack.py
예상 시간: ~3분
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import os, sys, glob, warnings
warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
DOCS_DIR = os.path.join(_BASE, '..', 'docs')

CKPT_46A = os.path.join(DOCS_DIR, 'model46a_ckpt')
CKPT_46C = os.path.join(DOCS_DIR, 'model46c_ckpt')
CKPT_Q   = os.path.join(DOCS_DIR, 'model45_ckpt', 'strat_c')

N_SPLITS     = 5
RANDOM_STATE = 42

META_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}
BASE_NAMES = ['lgbm', 'cb', 'tw15', 'et', 'rf', 'asym20']


def load_ckpt_dict(ckpt_dir, names):
    oof_d, test_d = {}, {}
    for n in names:
        op = os.path.join(ckpt_dir, f'{n}_oof.npy')
        tp = os.path.join(ckpt_dir, f'{n}_test.npy')
        if os.path.exists(op) and os.path.exists(tp):
            oof_d[n]  = np.load(op)
            test_d[n] = np.load(tp)
        else:
            print(f'  ⚠️ {n} 체크포인트 없음: {op}')
    return oof_d, test_d


def train_meta(oof_dict, test_dict, y_tr, grp, label):
    names   = list(oof_dict.keys())
    Xm_tr   = np.column_stack([oof_dict[n] for n in names])
    Xm_te   = np.column_stack([test_dict[n] for n in names])
    y_log   = np.log1p(y_tr)
    oof_meta = np.zeros(len(y_tr)); preds = []
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (ti, vi) in enumerate(gkf.split(Xm_tr, y_log, grp)):
        m = lgb.LGBMRegressor(**META_PARAMS)
        m.fit(Xm_tr[ti], y_log[ti],
              eval_set=[(Xm_tr[vi], y_log[vi])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[vi] = np.expm1(m.predict(Xm_tr[vi]))
        preds.append(np.expm1(m.predict(Xm_te)))
        del m
    test_meta = np.mean(preds, axis=0)
    cv = np.abs(oof_meta - y_tr).mean()
    print(f'  [{label}] CV={cv:.4f} | pred_std={test_meta.std():.2f} | test_mean={test_meta.mean():.2f}')
    for lo, hi in [(0,5),(5,20),(20,50),(50,80),(80,800)]:
        mask = (y_tr >= lo) & (y_tr < hi)
        if not mask.any(): continue
        mae = np.abs(oof_meta[mask] - y_tr[mask]).mean()
        pr  = oof_meta[mask].mean() / (y_tr[mask].mean() + 1e-8)
        print(f'    [{lo:3d},{hi:3d}) n={mask.sum():5d}  MAE={mae:.2f}  pred/actual={pr:.3f}')
    return cv, oof_meta, test_meta


def main():
    # ── 공통 데이터 ──
    print('▶ y_tr / grp / sample 로드')
    y_tr  = np.load(os.path.join(DOCS_DIR, 'y_tr_fe_order.npy'))
    grp   = np.load(os.path.join(DOCS_DIR, 'grp_fe_order.npy'), allow_pickle=True)
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    pred_col = [c for c in sample.columns if c != 'ID'][0]
    os.makedirs(SUB_DIR, exist_ok=True)

    # ── q95 OOF 로드 ──
    print('\n▶ q95 체크포인트 로드')
    q95_oof  = np.load(os.path.join(CKPT_Q, 'q95_oof.npy'))
    q95_test = np.load(os.path.join(CKPT_Q, 'q95_test.npy'))
    q95_mae  = np.abs(q95_oof - y_tr).mean()
    print(f'  q95 OOF MAE={q95_mae:.4f}')

    print(f'\n{"="*60}')
    print(f'  기준: 6모델(46c) CV=8.4600 | 6모델(46a) CV=8.4647')
    print(f'  기준: q95(7모델) CV=8.4684 | Public=9.7931')
    print(f'{"="*60}')
    results = {}

    # ── (A) model46c + q95 ──
    print('\n── (A) model46c(6) + q95 → 7모델 ──')
    oof_46c, test_46c = load_ckpt_dict(CKPT_46C, BASE_NAMES)
    print(f'  46c 로드: {list(oof_46c.keys())}')

    # LGBM 상관 확인
    lgbm_c = np.corrcoef(np.expm1(oof_46c['lgbm']), q95_oof)[0,1]
    print(f'  46c_lgbm - q95 상관: {lgbm_c:.4f}')

    od = dict(oof_46c); td = dict(test_46c)
    od['q95'] = q95_oof; td['q95'] = q95_test
    cv_a, _, tm_a = train_meta(od, td, y_tr, grp, 'A-46c+q95')
    results['A_46c_q95'] = (cv_a, tm_a)
    sub = sample.copy()
    sub[pred_col] = np.clip(tm_a, 0, None)
    fname_a = f'model46c_q7_q95_cv{cv_a:.4f}.csv'
    sub.to_csv(os.path.join(SUB_DIR, fname_a), index=False)
    print(f'  💾 {fname_a}')

    # ── (B) model46a + q95 ──
    print('\n── (B) model46a(6) + q95 → 7모델 ──')
    oof_46a, test_46a = load_ckpt_dict(CKPT_46A, BASE_NAMES)
    print(f'  46a 로드: {list(oof_46a.keys())}')

    lgbm_a = np.corrcoef(np.expm1(oof_46a['lgbm']), q95_oof)[0,1]
    print(f'  46a_lgbm - q95 상관: {lgbm_a:.4f}')

    od = dict(oof_46a); td = dict(test_46a)
    od['q95'] = q95_oof; td['q95'] = q95_test
    cv_b, _, tm_b = train_meta(od, td, y_tr, grp, 'B-46a+q95')
    results['B_46a_q95'] = (cv_b, tm_b)
    sub = sample.copy()
    sub[pred_col] = np.clip(tm_b, 0, None)
    fname_b = f'model46a_q7_q95_cv{cv_b:.4f}.csv'
    sub.to_csv(os.path.join(SUB_DIR, fname_b), index=False)
    print(f'  💾 {fname_b}')

    # ── (C) 46c × 46a OOF 상관 → 블렌드 가치 판단 ──
    print('\n── (C) model46c vs model46a 상관 분석 ──')
    lgbm_cc = np.corrcoef(np.expm1(oof_46c['lgbm']), np.expm1(oof_46a['lgbm']))[0,1]
    print(f'  46c_lgbm - 46a_lgbm 상관: {lgbm_cc:.4f}')

    # 46c와 46a 6모델 CSV 블렌드 (제출 파일 기반)
    def blend_csvs(path_c, path_a, w_c, label):
        pred_c = pd.read_csv(path_c)[pred_col].values
        pred_a = pd.read_csv(path_a)[pred_col].values
        corr = np.corrcoef(pred_c, pred_a)[0,1]
        blended = w_c * pred_c + (1 - w_c) * pred_a
        sub = sample.copy(); sub[pred_col] = np.clip(blended, 0, None)
        fname = f'blend_46c_46a_w{int(w_c*10)}_{label}.csv'
        sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f'  [{label}] 상관={corr:.4f} | std={blended.std():.2f} → {fname}')

    # 46c_q95 × 46a_q95 블렌드
    if os.path.exists(os.path.join(SUB_DIR, fname_a)) and \
       os.path.exists(os.path.join(SUB_DIR, fname_b)):
        print('  46c_q95 × 46a_q95 블렌드:')
        for w in [0.4, 0.5, 0.6, 0.7]:
            blend_csvs(os.path.join(SUB_DIR, fname_a),
                       os.path.join(SUB_DIR, fname_b), w, f'w{int(w*10)}')

    # ── 결과 요약 ──
    print(f'\n{"="*60}')
    print('결과 요약 (기준: q95 7모델 CV=8.4684 / Public=9.7931)')
    print(f'{"="*60}')
    ref_cv = 8.4684
    for k, (cv, tm) in results.items():
        delta = cv - ref_cv
        mark  = '✅' if delta < 0 else '❌'
        print(f'  {k:20s}: CV={cv:.4f} (Δ{delta:+.4f}) {mark} | pred_std={tm.std():.2f}')

    best_k = min(results, key=lambda k: results[k][0])
    best_cv = results[best_k][0]
    if best_cv < ref_cv:
        print(f'\n  🏆 최고: {best_k} (CV={best_cv:.4f}) → 제출 우선')
    else:
        print(f'\n  전 Config 기준 미달')


if __name__ == '__main__':
    main()
