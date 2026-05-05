"""
model47 + q95 7모델 스태킹 (경량 버전)
================================================================
model47 OOF 체크포인트(SC_AGG 확장 + Layout 교호작용 동시 적용) + q95 OOF를
합쳐 7모델 메타 스태킹 수행.

model47 6모델: CV=8.4649, pred_std=16.14
q95 OOF MAE: ~17.92 (기존과 동일)

현재 최고: model45c_q7_q95 CV=8.4684 / Public=9.7931
비교: model46a+q95 CV=8.4615 / Public=9.7997
    model46c+q95 CV=8.4639 / Public=9.7957

실행: python src/run_model47_q95_stack.py
예상 시간: ~3분
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import os, warnings
warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
DOCS_DIR = os.path.join(_BASE, '..', 'docs')

CKPT_47 = os.path.join(DOCS_DIR, 'model47_ckpt')
CKPT_Q  = os.path.join(DOCS_DIR, 'model45_ckpt', 'strat_c')

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
    names    = list(oof_dict.keys())
    Xm_tr    = np.column_stack([oof_dict[n] for n in names])
    Xm_te    = np.column_stack([test_dict[n] for n in names])
    y_log    = np.log1p(y_tr)
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
    print('▶ y_tr / grp / sample 로드')
    y_tr   = np.load(os.path.join(DOCS_DIR, 'y_tr_fe_order.npy'))
    grp    = np.load(os.path.join(DOCS_DIR, 'grp_fe_order.npy'), allow_pickle=True)
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    pred_col = [c for c in sample.columns if c != 'ID'][0]
    os.makedirs(SUB_DIR, exist_ok=True)

    # q95 OOF 로드
    print('\n▶ q95 체크포인트 로드')
    q95_oof  = np.load(os.path.join(CKPT_Q, 'q95_oof.npy'))
    q95_test = np.load(os.path.join(CKPT_Q, 'q95_test.npy'))
    q95_mae  = np.abs(q95_oof - y_tr).mean()
    print(f'  q95 OOF MAE={q95_mae:.4f}')

    # model47 OOF 로드
    print('\n▶ model47 체크포인트 로드')
    oof_47, test_47 = load_ckpt_dict(CKPT_47, BASE_NAMES)
    print(f'  model47 로드: {list(oof_47.keys())}')

    print(f'\n{"="*60}')
    print(f'  기준: model47 6모델 CV=8.4649 | pred_std=16.14')
    print(f'  기준: q95(7모델) CV=8.4684 | Public=9.7931')
    print(f'  비교: model46a+q95 CV=8.4615 | Public=9.7997')
    print(f'  비교: model46c+q95 CV=8.4639 | Public=9.7957')
    print(f'{"="*60}')

    # lgbm-q95 상관 확인
    lgbm_q = np.corrcoef(np.expm1(oof_47['lgbm']), q95_oof)[0,1]
    print(f'\n  model47_lgbm - q95 상관: {lgbm_q:.4f}')

    # model47 + q95 → 7모델
    print('\n── model47(6) + q95 → 7모델 ──')
    od = dict(oof_47); td = dict(test_47)
    od['q95'] = q95_oof; td['q95'] = q95_test

    cv, _, tm = train_meta(od, td, y_tr, grp, 'model47+q95')

    sub = sample.copy()
    sub[pred_col] = np.clip(tm, 0, None)
    fname = f'model47_q7_q95_cv{cv:.4f}.csv'
    sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
    print(f'  💾 {fname}')

    print(f'\n{"="*60}')
    ref_cv = 8.4684
    delta  = cv - ref_cv
    mark   = '✅' if delta < 0 else '❌'
    print(f'  model47+q95: CV={cv:.4f} (Δ{delta:+.4f}) {mark} | pred_std={tm.std():.2f}')
    print(f'  기준(q95 7모델): CV=8.4684 / Public=9.7931')
    if delta < 0:
        print(f'\n  🏆 기준 CV 돌파 → 제출 우선')
    else:
        print(f'\n  기준 CV 미달 — Public 제출 후 확인 필요')


if __name__ == '__main__':
    main()
