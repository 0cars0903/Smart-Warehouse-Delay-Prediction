"""
model47 + Multi-Q 스태킹 (경량 버전)
================================================================
model47 OOF(6모델) + q95 + q85/q90 조합으로 8/9모델 스태킹 시도.

근거:
  - model47+q95 pred_std=16.35 (역대 최고) → multi-Q로 pred_std 추가 확장 시도
  - q85/q90이 model34 base에서는 무효(높은 상관)였으나,
    model47의 다양한 FE(SC_AGG+lx_)와 결합 시 다른 결과 가능성
  - pred_std 추가 확장 → 배율 추가 개선 기대

기준: model47+q95 CV=8.4610 / Public=9.7901 (현재 최고)

실행: python src/run_model47_multi_q.py
예상 시간: ~5분
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
            print(f'  ⚠️ {n} 체크포인트 없음')
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
    print('▶ 공통 데이터 로드')
    y_tr   = np.load(os.path.join(DOCS_DIR, 'y_tr_fe_order.npy'))
    grp    = np.load(os.path.join(DOCS_DIR, 'grp_fe_order.npy'), allow_pickle=True)
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    pred_col = [c for c in sample.columns if c != 'ID'][0]
    os.makedirs(SUB_DIR, exist_ok=True)

    # ── Quantile OOF 로드 (q70~q95 전체) ──
    print('\n▶ Quantile OOF 로드')
    q_oofs = {}
    for qname in ['q70', 'q75', 'q80', 'q85', 'q90', 'q95']:
        op = os.path.join(CKPT_Q, f'{qname}_oof.npy')
        tp = os.path.join(CKPT_Q, f'{qname}_test.npy')
        if os.path.exists(op):
            q_oofs[qname] = (np.load(op), np.load(tp))
            mae = np.abs(q_oofs[qname][0] - y_tr).mean()
            print(f'  {qname} OOF MAE={mae:.4f}')
        else:
            print(f'  ⚠️ {qname} 없음')

    # ── model47 기본 OOF 로드 ──
    print('\n▶ model47 체크포인트 로드')
    oof_47, test_47 = load_ckpt_dict(CKPT_47, BASE_NAMES)

    # ── 상관 확인 (vs q95) ──
    print('\n▶ 각 Quantile ↔ q95 상관:')
    q95_oof = q_oofs['q95'][0] if 'q95' in q_oofs else None
    lgbm_oof = oof_47.get('lgbm')
    for qname in ['q70', 'q75', 'q80', 'q85', 'q90']:
        if qname in q_oofs and q95_oof is not None:
            c_q95  = np.corrcoef(q_oofs[qname][0], q95_oof)[0, 1]
            c_lgbm = np.corrcoef(q_oofs[qname][0], lgbm_oof)[0, 1] if lgbm_oof is not None else float('nan')
            print(f'  {qname}: q95 상관={c_q95:.4f}  lgbm 상관={c_lgbm:.4f}')

    print(f'\n{"="*60}')
    print(f'  기준: model47+q95 CV=8.4610 / Public=9.7901')
    print(f'  실패: q95+q85 Public=9.7995 / q95+q90 Public=9.7993')
    print(f'  주목: q70 상관=0.9588 (q85=0.9848보다 낮음 → 다양성 더 높음)')
    print(f'{"="*60}')

    results = {}

    def run_config(label, extra_qs):
        """extra_qs: list of quantile names to add"""
        od = dict(oof_47); td = dict(test_47)
        for qn in extra_qs:
            if qn in q_oofs:
                od[qn], td[qn] = q_oofs[qn]
        n_models = len(od)
        print(f'\n── {label} ({n_models}모델) ──')
        cv, _, tm = train_meta(od, td, y_tr, grp, label)
        results[label] = (cv, tm)
        sub = sample.copy()
        sub[pred_col] = np.clip(tm, 0, None)
        fname = f'model47_mq_{label.replace("+","_").replace(" ","")}_cv{cv:.4f}.csv'
        sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f'  💾 {fname}')
        return cv, tm

    # 기준 재현
    run_config('q95', ['q95'])

    # ── 하위 분위수 단독 추가 (q70이 핵심 — 가장 낮은 상관) ──
    run_config('q95+q70', ['q95', 'q70'])
    run_config('q95+q75', ['q95', 'q75'])
    run_config('q95+q80', ['q95', 'q80'])

    # ── 상·하 분위수 동시 (q95 상위 + q70 하위 양쪽 브라켓) ──
    run_config('q95+q70+q85', ['q95', 'q70', 'q85'])
    run_config('q70+q95',     ['q70', 'q95'])   # 이름 다를 뿐 동일

    # ── 결과 요약 ──
    print(f'\n{"="*60}')
    print('결과 요약 (기준: model47+q95 CV=8.4610 / Public=9.7901)')
    print(f'{"="*60}')
    ref_cv = 8.4610
    for k, (cv, tm) in results.items():
        delta = cv - ref_cv
        mark  = '✅' if delta < 0 else ('≈' if abs(delta) < 0.0005 else '❌')
        print(f'  {k:22s}: CV={cv:.4f} (Δ{delta:+.4f}) {mark} | pred_std={tm.std():.2f}')


if __name__ == '__main__':
    main()
