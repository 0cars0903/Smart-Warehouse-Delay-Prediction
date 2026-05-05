"""
model45c 다중 Quantile 스태킹 (경량 버전 — build_features 생략)
================================================================
OOF 체크포인트가 모두 존재하므로 feature engineering 없이 메타만 실행.
y_tr, grp = train CSV에서 직접 로드.

실험:
  (A) 8모델: model34_6 + q95 + q85
  (B) 8모델: model34_6 + q95 + q90
  (C) 9모델: model34_6 + q85 + q90 + q95

기준: q95(7모델) CV=8.4684 | Public=9.7931

실행: python src/run_model45c_multi_q_fast.py
예상 시간: ~4분
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
CKPT_C   = os.path.join(DOCS_DIR, 'model45_ckpt', 'strat_c')
RANDOM_STATE = 42
N_SPLITS     = 5

META_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}


def load_model34_config_b():
    m31 = os.path.join(DOCS_DIR, 'model31_ckpt')
    m34 = os.path.join(DOCS_DIR, 'model34_ckpt')
    mapping = {
        'lgbm':  (m31, 'lgbm'),
        'cb':    (m31, 'cb'),
        'et':    (m31, 'et'),
        'rf':    (m31, 'rf'),
        'tw15':  (m34, 'tw15'),
        'asym':  (m34, 'asym20'),
    }
    oof_d, test_d = {}, {}
    for name, (d, prefix) in mapping.items():
        op = os.path.join(d, f'{prefix}_oof.npy')
        tp = os.path.join(d, f'{prefix}_test.npy')
        if not os.path.exists(op):
            print(f"  ⚠️ {name} OOF 없음: {op}")
            continue
        oof_d[name]  = np.load(op)
        test_d[name] = np.load(tp)
        print(f"  ✅ {name}: OOF shape={oof_d[name].shape}")
    return oof_d, test_d


def load_q(qname):
    op = os.path.join(CKPT_C, f'{qname}_oof.npy')
    tp = os.path.join(CKPT_C, f'{qname}_test.npy')
    if os.path.exists(op) and os.path.exists(tp):
        return np.load(op), np.load(tp)
    print(f"  ⚠️ {qname} 체크포인트 없음")
    return None, None


def train_meta(oof_dict, test_dict, y_tr, grp, label):
    names = list(oof_dict.keys())
    Xm_tr = np.column_stack([oof_dict[n] for n in names])
    Xm_te = np.column_stack([test_dict[n] for n in names])
    y_log = np.log1p(y_tr)
    oof_meta, test_preds = np.zeros(len(y_tr)), []
    kf = GroupKFold(n_splits=N_SPLITS)
    for fold, (ti, vi) in enumerate(kf.split(Xm_tr, y_log, grp)):
        m = lgb.LGBMRegressor(**META_PARAMS)
        m.fit(Xm_tr[ti], y_log[ti],
              eval_set=[(Xm_tr[vi], y_log[vi])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[vi] = np.expm1(m.predict(Xm_tr[vi]))
        test_preds.append(np.expm1(m.predict(Xm_te)))
    test_meta = np.mean(test_preds, axis=0)
    cv = np.abs(oof_meta - y_tr).mean()
    print(f"  [{label}] CV={cv:.4f} | pred_std={test_meta.std():.2f} | "
          f"test_mean={test_meta.mean():.2f}")
    for lo, hi in [(0,5),(5,20),(20,50),(50,80),(80,800)]:
        mask = (y_tr >= lo) & (y_tr < hi)
        if not mask.any(): continue
        mae  = np.abs(oof_meta[mask] - y_tr[mask]).mean()
        pr   = oof_meta[mask].mean() / (y_tr[mask].mean() + 1e-8)
        print(f"    [{lo:3d},{hi:3d}) n={mask.sum():5d}  MAE={mae:.2f}  pred/actual={pr:.3f}")
    return cv, oof_meta, test_meta


def main():
    print("데이터 로드 중 (FE 정렬 순서 y_tr/grp 직접 로드)...")
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    # build_features()와 동일한 정렬 순서로 저장된 npy 로드
    y_tr_path  = os.path.join(DOCS_DIR, 'y_tr_fe_order.npy')
    grp_path   = os.path.join(DOCS_DIR, 'grp_fe_order.npy')
    if not os.path.exists(y_tr_path):
        raise FileNotFoundError(
            "docs/y_tr_fe_order.npy 없음. 먼저 다음 명령 실행:\n"
            "  python3 -c \"import sys;sys.path.insert(0,'src');import pandas as pd,numpy as np;"
            "from feature_engineering import build_features;...\" 또는\n"
            "  python src/run_model45c_q_stack.py 내부에서 자동 저장됨"
        )
    y_tr  = np.load(y_tr_path)
    grp   = np.load(grp_path, allow_pickle=True)
    pred_col = [c for c in sample.columns if c != 'ID'][0]
    os.makedirs(SUB_DIR, exist_ok=True)

    print("\n▶ model34 Config B 체크포인트 로드")
    base_oof, base_test = load_model34_config_b()
    print(f"  base 모델 수: {len(base_oof)}")

    print("\n▶ Quantile OOF 로드")
    q_data = {}
    for qn in ['q85', 'q90', 'q95']:
        o, t = load_q(qn)
        if o is not None:
            q_data[qn] = (o, t)
            corr = np.corrcoef(o, base_oof['lgbm'])[0,1]
            print(f"  {qn}: LGBM-corr={corr:.4f} | OOF_MAE={np.abs(o - y_tr).mean():.4f}")

    # 분위수 간 상관
    if 'q85' in q_data and 'q95' in q_data:
        c = np.corrcoef(q_data['q85'][0], q_data['q95'][0])[0,1]
        print(f"  q85-q95 상관: {c:.4f}")
    if 'q90' in q_data and 'q95' in q_data:
        c = np.corrcoef(q_data['q90'][0], q_data['q95'][0])[0,1]
        print(f"  q90-q95 상관: {c:.4f}")
    if 'q85' in q_data and 'q90' in q_data:
        c = np.corrcoef(q_data['q85'][0], q_data['q90'][0])[0,1]
        print(f"  q85-q90 상관: {c:.4f}")

    print(f"\n{'='*60}")
    print(f"  다중 Quantile 스태킹 실험")
    print(f"  기준: q95(7모델) CV=8.4684 | Public=9.7931")
    print(f"{'='*60}")

    results = {}

    # ── (A) 8모델: 6 + q95 + q85 ──────────────────────────────────
    if 'q85' in q_data and 'q95' in q_data:
        print("\n  ── (A) 8모델: model34_6 + q95 + q85 ──")
        od = dict(base_oof); td = dict(base_test)
        od['q95'], td['q95'] = q_data['q95']
        od['q85'], td['q85'] = q_data['q85']
        cv, _, test_meta = train_meta(od, td, y_tr, grp, "A-8m(q95+q85)")
        results['A'] = cv
        sub = sample.copy()
        sub[pred_col] = np.clip(test_meta, 0, None)
        fname = f"model45c_q8_q95q85_cv{cv:.4f}.csv"
        sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f"  💾 {fname}")

    # ── (B) 8모델: 6 + q95 + q90 ──────────────────────────────────
    if 'q90' in q_data and 'q95' in q_data:
        print("\n  ── (B) 8모델: model34_6 + q95 + q90 ──")
        od = dict(base_oof); td = dict(base_test)
        od['q95'], td['q95'] = q_data['q95']
        od['q90'], td['q90'] = q_data['q90']
        cv, _, test_meta = train_meta(od, td, y_tr, grp, "B-8m(q95+q90)")
        results['B'] = cv
        sub = sample.copy()
        sub[pred_col] = np.clip(test_meta, 0, None)
        fname = f"model45c_q8_q95q90_cv{cv:.4f}.csv"
        sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f"  💾 {fname}")

    # ── (C) 9모델: 6 + q85 + q90 + q95 ───────────────────────────
    if all(q in q_data for q in ['q85', 'q90', 'q95']):
        print("\n  ── (C) 9모델: model34_6 + q85 + q90 + q95 ──")
        od = dict(base_oof); td = dict(base_test)
        od['q85'], td['q85'] = q_data['q85']
        od['q90'], td['q90'] = q_data['q90']
        od['q95'], td['q95'] = q_data['q95']
        cv, _, test_meta = train_meta(od, td, y_tr, grp, "C-9m(all-q)")
        results['C'] = cv
        sub = sample.copy()
        sub[pred_col] = np.clip(test_meta, 0, None)
        fname = f"model45c_q9_all_cv{cv:.4f}.csv"
        sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f"  💾 {fname}")

    # ── (D) q95 × blend_m34bd_b60 블렌드 ─────────────────────────
    print("\n  ── (D) q95 × blend_m34bd_b60 블렌드 상관 확인 ──")
    b60_path = os.path.join(SUB_DIR, 'blend_m34bd_b60.csv')
    q95_path  = None
    import glob
    q95_files = sorted(glob.glob(os.path.join(SUB_DIR, 'model45c_q7_q95*.csv')))
    if q95_files:
        q95_path = q95_files[-1]

    if os.path.exists(str(b60_path)) and q95_path and os.path.exists(q95_path):
        pred_b60 = pd.read_csv(b60_path)[pred_col].values
        pred_q95 = pd.read_csv(q95_path)[pred_col].values
        corr = np.corrcoef(pred_b60, pred_q95)[0,1]
        print(f"  b60-q95 상관: {corr:.4f}")
        if corr < 0.999:
            print("  → 상관 < 0.999, 블렌드 파일 생성")
            for w_q in [0.3, 0.4, 0.5, 0.6, 0.7]:
                blended = w_q * pred_q95 + (1 - w_q) * pred_b60
                sub = sample.copy()
                sub[pred_col] = np.clip(blended, 0, None)
                fname = f"blend_q95_b60_w{int(w_q*10)}.csv"
                sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
                print(f"    💾 {fname}  (q95×{w_q:.1f} + b60×{1-w_q:.1f})")
        else:
            print("  → 상관 ≥ 0.999, 블렌드 효과 없음 — 스킵")
    else:
        print(f"  ⚠️ 파일 없음: b60={os.path.exists(str(b60_path))}, q95={bool(q95_path)}")

    # ── 결과 요약 ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  결과 요약 (기준: q95(7모델) CV=8.4684)")
    print(f"{'='*60}")
    ref = 8.4684
    for k, cv in results.items():
        delta = cv - ref
        mark = "✅" if delta < 0 else "❌"
        print(f"  Config {k}: CV={cv:.4f} (Δ{delta:+.4f}) {mark}")

    if results:
        best_k = min(results, key=results.get)
        best_cv = results[best_k]
        if best_cv < ref:
            print(f"\n  🏆 최고: Config {best_k} (CV={best_cv:.4f}) → 제출 우선")
        else:
            print(f"\n  전 Config 기준 미달 — 현재 q95 7모델이 최강")


if __name__ == '__main__':
    main()
