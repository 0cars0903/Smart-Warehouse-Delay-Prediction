"""
모델실험27: 7모델 하이브리드 스태킹 (v3.0 Phase 2)
=============================================================
model22(11통계 시나리오 집계, Public 최고 9.9385) base에
model26 시퀀스 모델(CNN + BiLSTM) OOF를 추가한 7모델 스태킹.

핵심 변경:
  - Base learner 7종: LGBM, TW1.8, CB, ET, RF (model22 체크포인트)
                      + CNN, LSTM (model26 체크포인트)
  - 메타 입력: 7차원 (기존 5차원 + CNN OOF + LSTM OOF)
  - model22 11통계 시나리오 집계 피처 (198 sc피처) 기반
  - Phase 1 결과: CNN-LGBM 0.9063, LSTM-LGBM 0.9386 (다양성 ✅)
    CNN-TW 0.8253 (GBDT 쌍 최저 TW-ET 0.8994보다 낮음 — 이질적 신호)

예상:
  - model22 메타 CV ~8.51 → 시퀀스 모델 다양성으로 개선 기대
  - 특히 시나리오 내 시계열 패턴(급등/급락)에서 시퀀스 모델 보완 효과
  - CNN-TW 0.8253: 기존 GBDT 어떤 쌍보다 낮은 상관 → 메타에서 새 정보

의존성:
  - docs/model22_ckpt/ (GBDT 5모델 OOF + test)
  - docs/model26_ckpt/ (CNN + LSTM flat OOF + test)

실행: python src/run_exp_model27_hybrid_stacking.py
예상 시간: ~5분 (체크포인트 로드 + 메타 학습만)
출력: submissions/model27_hybrid_stacking.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import warnings, gc, os, sys, time

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
M22_CKPT = os.path.join(_BASE, '..', 'docs', 'model22_ckpt')
M26_CKPT = os.path.join(_BASE, '..', 'docs', 'model26_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 메타 학습기 파라미터
# ─────────────────────────────────────────────
META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}


# ─────────────────────────────────────────────
# 체크포인트 로드
# ─────────────────────────────────────────────
def load_all_oofs():
    """
    model22 (GBDT 5종) + model26 (시퀀스 2종) 체크포인트 로드.

    OOF 순서 정렬:
      - model22 GBDT: build_features → scenario_id+ts_idx 정렬 순서
      - model26 시퀀스: sorted(scenario_id) × 25 ts → 동일한 정렬 순서
      ∴ 직접 column_stack 가능 (재정렬 불필요)

    공간:
      - LGBM, CB, ET, RF: log1p space
      - TW1.8: raw space
      - CNN, LSTM flat: raw space (model26에서 expm1 적용 후 저장)
    """
    print('  [model22] GBDT 체크포인트 로드...')
    oof_lgbm = np.load(os.path.join(M22_CKPT, 'lgbm_oof.npy'))    # log1p
    oof_cb   = np.load(os.path.join(M22_CKPT, 'cb_oof.npy'))      # log1p
    oof_tw   = np.load(os.path.join(M22_CKPT, 'tw18_oof.npy'))    # raw
    oof_et   = np.load(os.path.join(M22_CKPT, 'et_oof.npy'))      # log1p
    oof_rf   = np.load(os.path.join(M22_CKPT, 'rf_oof.npy'))      # log1p

    test_lgbm = np.load(os.path.join(M22_CKPT, 'lgbm_test.npy'))
    test_cb   = np.load(os.path.join(M22_CKPT, 'cb_test.npy'))
    test_tw   = np.load(os.path.join(M22_CKPT, 'tw18_test.npy'))
    test_et   = np.load(os.path.join(M22_CKPT, 'et_test.npy'))
    test_rf   = np.load(os.path.join(M22_CKPT, 'rf_test.npy'))

    print('  [model26] 시퀀스 체크포인트 로드...')
    oof_cnn  = np.load(os.path.join(M26_CKPT, 'cnn_oof_flat.npy'))   # raw
    oof_lstm = np.load(os.path.join(M26_CKPT, 'lstm_oof_flat.npy'))  # raw

    test_cnn  = np.load(os.path.join(M26_CKPT, 'cnn_test_flat.npy'))   # raw
    test_lstm = np.load(os.path.join(M26_CKPT, 'lstm_test_flat.npy'))  # raw

    # 크기 검증
    n_train = len(oof_lgbm)
    n_test  = len(test_lgbm)
    assert len(oof_cnn) == n_train, f'CNN OOF 크기 불일치: {len(oof_cnn)} != {n_train}'
    assert len(oof_lstm) == n_train, f'LSTM OOF 크기 불일치: {len(oof_lstm)} != {n_train}'
    assert len(test_cnn) == n_test, f'CNN test 크기 불일치: {len(test_cnn)} != {n_test}'
    assert len(test_lstm) == n_test, f'LSTM test 크기 불일치: {len(test_lstm)} != {n_test}'
    print(f'  OOF 크기: train={n_train}, test={n_test} ✅')

    return {
        'oof': {
            'lgbm': oof_lgbm, 'cb': oof_cb, 'tw': oof_tw,
            'et': oof_et, 'rf': oof_rf,
            'cnn': oof_cnn, 'lstm': oof_lstm,
        },
        'test': {
            'lgbm': test_lgbm, 'cb': test_cb, 'tw': test_tw,
            'et': test_et, 'rf': test_rf,
            'cnn': test_cnn, 'lstm': test_lstm,
        },
    }


# ─────────────────────────────────────────────
# 상관 분석
# ─────────────────────────────────────────────
def analyze_correlations(oofs_raw):
    """7모델 OOF 상관 행렬 출력"""
    names = ['LGBM', 'TW', 'CB', 'ET', 'RF', 'CNN', 'LSTM']
    arrays = [oofs_raw[n.lower()] for n in names]

    print(f'\n{"":8s}', end='')
    for n in names:
        print(f'{n:>8s}', end='')
    print()

    corr_matrix = {}
    for i, (n1, o1) in enumerate(zip(names, arrays)):
        print(f'{n1:8s}', end='')
        for j, (n2, o2) in enumerate(zip(names, arrays)):
            c = np.corrcoef(o1, o2)[0, 1]
            corr_matrix[(n1, n2)] = c
            print(f'{c:8.4f}', end='')
        print()

    # 시퀀스 vs GBDT 핵심 상관
    print(f'\n시퀀스-GBDT 핵심 상관:')
    for seq in ['CNN', 'LSTM']:
        for gbdt in ['LGBM', 'TW', 'CB', 'ET', 'RF']:
            c = corr_matrix[(seq, gbdt)]
            status = '✅' if c < 0.95 else '❌'
            print(f'  {seq}-{gbdt}: {c:.4f} {status}')
    print(f'  CNN-LSTM: {corr_matrix[("CNN", "LSTM")]:.4f}')

    return corr_matrix


# ─────────────────────────────────────────────
# 가중치 앙상블 (비교용)
# ─────────────────────────────────────────────
def weighted_ensemble(oofs_raw, y_raw):
    """7모델 가중 앙상블 (Nelder-Mead 최적화)"""
    names = ['lgbm', 'tw', 'cb', 'et', 'rf', 'cnn', 'lstm']
    arrays = [oofs_raw[n] for n in names]

    def loss7(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        blend = sum(w[i] * arrays[i] for i in range(7))
        return np.mean(np.abs(blend - y_raw))

    best_loss, best_w = np.inf, np.ones(7) / 7
    for _ in range(500):
        w0 = np.random.dirichlet(np.ones(7))
        res = minimize(loss7, w0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-6})
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = np.abs(res.x) / np.abs(res.x).sum()

    labels = ['LGBM', 'TW', 'CB', 'ET', 'RF', 'CNN', 'LSTM']
    print(f'\n  7모델 가중 앙상블 CV MAE: {best_loss:.4f}')
    w_str = ', '.join(f'{labels[i]}={best_w[i]:.3f}' for i in range(7))
    print(f'    {w_str}')

    # 5모델(GBDT만) 비교
    def loss5(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        blend = sum(w[i] * arrays[i] for i in range(5))
        return np.mean(np.abs(blend - y_raw))

    best5_loss, best5_w = np.inf, np.ones(5) / 5
    for _ in range(500):
        w0 = np.random.dirichlet(np.ones(5))
        res = minimize(loss5, w0, method='Nelder-Mead')
        if res.fun < best5_loss:
            best5_loss = res.fun
            best5_w = np.abs(res.x) / np.abs(res.x).sum()

    print(f'  5모델 가중 앙상블 CV MAE: {best5_loss:.4f} (비교용)')
    delta = best_loss - best5_loss
    print(f'  시퀀스 모델 추가 효과: {delta:+.4f}')

    return best_loss, best_w, best5_loss


# ─────────────────────────────────────────────
# Layer 2: LGBM 메타 학습기
# ─────────────────────────────────────────────
def run_meta_lgbm(meta_train, meta_test, y_raw, groups, label='7모델-meta'):
    """
    GroupKFold 5-fold 메타 학습기.
    입력: log1p space OOF 7차원 (model22 패턴 유지)
    타겟: log1p(y_raw)
    출력: raw space 예측
    """
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
        mae = np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_mae = np.abs(oof_meta - y_raw.values).mean()
    pred_std = oof_meta.std()
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | pred_std={pred_std:.2f}')
    return oof_meta, test_meta, oof_mae


def run_meta_lgbm_5model(meta_train_5, meta_test_5, y_raw, groups, label='5모델-meta'):
    """5모델 메타 (model22 재현, 비교 기준)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_meta  = np.zeros(len(y_raw))
    test_meta = np.zeros(meta_test_5.shape[0])

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train_5, y_raw, groups)):
        X_tr, X_va = meta_train_5[tr_idx], meta_train_5[va_idx]
        y_tr_log = np.log1p(y_raw.iloc[tr_idx].values)
        y_va_log = np.log1p(y_raw.iloc[va_idx].values)

        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(X_tr, y_tr_log,
              eval_set=[(X_va, y_va_log)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        oof_meta[va_idx] = np.expm1(m.predict(X_va))
        test_meta       += np.expm1(m.predict(meta_test_5)) / N_SPLITS
        mae = np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_mae = np.abs(oof_meta - y_raw.values).mean()
    pred_std = oof_meta.std()
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | pred_std={pred_std:.2f}')
    return oof_meta, test_meta, oof_mae


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 60)
    print('모델실험27: 7모델 하이브리드 스태킹 (v3.0 Phase 2)')
    print('기준: model22 CV ~8.51 / Public 9.9385')
    print('변경: 5모델 → 7모델 (+ CNN + LSTM)')
    print('Phase 1: CNN-LGBM 0.9063, LSTM-LGBM 0.9386 (다양성 확인)')
    print('=' * 60)

    os.makedirs(SUB_DIR, exist_ok=True)

    # ══════════════════════════════════════════
    # 데이터 로드 (y_raw, groups 추출용)
    # ══════════════════════════════════════════
    print('\n[1/6] 데이터 로드...')
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    # build_features → scenario_id+ts_idx 정렬됨 (OOF와 동일 순서)
    train, test = build_features(
        train, test, layout,
        lag_lags=[1,2,3,4,5,6],
        rolling_windows=[3,5,10],
    )
    y_raw  = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']
    print(f'  train: {len(train)}, test: {len(test)}')
    print(f'  y_raw: mean={y_raw.mean():.2f}, std={y_raw.std():.2f}, max={y_raw.max():.1f}')

    # ══════════════════════════════════════════
    # 체크포인트 로드
    # ══════════════════════════════════════════
    print('\n[2/6] 체크포인트 로드...')
    data = load_all_oofs()

    # ══════════════════════════════════════════
    # OOF MAE 개별 확인
    # ══════════════════════════════════════════
    print('\n[3/6] 개별 모델 OOF MAE 확인')
    print('─' * 60)

    # raw space 변환
    oofs_raw = {}
    oofs_raw['lgbm'] = np.expm1(data['oof']['lgbm'])
    oofs_raw['cb']   = np.expm1(data['oof']['cb'])
    oofs_raw['tw']   = data['oof']['tw']           # 이미 raw
    oofs_raw['et']   = np.expm1(data['oof']['et'])
    oofs_raw['rf']   = np.expm1(data['oof']['rf'])
    oofs_raw['cnn']  = data['oof']['cnn']           # 이미 raw
    oofs_raw['lstm'] = data['oof']['lstm']           # 이미 raw

    test_raw = {}
    test_raw['lgbm'] = np.expm1(data['test']['lgbm'])
    test_raw['cb']   = np.expm1(data['test']['cb'])
    test_raw['tw']   = data['test']['tw']
    test_raw['et']   = np.expm1(data['test']['et'])
    test_raw['rf']   = np.expm1(data['test']['rf'])
    test_raw['cnn']  = data['test']['cnn']
    test_raw['lstm'] = data['test']['lstm']

    names_ordered = ['lgbm', 'tw', 'cb', 'et', 'rf', 'cnn', 'lstm']
    labels = ['LGBM', 'TW1.8', 'CB', 'ET', 'RF', 'CNN', 'LSTM']
    for name, label in zip(names_ordered, labels):
        mae = np.abs(oofs_raw[name] - y_raw.values).mean()
        std = oofs_raw[name].std()
        print(f'  {label:6s} OOF MAE={mae:.4f}  pred_std={std:.2f}')

    # ══════════════════════════════════════════
    # 상관 분석
    # ══════════════════════════════════════════
    print('\n[4/6] 상관 분석')
    print('─' * 60)
    corr_matrix = analyze_correlations(oofs_raw)

    # ══════════════════════════════════════════
    # 가중 앙상블 (비교용)
    # ══════════════════════════════════════════
    print('\n[5/6] 가중 앙상블 비교')
    print('─' * 60)
    ens7_mae, ens7_w, ens5_mae = weighted_ensemble(oofs_raw, y_raw.values)

    # ══════════════════════════════════════════
    # Layer 2: 메타 학습기
    # ══════════════════════════════════════════
    print('\n[6/6] 메타 학습기')
    print('─' * 60)

    # ── 메타 입력 구성 (모두 log1p space) ──
    # GBDT: LGBM, CB, ET, RF는 이미 log1p / TW는 raw → log1p 변환
    # 시퀀스: CNN, LSTM는 raw → log1p 변환
    oof_tw_log  = np.log1p(np.maximum(data['oof']['tw'], 0))
    oof_cnn_log = np.log1p(np.maximum(data['oof']['cnn'], 0))
    oof_lstm_log = np.log1p(np.maximum(data['oof']['lstm'], 0))

    test_tw_log  = np.log1p(np.maximum(data['test']['tw'], 0))
    test_cnn_log = np.log1p(np.maximum(data['test']['cnn'], 0))
    test_lstm_log = np.log1p(np.maximum(data['test']['lstm'], 0))

    # 7모델 메타 입력
    meta_train_7 = np.column_stack([
        data['oof']['lgbm'],    # log1p
        data['oof']['cb'],      # log1p
        oof_tw_log,             # log1p
        data['oof']['et'],      # log1p
        data['oof']['rf'],      # log1p
        oof_cnn_log,            # log1p
        oof_lstm_log,           # log1p
    ])
    meta_test_7 = np.column_stack([
        data['test']['lgbm'],
        data['test']['cb'],
        test_tw_log,
        data['test']['et'],
        data['test']['rf'],
        test_cnn_log,
        test_lstm_log,
    ])

    # 5모델 메타 입력 (model22 재현 — 비교 기준)
    meta_train_5 = np.column_stack([
        data['oof']['lgbm'], data['oof']['cb'], oof_tw_log,
        data['oof']['et'], data['oof']['rf'],
    ])
    meta_test_5 = np.column_stack([
        data['test']['lgbm'], data['test']['cb'], test_tw_log,
        data['test']['et'], data['test']['rf'],
    ])

    print(f'  메타 입력: 7모델 {meta_train_7.shape}, 5모델 {meta_train_5.shape}')

    # ── 5모델 메타 (비교 기준) ──
    print(f'\n  ── 5모델 메타 (model22 재현) ──')
    oof_meta5, test_meta5, mae_meta5 = run_meta_lgbm_5model(
        meta_train_5, meta_test_5, y_raw, groups
    )

    # ── 7모델 메타 (핵심 실험) ──
    print(f'\n  ── 7모델 메타 (CNN + LSTM 추가) ──')
    oof_meta7, test_meta7, mae_meta7 = run_meta_lgbm(
        meta_train_7, meta_test_7, y_raw, groups
    )

    # ══════════════════════════════════════════
    # 제출 파일 생성
    # ══════════════════════════════════════════
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    # 7모델 메타 제출
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta7, 0)
    sub_path_7 = os.path.join(SUB_DIR, 'model27_hybrid_stacking.csv')
    sample.to_csv(sub_path_7, index=False)
    print(f'\n  7모델 제출: {sub_path_7}')

    # 5모델 메타 제출 (비교용)
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta5, 0)
    sub_path_5 = os.path.join(SUB_DIR, 'model27_5model_baseline.csv')
    sample.to_csv(sub_path_5, index=False)
    print(f'  5모델 제출: {sub_path_5} (비교용)')

    # ══════════════════════════════════════════
    # 타겟 구간별 분석
    # ══════════════════════════════════════════
    print('\n' + '─' * 60)
    print('[분석] 타겟 구간별 MAE (7모델 vs 5모델)')
    print('─' * 60)
    bins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 800)]
    print(f'  {"구간":>10s}  {"n":>6s}  {"5모델":>8s}  {"7모델":>8s}  {"Δ":>8s}')
    for lo, hi in bins:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            mae5 = np.abs(oof_meta5[mask] - y_raw.values[mask]).mean()
            mae7 = np.abs(oof_meta7[mask] - y_raw.values[mask]).mean()
            delta = mae7 - mae5
            print(f'  [{lo:3d},{hi:3d})  {mask.sum():6d}  {mae5:8.2f}  {mae7:8.2f}  {delta:+8.4f}')

    # ══════════════════════════════════════════
    # 예측 분포 분석
    # ══════════════════════════════════════════
    print('\n' + '─' * 60)
    print('[분석] 예측 분포')
    print('─' * 60)
    for label, oof, test_pred in [('5모델', oof_meta5, test_meta5),
                                   ('7모델', oof_meta7, test_meta7)]:
        print(f'  {label} OOF:  mean={oof.mean():.2f}, std={oof.std():.2f}, '
              f'min={oof.min():.2f}, max={oof.max():.2f}')
        print(f'  {label} test: mean={test_pred.mean():.2f}, std={test_pred.std():.2f}, '
              f'min={test_pred.min():.2f}, max={test_pred.max():.2f}')

    # ══════════════════════════════════════════
    # 최종 요약
    # ══════════════════════════════════════════
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 60)
    print(f'모델실험27 결과 ({elapsed:.1f}분 소요)')
    print('=' * 60)
    delta_meta = mae_meta7 - mae_meta5
    delta_ens  = ens7_mae - ens5_mae

    print(f'  가중 앙상블  5모델: {ens5_mae:.4f}')
    print(f'  가중 앙상블  7모델: {ens7_mae:.4f}  ({delta_ens:+.4f})')
    print(f'  메타 LGBM   5모델: {mae_meta5:.4f}  pred_std={oof_meta5.std():.2f}')
    print(f'  메타 LGBM   7모델: {mae_meta7:.4f}  pred_std={oof_meta7.std():.2f}  ({delta_meta:+.4f})')
    print()
    print(f'  model22 기준: ~8.51 / Public 9.9385 (배율 ~1.168)')
    print(f'  model27 기대 Public (×1.168): {mae_meta7 * 1.168:.4f}')
    print(f'  model27 기대 Public (×1.170): {mae_meta7 * 1.170:.4f}')

    if delta_meta < 0:
        print(f'\n  ✅ 시퀀스 모델 추가 효과: CV {delta_meta:+.4f} 개선')
        print(f'     → model27_hybrid_stacking.csv 제출 권장')
    else:
        print(f'\n  ⚠️ 시퀀스 모델 추가 효과 없음: CV {delta_meta:+.4f}')
        print(f'     → model22 유지, 시퀀스 모델 하이퍼파라미터 조정 검토')

    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
