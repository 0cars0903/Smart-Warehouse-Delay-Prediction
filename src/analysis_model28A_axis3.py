"""
축3 극값 분석: model28A 체크포인트 기반
=============================================================
model28A base learner OOF를 사용하여 극값 예측 실패 원인을 정밀 진단.
model22 → model28A 개선 효과와 잔여 갭을 정량화.

핵심 질문:
  1. 비율 피처가 극값 구간 base learner 예측을 얼마나 개선했는가?
  2. 메타 학습기의 극값 복원 능력 한계는 어디인가?
  3. 잔여 갭(8.47 → 목표 8.30)을 줄이려면 어떤 방향이 유효한가?

실행: python src/analysis_model28A_axis3.py
예상 시간: ~1분 (체크포인트 재사용)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings, os, sys, gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
CKPT_28A = os.path.join(_BASE, '..', 'docs', 'model28A_ckpt')
CKPT_22  = os.path.join(_BASE, '..', 'docs', 'model22_ckpt')
N_SPLITS = 5


def load_ckpt(ckpt_dir, name):
    oof  = np.load(os.path.join(ckpt_dir, f'{name}_oof.npy'))
    test = np.load(os.path.join(ckpt_dir, f'{name}_test.npy'))
    return oof, test


def main():
    print('=' * 70)
    print('축3 극값 분석: model28A vs model22 체크포인트 비교')
    print('=' * 70)

    # ── 데이터 로드 (정렬 필수) ──
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    train['ts_idx'] = train.groupby('scenario_id').cumcount()
    train = train.sort_values(['scenario_id', 'ts_idx']).reset_index(drop=True)
    y_raw = train['avg_delay_minutes_next_30m'].values
    groups = train['scenario_id']

    print(f'train: {len(train)} rows')
    print(f'target: mean={y_raw.mean():.2f}, std={y_raw.std():.2f}, max={y_raw.max():.2f}')

    # ── 체크포인트 로드 ──
    models = ['lgbm', 'tw18', 'cb', 'et', 'rf']
    labels = ['LGBM', 'TW1.8', 'CB', 'ET', 'RF']

    oof_22, oof_28A = {}, {}
    test_22, test_28A = {}, {}
    for m in models:
        oof_22[m], test_22[m] = load_ckpt(CKPT_22, m)
        oof_28A[m], test_28A[m] = load_ckpt(CKPT_28A, m)

    # OOF를 raw space로 변환
    def to_raw(oof_dict):
        return {
            'LGBM': np.expm1(oof_dict['lgbm']),
            'TW1.8': oof_dict['tw18'],
            'CB': np.expm1(oof_dict['cb']),
            'ET': np.expm1(oof_dict['et']),
            'RF': np.expm1(oof_dict['rf']),
        }

    raw_22  = to_raw(oof_22)
    raw_28A = to_raw(oof_28A)

    # ══════════════════════════════════════════════
    # 1. 전체 OOF MAE 비교
    # ══════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('[1] Base Learner OOF MAE: model22 → model28A')
    print('═' * 70)
    for label in labels:
        mae_22  = np.abs(raw_22[label] - y_raw).mean()
        mae_28A = np.abs(raw_28A[label] - y_raw).mean()
        delta = mae_28A - mae_22
        print(f'  {label:6s}: {mae_22:.4f} → {mae_28A:.4f}  Δ={delta:+.4f} {"✅" if delta < 0 else "⚠️"}')

    # ══════════════════════════════════════════════
    # 2. 타겟 구간별 Base Learner 예측 분석
    # ══════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('[2] 타겟 구간별 Base Learner MAE 비교')
    print('═' * 70)

    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]

    for label in labels:
        print(f'\n  ── {label} ──')
        for lo, hi in bins:
            mask = (y_raw >= lo) & (y_raw < hi)
            n = mask.sum()
            if n == 0:
                continue
            mae_22  = np.abs(raw_22[label][mask] - y_raw[mask]).mean()
            mae_28A = np.abs(raw_28A[label][mask] - y_raw[mask]).mean()
            pred_22  = raw_22[label][mask].mean()
            pred_28A = raw_28A[label][mask].mean()
            actual   = y_raw[mask].mean()
            delta = mae_28A - mae_22
            print(f'    [{lo:3d},{hi:3d}) n={n:6d} | '
                  f'MAE: {mae_22:6.2f}→{mae_28A:6.2f} Δ={delta:+5.2f} | '
                  f'pred: {pred_22:6.2f}→{pred_28A:6.2f} | actual={actual:6.2f}')

    # ══════════════════════════════════════════════
    # 3. 극값 구간 정밀 분석 (target >= 50)
    # ══════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('[3] 극값 구간 정밀 분석 (target ≥ 50)')
    print('═' * 70)

    mask_50 = y_raw >= 50
    mask_80 = y_raw >= 80
    actual_50_mean = y_raw[mask_50].mean()
    actual_80_mean = y_raw[mask_80].mean()

    print(f'\n  target≥50: n={mask_50.sum()} ({mask_50.mean()*100:.1f}%), mean={actual_50_mean:.2f}')
    print(f'  target≥80: n={mask_80.sum()} ({mask_80.mean()*100:.1f}%), mean={actual_80_mean:.2f}')

    print(f'\n  ── model22 vs model28A: 예측 비율 (pred/actual) ──')
    for label in labels:
        ratio_22_50  = raw_22[label][mask_50].mean() / actual_50_mean
        ratio_28A_50 = raw_28A[label][mask_50].mean() / actual_50_mean
        ratio_22_80  = raw_22[label][mask_80].mean() / actual_80_mean
        ratio_28A_80 = raw_28A[label][mask_80].mean() / actual_80_mean
        print(f'  {label:6s}: [≥50] {ratio_22_50:.3f}→{ratio_28A_50:.3f}  '
              f'[≥80] {ratio_22_80:.3f}→{ratio_28A_80:.3f}')

    # ══════════════════════════════════════════════
    # 4. 극값 시나리오 특성 분석
    # ══════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('[4] 극값 시나리오 특성 (target ≥ 50 시나리오 vs 전체)')
    print('═' * 70)

    # 시나리오별 평균 target
    sc_mean = train.groupby('scenario_id')['avg_delay_minutes_next_30m'].mean()
    extreme_sc = sc_mean[sc_mean >= 50].index
    normal_sc  = sc_mean[sc_mean < 20].index
    mid_sc     = sc_mean[(sc_mean >= 20) & (sc_mean < 50)].index

    print(f'  극값 시나리오 (mean≥50): {len(extreme_sc)} ({len(extreme_sc)/len(sc_mean)*100:.1f}%)')
    print(f'  중간 시나리오 (20≤mean<50): {len(mid_sc)} ({len(mid_sc)/len(sc_mean)*100:.1f}%)')
    print(f'  일반 시나리오 (mean<20): {len(normal_sc)} ({len(normal_sc)/len(sc_mean)*100:.1f}%)')

    # 극값 시나리오의 피처 통계 (원본 피처)
    numeric_cols = [c for c in train.columns
                    if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m', 'ts_idx'}
                    and train[c].dtype in ['float64', 'float32', 'int64']]

    ext_mask = train['scenario_id'].isin(extreme_sc)
    norm_mask = train['scenario_id'].isin(normal_sc)

    print(f'\n  ── 극값 vs 일반 시나리오 피처 차이 (상위 15개) ──')
    diffs = {}
    for col in numeric_cols[:30]:  # 원본 피처 위주
        ext_m  = train.loc[ext_mask, col].mean()
        norm_m = train.loc[norm_mask, col].mean()
        overall_std = train[col].std()
        if overall_std > 0:
            shift = abs(ext_m - norm_m) / overall_std
            diffs[col] = (shift, ext_m, norm_m)

    for col, (shift, ext_m, norm_m) in sorted(diffs.items(), key=lambda x: -x[1][0])[:15]:
        print(f'    {col:35s}: extreme={ext_m:8.3f}  normal={norm_m:8.3f}  diff={shift:.2f}σ')

    # ══════════════════════════════════════════════
    # 5. 메타 학습기 극값 복원 분석
    # ══════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('[5] 메타 학습기 극값 복원 한계 분석')
    print('═' * 70)

    # model28A 메타 OOF 재현
    META_PARAMS = {
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'objective': 'regression_l1', 'n_estimators': 500,
        'bagging_freq': 1, 'random_state': 42,
        'verbosity': -1, 'n_jobs': -1,
    }

    meta_train = np.column_stack([
        oof_28A['lgbm'],
        oof_28A['cb'],
        np.log1p(np.maximum(oof_28A['tw18'], 0)),
        oof_28A['et'],
        oof_28A['rf'],
    ])

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_meta = np.zeros(len(y_raw))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        m = lgb.LGBMRegressor(**META_PARAMS)
        y_tr_log = np.log1p(y_raw[tr_idx])
        y_va_log = np.log1p(y_raw[va_idx])
        m.fit(meta_train[tr_idx], y_tr_log,
              eval_set=[(meta_train[va_idx], y_va_log)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[va_idx] = np.expm1(m.predict(meta_train[va_idx]))
        del m; gc.collect()

    meta_mae = np.abs(oof_meta - y_raw).mean()
    print(f'  메타 OOF MAE: {meta_mae:.4f}')

    # 구간별 메타 vs base 비교
    print(f'\n  ── 구간별 메타 복원 효과 (base 최고 vs 메타) ──')
    for lo, hi in bins:
        mask = (y_raw >= lo) & (y_raw < hi)
        n = mask.sum()
        if n == 0:
            continue
        # base 최고 = 각 행에서 5모델 중 가장 정확한 예측의 평균 MAE
        base_maes = np.stack([np.abs(raw_28A[l][mask] - y_raw[mask]) for l in labels], axis=1)
        best_base_mae = base_maes.min(axis=1).mean()  # oracle best
        avg_base_mae  = np.abs(raw_28A['LGBM'][mask] - y_raw[mask]).mean()  # LGBM 단독
        meta_seg_mae  = np.abs(oof_meta[mask] - y_raw[mask]).mean()
        actual_mean   = y_raw[mask].mean()
        meta_pred     = oof_meta[mask].mean()
        ratio = meta_pred / actual_mean

        print(f'    [{lo:3d},{hi:3d}) n={n:6d} | '
              f'LGBM={avg_base_mae:6.2f}  oracle={best_base_mae:6.2f}  '
              f'meta={meta_seg_mae:6.2f} | '
              f'pred={meta_pred:6.2f} actual={actual_mean:6.2f} ratio={ratio:.3f}')

    # ══════════════════════════════════════════════
    # 6. 극값 잔여 MAE 기여도
    # ══════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('[6] 구간별 전체 MAE 기여도 (잔여 갭 분석)')
    print('═' * 70)

    total_mae = np.abs(oof_meta - y_raw).mean()
    print(f'  전체 메타 OOF MAE: {total_mae:.4f}')
    print(f'  목표 (1위 달성): ~8.30 (갭 {total_mae - 8.30:.4f})')
    print()

    for lo, hi in bins:
        mask = (y_raw >= lo) & (y_raw < hi)
        n = mask.sum()
        if n == 0:
            continue
        seg_mae  = np.abs(oof_meta[mask] - y_raw[mask]).mean()
        seg_contribution = seg_mae * n / len(y_raw)
        pct = seg_contribution / total_mae * 100
        print(f'    [{lo:3d},{hi:3d}) n={n:6d} ({n/len(y_raw)*100:5.1f}%) | '
              f'MAE={seg_mae:6.2f} | '
              f'기여={seg_contribution:.4f} ({pct:5.1f}%)')

    # ══════════════════════════════════════════════
    # 7. 개선 시뮬레이션: 극값 MAE를 X% 줄이면?
    # ══════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('[7] 개선 시뮬레이션: 구간별 MAE 감소 → 전체 MAE 영향')
    print('═' * 70)

    for target_seg, pct_improve in [
        ('[50,80)', 0.20), ('[50,80)', 0.50),
        ('[80,800)', 0.20), ('[80,800)', 0.50),
        ('[30,50)', 0.20),
        ('[0,5)', 0.10),
    ]:
        lo, hi = int(target_seg.split(',')[0][1:]), int(target_seg.split(',')[1][:-1])
        mask = (y_raw >= lo) & (y_raw < hi)
        n = mask.sum()
        seg_mae = np.abs(oof_meta[mask] - y_raw[mask]).mean()
        reduced_mae = seg_mae * (1 - pct_improve)
        delta_total = (seg_mae - reduced_mae) * n / len(y_raw)
        new_total = total_mae - delta_total
        print(f'  {target_seg} MAE {pct_improve*100:.0f}%↓ ({seg_mae:.2f}→{reduced_mae:.2f}): '
              f'전체 MAE {total_mae:.4f}→{new_total:.4f} (Δ={-delta_total:.4f})')

    # ══════════════════════════════════════════════
    # 8. 결론 및 권장 방향
    # ══════════════════════════════════════════════
    print('\n' + '═' * 70)
    print('[8] 결론')
    print('═' * 70)

    # 구간별 개선 여지 순위
    print('  구간별 "개선 가능량 × 실현 가능성" 순위:')
    for lo, hi in bins:
        mask = (y_raw >= lo) & (y_raw < hi)
        n = mask.sum()
        seg_mae = np.abs(oof_meta[mask] - y_raw[mask]).mean()
        contribution = seg_mae * n / len(y_raw)
        # 실현 가능성: 현재 ratio가 낮을수록 개선 가능성 높음
        ratio = oof_meta[mask].mean() / (y_raw[mask].mean() + 1e-8)
        headroom = (1 - ratio) if ratio < 1 else (ratio - 1)
        score = contribution * headroom
        print(f'    [{lo:3d},{hi:3d}): 기여={contribution:.4f}  '
              f'ratio={ratio:.3f}  headroom={headroom:.3f}  '
              f'score={score:.4f}')

    print(f'\n{"=" * 70}')


if __name__ == '__main__':
    main()
