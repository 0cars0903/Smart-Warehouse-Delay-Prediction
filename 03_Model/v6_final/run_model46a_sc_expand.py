"""
model46a: SC_AGG 확장 — 시나리오 집계 컬럼 18 → 23개
================================================================
발견: SC_AGG는 90개 원본 중 18개만 집계. 나머지 72개 중 target 상관
      > 0.12인 고상관 피처 5개가 미활용 상태.

추가 SC_AGG 컬럼 (target 상관 순):
  avg_charge_wait       r=0.251  충전 대기 시간 — 배터리 병목 결과
  unique_sku_15m        r=0.229  SKU 다양성 — 피킹 복잡도
  loading_dock_util     r=0.213  출고 도크 가동률
  maintenance_schedule_score r=0.197  정비 계획 준수율 (음의 상관)
  manual_override_ratio r=0.196  수동 개입 비율

효과 예상:
  - 5개 × 11통계 = 55 sc 피처 추가 (총 253 sc 피처, 기존 198)
  - 모델21이 기존 FE에서 SC_AGG 추가로 CV 9.00→8.51 돌파한 메커니즘과 동일
  - 새 5개 컬럼이 시나리오 레벨에서 포착하지 못했던 물류 흐름 패턴 제공

리스크:
  - SC_AGG 컬럼 수 증가 → 피처 수 증가 (427 → 482) → 과적합 우려
  - maintenance_score는 정비 수준 측정치로 시나리오 변동이 적을 수 있음
  - 기존 모델 ckpt 재활용 불가 → 전체 재학습 필요 (~45분)

기준: model34 Config B  CV=8.4803 / Public=9.8078
      model45c q95(7모델) CV=8.4684 / Public=9.7931 ← 현 최고

실행: python src/run_model46a_sc_expand.py
예상 시간: ~45~55분 (6모델 전체 재학습)
"""

import numpy as np
import pandas as pd
import os, sys, time
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)

from run_model46_base import (
    DATA_DIR, SUB_DIR, DOCS_DIR, N_SPLITS,
    SC_AGG_BASE, add_scenario_agg, add_ratio_tier1, add_ratio_tier2,
    load_base_fe, get_feat_cols,
    train_lgbm, train_cb, train_tw15, train_et, train_rf, train_asym20,
    run_meta, segment_report, diversity_report,
    ckpt_exists, load_ckpt, save_ckpt,
)
import feature_engineering as fe_module
from feature_engineering import build_features
import warnings
warnings.filterwarnings('ignore')

# ── 실험 설정 ──
EXP_NAME  = 'model46a'
CKPT_DIR  = os.path.join(DOCS_DIR, 'model46a_ckpt')

# ★ 핵심 변경: SC_AGG 5개 추가
SC_AGG_EXPAND = SC_AGG_BASE + [
    'avg_charge_wait',            # r=0.251
    'unique_sku_15m',             # r=0.229
    'loading_dock_util',          # r=0.213
    'maintenance_schedule_score', # r=0.197 (음의 상관)
    'manual_override_ratio',      # r=0.196
]
print(f'SC_AGG 컬럼: {len(SC_AGG_BASE)} → {len(SC_AGG_EXPAND)}개 '
      f'(+{len(SC_AGG_EXPAND)-len(SC_AGG_BASE)})')


def load_data():
    """SC_AGG 확장 FE 파이프라인"""
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    # build_features: model34와 동일 (lag [1-6], rolling [3,5,10])
    train, test = build_features(train, test, layout,
                                 lag_lags=[1, 2, 3, 4, 5, 6],
                                 rolling_windows=[3, 5, 10],
                                 verbose=True)
    print(f'  SC_AGG 컬럼 수: {len(SC_AGG_EXPAND)} → sc 피처: {len(SC_AGG_EXPAND)*11}')

    # ★ SC_AGG 확장 적용
    for fn in [lambda df: add_scenario_agg(df, SC_AGG_EXPAND),
               add_ratio_tier1, add_ratio_tier2]:
        train = fn(train)
        test  = fn(test)

    return train, test


def main():
    t0 = time.time()
    print('=' * 70)
    print(f'[{EXP_NAME}] SC_AGG 확장: 18 → {len(SC_AGG_EXPAND)}개')
    print(f'  추가: avg_charge_wait / unique_sku_15m / loading_dock_util')
    print(f'        maintenance_schedule_score / manual_override_ratio')
    print(f'  기준: CV=8.4803 (model34) / CV=8.4684 (q95 7모델)')
    print('=' * 70)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    print('\n[데이터 로드 + FE]')
    train, test = load_data()
    feat_cols = get_feat_cols(train)
    y_raw  = train['avg_delay_minutes_next_30m']
    y_log  = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'  최종 피처 수: {len(feat_cols)}')
    print(f'  예상 sc 피처: {len(SC_AGG_EXPAND)*11} (기존 {len(SC_AGG_BASE)*11})')

    oof_d, test_d = {}, {}

    # ── Layer 1: 6모델 학습 ──
    configs = [
        ('lgbm',   '▶ LGBM (MAE + log1p)',          train_lgbm,   y_log,  False),
        ('cb',     '▶ CatBoost (MAE + log1p)',       train_cb,     y_log,  False),
        ('tw15',   '▶ CatBoost Tweedie 1.5 (raw)',   train_tw15,   y_raw,  True),
        ('et',     '▶ ExtraTrees (log1p)',            train_et,     y_log,  False),
        ('rf',     '▶ RandomForest (log1p)',          train_rf,     y_log,  False),
        ('asym20', '▶ Asymmetric MAE α=2.0 (log1p)', train_asym20, y_log,  False),
    ]

    for name, label, fn, target, is_raw in configs:
        print(f'\n{"-"*60}\n{label}')
        oof_d[name], test_d[name] = fn(train, test, target, groups, feat_cols, CKPT_DIR, name)
        if is_raw:
            mae = np.abs(oof_d[name] - y_raw.values).mean()
        else:
            mae = np.abs(np.expm1(oof_d[name]) - y_raw.values).mean()
        print(f'  OOF MAE: {mae:.4f}')

    # ── 다양성 분석 ──
    print(f'\n{"="*60}')
    diversity_report(oof_d)

    # ── Layer 2: 메타 스태킹 ──
    print(f'\n{"="*60}')
    print('[Layer 2] 메타 스태킹 (6모델)')
    cv, oof_meta, test_meta = run_meta(oof_d, test_d, y_raw, groups, label=EXP_NAME)
    segment_report(oof_meta, y_raw.values, label=EXP_NAME)

    # ── 제출 파일 ──
    sample  = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    pred_col = [c for c in sample.columns if c != 'ID'][0]
    sample[pred_col] = np.clip(test_meta, 0, None)
    fname = f'{EXP_NAME}_cv{cv:.4f}.csv'
    sample.to_csv(os.path.join(SUB_DIR, fname), index=False)

    # ── y_tr / grp 갱신 저장 ──
    np.save(os.path.join(DOCS_DIR, 'y_tr_fe_order.npy'), y_raw.values)
    np.save(os.path.join(DOCS_DIR, 'grp_fe_order.npy'), groups.values)

    elapsed = (time.time() - t0) / 60
    print(f'\n{"="*70}')
    print(f'[{EXP_NAME}] 완료  CV={cv:.4f}  피처={len(feat_cols)}개  '
          f'pred_std={test_meta.std():.2f}  ({elapsed:.1f}분)')
    print(f'  기준 model34 CV=8.4803 → Δ{cv-8.4803:+.4f}')
    print(f'  기준 q95(7m) CV=8.4684 → Δ{cv-8.4684:+.4f}')
    print(f'  💾 {fname}')
    print('=' * 70)


if __name__ == '__main__':
    main()
