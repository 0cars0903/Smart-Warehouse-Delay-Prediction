"""
model46b: KEY_COLS 확장 — lag/rolling 대상 8 → 10개
================================================================
발견: lag/rolling 피처는 8개(KEY_COLS)에만 적용되는데, target 상관
      0.30 이상이면서 KEY_COLS 미포함인 피처가 존재.

추가 KEY_COLS (target 상관 순):
  robot_charging   r=0.320  [충전 중 로봇 수] battery_mean(0.359)에 필적
  battery_std      r=0.308  [배터리 분산]     battery_mean과 쌍 정보

선택 근거:
  - robot_charging: KEY_COLS의 robot_idle(0.349)과 비슷한 상관이지만
    lag/rolling에 포함 안 됨. 충전 대기 동태가 시계열로 중요
  - battery_std: battery_mean은 이미 KEY_COLS 포함. 평균만으로는
    충전 상태의 불균일성을 못 잡음 → 분산 정보가 보완
  - 두 피처 모두 SC_AGG에는 이미 포함(시나리오 레벨 정보 있음)
    → lag/rolling 추가는 시계열 동태 정보를 별도로 포착

model32와의 차이:
  - model32: shift-safe 12종 한꺼번에 추가 → CV 악화
  - 이번: 고상관 2개만 선별 → 배율 위험 최소화

기준: model34 Config B  CV=8.4803 / Public=9.8078
      model45c q95(7모델) CV=8.4684 / Public=9.7931 ← 현 최고

실행: python src/run_model46b_key_expand.py
예상 시간: ~50~60분 (lag/rolling 피처 증가로 FE 시간 증가)
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
    get_feat_cols,
    train_lgbm, train_cb, train_tw15, train_et, train_rf, train_asym20,
    run_meta, segment_report, diversity_report,
    ckpt_exists, load_ckpt, save_ckpt,
)
from feature_engineering import build_features
import warnings
warnings.filterwarnings('ignore')

# ── 실험 설정 ──
EXP_NAME = 'model46b'
CKPT_DIR = os.path.join(DOCS_DIR, 'model46b_ckpt')

# ★ 핵심 변경: KEY_COLS에 2개 추가
KEY_COLS_BASE = [
    'low_battery_ratio', 'battery_mean', 'charge_queue_length',
    'robot_idle', 'order_inflow_15m', 'congestion_score',
    'max_zone_density', 'avg_trip_distance',
]
KEY_COLS_EXPANDED = KEY_COLS_BASE + [
    'robot_charging',  # r=0.320 (배터리 crisis chain의 핵심)
    'battery_std',     # r=0.308 (배터리 불균일성 — mean만으로 부족)
]
print(f'KEY_COLS: {len(KEY_COLS_BASE)} → {len(KEY_COLS_EXPANDED)}개 '
      f'(+{len(KEY_COLS_EXPANDED)-len(KEY_COLS_BASE)})')

# 추가되는 lag/rolling 피처 수
# 2컬럼 × 6 lag + 2컬럼 × (3+5+10) rolling × 2 (mean/std) = 12 + 36 = 48 추가
EXTRA_LAG   = len(KEY_COLS_EXPANDED) - len(KEY_COLS_BASE)
EXTRA_FEATS = EXTRA_LAG * 6 + EXTRA_LAG * 3 * 2  # lag6 + roll(3,5,10) × mean+std
print(f'예상 추가 피처: lag {EXTRA_LAG*6} + rolling {EXTRA_LAG*3*2} = {EXTRA_FEATS}개')


def load_data():
    """KEY_COLS 확장 FE 파이프라인"""
    import feature_engineering as fe_module

    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    # ★ KEY_COLS 임시 교체 후 build_features 호출
    _orig = fe_module.KEY_COLS[:]
    fe_module.KEY_COLS = KEY_COLS_EXPANDED
    print(f'  KEY_COLS 임시 확장: {KEY_COLS_BASE} + {KEY_COLS_EXPANDED[len(KEY_COLS_BASE):]}')

    train, test = build_features(train, test, layout,
                                 lag_lags=[1, 2, 3, 4, 5, 6],
                                 rolling_windows=[3, 5, 10],
                                 verbose=True)
    fe_module.KEY_COLS = _orig  # 원복

    # SC_AGG는 model34와 동일 (18개)
    for fn in [lambda df: add_scenario_agg(df, SC_AGG_BASE),
               add_ratio_tier1, add_ratio_tier2]:
        train = fn(train)
        test  = fn(test)

    return train, test


def main():
    t0 = time.time()
    print('=' * 70)
    print(f'[{EXP_NAME}] KEY_COLS 확장: 8 → {len(KEY_COLS_EXPANDED)}개')
    print(f'  추가: robot_charging (r=0.320) / battery_std (r=0.308)')
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

    # 새로 추가된 lag/rolling 피처 확인
    new_lag_feats = [c for c in feat_cols
                     if any(kc in c for kc in ['robot_charging_lag', 'robot_charging_roll',
                                                'battery_std_lag', 'battery_std_roll'])]
    print(f'  신규 lag/rolling 피처 ({len(new_lag_feats)}개): {new_lag_feats[:6]}...')

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
    sample   = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    pred_col = [c for c in sample.columns if c != 'ID'][0]
    sample[pred_col] = np.clip(test_meta, 0, None)
    fname = f'{EXP_NAME}_cv{cv:.4f}.csv'
    sample.to_csv(os.path.join(SUB_DIR, fname), index=False)

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
