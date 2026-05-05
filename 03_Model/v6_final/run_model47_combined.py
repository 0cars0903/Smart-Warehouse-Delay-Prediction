"""
model47: SC_AGG 확장(18→23) + Layout 교호작용 피처(6종) 동시 적용
================================================================
model46a (SC_AGG 확장): CV=8.4647 ✅ (Δ-0.0037 vs q95 7모델)
model46c (Layout 교호작용): CV=8.4600 ✅ (Δ-0.0084 vs q95 7모델, CV 역대 최고)

두 FE 개선을 동시 적용했을 때 상호보완적 효과가 있는지 검증.
- SC_AGG 확장: 시나리오 레벨에서 avg_charge_wait, unique_sku_15m 등 5개 집계 추가
- Layout 교호작용: 물리 구조 × 운영 지표 교차 피처 (lx_orders_per_pack_station 등)

기준: model46c 6모델 CV=8.4600 (역대 최고)
      model46a+q95 7모델 CV=8.4615 (역대 최고)

실행: python src/run_model47_combined.py
예상 시간: ~50~60분 (6모델 전체 재학습)
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
    ckpt_exists, load_ckpt, save_ckpt, safe_div,
)
from feature_engineering import build_features
import warnings
warnings.filterwarnings('ignore')

# ── 실험 설정 ──
EXP_NAME = 'model47'
CKPT_DIR = os.path.join(DOCS_DIR, 'model47_ckpt')

# ★ 핵심 변경 1: SC_AGG 확장 (model46a와 동일)
SC_AGG_EXPAND = SC_AGG_BASE + [
    'avg_charge_wait',            # r=0.251
    'unique_sku_15m',             # r=0.229
    'loading_dock_util',          # r=0.213
    'maintenance_schedule_score', # r=0.197 (음의 상관)
    'manual_override_ratio',      # r=0.196
]
print(f'SC_AGG 컬럼: {len(SC_AGG_BASE)} → {len(SC_AGG_EXPAND)}개 '
      f'(+{len(SC_AGG_EXPAND)-len(SC_AGG_BASE)})')


def add_layout_cross_features(df):
    """
    Layout 물리 구조 × 운영 지표 교호작용 피처 (6종) — model46c와 동일

    ⚠️ add_scenario_agg() 이후 호출해야 sc_* 컬럼이 존재함
    """
    df = df.copy()

    # (1) 팩 스테이션 당 주문 부하 (r=0.4186 — 데이터셋 최강 신규 피처)
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['lx_orders_per_pack_station'] = safe_div(
            df['sc_order_inflow_15m_mean'], df['pack_station_count'])

    # (2) 전체 로봇 대비 충전 중 로봇 비율 (절대 수 고려)
    if 'robot_charging' in df.columns and 'robot_total' in df.columns:
        df['lx_charging_ratio_abs'] = safe_div(
            df['robot_charging'], df['robot_total'])

    # (3) 통로 폭 대비 혼잡도 증폭기 (좁을수록 혼잡 페널티 ↑)
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['lx_congestion_aisle_amp'] = (
            df['sc_congestion_score_mean'] / (df['aisle_width_avg'] + 0.1))

    # (4) 로봇 1대 담당 면적 (운영 밀도 역수)
    if 'floor_area_sqm' in df.columns and 'robot_total' in df.columns:
        df['lx_area_per_robot'] = safe_div(df['floor_area_sqm'], df['robot_total'])

    # (5) 충전기 수 대비 평균 충전 대기 시간 (충전 병목 강도)
    #     model46a SC_AGG에 avg_charge_wait 포함 → sc_avg_charge_wait_mean 사용 가능
    if 'sc_avg_charge_wait_mean' in df.columns and 'charger_count' in df.columns:
        df['lx_wait_per_charger'] = safe_div(
            df['sc_avg_charge_wait_mean'], df['charger_count'])
    elif 'avg_charge_wait' in df.columns and 'charger_count' in df.columns:
        df['lx_wait_per_charger'] = safe_div(df['avg_charge_wait'], df['charger_count'])

    # (6) 창고 밀집도 × 최고 구역 밀도 (layout 구조 × 실시간 밀도)
    if 'layout_compactness' in df.columns and 'sc_max_zone_density_mean' in df.columns:
        df['lx_compact_density'] = (
            df['layout_compactness'] * df['sc_max_zone_density_mean'])

    new_cols = [c for c in df.columns if c.startswith('lx_')]
    return df, new_cols


def load_data():
    """SC_AGG 확장 + Layout 교호작용 FE 파이프라인"""
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    # build_features: model34와 동일 (lag [1-6], rolling [3,5,10])
    train, test = build_features(train, test, layout,
                                 lag_lags=[1, 2, 3, 4, 5, 6],
                                 rolling_windows=[3, 5, 10],
                                 verbose=True)

    # ★ SC_AGG 확장 적용 (avg_charge_wait 포함 → lx_wait_per_charger에서 sc_mean 사용 가능)
    print(f'  SC_AGG 컬럼 수: {len(SC_AGG_EXPAND)} → sc 피처: {len(SC_AGG_EXPAND)*11}')
    for fn in [lambda df: add_scenario_agg(df, SC_AGG_EXPAND),
               add_ratio_tier1, add_ratio_tier2]:
        train = fn(train)
        test  = fn(test)

    # ★ Layout 교호작용 피처 추가 (SC_AGG 후 호출 필수)
    train, new_cols_tr = add_layout_cross_features(train)
    test,  new_cols_te = add_layout_cross_features(test)
    print(f'  Layout 교호작용 피처 추가: {new_cols_tr}')

    return train, test


def main():
    t0 = time.time()
    print('=' * 70)
    print(f'[{EXP_NAME}] SC_AGG 확장(18→23) + Layout 교호작용(6종) 동시 적용')
    print(f'  SC_AGG 추가: avg_charge_wait / unique_sku_15m / loading_dock_util')
    print(f'               maintenance_schedule_score / manual_override_ratio')
    print(f'  Layout 추가: lx_orders_per_pack_station / lx_charging_ratio_abs')
    print(f'               lx_congestion_aisle_amp / lx_area_per_robot')
    print(f'               lx_wait_per_charger / lx_compact_density')
    print(f'  기준: CV=8.4600 (model46c) / CV=8.4615 (model46a+q95)')
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
    print(f'  예상: ~477(46a) + 6(lx) - 중복제거 ≈ 483피처')

    # lx_ 피처 상관 확인
    lx_cols = [c for c in feat_cols if c.startswith('lx_')]
    print(f'  lx_ 피처 ({len(lx_cols)}개): {lx_cols}')
    if lx_cols:
        corr_lx = train[lx_cols + ['avg_delay_minutes_next_30m']].corr()[
            'avg_delay_minutes_next_30m'].drop('avg_delay_minutes_next_30m').abs()
        print('  lx_ × target 상관:')
        for c, r in corr_lx.sort_values(ascending=False).items():
            print(f'    {c:35s}  r={r:.4f}')

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

    # ── y_tr / grp 갱신 저장 (model47 FE 순서 기준) ──
    np.save(os.path.join(DOCS_DIR, 'y_tr_fe_order.npy'), y_raw.values)
    np.save(os.path.join(DOCS_DIR, 'grp_fe_order.npy'), groups.values)

    elapsed = (time.time() - t0) / 60
    print(f'\n{"="*70}')
    print(f'[{EXP_NAME}] 완료  CV={cv:.4f}  피처={len(feat_cols)}개  '
          f'pred_std={test_meta.std():.2f}  ({elapsed:.1f}분)')
    print(f'  기준 model46c CV=8.4600 → Δ{cv-8.4600:+.4f}')
    print(f'  기준 q95(7m)  CV=8.4684 → Δ{cv-8.4684:+.4f}')
    print(f'  기준 46a+q95  CV=8.4615 → Δ{cv-8.4615:+.4f}')
    print(f'  💾 {fname}')
    print('=' * 70)


if __name__ == '__main__':
    main()
