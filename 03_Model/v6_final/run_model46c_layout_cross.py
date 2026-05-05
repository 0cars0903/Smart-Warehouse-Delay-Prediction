"""
model46c: Layout × 운영 교호작용 피처
================================================================
발견: layout_info 14개 컬럼이 merge되어 개별 피처로는 쓰이지만,
      '창고 물리 용량 대비 실시간 운영 부하' 교호작용은 미탐색.

      hub_spoke layout: mean=22.28, [80+]=3.1% — 다른 layout(18~18.4, 2.3~2.6%)
      대비 유의미하게 지연이 높음 → layout별 bottleneck 패턴이 다름.

추가 교호작용 피처 (5종):
  (1) orders_per_pack_station
      = sc_order_inflow_15m_mean / pack_station_count
      [의미] 팩 스테이션 당 처리해야 할 주문 부하
      [근거] pack_station_count r=0.186(layout 피처 중 최고)
             sc_order_inflow_15m_mean이 가장 강한 시나리오 구분자

  (2) charging_ratio_abs
      = robot_charging / robot_total
      [의미] 전체 로봇 중 실제 충전 중인 비율 (절대 수 기반)
      [근거] 기존 ratio_idle_fraction은 idle/total. charging 비율은 배터리
             위기 심각도를 직접 나타내며 robot_total(레이아웃)까지 반영

  (3) congestion_aisle_amplifier
      = sc_congestion_score_mean × (1 / aisle_width_avg)
      [의미] 통로 폭이 좁을수록 동일 혼잡도가 더 큰 지연을 만듦
      [근거] hub_spoke는 narrow aisle이 많아 congestion 단위 영향이 큼
             기존 ratio_congestion_per_aisle은 단순 나누기지만 이 피처는
             sc_mean을 사용해 시나리오 레벨 혼잡을 layout 폭에 맵핑

  (4) area_per_robot
      = floor_area_sqm / robot_total
      [의미] 로봇 1대가 커버해야 할 바닥 면적 (운영 밀도 역수)
      [근거] 넓은 창고에 로봇 수가 부족하면 trip 거리 증가 → 지연
             avg_trip_distance(r=0.08)보다 인과적으로 앞선 구조 피처

  (5) charger_utilization_rate
      = sc_robot_charging_mean / charger_count
      [의미] 충전기 점유율 (기존 ratio_charge_competition와 유사하나
             sc_mean 기반으로 시나리오 평균 충전기 부하)
      [보정] ratio_charge_competition이 이미 있으므로 대신
             sc_avg_charge_wait_mean / (charger_count + 1) 로 교체
             = 충전기 수 대비 평균 대기 시간 → 충전 병목 강도

  (6) layout_type_bottleneck_score (보너스)
      = layout_compactness × sc_max_zone_density_mean
      [의미] 창고가 촘촘할수록 고밀도 구역이 더 심각한 병목
      [근거] layout_compactness(r=0.022)는 단독으로 약하지만
             sc_max_zone_density_mean(강한 시나리오 신호)와 곱하면
             창고 구조 × 실시간 밀도 교호작용 포착

기준: model34 Config B  CV=8.4803 / Public=9.8078
      model45c q95(7모델) CV=8.4684 / Public=9.7931 ← 현 최고

실행: python src/run_model46c_layout_cross.py
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
    get_feat_cols,
    train_lgbm, train_cb, train_tw15, train_et, train_rf, train_asym20,
    run_meta, segment_report, diversity_report,
    ckpt_exists, load_ckpt, save_ckpt, safe_div,
)
from feature_engineering import build_features
import warnings
warnings.filterwarnings('ignore')

# ── 실험 설정 ──
EXP_NAME = 'model46c'
CKPT_DIR = os.path.join(DOCS_DIR, 'model46c_ckpt')


def add_layout_cross_features(df):
    """
    Layout 물리 구조 × 운영 지표 교호작용 피처 (6종)

    ⚠️ add_scenario_agg() 이후 호출해야 sc_* 컬럼이 존재함
    """
    df = df.copy()

    # (1) 팩 스테이션 당 주문 부하
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['lx_orders_per_pack_station'] = safe_div(
            df['sc_order_inflow_15m_mean'], df['pack_station_count'])

    # (2) 전체 로봇 대비 충전 중 로봇 비율 (절대 수 고려)
    if 'robot_charging' in df.columns and 'robot_total' in df.columns:
        df['lx_charging_ratio_abs'] = safe_div(
            df['robot_charging'], df['robot_total'])

    # (3) 통로 폭 대비 혼잡도 증폭기 (좁을수록 혼잡 페널티 ↑)
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        # 역수를 취해 "좁을수록 높은 값" 표현
        df['lx_congestion_aisle_amp'] = (
            df['sc_congestion_score_mean'] / (df['aisle_width_avg'] + 0.1))

    # (4) 로봇 1대 담당 면적 (운영 밀도 역수)
    if 'floor_area_sqm' in df.columns and 'robot_total' in df.columns:
        df['lx_area_per_robot'] = safe_div(df['floor_area_sqm'], df['robot_total'])

    # (5) 충전기 수 대비 평균 충전 대기 시간 (충전 병목 강도)
    if 'sc_avg_charge_wait_mean' in df.columns and 'charger_count' in df.columns:
        df['lx_wait_per_charger'] = safe_div(
            df['sc_avg_charge_wait_mean'], df['charger_count'])
    elif 'avg_charge_wait' in df.columns and 'charger_count' in df.columns:
        # sc_ 집계 전에 호출된 경우 fallback
        df['lx_wait_per_charger'] = safe_div(df['avg_charge_wait'], df['charger_count'])

    # (6) 창고 밀집도 × 최고 구역 밀도 (layout 구조 × 실시간 밀도)
    if 'layout_compactness' in df.columns and 'sc_max_zone_density_mean' in df.columns:
        df['lx_compact_density'] = (
            df['layout_compactness'] * df['sc_max_zone_density_mean'])

    new_cols = [c for c in df.columns if c.startswith('lx_')]
    return df, new_cols


def load_data():
    """Layout 교호작용 FE 파이프라인"""
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    # SC_AGG에 avg_charge_wait 추가 (lx_wait_per_charger를 위해)
    SC_WITH_CHARGEWAIT = SC_AGG_BASE + ['avg_charge_wait']

    # build_features: model34와 동일 (lag [1-6], rolling [3,5,10])
    train, test = build_features(train, test, layout,
                                 lag_lags=[1, 2, 3, 4, 5, 6],
                                 rolling_windows=[3, 5, 10],
                                 verbose=True)

    # SC_AGG + ratio
    for fn in [lambda df: add_scenario_agg(df, SC_WITH_CHARGEWAIT),
               add_ratio_tier1, add_ratio_tier2]:
        train = fn(train)
        test  = fn(test)

    # ★ Layout 교호작용 피처 추가
    train, new_cols_tr = add_layout_cross_features(train)
    test,  new_cols_te = add_layout_cross_features(test)
    print(f'  Layout 교호작용 피처 추가: {new_cols_tr}')

    return train, test


def main():
    t0 = time.time()
    print('=' * 70)
    print(f'[{EXP_NAME}] Layout × 운영 교호작용 피처 (6종)')
    print(f'  lx_orders_per_pack_station  / lx_charging_ratio_abs')
    print(f'  lx_congestion_aisle_amp     / lx_area_per_robot')
    print(f'  lx_wait_per_charger         / lx_compact_density')
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

    # layout 교호작용 피처 통계 확인
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
