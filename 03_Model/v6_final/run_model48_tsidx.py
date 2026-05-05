"""
model48: model47 FE + ts_ratio × SC_AGG 교호작용 피처
================================================================
기존 파이프라인(build_features)에서 이미 생성되는 피처:
  ts_idx (0~24), ts_ratio (0~1), ts_sin, ts_cos

→ 위치 피처는 이미 있음. 새로 추가할 정보는 "교호작용(interaction)":
  ts_ratio × sc_order_inflow_15m_mean = "시나리오 평균 주문량 × 현재 시점"
  → 얼마나 오래 고부하 상태가 지속됐는가 (누적 압박 근사)

  트리 모델은 ts_ratio와 sc_order를 별도 피처로 학습하며,
  둘의 곱(product interaction)은 여러 단계 분기를 통해서만 근사 가능.
  명시적 교호작용 피처 제공 시 단일 분기로 포착 가능 → 효율 향상 기대.

추가 피처 (10종):
  ts_ratio_sq, ts_late_flag, ts_early_flag          (비선형/임계값)
  ts_x_order, ts_x_cong, ts_x_util, ts_x_battery,  (SC_AGG 교호작용)
  ts_x_fault, ts_x_demand, ts_x_cong_isect          (ratio 교호작용)

기준: model47+q95 CV=8.4610 / Public=9.7901

실행: python src/run_model48_tsidx.py
예상 시간: ~50~60분 (6모델 전체 재학습)
"""

import numpy as np
import pandas as pd
import os, sys, time
import warnings
warnings.filterwarnings('ignore')
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

EXP_NAME = 'model48'
CKPT_DIR = os.path.join(DOCS_DIR, 'model48_ckpt')
CKPT_Q   = os.path.join(DOCS_DIR, 'model45_ckpt', 'strat_c')

# model47와 동일한 SC_AGG 확장 (23컬럼)
SC_AGG_EXPAND = SC_AGG_BASE + [
    'avg_charge_wait',
    'unique_sku_15m',
    'loading_dock_util',
    'maintenance_schedule_score',
    'manual_override_ratio',
]


# ── model47 Layout 교호작용 피처 (동일) ──
def add_layout_cross_features(df):
    df = df.copy()
    new_cols = []
    def _add(name, expr):
        df[name] = expr
        new_cols.append(name)

    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        _add('lx_orders_per_pack_station',
             safe_div(df['sc_order_inflow_15m_mean'], df['pack_station_count']))
    if 'robot_charging' in df.columns and 'robot_total' in df.columns:
        _add('lx_charging_ratio_abs',
             safe_div(df['robot_charging'], df['robot_total']))
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        _add('lx_congestion_aisle_amp',
             df['sc_congestion_score_mean'] / (df['aisle_width_avg'] + 0.1))
    if 'floor_area_sqm' in df.columns and 'robot_total' in df.columns:
        _add('lx_area_per_robot',
             safe_div(df['floor_area_sqm'], df['robot_total']))
    if 'sc_avg_charge_wait_mean' in df.columns and 'charger_count' in df.columns:
        _add('lx_wait_per_charger',
             safe_div(df['sc_avg_charge_wait_mean'], df['charger_count']))
    if 'layout_compactness' in df.columns and 'sc_max_zone_density_mean' in df.columns:
        _add('lx_compact_density',
             df['layout_compactness'] * df['sc_max_zone_density_mean'])
    return df, new_cols


# ── ★ 핵심 신규: ts_ratio × SC_AGG 교호작용 피처 ──
def add_tsidx_features(df):
    """
    ts_ratio(0~1, 이미 존재)를 활용한 교호작용 피처.

    build_features()에서 이미 생성됨:
      ts_idx (0~24), ts_ratio (0~1), ts_sin, ts_cos

    새로 추가하는 것:
      1. ts_ratio_sq     : 2차 다항식 — 후반 가속 효과를 비선형으로 포착
      2. ts_late_flag    : ts_idx >= 18 임계값 플래그
      3. ts_early_flag   : ts_idx <= 6 임계값 플래그
      4. ts_x_*          : ts_ratio × SC_AGG 피처 — 핵심 신규!
                           트리가 단일 분기로 포착하기 어려운
                           "누적 압박 = 부하 × 지속 시간"을 직접 계산

    리크 안전성: test 데이터도 동일 ts_idx 0~24 분포 → shift 없음
    """
    df = df.copy()

    # ts_ratio는 이미 존재 → 사용만
    if 'ts_ratio' not in df.columns:
        df['ts_ratio'] = df['ts_idx'] / 24.0

    # ─ 비선형/임계값 피처 ─
    df['ts_ratio_sq']   = df['ts_ratio'] ** 2          # 후반 가속
    df['ts_late_flag']  = (df['ts_idx'] >= 18).astype(float)  # 후반 25%
    df['ts_early_flag'] = (df['ts_idx'] <= 6).astype(float)   # 초반 25%

    # ─ SC_AGG 교호작용 (ts_ratio × 시나리오 수준 부하) ─
    sc_cross = {
        'ts_x_order':   'sc_order_inflow_15m_mean',
        'ts_x_cong':    'sc_congestion_score_mean',
        'ts_x_util':    'sc_robot_utilization_mean',
        'ts_x_battery': 'sc_low_battery_ratio_mean',
        'ts_x_fault':   'sc_fault_count_15m_mean',
    }
    for feat_name, sc_col in sc_cross.items():
        if sc_col in df.columns:
            df[feat_name] = df['ts_ratio'] * df[sc_col]

    # ─ Ratio 피처 × 위치 ─
    ratio_cross = {
        'ts_x_demand':    'ratio_demand_per_robot',
        'ts_x_cong_isect': 'ratio_congestion_per_intersection',
    }
    for feat_name, r_col in ratio_cross.items():
        if r_col in df.columns:
            df[feat_name] = df['ts_ratio'] * df[r_col]

    return df


def load_data():
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    train, test = build_features(train, test, layout,
                                 lag_lags=[1, 2, 3, 4, 5, 6],
                                 rolling_windows=[3, 5, 10],
                                 verbose=True)
    for fn in [lambda df: add_scenario_agg(df, SC_AGG_EXPAND),
               add_ratio_tier1, add_ratio_tier2]:
        train = fn(train); test = fn(test)

    train, lx_cols = add_layout_cross_features(train)
    test,  _       = add_layout_cross_features(test)
    print(f'  Layout 교호작용: {lx_cols}')

    # ★ ts_idx 위치 피처 추가
    train = add_tsidx_features(train)
    test  = add_tsidx_features(test)
    tsidx_cols = [c for c in train.columns if c.startswith('ts_idx') or c.startswith('tsidx_x')]
    print(f'  ts_idx 위치 피처 ({len(tsidx_cols)}개): {tsidx_cols}')

    return train, test


def main():
    t0 = time.time()
    print('=' * 70)
    print(f'[{EXP_NAME}] model47 FE + ts_idx 위치 피처 추가')
    print(f'  SC_AGG 23컬럼 + Layout 교호작용 6종 + ts_idx 위치 ~11종')
    print(f'  기준: model47+q95 CV=8.4610 / Public=9.7901')
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

    # 신규 교호작용 피처 상관 확인
    new_ts_cols = [c for c in feat_cols if c.startswith('ts_x_') or
                   c in ('ts_ratio_sq', 'ts_late_flag', 'ts_early_flag')]
    existing_ts = [c for c in feat_cols if c in ('ts_idx', 'ts_ratio', 'ts_sin', 'ts_cos')]
    print(f'\n  기존 ts 피처 (model47에 이미 있음): {existing_ts}')
    print(f'  신규 교호작용 피처 ({len(new_ts_cols)}종):')
    for c in new_ts_cols:
        r = train[c].corr(y_raw)
        print(f'    {c:40s}  r={r:.4f}')

    oof_d, test_d = {}, {}

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

    print(f'\n{"="*60}')
    diversity_report(oof_d)

    # ── 6모델 메타 스태킹 ──
    print(f'\n{"="*60}\n[Layer 2] 6모델 메타 스태킹')
    cv6, oof6, test6 = run_meta(oof_d, test_d, y_raw, groups, label=f'{EXP_NAME}(6모델)')
    segment_report(oof6, y_raw.values, label=f'{EXP_NAME}(6모델)')

    # ── + q95 7모델 스태킹 ──
    q95_path_oof  = os.path.join(CKPT_Q, 'q95_oof.npy')
    q95_path_test = os.path.join(CKPT_Q, 'q95_test.npy')
    if os.path.exists(q95_path_oof):
        print(f'\n{"="*60}\n[Layer 2] 7모델 (6 + q95) 스태킹')
        oof_q = dict(oof_d); test_q = dict(test_d)
        oof_q['q95']  = np.load(q95_path_oof)
        test_q['q95'] = np.load(q95_path_test)
        q95_mae = np.abs(oof_q['q95'] - y_raw.values).mean()
        lgbm_q95_corr = np.corrcoef(oof_d['lgbm'], oof_q['q95'])[0, 1]
        print(f'  q95 OOF MAE={q95_mae:.4f} | lgbm-q95 상관={lgbm_q95_corr:.4f}')

        cv7, oof7, test7 = run_meta(oof_q, test_q, y_raw, groups,
                                    label=f'{EXP_NAME}(6+q95)')
        segment_report(oof7, y_raw.values, label=f'{EXP_NAME}(6+q95)')

        sample   = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
        pred_col = [c for c in sample.columns if c != 'ID'][0]
        sample[pred_col] = np.clip(test7, 0, None)
        fname = f'{EXP_NAME}_q7_q95_cv{cv7:.4f}.csv'
        sample.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f'  💾 {fname}')
    else:
        print('  ⚠️ q95 체크포인트 없음 — 6모델만 제출')
        cv7, test7 = None, None

    # ── 6모델 제출 파일 ──
    sample   = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    pred_col = [c for c in sample.columns if c != 'ID'][0]
    sample[pred_col] = np.clip(test6, 0, None)
    fname6 = f'{EXP_NAME}_cv{cv6:.4f}.csv'
    sample.to_csv(os.path.join(SUB_DIR, fname6), index=False)
    print(f'  💾 {fname6}')

    # ── y_tr / grp 갱신 (model48 FE 기준) ──
    np.save(os.path.join(DOCS_DIR, 'y_tr_model48.npy'), y_raw.values)
    np.save(os.path.join(DOCS_DIR, 'grp_model48.npy'), groups.values)

    elapsed = (time.time() - t0) / 60
    print(f'\n{"="*70}')
    print(f'[{EXP_NAME}] 완료  피처={len(feat_cols)}개  ({elapsed:.1f}분)')
    print(f'  6모델  CV={cv6:.4f}  (기준 model47  8.4649 → Δ{cv6-8.4649:+.4f})')
    if cv7:
        print(f'  +q95   CV={cv7:.4f}  (기준 model47+q95 8.4610 → Δ{cv7-8.4610:+.4f})')
    print(f'  pred_std(6모델)={test6.std():.2f}')
    if cv7:
        print(f'  pred_std(7모델)={test7.std():.2f}  (기준 16.35)')


if __name__ == '__main__':
    main()
