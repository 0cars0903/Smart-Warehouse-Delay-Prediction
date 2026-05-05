"""
q80 / q75 / q70 LGBM Quantile 모델 훈련 + OOF 저장
================================================================
기존 q85/q90/q95(strat_c)와 동일한 feature set(model45c FE)으로
더 낮은 분위수 모델을 훈련하고 체크포인트를 저장한다.

근거:
  - model34 base: q85 추가 시 Public 악화 (q85-q95 상관 0.9848로 다양성 부족)
  - model47 base: q85 추가 시 CV 개선 + pred_std 16.35→16.47 확장
  → model47의 다양한 FE(SC_AGG 23 + lx_6)가 낮은 분위수의 다양성을 흡수
  → q80/q75/q70은 q95와의 상관이 더 낮아 추가 pred_std 확장 기대

훈련 후: python src/run_model47_multi_q.py 의 CKPT_Q 경로에
  q80_oof.npy, q80_test.npy 등이 추가되어 자동으로 스태킹에 활용 가능.

저장 경로: docs/model45_ckpt/strat_c/  (기존 q85~q95와 동일 위치)

실행: python src/run_train_quantile_low.py
예상 시간: ~15~20분 (3개 분위수 × 5-fold)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import os, sys, time, warnings
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
warnings.filterwarnings('ignore')

from feature_engineering import build_features

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
DOCS_DIR = os.path.join(_BASE, '..', 'docs')
CKPT_DIR = os.path.join(DOCS_DIR, 'model45_ckpt', 'strat_c')

N_SPLITS     = 5
RANDOM_STATE = 42

# 기존 q85/q90/q95와 동일한 LGBM Quantile 파라미터 (동일 feature set에서 일관성 유지)
def make_q_params(alpha):
    return {
        'n_estimators': 3000,
        'learning_rate': 0.02,
        'num_leaves': 127,
        'feature_fraction': 0.51,
        'bagging_fraction': 0.90,
        'bagging_freq': 1,
        'min_child_samples': 26,
        'reg_alpha': 0.38,
        'reg_lambda': 0.36,
        'objective': 'quantile',
        'alpha': alpha,
        'random_state': RANDOM_STATE,
        'verbosity': -1,
        'n_jobs': -1,
    }

SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]


def add_scenario_agg(df):
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns: continue
        grp = df.groupby('scenario_id')[col]
        df[f'sc_{col}_mean']     = grp.transform('mean')
        df[f'sc_{col}_std']      = grp.transform('std').fillna(0)
        df[f'sc_{col}_max']      = grp.transform('max')
        df[f'sc_{col}_min']      = grp.transform('min')
        df[f'sc_{col}_diff']     = df[col] - df[f'sc_{col}_mean']
        df[f'sc_{col}_median']   = grp.transform('median')
        df[f'sc_{col}_p10']      = grp.transform(lambda x: x.quantile(0.10))
        df[f'sc_{col}_p90']      = grp.transform(lambda x: x.quantile(0.90))
        df[f'sc_{col}_skew']     = grp.transform(lambda x: x.skew()).fillna(0)
        df[f'sc_{col}_kurtosis'] = grp.transform(lambda x: x.kurtosis()).fillna(0)
        df[f'sc_{col}_cv']       = (df[f'sc_{col}_std'] /
                                    (df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)
    return df


def safe_div(a, b, fill=0):
    return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)


def add_ratio_features(df):
    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = safe_div(df['sc_order_inflow_15m_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = safe_div(df['sc_congestion_score_mean'], df['intersection_count'])
    if 'sc_low_battery_ratio_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_stress'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df.get('sc_charge_queue_length_mean', 0), df['charger_count'])
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = safe_div(df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']
    if all(c in df.columns for c in ['sc_congestion_score_mean', 'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'], df['robot_total'] ** 2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm'] / 100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(df['pack_station_count'], df['floor_area_sqm'] / 1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns:
        df['ss_congestion_x_battery'] = (
            df.get('sc_congestion_score_mean', 0) * df.get('sc_low_battery_ratio_mean', 0))
        df['ss_order_x_util'] = (
            df.get('sc_order_inflow_15m_mean', 0) * df.get('sc_robot_utilization_mean', 0))
        df['ss_demand_x_congestion'] = (
            df.get('ratio_demand_per_robot', 0) * df.get('ratio_congestion_per_intersection', 0))
        df['ss_stress_x_pressure'] = (
            df.get('ratio_battery_stress', 0) * df.get('ratio_packing_pressure', 0))
        df['ss_idle_x_demand'] = (
            df.get('ratio_idle_fraction', 0) * df.get('ratio_demand_per_robot', 0))
    # Tier 2 추가 비율 피처 (model29A)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm'] / 100)
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    if 'sc_robot_charging_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_battery_per_robot'] = safe_div(
            df.get('sc_low_battery_ratio_mean', 0) * df['robot_total'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_count' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(df['sc_congestion_score_mean'], df['aisle_count'])
    return df


def get_feat_cols(df):
    exclude = {'scenario_id', 'ts_idx', 'avg_delay_minutes_next_30m',
               'warehouse_id', 'timestamp'}
    return [c for c in df.columns
            if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]


def train_quantile(train_fe, test_fe, y_raw, groups, feat_cols, alpha, qname):
    """GroupKFold 5-fold Quantile LGBM 훈련 + OOF/test 저장"""
    oof_path  = os.path.join(CKPT_DIR, f'{qname}_oof.npy')
    test_path = os.path.join(CKPT_DIR, f'{qname}_test.npy')

    if os.path.exists(oof_path) and os.path.exists(test_path):
        print(f'  ✅ {qname} 이미 존재, 건너뜀')
        oof  = np.load(oof_path)
        mae  = np.abs(oof - y_raw).mean()
        print(f'     OOF MAE={mae:.4f} | 상관(vs q95)=', end='')
        q95_oof = np.load(os.path.join(CKPT_DIR, 'q95_oof.npy'))
        print(f'{np.corrcoef(oof, q95_oof)[0,1]:.4f}')
        return oof, np.load(test_path)

    params  = make_q_params(alpha)
    X_tr    = train_fe[feat_cols].values
    X_te    = test_fe[feat_cols].values
    oof     = np.zeros(len(y_raw))
    preds   = []
    gkf     = GroupKFold(n_splits=N_SPLITS)
    t0      = time.time()

    for fold, (ti, vi) in enumerate(gkf.split(X_tr, y_raw, groups)):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr[ti], y_raw.values[ti],
              eval_set=[(X_tr[vi], y_raw.values[vi])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = m.predict(X_tr[vi])
        preds.append(m.predict(X_te))
        elapsed = (time.time()-t0)/60
        print(f'    {qname} Fold {fold+1}  MAE={np.abs(oof[vi]-y_raw.values[vi]).mean():.4f}'
              f'  iter={m.best_iteration_}  ({elapsed:.1f}m)')
        del m

    test_pred = np.mean(preds, axis=0)
    mae = np.abs(oof - y_raw.values).mean()
    q95_oof = np.load(os.path.join(CKPT_DIR, 'q95_oof.npy'))
    corr = np.corrcoef(oof, q95_oof)[0,1]
    print(f'  {qname} OOF MAE={mae:.4f} | q95 상관={corr:.4f}')

    np.save(oof_path,  oof)
    np.save(test_path, test_pred)
    print(f'  💾 {qname}_oof.npy / {qname}_test.npy 저장')
    return oof, test_pred


def main():
    t0 = time.time()
    print('=' * 60)
    print('q80 / q75 / q70 Quantile LGBM 훈련')
    print('  feature set: model45c (SC_AGG 18컬럼 + ratio Tier1/2)')
    print('  저장: docs/model45_ckpt/strat_c/')
    print('=' * 60)

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── 데이터 로드 ──
    print('\n[데이터 로드 + FE]')
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    train_fe, test_fe = build_features(train, test, layout,
                                       lag_lags=[1,2,3,4,5,6],
                                       rolling_windows=[3,5,10],
                                       verbose=True)
    train_fe = add_scenario_agg(train_fe)
    test_fe  = add_scenario_agg(test_fe)
    train_fe = add_ratio_features(train_fe)
    test_fe  = add_ratio_features(test_fe)

    feat_cols = get_feat_cols(train_fe)
    y_raw     = train_fe['avg_delay_minutes_next_30m']
    groups    = train_fe['scenario_id']
    print(f'  피처 수: {len(feat_cols)}')

    # ── 기존 q85 상관 기준 확인 ──
    q85_oof = np.load(os.path.join(CKPT_DIR, 'q85_oof.npy'))
    q95_oof = np.load(os.path.join(CKPT_DIR, 'q95_oof.npy'))
    print(f'\n  기존 q85-q95 상관: {np.corrcoef(q85_oof, q95_oof)[0,1]:.4f}')

    # ── 훈련: q80, q75, q70 ──
    QUANTILES = [
        (0.80, 'q80'),
        (0.75, 'q75'),
        (0.70, 'q70'),
    ]

    for alpha, qname in QUANTILES:
        print(f'\n{"─"*50}')
        print(f'▶ {qname} (alpha={alpha})')
        train_quantile(train_fe, test_fe, y_raw, groups, feat_cols, alpha, qname)

    # ── 저장 확인 + 상관 요약 ──
    print(f'\n{"="*60}')
    print('저장된 quantile OOF 상관 요약 (vs q95):')
    for qname in ['q70', 'q75', 'q80', 'q85', 'q90']:
        p = os.path.join(CKPT_DIR, f'{qname}_oof.npy')
        if os.path.exists(p):
            oof = np.load(p)
            mae  = np.abs(oof - y_raw.values).mean()
            corr = np.corrcoef(oof, q95_oof)[0,1]
            print(f'  {qname}: MAE={mae:.4f} | q95 상관={corr:.4f}')

    elapsed = (time.time()-t0)/60
    print(f'\n완료 ({elapsed:.1f}분)')
    print('다음 단계: python src/run_model47_multi_q.py  (q80/q75/q70 자동 인식)')


if __name__ == '__main__':
    main()
