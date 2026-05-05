"""Step 1 (fast): 시나리오 집계를 vectorized로 수행"""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np

with open('/tmp/bc_features.pkl', 'rb') as f:
    train, test = pickle.load(f)

SC_AGG_COLS = [
    'robot_utilization','order_inflow_15m','low_battery_ratio','congestion_score',
    'max_zone_density','charge_queue_length','battery_mean','battery_std',
    'robot_idle','robot_active','robot_charging','near_collision_15m',
    'fault_count_15m','avg_recovery_time','blocked_path_15m','sku_concentration',
    'urgent_order_ratio','pack_utilization',
]

def add_scenario_agg_fast(df):
    df = df.copy()
    for i, col in enumerate(SC_AGG_COLS):
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]
        # 빠른 built-in transforms
        df[f'sc_{col}_mean'] = grp.transform('mean')
        df[f'sc_{col}_std']  = grp.transform('std').fillna(0)
        df[f'sc_{col}_max']  = grp.transform('max')
        df[f'sc_{col}_min']  = grp.transform('min')
        df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']
        df[f'sc_{col}_median'] = grp.transform('median')

        # p10/p90/skew/kurtosis: agg → map (lambda transform보다 10x 빠름)
        sc_agg = grp.agg(['skew', lambda x: x.kurtosis(),
                           lambda x: x.quantile(0.10),
                           lambda x: x.quantile(0.90)])
        sc_agg.columns = ['skew', 'kurtosis', 'p10', 'p90']
        sc_agg = sc_agg.fillna(0)
        sid = df['scenario_id']
        df[f'sc_{col}_p10'] = sid.map(sc_agg['p10'])
        df[f'sc_{col}_p90'] = sid.map(sc_agg['p90'])
        df[f'sc_{col}_skew'] = sid.map(sc_agg['skew']).fillna(0)
        df[f'sc_{col}_kurtosis'] = sid.map(sc_agg['kurtosis']).fillna(0)

        cv = df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)
        df[f'sc_{col}_cv'] = cv.fillna(0)

        if (i + 1) % 6 == 0:
            print(f'  집계 진행: {i+1}/{len(SC_AGG_COLS)}')
    return df

t0 = time.time()
print('시나리오 집계 피처 추가 (fast)...')
train = add_scenario_agg_fast(train)
print(f'  train 완료: {train.shape}, {time.time()-t0:.1f}s')
t1 = time.time()
test  = add_scenario_agg_fast(test)
print(f'  test 완료: {test.shape}, {time.time()-t1:.1f}s')

# 비율 피처
def safe_div(a, b, fill=0):
    return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

for df in [train, test]:
    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = safe_div(df['sc_order_inflow_15m_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = safe_div(df['sc_congestion_score_mean'], df['intersection_count'])
    if 'sc_low_battery_ratio_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_stress'] = safe_div(df['sc_low_battery_ratio_mean'] * df.get('sc_charge_queue_length_mean', 0), df['charger_count'])
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = safe_div(df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']
    if all(c in df.columns for c in ['sc_congestion_score_mean','sc_order_inflow_15m_mean','robot_total']):
        df['ratio_cross_stress'] = safe_div(df['sc_congestion_score_mean']*df['sc_order_inflow_15m_mean'], df['robot_total']**2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm']/100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(df['pack_station_count'], df['floor_area_sqm']/1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if 'sc_battery_mean_mean' in df.columns and 'sc_robot_utilization_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_per_robot'] = safe_div(df['sc_battery_mean_mean']*df['sc_robot_utilization_mean'], df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])

feat_cols = [c for c in train.columns
             if c not in {'ID','scenario_id','layout_id','avg_delay_minutes_next_30m'}
             and train[c].dtype != object]
print(f'최종 피처 수: {len(feat_cols)}')

with open('/tmp/bc_full_features.pkl', 'wb') as f:
    pickle.dump((train, test, feat_cols), f)
print(f'전체 캐시 저장 완료. 총 소요: {time.time()-t0:.1f}s')
