"""Step 1a: train 시나리오 집계만"""
import pickle, time
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

t0 = time.time()
for i, col in enumerate(SC_AGG_COLS):
    if col not in train.columns:
        continue
    grp = train.groupby('scenario_id')[col]
    train[f'sc_{col}_mean'] = grp.transform('mean')
    train[f'sc_{col}_std']  = grp.transform('std').fillna(0)
    train[f'sc_{col}_max']  = grp.transform('max')
    train[f'sc_{col}_min']  = grp.transform('min')
    train[f'sc_{col}_diff'] = train[col] - train[f'sc_{col}_mean']
    train[f'sc_{col}_median'] = grp.transform('median')
    # agg → map for quantiles/skew/kurtosis
    sc_agg = grp.agg(
        skew='skew',
        p10=lambda x: x.quantile(0.10),
        p90=lambda x: x.quantile(0.90),
    )
    sc_agg['kurtosis'] = grp.apply(lambda x: x.kurtosis())
    sc_agg = sc_agg.fillna(0)
    sid = train['scenario_id']
    train[f'sc_{col}_p10'] = sid.map(sc_agg['p10'])
    train[f'sc_{col}_p90'] = sid.map(sc_agg['p90'])
    train[f'sc_{col}_skew'] = sid.map(sc_agg['skew']).fillna(0)
    train[f'sc_{col}_kurtosis'] = sid.map(sc_agg['kurtosis']).fillna(0)
    cv = train[f'sc_{col}_std'] / (train[f'sc_{col}_mean'].abs() + 1e-8)
    train[f'sc_{col}_cv'] = cv.fillna(0)
    print(f'  [{i+1}/{len(SC_AGG_COLS)}] {col} done ({time.time()-t0:.1f}s)')

print(f'train 집계 완료: {train.shape}')
with open('/tmp/bc_train_agg.pkl', 'wb') as f:
    pickle.dump(train, f)
print(f'저장 완료. 총 {time.time()-t0:.1f}s')
