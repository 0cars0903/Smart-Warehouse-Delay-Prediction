"""시나리오 집계 - 메모리 최적화 (merge 대신 map)"""
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
import numpy as np
import gc

WHICH = os.environ.get('WHICH', 'train')
BATCH = int(os.environ.get('BATCH', '0'))  # 0=first 9, 1=last 9

SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

if BATCH == 0:
    cols_to_process = SC_AGG_COLS[:9]
else:
    cols_to_process = SC_AGG_COLS[9:]

t0 = time.time()

# 이전 배치 결과가 있으면 그걸 로드, 없으면 bf 로드
if BATCH == 0:
    df = pd.read_feather(f'docs/_v4_{WHICH}_bf.feather')
else:
    df = pd.read_feather(f'docs/_v4_{WHICH}_agg_b0.feather')
print(f'{WHICH} batch={BATCH} 로드: {time.time()-t0:.1f}s, shape={df.shape}')

sid = df['scenario_id']

for i, col in enumerate(cols_to_process):
    if col not in df.columns:
        continue
    t1 = time.time()

    grp = df.groupby('scenario_id')[col]

    # 빠른 transform 5종
    df[f'sc_{col}_mean'] = grp.transform('mean')
    df[f'sc_{col}_std']  = grp.transform('std').fillna(0)
    df[f'sc_{col}_max']  = grp.transform('max')
    df[f'sc_{col}_min']  = grp.transform('min')
    df[f'sc_{col}_median'] = grp.transform('median')
    df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']

    # 느린 통계: 시나리오 레벨에서 계산 후 map
    sc_stats = df.groupby('scenario_id')[col].agg(['mean']).copy()  # just for index
    sc_vals = df.groupby('scenario_id')[col]

    p10 = sc_vals.quantile(0.10)
    p90 = sc_vals.quantile(0.90)
    skew = sc_vals.skew().fillna(0)
    kurt = sc_vals.apply(pd.Series.kurtosis).fillna(0)

    df[f'sc_{col}_p10'] = sid.map(p10)
    df[f'sc_{col}_p90'] = sid.map(p90)
    df[f'sc_{col}_skew'] = sid.map(skew)
    df[f'sc_{col}_kurtosis'] = sid.map(kurt)

    # cv
    df[f'sc_{col}_cv'] = (df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)

    gc.collect()
    print(f'  [{i+1:2d}/{len(cols_to_process)}] {col}: {time.time()-t1:.1f}s')

print(f'{WHICH} batch={BATCH} 완료: {time.time()-t0:.1f}s, shape={df.shape}')
df.to_feather(f'docs/_v4_{WHICH}_agg_b{BATCH}.feather')
print(f'저장 완료: {time.time()-t0:.1f}s')
