"""
블렌드 2: model34-B(6model+Asym2.0) × model34-A(5model-TW15)
================================================================
근거:
  - model34-B: Public 9.8078, pred_std=16.15, 6모델(+Asym2.0)
  - model34-A: Public 9.8144, pred_std=15.87, 5모델(TW1.5 교체)
  - B가 Asym2.0 추가로 극값 구간 예측 확장, A는 TW1.5로 극값 정밀도
  - 두 모델 모두 같은 base에서 출발하므로 상관 높을 수 있으나,
    Asym2.0 유무가 극값 구간에서 차이 발생

실행: python src/blend_m34_internal.py
"""

import numpy as np
import pandas as pd
import os

_BASE   = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.join(_BASE, '..', 'submissions')

TARGET = 'avg_delay_minutes_next_30m'

# ── 제출 파일 로드 ──
m34b = pd.read_csv(os.path.join(SUB_DIR, 'model34_6asym20.csv'))
m34a = pd.read_csv(os.path.join(SUB_DIR, 'model34_5tw15.csv'))
m34d = pd.read_csv(os.path.join(SUB_DIR, 'model34_6ltw15.csv'))

print('='*60)
print('블렌드 2: model34 내부 조합')
print('='*60)
print(f'  34-B(6+Asym2.0): mean={m34b[TARGET].mean():.2f}, std={m34b[TARGET].std():.2f}, max={m34b[TARGET].max():.2f}')
print(f'  34-A(5+TW15)   : mean={m34a[TARGET].mean():.2f}, std={m34a[TARGET].std():.2f}, max={m34a[TARGET].max():.2f}')
print(f'  34-D(6+LTW15)  : mean={m34d[TARGET].mean():.2f}, std={m34d[TARGET].std():.2f}, max={m34d[TARGET].max():.2f}')

# ── 예측값 상관 확인 ──
corr_ba = np.corrcoef(m34b[TARGET].values, m34a[TARGET].values)[0,1]
corr_bd = np.corrcoef(m34b[TARGET].values, m34d[TARGET].values)[0,1]
corr_ad = np.corrcoef(m34a[TARGET].values, m34d[TARGET].values)[0,1]
print(f'\n  예측값 상관: B-A={corr_ba:.4f}, B-D={corr_bd:.4f}, A-D={corr_ad:.4f}')

# ── B × A 블렌드 ──
print(f'\n[B × A 블렌드]')
print(f'  {"B weight":>10s}  {"mean":>8s}  {"std":>8s}  {"max":>8s}  파일')
print(f'  {"-"*10}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*30}')

for wb in [0.5, 0.6, 0.7, 0.8]:
    wa = 1 - wb
    blend = m34b.copy()
    blend[TARGET] = wb * m34b[TARGET].values + wa * m34a[TARGET].values
    blend[TARGET] = blend[TARGET].clip(lower=0)

    fname = f'blend_m34ba_b{int(wb*100)}.csv'
    blend.to_csv(os.path.join(SUB_DIR, fname), index=False)
    print(f'  {wb:>10.1%}  {blend[TARGET].mean():>8.2f}  '
          f'{blend[TARGET].std():>8.2f}  {blend[TARGET].max():>8.2f}  {fname}')

# ── B × D 블렌드 ──
print(f'\n[B × D 블렌드]')
print(f'  {"B weight":>10s}  {"mean":>8s}  {"std":>8s}  {"max":>8s}  파일')
print(f'  {"-"*10}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*30}')

for wb in [0.5, 0.6, 0.7]:
    wd = 1 - wb
    blend = m34b.copy()
    blend[TARGET] = wb * m34b[TARGET].values + wd * m34d[TARGET].values
    blend[TARGET] = blend[TARGET].clip(lower=0)

    fname = f'blend_m34bd_b{int(wb*100)}.csv'
    blend.to_csv(os.path.join(SUB_DIR, fname), index=False)
    print(f'  {wb:>10.1%}  {blend[TARGET].mean():>8.2f}  '
          f'{blend[TARGET].std():>8.2f}  {blend[TARGET].max():>8.2f}  {fname}')

# ── 3종 균등 블렌드 ──
blend3 = m34b.copy()
blend3[TARGET] = (m34b[TARGET].values + m34a[TARGET].values + m34d[TARGET].values) / 3
blend3[TARGET] = blend3[TARGET].clip(lower=0)
blend3.to_csv(os.path.join(SUB_DIR, 'blend_m34_3way.csv'), index=False)
print(f'\n[3종 균등] mean={blend3[TARGET].mean():.2f}, std={blend3[TARGET].std():.2f}, '
      f'max={blend3[TARGET].max():.2f}  → blend_m34_3way.csv')

print(f'\n추천 제출: blend_m34ba_b70 (B 70% + A 30%)')
print(f'  → B의 높은 pred_std 유지하면서 A의 TW1.5 정밀도 blend')
