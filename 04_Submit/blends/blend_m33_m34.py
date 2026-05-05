"""
블렌드 1: model33(Asym α=1.5) × model34(Asym α=2.0)
================================================================
근거:
  - model33: Public 9.8223, pred_std ~15.5, Asym α=1.5 (보수적 극값)
  - model34-B: Public 9.8078, pred_std 16.15, Asym α=2.0 (대담한 극값)
  - 두 모델의 극값 예측 패턴이 다름 → 블렌드로 최적점 탐색
  - model34-B가 Public 최고이므로 가중치 0.5~0.8 범위 탐색

실행: python src/blend_m33_m34.py
"""

import numpy as np
import pandas as pd
import os

_BASE   = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.join(_BASE, '..', 'submissions')

# ── 제출 파일 로드 ──
m33 = pd.read_csv(os.path.join(SUB_DIR, 'model33_6model_asym.csv'))
m34 = pd.read_csv(os.path.join(SUB_DIR, 'model34_6asym20.csv'))

TARGET = 'avg_delay_minutes_next_30m'

print('='*60)
print('블렌드 1: model33(α=1.5) × model34-B(α=2.0)')
print('='*60)
print(f'  model33: mean={m33[TARGET].mean():.2f}, std={m33[TARGET].std():.2f}, '
      f'max={m33[TARGET].max():.2f}')
print(f'  model34: mean={m34[TARGET].mean():.2f}, std={m34[TARGET].std():.2f}, '
      f'max={m34[TARGET].max():.2f}')

# ── 다양한 비율 블렌드 ──
ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # model34 가중치

print(f'\n  {"m34 weight":>10s}  {"mean":>8s}  {"std":>8s}  {"max":>8s}  파일')
print(f'  {"-"*10}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*30}')

for w34 in ratios:
    w33 = 1 - w34
    blend = m33.copy()
    blend[TARGET] = w33 * m33[TARGET].values + w34 * m34[TARGET].values
    blend[TARGET] = blend[TARGET].clip(lower=0)

    fname = f'blend_m33m34_w{int(w34*100)}.csv'
    blend.to_csv(os.path.join(SUB_DIR, fname), index=False)

    print(f'  {w34:>10.1%}  {blend[TARGET].mean():>8.2f}  '
          f'{blend[TARGET].std():>8.2f}  {blend[TARGET].max():>8.2f}  {fname}')

# ── 추천 ──
print(f'\n추천 제출: w34=0.7 (model34 70%, model33 30%)')
print(f'  → model34가 Public 최고이므로 높은 가중치 부여')
print(f'  → pred_std가 model34(16.15)에 가까울수록 배율 유리')
print(f'\n저장 완료: {SUB_DIR}/blend_m33m34_w*.csv (6종)')
