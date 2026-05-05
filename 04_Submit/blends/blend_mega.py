"""
전략 1: 메가 블렌드 (Mega Blend)
================================================================
근거:
  - 현재 blend_m33m34_w80 = Public 9.8073 (model34 80% + model33 20%)
  - model30, model31은 FE/파라미터가 다름 → model33/34와 상관이 낮을 수 있음
  - 3~4종 제출물 조합으로 추가 개선 탐색

제출 파일 후보:
  - model30_combined.csv : 422피처 + 29B Optuna, Public 9.8279, 배율 1.1584
  - model31_selected_fe.csv : 429피처 + shift-safe FE 7종, Public 9.8255, 배율 1.1589
  - model33_6model_asym.csv : 6model + Asym α=1.5, Public 9.8223, 배율 1.1588
  - model34_6asym20.csv : 6model + Asym α=2.0, Public 9.8078, 배율 1.1565
  - blend_m33m34_w80.csv : m33×20% + m34×80%, Public 9.8073

실행: python src/blend_mega.py
"""

import numpy as np
import pandas as pd
import os
from itertools import combinations

_BASE   = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.join(_BASE, '..', 'submissions')

TARGET = 'avg_delay_minutes_next_30m'

# ── 제출 파일 로드 ──
files = {
    'm30':      ('model30_combined.csv',      9.8279),
    'm31':      ('model31_selected_fe.csv',    9.8255),
    'm33':      ('model33_6model_asym.csv',    9.8223),
    'm34':      ('model34_6asym20.csv',        9.8078),
    'bw80':     ('blend_m33m34_w80.csv',       9.8073),
}

preds = {}
for key, (fname, pub) in files.items():
    df = pd.read_csv(os.path.join(SUB_DIR, fname))
    preds[key] = df[TARGET].values
    print(f'  {key:>6s}: {fname:<35s} Public={pub:.4f}, '
          f'mean={df[TARGET].mean():.2f}, std={df[TARGET].std():.2f}, max={df[TARGET].max():.2f}')

# ── 상관 분석 ──
keys = list(preds.keys())
print('\n' + '='*60)
print('예측값 상관 행렬')
print('='*60)
header = f'{"":>6s}  ' + '  '.join(f'{k:>6s}' for k in keys)
print(header)
for i, ki in enumerate(keys):
    row = f'{ki:>6s}  '
    for j, kj in enumerate(keys):
        corr = np.corrcoef(preds[ki], preds[kj])[0, 1]
        row += f'{corr:>6.4f}  '
    print(row)

# ── 2종 블렌드 (모든 조합) ──
print('\n' + '='*60)
print('2종 블렌드 탐색 (w1 = 0.3~0.8)')
print('='*60)
print(f'  {"조합":>12s}  {"w1":>5s}  {"mean":>8s}  {"std":>8s}  {"max":>8s}  파일')
print(f'  {"-"*12}  {"-"*5}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*35}')

ref = pd.read_csv(os.path.join(SUB_DIR, files['bw80'][0]))  # 템플릿

blend_results = []

for (k1, k2) in combinations(keys, 2):
    # bw80은 m33+m34의 블렌드이므로 m33/m34와 조합 시 중복 → 스킵 일부
    for w1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        w2 = 1 - w1
        blended = w1 * preds[k1] + w2 * preds[k2]
        blended = np.clip(blended, 0, None)

        mean_v = np.mean(blended)
        std_v = np.std(blended)
        max_v = np.max(blended)

        fname = f'mega_{k1}_{k2}_w{int(w1*100)}.csv'

        out = ref.copy()
        out[TARGET] = blended
        out.to_csv(os.path.join(SUB_DIR, fname), index=False)

        blend_results.append({
            'combo': f'{k1}+{k2}',
            'w1': w1,
            'mean': mean_v,
            'std': std_v,
            'max': max_v,
            'fname': fname,
        })

        print(f'  {k1}+{k2:>5s}  {w1:>5.1%}  {mean_v:>8.2f}  '
              f'{std_v:>8.2f}  {max_v:>8.2f}  {fname}')

# ── 3종 블렌드 (핵심 조합만) ──
print('\n' + '='*60)
print('3종 블렌드 탐색')
print('='*60)

# 가장 유망한 3종 조합: m30 + m34 + bw80 (FE 다양성 + 최고 성능)
triple_combos = [
    ('m30', 'm34', 'bw80'),
    ('m30', 'm33', 'm34'),
    ('m31', 'm33', 'm34'),
    ('m30', 'm31', 'bw80'),
    ('m30', 'm31', 'm34'),
]

for (k1, k2, k3) in triple_combos:
    # 균등 + 비균등 비율
    weight_sets = [
        (1/3, 1/3, 1/3),
        (0.2, 0.3, 0.5),
        (0.1, 0.2, 0.7),
        (0.1, 0.3, 0.6),
        (0.2, 0.2, 0.6),
        (0.15, 0.15, 0.7),
    ]

    for (w1, w2, w3) in weight_sets:
        blended = w1 * preds[k1] + w2 * preds[k2] + w3 * preds[k3]
        blended = np.clip(blended, 0, None)

        mean_v = np.mean(blended)
        std_v = np.std(blended)
        max_v = np.max(blended)

        fname = f'mega3_{k1}_{k2}_{k3}_{int(w1*100)}_{int(w2*100)}_{int(w3*100)}.csv'

        out = ref.copy()
        out[TARGET] = blended
        out.to_csv(os.path.join(SUB_DIR, fname), index=False)

        blend_results.append({
            'combo': f'{k1}+{k2}+{k3}',
            'w1': w1,
            'mean': mean_v,
            'std': std_v,
            'max': max_v,
            'fname': fname,
        })

        print(f'  {k1}+{k2}+{k3}  w=({w1:.2f},{w2:.2f},{w3:.2f})  '
              f'mean={mean_v:.2f}  std={std_v:.2f}  max={max_v:.2f}  {fname}')

# ── 4종 블렌드 ──
print('\n' + '='*60)
print('4종 블렌드 (핵심 조합)')
print('='*60)

quad_combos = [
    ('m30', 'm31', 'm33', 'm34'),
    ('m30', 'm33', 'm34', 'bw80'),
]

for (k1, k2, k3, k4) in quad_combos:
    weight_sets = [
        (0.25, 0.25, 0.25, 0.25),
        (0.1, 0.1, 0.2, 0.6),
        (0.1, 0.1, 0.1, 0.7),
        (0.15, 0.15, 0.2, 0.5),
        (0.1, 0.2, 0.2, 0.5),
    ]

    for (w1, w2, w3, w4) in weight_sets:
        blended = w1*preds[k1] + w2*preds[k2] + w3*preds[k3] + w4*preds[k4]
        blended = np.clip(blended, 0, None)

        mean_v = np.mean(blended)
        std_v = np.std(blended)
        max_v = np.max(blended)

        fname = f'mega4_{k1}_{k2}_{k3}_{k4}_{int(w1*100)}_{int(w2*100)}_{int(w3*100)}_{int(w4*100)}.csv'

        out = ref.copy()
        out[TARGET] = blended
        out.to_csv(os.path.join(SUB_DIR, fname), index=False)

        print(f'  {k1}+{k2}+{k3}+{k4}  w=({w1:.2f},{w2:.2f},{w3:.2f},{w4:.2f})  '
              f'mean={mean_v:.2f}  std={std_v:.2f}  max={max_v:.2f}  {fname}')

# ── 추천 ──
print('\n' + '='*60)
print('추천 제출 후보 (pred_std 기준 내림차순 Top 10)')
print('='*60)

# std가 높을수록 배율 유리 → 상위 10개 출력
df_results = pd.DataFrame(blend_results)
df_top = df_results.nlargest(10, 'std')
for _, row in df_top.iterrows():
    print(f'  {row["combo"]:<20s} w1={row["w1"]:.2f}  '
          f'std={row["std"]:.2f}  max={row["max"]:.2f}  → {row["fname"]}')

print(f'\n핵심: m34(Public 최고)에 가중치 50~70% 부여하면서')
print(f'm30/m31(다른 FE)을 10~20% 혼합하는 것이 유리할 것으로 예상')
print(f'총 생성 파일 수: {len(blend_results)}개')
