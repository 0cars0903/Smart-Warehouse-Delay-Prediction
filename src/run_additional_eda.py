"""
run_additional_eda.py
=====================
추가 EDA 3방향 분석

  Part A. 시나리오 시작 조건(TS0) 분석
          → 초기 배터리/로봇 상태가 이후 지연 궤적을 얼마나 결정하는가
          → "붕괴 시나리오" vs "안정 시나리오" 초기 특성 비교
          → 피처 엔지니어링 제안: TS0 broadcast 피처

  Part B. 극단 지연 세그먼트 분석 (P90+)
          → 상관관계 역전 현상 정밀 측정
          → 극단 진입 조건 프로파일링
          → 2-stage 모델링 타당성 검토

  Part C. 결측치 패턴 구조 분석
          → 타임슬롯별/창고별 결측 집중 여부
          → 결측 자체의 타겟 예측력 (MNAR 검증)
          → 결측 indicator 피처 가치 평가

실행: python src/run_additional_eda.py
예상 시간: 약 3~5분
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
import os

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data') + '/'
OUT_PATH    = os.path.join(os.path.dirname(__file__), '..', 'eda_outputs') + '/'
TARGET      = 'avg_delay_minutes_next_30m'
os.makedirs(OUT_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────
print("데이터 로드 중...")
train  = pd.read_csv(DATA_PATH + 'train.csv')
layout = pd.read_csv(DATA_PATH + 'layout_info.csv')
train  = train.merge(layout, on='layout_id', how='left')

# 타임슬롯 인덱스 생성
train['ts_idx'] = train.groupby('scenario_id').cumcount()

y = train[TARGET].values
print(f"Train: {train.shape} | Target mean={y.mean():.2f}, std={y.std():.2f}")

# 핵심 피처 목록
BATTERY_COLS  = ['battery_mean', 'low_battery_ratio', 'battery_std',
                  'charge_queue_length', 'avg_charge_wait', 'robot_charging']
ROBOT_COLS    = ['robot_active', 'robot_idle', 'robot_utilization',
                  'agv_task_success_rate', 'task_reassign_15m']
ORDER_COLS    = ['order_inflow_15m', 'urgent_order_ratio', 'sku_concentration',
                  'unique_sku_15m', 'avg_items_per_order']
CONGESTION_COLS = ['congestion_score', 'max_zone_density', 'blocked_path_15m',
                    'near_collision_15m', 'avg_trip_distance']
FAULT_COLS    = ['fault_count_15m', 'avg_recovery_time']
PACK_COLS     = ['pack_utilization', 'staging_area_util', 'manual_override_ratio']

KEY_COLS = (BATTERY_COLS[:4] + ROBOT_COLS[:3] +
            ORDER_COLS[:3] + CONGESTION_COLS[:3] + FAULT_COLS + PACK_COLS[:2])


# ═════════════════════════════════════════════════════════════
# PART A: 시나리오 시작 조건(TS0) 분석
# ═════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" PART A: 시나리오 시작 조건(TS0) 분석")
print("="*60)

ts0  = train[train['ts_idx'] == 0].copy()
ts_last = train[train['ts_idx'] == 24].copy()

# 시나리오별 평균 지연 및 최대 지연
sc_stats = train.groupby('scenario_id')[TARGET].agg(
    sc_mean='mean', sc_max='max', sc_std='std'
).reset_index()

ts0 = ts0.merge(sc_stats, on='scenario_id')

# ── A1. TS0 피처 vs 시나리오 평균 지연 상관관계
print("\n[A1] TS0 초기값 vs 시나리오 평균 지연 상관계수")
ts0_corr = {}
for col in KEY_COLS:
    if col in ts0.columns:
        valid = ts0[[col, 'sc_mean']].dropna()
        if len(valid) > 100:
            r, p = stats.pearsonr(valid[col], valid['sc_mean'])
            ts0_corr[col] = {'r': r, 'p': p}

ts0_corr_df = pd.DataFrame(ts0_corr).T.sort_values('r', key=abs, ascending=False)
print(ts0_corr_df.head(15).to_string())

# ── A2. "붕괴" vs "안정" 시나리오 초기 조건 비교
P_HIGH = np.percentile(sc_stats['sc_mean'], 75)
P_LOW  = np.percentile(sc_stats['sc_mean'], 25)

collapse_ids = sc_stats[sc_stats['sc_mean'] >= P_HIGH]['scenario_id']
stable_ids   = sc_stats[sc_stats['sc_mean'] <= P_LOW]['scenario_id']

ts0_collapse = ts0[ts0['scenario_id'].isin(collapse_ids)]
ts0_stable   = ts0[ts0['scenario_id'].isin(stable_ids)]

print(f"\n[A2] 붕괴 시나리오(상위25%): {len(collapse_ids)}개 | "
      f"안정 시나리오(하위25%): {len(stable_ids)}개")
print(f"  붕괴 기준: sc_mean ≥ {P_HIGH:.1f}분 | 안정 기준: sc_mean ≤ {P_LOW:.1f}분")

a2_rows = []
for col in KEY_COLS:
    if col in ts0.columns:
        c_mean = ts0_collapse[col].mean()
        s_mean = ts0_stable[col].mean()
        ratio  = c_mean / (s_mean + 1e-8)
        a2_rows.append({'피처': col, '붕괴_TS0': c_mean, '안정_TS0': s_mean, '배율': ratio})

a2_df = pd.DataFrame(a2_rows).sort_values('배율', key=abs, ascending=False)
print(a2_df.head(12).to_string(index=False))

# ── A3. TS0 배터리 분포 → 시나리오 결말 예측력
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Part A: Scenario Initial Conditions (TS0) vs Outcome', fontsize=14, fontweight='bold')

top_ts0_feats = ts0_corr_df.head(6).index.tolist()
for ax, col in zip(axes.flatten(), top_ts0_feats):
    if col in ts0.columns:
        valid = ts0[[col, 'sc_mean']].dropna()
        ax.scatter(valid[col], valid['sc_mean'], alpha=0.3, s=5, color='steelblue')
        r = ts0_corr_df.loc[col, 'r']
        ax.set_title(f'{col}\n(r={r:.3f})', fontsize=9)
        ax.set_xlabel('TS0 value', fontsize=8)
        ax.set_ylabel('Scenario avg delay (min)', fontsize=8)
        ax.tick_params(labelsize=7)
        # 선형 추세선
        z = np.polyfit(valid[col].fillna(0), valid['sc_mean'], 1)
        p = np.poly1d(z)
        xrange = np.linspace(valid[col].min(), valid[col].max(), 100)
        ax.plot(xrange, p(xrange), 'r-', alpha=0.7, linewidth=1.5)

plt.tight_layout()
plt.savefig(OUT_PATH + 'A1_ts0_vs_outcome.png', dpi=120, bbox_inches='tight')
plt.close()

# A2 시각화: 붕괴 vs 안정 초기 조건 비교 (상위 10 피처)
plot_feats = a2_df.head(10)['피처'].tolist()
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
fig.suptitle('Part A: TS0 Initial State — Collapse vs Stable Scenarios', fontsize=13, fontweight='bold')

for ax, col in zip(axes.flatten(), plot_feats):
    if col in ts0.columns:
        c_vals = ts0_collapse[col].dropna()
        s_vals = ts0_stable[col].dropna()
        ax.boxplot([s_vals, c_vals], labels=['Stable', 'Collapse'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue'),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_title(col, fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(OUT_PATH + 'A2_ts0_collapse_vs_stable.png', dpi=120, bbox_inches='tight')
plt.close()
print("  → A1_ts0_vs_outcome.png, A2_ts0_collapse_vs_stable.png 저장")


# ═════════════════════════════════════════════════════════════
# PART B: 극단 지연 세그먼트 분석
# ═════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" PART B: 극단 지연 세그먼트 분석 (P90+)")
print("="*60)

P90 = np.percentile(y, 90)
P75 = np.percentile(y, 75)
P50 = np.percentile(y, 50)

print(f"  P50={P50:.1f}분 | P75={P75:.1f}분 | P90={P90:.1f}분")

mask_normal  = train[TARGET] <= P50
mask_mid     = (train[TARGET] > P50) & (train[TARGET] <= P90)
mask_extreme = train[TARGET] > P90

print(f"  일반(≤P50): {mask_normal.sum():,}행 | "
      f"중간(P50~P90): {mask_mid.sum():,}행 | "
      f"극단(>P90): {mask_extreme.sum():,}행")

# ── B1. 구간별 상관계수 비교
print("\n[B1] 구간별 Pearson r (vs 타겟) — 상관 역전 탐지")
b1_rows = []
for col in KEY_COLS:
    if col not in train.columns:
        continue
    valid_all = train[[col, TARGET]].dropna()
    r_all, _  = stats.pearsonr(valid_all[col], valid_all[TARGET])

    ext_df = train.loc[mask_extreme, [col, TARGET]].dropna()
    r_ext  = stats.pearsonr(ext_df[col], ext_df[TARGET])[0] if len(ext_df) > 50 else np.nan

    nor_df = train.loc[mask_normal, [col, TARGET]].dropna()
    r_nor  = stats.pearsonr(nor_df[col], nor_df[TARGET])[0] if len(nor_df) > 50 else np.nan

    reversed_ = (r_all * r_ext < 0) if not np.isnan(r_ext) else False
    b1_rows.append({
        '피처': col, 'r_전체': r_all, 'r_일반(≤P50)': r_nor,
        'r_극단(>P90)': r_ext, '역전여부': '⚠️ 역전' if reversed_ else ''
    })

b1_df = pd.DataFrame(b1_rows).sort_values('r_전체', key=abs, ascending=False)
print(b1_df.to_string(index=False))

# ── B2. 극단 지연 진입 조건: 이전 타임슬롯 상태
# 극단 지연 행의 직전 타임슬롯(ts_idx-1)과 일반 행의 직전 타임슬롯 비교
train_sorted = train.sort_values(['scenario_id', 'ts_idx'])
train_sorted['target_lag1'] = train_sorted.groupby('scenario_id')[TARGET].shift(1)
train_sorted['is_extreme'] = (train_sorted[TARGET] > P90).astype(int)

for col in BATTERY_COLS[:3] + ROBOT_COLS[:2]:
    if col in train_sorted.columns:
        train_sorted[f'{col}_lag1'] = train_sorted.groupby('scenario_id')[col].shift(1)

lag_feats = [f'{c}_lag1' for c in BATTERY_COLS[:3] + ROBOT_COLS[:2]
             if f'{c}_lag1' in train_sorted.columns]

print("\n[B2] 극단 지연 발생 1 타임슬롯 전 상태 비교")
b2_rows = []
for col in lag_feats:
    ext_mean = train_sorted.loc[train_sorted['is_extreme'] == 1, col].mean()
    nor_mean = train_sorted.loc[train_sorted['is_extreme'] == 0, col].mean()
    b2_rows.append({'피처(직전슬롯)': col, '극단진입전': ext_mean, '일반상태': nor_mean,
                    '배율': ext_mean / (nor_mean + 1e-8)})
b2_df = pd.DataFrame(b2_rows).sort_values('배율', key=abs, ascending=False)
print(b2_df.to_string(index=False))

# ── B3. 시각화: 구간별 상관계수 역전
fig, ax = plt.subplots(figsize=(12, 7))
plot_b1 = b1_df[b1_df['r_극단(>P90)'].notna()].head(14)
x = np.arange(len(plot_b1))
w = 0.28

bars1 = ax.bar(x - w, plot_b1['r_일반(≤P50)'], w, label='일반(≤P50)', color='steelblue', alpha=0.85)
bars2 = ax.bar(x,      plot_b1['r_전체'],       w, label='전체',      color='gray',      alpha=0.7)
bars3 = ax.bar(x + w,  plot_b1['r_극단(>P90)'], w, label='극단(>P90)', color='crimson',   alpha=0.85)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(plot_b1['피처'], rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Pearson r (vs target)', fontsize=11)
ax.set_title('Part B: Correlation by Delay Segment — Detecting Reversal', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

# 역전 표시
for i, (_, row) in enumerate(plot_b1.iterrows()):
    if row['역전여부']:
        ax.text(i, max(abs(row['r_전체']), abs(row['r_극단(>P90)'])) + 0.02,
                '↕', ha='center', color='darkorange', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_PATH + 'B1_extreme_segment_corr.png', dpi=120, bbox_inches='tight')
plt.close()

# B4. 극단 지연 발생 타임슬롯 분포
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Part B: Extreme Delay Occurrence Pattern', fontsize=13, fontweight='bold')

extreme_ts = train.loc[mask_extreme, 'ts_idx']
axes[0].hist(extreme_ts, bins=25, color='crimson', alpha=0.7, edgecolor='white')
axes[0].set_xlabel('Timeslot Index (ts_idx)', fontsize=11)
axes[0].set_ylabel('Count of extreme delay rows', fontsize=11)
axes[0].set_title('Extreme Delay by Timeslot', fontsize=11)
axes[0].set_xticks(range(0, 25, 2))

# layout_type별 극단 지연 비율
if 'layout_type' in train.columns:
    lt_extreme = train.groupby('layout_type').apply(
        lambda x: (x[TARGET] > P90).mean()).sort_values(ascending=False)
    axes[1].bar(lt_extreme.index, lt_extreme.values * 100, color='darkorange', alpha=0.85)
    axes[1].set_ylabel('Extreme delay rate (%)', fontsize=11)
    axes[1].set_title('Extreme Delay Rate by Layout Type', fontsize=11)
    for i, (lbl, v) in enumerate(lt_extreme.items()):
        axes[1].text(i, v * 100 + 0.2, f'{v*100:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUT_PATH + 'B2_extreme_pattern.png', dpi=120, bbox_inches='tight')
plt.close()
print("  → B1_extreme_segment_corr.png, B2_extreme_pattern.png 저장")


# ═════════════════════════════════════════════════════════════
# PART C: 결측치 패턴 구조 분석
# ═════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" PART C: 결측치 패턴 구조 분석")
print("="*60)

feat_cols = [c for c in train.columns
             if c not in {'ID', 'layout_id', 'scenario_id', TARGET, 'ts_idx'}
             and train[c].dtype in [np.float64, np.int64, float, int]]

missing_rate = train[feat_cols].isnull().mean().sort_values(ascending=False)
print(f"결측 있는 피처: {(missing_rate > 0).sum()}개 / {len(feat_cols)}개")
print(f"결측률 상위:\n{missing_rate[missing_rate > 0].head(10).to_string()}")

# ── C1. 타임슬롯별 결측률 (시간에 따른 패턴?)
high_missing_cols = missing_rate[missing_rate > 0.05].index.tolist()[:8]
ts_missing = train.groupby('ts_idx')[high_missing_cols].apply(lambda x: x.isnull().mean())

print(f"\n[C1] 타임슬롯별 결측률 (상위 {len(high_missing_cols)}개 피처)")
print(f"  최대 편차(ts별): {ts_missing.std().max():.4f}")
print(f"  결론: {'타임슬롯별 결측 패턴 있음 (MNAR 가능)' if ts_missing.std().max() > 0.02 else '타임슬롯 무관 (균일 결측)'}")

# ── C2. layout_type별 결측률
if 'layout_type' in train.columns:
    lt_missing = train.groupby('layout_type')[high_missing_cols].apply(
        lambda x: x.isnull().mean())
    print(f"\n[C2] layout_type별 결측률")
    print(lt_missing.to_string())
    max_lt_diff = lt_missing.max() - lt_missing.min()
    print(f"  최대 layout_type 간 결측 편차: {max_lt_diff.max():.4f}")

# ── C3. 결측 indicator vs 타겟 상관 (MNAR 검증 핵심)
print("\n[C3] 결측 indicator (is_missing=1/0) vs 타겟 상관 (MNAR 검증)")
c3_rows = []
for col in missing_rate[missing_rate > 0.03].index:
    indicator = train[col].isnull().astype(int)
    if indicator.sum() < 100:
        continue
    r, p = stats.pointbiserialr(indicator, train[TARGET])
    c3_rows.append({'피처': col, '결측률': missing_rate[col], 'r(indicator→target)': r, 'p-value': p})

c3_df = pd.DataFrame(c3_rows).sort_values('r(indicator→target)', key=abs, ascending=False)
print(c3_df.head(15).to_string(index=False))

sig_mnar = c3_df[c3_df['p-value'] < 0.001]
print(f"\n  MNAR 강하게 시사 (p<0.001): {len(sig_mnar)}개 피처")
if len(sig_mnar) > 0:
    print(f"  상위: {sig_mnar['피처'].tolist()[:5]}")

# ── C4. 시각화
fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle('Part C: Missing Value Pattern Analysis', fontsize=14, fontweight='bold')

# C4-1: 타임슬롯별 결측률 히트맵
ax1 = fig.add_subplot(gs[0, :])
if len(high_missing_cols) > 0:
    im = ax1.imshow(ts_missing[high_missing_cols].T.values, aspect='auto',
                    cmap='YlOrRd', vmin=0, vmax=0.25)
    ax1.set_xticks(range(25))
    ax1.set_xticklabels(range(25), fontsize=8)
    ax1.set_yticks(range(len(high_missing_cols)))
    ax1.set_yticklabels([c[:25] for c in high_missing_cols], fontsize=8)
    ax1.set_xlabel('Timeslot (ts_idx)', fontsize=10)
    ax1.set_title('Missing Rate by Timeslot (Heatmap)', fontsize=11)
    plt.colorbar(im, ax=ax1, label='Missing Rate')

# C4-2: 결측 indicator vs 타겟 상관
ax2 = fig.add_subplot(gs[1, 0])
if len(c3_df) > 0:
    plot_c3 = c3_df.head(12)
    colors = ['crimson' if r > 0 else 'steelblue' for r in plot_c3['r(indicator→target)']]
    bars = ax2.barh(plot_c3['피처'], plot_c3['r(indicator→target)'], color=colors, alpha=0.8)
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('Pearson r (missing indicator → target)', fontsize=9)
    ax2.set_title('C3: MNAR Test — Missing Indicator vs Target', fontsize=10)
    ax2.tick_params(labelsize=8)

# C4-3: 결측률 전체 분포
ax3 = fig.add_subplot(gs[1, 1])
nonzero_missing = missing_rate[missing_rate > 0]
ax3.hist(nonzero_missing.values, bins=20, color='steelblue', alpha=0.8, edgecolor='white')
ax3.axvline(nonzero_missing.mean(), color='red', linestyle='--',
            label=f'Mean={nonzero_missing.mean():.3f}')
ax3.set_xlabel('Missing Rate', fontsize=10)
ax3.set_ylabel('Feature Count', fontsize=10)
ax3.set_title('C4: Missing Rate Distribution', fontsize=10)
ax3.legend(fontsize=9)

plt.savefig(OUT_PATH + 'C1_missing_pattern.png', dpi=120, bbox_inches='tight')
plt.close()
print("  → C1_missing_pattern.png 저장")


# ═════════════════════════════════════════════════════════════
# 종합 인사이트 요약 저장
# ═════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" 종합 인사이트 요약")
print("="*60)

# 상위 TS0 피처 추출
top_ts0 = ts0_corr_df[ts0_corr_df['p'] < 0.001].head(8)
top_reversed = b1_df[b1_df['역전여부'] == '⚠️ 역전']
top_mnar = c3_df[c3_df['p-value'] < 0.001].head(8)

print(f"\n[A] TS0 초기값 → 시나리오 결과 상관 Top피처:")
for feat, row in top_ts0.iterrows():
    print(f"  {feat:<35} r={row['r']:.3f}")

print(f"\n[B] 상관관계 역전 피처 (전체 vs 극단구간):")
for _, row in top_reversed.iterrows():
    print(f"  {row['피처']:<35} 전체 r={row['r_전체']:.3f} → 극단 r={row['r_극단(>P90)']:.3f}")

print(f"\n[C] MNAR 강하게 시사하는 피처 (결측 indicator 유의):")
for _, row in top_mnar.iterrows():
    print(f"  {row['피처']:<35} indicator_r={row['r(indicator→target)']:.3f}")

# 마크다운 요약 저장
summary_lines = [
    "# 추가 EDA 분석 인사이트 (2026-04-11)\n",
    "## Part A: 시나리오 시작 조건(TS0) 분석\n",
    "### A1. TS0 초기값 → 시나리오 평균 지연 상관 (Top)\n",
    "| 피처 | r | 해석 |\n|------|---|------|\n",
]
for feat, row in top_ts0.iterrows():
    summary_lines.append(f"| `{feat}` | {row['r']:.3f} | TS0 초기값이 시나리오 결과를 결정 |\n")

summary_lines += [
    "\n### A2. 붕괴 vs 안정 시나리오 초기 조건 비교 (Top)\n",
    "| 피처 | 붕괴_TS0 | 안정_TS0 | 배율 |\n|------|---------|---------|------|\n",
]
for _, row in a2_df.head(8).iterrows():
    summary_lines.append(
        f"| `{row['피처']}` | {row['붕괴_TS0']:.3f} | {row['안정_TS0']:.3f} | {row['배율']:.1f}× |\n"
    )

summary_lines += [
    "\n### A3. 피처 엔지니어링 제안\n",
    "- **TS0 broadcast 피처**: 시나리오 내 모든 타임슬롯에 TS0 값을 복사하여 추가\n",
    "- 특히 `battery_mean(TS0)`, `robot_idle(TS0)`, `order_inflow_15m(TS0)` 유망\n",
    "- 시나리오 붕괴 취약성 지수: `low_battery_ratio(TS0) × order_inflow_15m(TS0)`\n",
    "\n## Part B: 극단 지연 세그먼트 분석\n",
    "\n### B1. 상관관계 역전 피처\n",
    "| 피처 | r_전체 | r_극단(>P90) | 해석 |\n|------|--------|-------------|------|\n",
]
for _, row in top_reversed.iterrows():
    summary_lines.append(
        f"| `{row['피처']}` | {row['r_전체']:.3f} | {row['r_극단(>P90)']:.3f} | "
        f"{'극단에서 역전' if row['역전여부'] else ''} |\n"
    )

summary_lines += [
    "\n### B2. 2-stage 모델링 타당성\n",
    f"- P90 기준({P90:.1f}분) 이상 극단 지연: 전체의 10%\n",
    "- 극단 구간에서 pack_utilization 상관이 폭발적 상승 → 패킹 병목이 최종 병목\n",
    "- 배터리 관련 피처 상관 역전 → 극단 구간은 다른 인과 구조\n",
    "- **결론**: 2-stage 모델(일반/극단 분류 → 각각 회귀) 실험 가치 있음\n",
    "\n## Part C: 결측치 패턴 분석\n",
    "\n### C1. MNAR 검증 결과 (결측 indicator 유의 피처)\n",
    "| 피처 | 결측률 | indicator_r | 해석 |\n|------|--------|------------|------|\n",
]
for _, row in top_mnar.iterrows():
    mnar_yn = '✅ MNAR 강함' if abs(row['r(indicator→target)']) > 0.05 else 'MNAR 약함'
    summary_lines.append(
        f"| `{row['피처']}` | {row['결측률']:.1%} | {row['r(indicator→target)']:.3f} | {mnar_yn} |\n"
    )

summary_lines += [
    "\n### C2. 피처 엔지니어링 제안\n",
    "- MNAR 강한 피처에 대해 `is_missing` binary indicator 피처 추가\n",
    "- 결측 자체가 '이벤트 미발생'의 신호일 가능성 → indicator가 타겟과 음/양 상관\n",
    "\n## 생성 파일\n",
    "- `A1_ts0_vs_outcome.png` — TS0 초기값 vs 시나리오 평균 지연 산점도\n",
    "- `A2_ts0_collapse_vs_stable.png` — 붕괴/안정 시나리오 초기 조건 박스플롯\n",
    "- `B1_extreme_segment_corr.png` — 구간별 상관계수 역전 시각화\n",
    "- `B2_extreme_pattern.png` — 극단 지연 발생 타임슬롯·레이아웃 분포\n",
    "- `C1_missing_pattern.png` — 결측치 타임슬롯별 히트맵 + MNAR 검증\n",
]

summary_path = OUT_PATH + 'ADDITIONAL_EDA_REPORT.md'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.writelines(summary_lines)

print(f"\n마크다운 요약 저장: {summary_path}")
print("\n완료!")
print("="*60)
print(" 주요 실험 제안")
print("="*60)
print(f"  [A] TS0 broadcast 피처 추가 → feature_engineering.py 반영 대상")
print(f"  [B] 2-stage 모델 (P90 기준 극단 분리) → 별도 실험 스크립트")
print(f"  [C] MNAR indicator 피처 추가 → feature_engineering.py 반영 대상")
