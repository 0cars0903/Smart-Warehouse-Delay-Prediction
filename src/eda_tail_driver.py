"""
§1 Tail Driver Decomposition EDA
================================================================
Notion 전략문서 "다시 EDA" §1 실행:
  핵심 질문: 타겟의 right-tail을 만드는 것은 수치 피처의 값인가,
             범주형 피처의 조합인가?

분석:
  1) 구간별 수치 피처 분포 비교 (KS-test)
  2) 범주형 피처 lift 분석
  3) Mutual Information ranking (피처 vs tail 여부)
  4) SHAP on tail-only subset
  5) 범주형 조합 lift 탐색
  6) 큐잉이론 physics-informed feature 후보 탐색

실행: python src/eda_tail_driver.py
출력: docs/eda_tail_driver_report.txt + docs/ 시각화 PNG
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GroupKFold
from itertools import combinations
import warnings, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
DOCS_DIR = os.path.join(_BASE, '..', 'docs')
os.makedirs(DOCS_DIR, exist_ok=True)

# ── 구간 정의 (Notion 문서 기준) ──
BINS = [(0, 5), (5, 40), (40, 80), (80, 800)]
BIN_LABELS = ['normal_low', 'normal_mid', 'high', 'extreme']

# tail 기준
TAIL_THRESHOLD = 50  # target >= 50 을 tail로 분류 (8% 데이터, 45% MAE)
EXTREME_THRESHOLD = 80


def load_data():
    """model31 피처 파이프라인으로 데이터 로드"""
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    # 원본 컬럼도 보존하기 위해 build_features 전 원본 저장
    train_raw_cols = [c for c in train.columns if c not in ['ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m']]
    train, test = build_features(train, test, layout, lag_lags=[1,2,3,4,5,6], rolling_windows=[3,5,10])
    return train, test, train_raw_cols


def identify_col_types(df):
    """수치형/범주형 분리"""
    exclude = {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
    numeric_cols = []
    cat_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object:
            cat_cols.append(c)
        elif df[c].nunique() <= 20 and df[c].dtype in ['int64', 'int32', 'float64']:
            # 범주형으로 취급 가능한 low-cardinality 수치형
            cat_cols.append(c)
        else:
            numeric_cols.append(c)
    return numeric_cols, cat_cols


# ─────────────────────────────────────────────
# 1. 수치 피처 KS-test (구간별 분포 차이)
# ─────────────────────────────────────────────
def ks_test_analysis(train, y, numeric_cols, report):
    """[0,5) vs [80,800) 의 수치 피처 분포 비교"""
    report.append('\n' + '=' * 70)
    report.append('§1-1. 수치 피처 KS-test: [0,5) vs [80,800)')
    report.append('=' * 70)

    mask_low = y < 5
    mask_ext = y >= EXTREME_THRESHOLD

    results = []
    for col in numeric_cols:
        vals_low = train.loc[mask_low, col].dropna()
        vals_ext = train.loc[mask_ext, col].dropna()
        if len(vals_low) < 10 or len(vals_ext) < 10:
            continue
        ks_stat, p_val = stats.ks_2samp(vals_low, vals_ext)
        # 효과 크기: 평균 차이 / pooled std
        pooled_std = np.sqrt((vals_low.var() + vals_ext.var()) / 2)
        if pooled_std > 1e-8:
            cohen_d = (vals_ext.mean() - vals_low.mean()) / pooled_std
        else:
            cohen_d = 0
        results.append({
            'feature': col,
            'ks_stat': ks_stat,
            'p_value': p_val,
            'mean_low': vals_low.mean(),
            'mean_ext': vals_ext.mean(),
            'cohen_d': cohen_d,
        })

    df_ks = pd.DataFrame(results).sort_values('ks_stat', ascending=False)

    report.append(f'\n총 {len(df_ks)}개 피처 검사, p < 0.001 & KS > 0.3 필터:')
    report.append(f'{"피처":<40s} {"KS":>6s} {"Cohen_d":>8s} {"mean_low":>10s} {"mean_ext":>10s}')
    report.append('-' * 80)

    sig_features = []
    for _, row in df_ks.iterrows():
        if row['ks_stat'] > 0.3 and row['p_value'] < 0.001:
            sig_features.append(row['feature'])
            report.append(f'{row["feature"]:<40s} {row["ks_stat"]:6.3f} {row["cohen_d"]:+8.3f} '
                         f'{row["mean_low"]:10.3f} {row["mean_ext"]:10.3f}')

    report.append(f'\n통계적으로 유의한 피처 수: {len(sig_features)}')
    report.append(f'상위 10: {sig_features[:10]}')

    return df_ks, sig_features


# ─────────────────────────────────────────────
# 2. 범주형 피처 Lift 분석
# ─────────────────────────────────────────────
def categorical_lift_analysis(train, y, cat_cols, report):
    """P(category | tail) / P(category | non-tail) → lift 계산"""
    report.append('\n' + '=' * 70)
    report.append('§1-2. 범주형 피처 Lift: P(cat|tail) / P(cat|non-tail)')
    report.append('=' * 70)

    is_tail = (y >= TAIL_THRESHOLD).astype(int)
    tail_rate = is_tail.mean()

    all_lifts = []
    for col in cat_cols:
        if train[col].nunique() > 50:
            continue
        for val in train[col].unique():
            mask_val = train[col] == val
            n_val = mask_val.sum()
            if n_val < 50:
                continue
            tail_rate_given_val = is_tail[mask_val].mean()
            lift = tail_rate_given_val / (tail_rate + 1e-8)
            all_lifts.append({
                'feature': col,
                'value': val,
                'n': n_val,
                'tail_rate': tail_rate_given_val,
                'lift': lift,
            })

    df_lift = pd.DataFrame(all_lifts).sort_values('lift', ascending=False)

    report.append(f'\ntail 기준: target >= {TAIL_THRESHOLD} (전체 tail 비율: {tail_rate:.3f})')
    report.append(f'\nLift >= 2.0 인 범주:')
    report.append(f'{"피처":<30s} {"값":>8s} {"n":>7s} {"tail_rate":>10s} {"lift":>6s}')
    report.append('-' * 70)

    high_lift = df_lift[df_lift['lift'] >= 2.0]
    for _, row in high_lift.head(30).iterrows():
        report.append(f'{row["feature"]:<30s} {str(row["value"]):>8s} {row["n"]:7d} '
                     f'{row["tail_rate"]:10.3f} {row["lift"]:6.2f}')

    report.append(f'\nLift >= 2 범주 수: {len(high_lift)}')
    report.append(f'Lift >= 3 범주 수: {len(df_lift[df_lift["lift"] >= 3.0])}')

    return df_lift


# ─────────────────────────────────────────────
# 3. Mutual Information ranking
# ─────────────────────────────────────────────
def mutual_info_analysis(train, y, numeric_cols, cat_cols, report):
    """각 피처와 "tail 여부(binary)" 사이 MI ranking"""
    report.append('\n' + '=' * 70)
    report.append('§1-3. Mutual Information: 피처 vs tail 여부')
    report.append('=' * 70)

    is_tail = (y >= TAIL_THRESHOLD).astype(int)
    all_cols = numeric_cols + cat_cols
    X = train[all_cols].fillna(0)

    # MI 계산 (discrete_features for categorical)
    discrete_mask = [False] * len(numeric_cols) + [True] * len(cat_cols)
    mi = mutual_info_classif(X, is_tail, discrete_features=discrete_mask, random_state=42, n_neighbors=5)

    df_mi = pd.DataFrame({'feature': all_cols, 'MI': mi}).sort_values('MI', ascending=False)

    report.append(f'\n상위 30 MI 피처:')
    report.append(f'{"순위":>4s} {"피처":<40s} {"MI":>8s} {"유형":>6s}')
    report.append('-' * 65)
    for i, (_, row) in enumerate(df_mi.head(30).iterrows()):
        ftype = 'CAT' if row['feature'] in cat_cols else 'NUM'
        report.append(f'{i+1:4d} {row["feature"]:<40s} {row["MI"]:8.4f} {ftype:>6s}')

    # 수치 vs 범주 MI 비교
    mi_num = df_mi[df_mi['feature'].isin(numeric_cols)]['MI']
    mi_cat = df_mi[df_mi['feature'].isin(cat_cols)]['MI']
    report.append(f'\n수치 피처 평균 MI: {mi_num.mean():.4f} (상위10 평균: {mi_num.nlargest(10).mean():.4f})')
    report.append(f'범주 피처 평균 MI: {mi_cat.mean():.4f} (상위10 평균: {mi_cat.nlargest(10).mean():.4f})')

    return df_mi


# ─────────────────────────────────────────────
# 4. SHAP on tail-only subset
# ─────────────────────────────────────────────
def shap_tail_analysis(train, y, numeric_cols, cat_cols, report):
    """LGBM 학습 후 [80,800) 구간 SHAP 기여도 추출"""
    report.append('\n' + '=' * 70)
    report.append('§1-4. SHAP 분석: [80,800) 구간 기여도 상위 피처')
    report.append('=' * 70)

    all_cols = [c for c in numeric_cols + cat_cols if c in train.columns]
    X = train[all_cols].fillna(0)
    y_log = np.log1p(y)

    # 빠른 LGBM 학습 (전체 데이터, 500 rounds)
    dtrain = lgb.Dataset(X, label=y_log)
    params = {
        'num_leaves': 127, 'learning_rate': 0.05,
        'feature_fraction': 0.7, 'bagging_fraction': 0.8,
        'objective': 'regression_l1', 'bagging_freq': 1,
        'verbosity': -1, 'n_jobs': -1, 'random_state': 42,
    }
    bst = lgb.train(params, dtrain, num_boost_round=500)

    # SHAP 값 (tree SHAP — LightGBM native)
    extreme_mask = y >= EXTREME_THRESHOLD
    X_ext = X[extreme_mask]

    print(f'  SHAP 계산 중 (n={extreme_mask.sum()})...')
    shap_vals = bst.predict(X_ext, pred_contrib=True)
    # pred_contrib: shape (n, n_features + 1), 마지막이 bias
    shap_vals = shap_vals[:, :-1]  # bias 제거

    # 평균 |SHAP| per feature (극값 구간)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    df_shap = pd.DataFrame({'feature': all_cols, 'mean_abs_shap': mean_abs_shap})
    df_shap = df_shap.sort_values('mean_abs_shap', ascending=False)

    # 전체 데이터 SHAP도 비교
    shap_all = bst.predict(X, pred_contrib=True)[:, :-1]
    mean_abs_shap_all = np.abs(shap_all).mean(axis=0)
    df_shap['mean_abs_shap_all'] = mean_abs_shap_all
    df_shap['tail_ratio'] = df_shap['mean_abs_shap'] / (df_shap['mean_abs_shap_all'] + 1e-8)

    report.append(f'\n[80,800) SHAP 상위 20 (tail 구간에서 특히 중요한 피처):')
    report.append(f'{"순위":>4s} {"피처":<40s} {"SHAP_tail":>10s} {"SHAP_all":>10s} {"ratio":>7s}')
    report.append('-' * 78)
    for i, (_, row) in enumerate(df_shap.head(20).iterrows()):
        report.append(f'{i+1:4d} {row["feature"]:<40s} {row["mean_abs_shap"]:10.4f} '
                     f'{row["mean_abs_shap_all"]:10.4f} {row["tail_ratio"]:7.2f}')

    # tail에서만 특히 중요한 피처 (ratio > 1.5)
    tail_specific = df_shap[df_shap['tail_ratio'] > 1.5].sort_values('tail_ratio', ascending=False)
    report.append(f'\ntail 특이 피처 (SHAP ratio > 1.5): {len(tail_specific)}개')
    for _, row in tail_specific.head(15).iterrows():
        report.append(f'  {row["feature"]:<40s} ratio={row["tail_ratio"]:.2f}')

    # SHAP 방향 분석 (극값에서 어느 방향?)
    report.append(f'\n[80,800) SHAP 방향 (양수=delay 증가, 음수=delay 감소):')
    mean_shap_signed = shap_vals.mean(axis=0)
    df_dir = pd.DataFrame({'feature': all_cols, 'mean_shap_signed': mean_shap_signed})
    df_dir = df_dir.sort_values('mean_shap_signed', ascending=False)
    report.append(f'  delay 증가 방향 상위 10:')
    for _, row in df_dir.head(10).iterrows():
        report.append(f'    {row["feature"]:<40s} {row["mean_shap_signed"]:+.4f}')
    report.append(f'  delay 감소 방향 상위 10:')
    for _, row in df_dir.tail(10).iterrows():
        report.append(f'    {row["feature"]:<40s} {row["mean_shap_signed"]:+.4f}')

    return df_shap


# ─────────────────────────────────────────────
# 5. 범주형 조합 Lift
# ─────────────────────────────────────────────
def combo_lift_analysis(train, y, cat_cols, report):
    """상위 MI 범주 2~3개의 교차 조합 lift 탐색"""
    report.append('\n' + '=' * 70)
    report.append('§1-5. 범주형 조합 Lift (2-way, 3-way)')
    report.append('=' * 70)

    is_tail = (y >= TAIL_THRESHOLD).astype(int)
    tail_rate = is_tail.mean()

    # low cardinality 범주만 (< 15 unique)
    low_card = [c for c in cat_cols if train[c].nunique() <= 15 and c in train.columns]
    report.append(f'\nlow-cardinality 범주 ({len(low_card)}개): {low_card[:10]}...')

    # 2-way 조합
    report.append(f'\n[2-way 조합] Lift >= 3.0:')
    report.append(f'{"조합":<50s} {"n":>6s} {"tail_rate":>10s} {"lift":>6s}')
    report.append('-' * 78)

    combo_results = []
    for c1, c2 in combinations(low_card[:15], 2):  # 상위 15개만
        combo = train[c1].astype(str) + '_' + train[c2].astype(str)
        for val in combo.unique():
            mask = combo == val
            n = mask.sum()
            if n < 30:
                continue
            tr = is_tail[mask].mean()
            lift = tr / (tail_rate + 1e-8)
            if lift >= 3.0:
                combo_results.append({
                    'combo': f'{c1}={val.split("_")[0]} × {c2}={val.split("_")[1]}',
                    'features': (c1, c2),
                    'n': n,
                    'tail_rate': tr,
                    'lift': lift,
                })

    combo_results.sort(key=lambda x: x['lift'], reverse=True)
    for r in combo_results[:20]:
        report.append(f'{r["combo"]:<50s} {r["n"]:6d} {r["tail_rate"]:10.3f} {r["lift"]:6.2f}')

    report.append(f'\nLift >= 3 조합 수: {len(combo_results)}')

    return combo_results


# ─────────────────────────────────────────────
# 6. 시나리오 레벨 분석
# ─────────────────────────────────────────────
def scenario_level_analysis(train, y, report):
    """시나리오별 극단값 빈도/패턴 분석"""
    report.append('\n' + '=' * 70)
    report.append('§1-6. 시나리오 레벨 극단값 패턴')
    report.append('=' * 70)

    train['_target'] = y
    train['_is_tail'] = (y >= TAIL_THRESHOLD).astype(int)
    train['_is_extreme'] = (y >= EXTREME_THRESHOLD).astype(int)

    sc = train.groupby('scenario_id').agg(
        n=('_target', 'size'),
        mean_target=('_target', 'mean'),
        max_target=('_target', 'max'),
        std_target=('_target', 'std'),
        tail_count=('_is_tail', 'sum'),
        extreme_count=('_is_extreme', 'sum'),
    ).reset_index()
    sc['tail_frac'] = sc['tail_count'] / sc['n']
    sc['extreme_frac'] = sc['extreme_count'] / sc['n']

    # 시나리오 분류
    sc['type'] = 'normal'
    sc.loc[sc['mean_target'] >= 40, 'type'] = 'extreme_scenario'
    sc.loc[(sc['mean_target'] >= 20) & (sc['mean_target'] < 40), 'type'] = 'high_scenario'

    type_counts = sc['type'].value_counts()
    report.append(f'\n시나리오 분류:')
    for t, c in type_counts.items():
        mean_t = sc[sc['type'] == t]['mean_target'].mean()
        report.append(f'  {t:20s}: {c:5d}개, 평균 target = {mean_t:.2f}')

    # 극단 시나리오 특성
    extreme_sc = sc[sc['type'] == 'extreme_scenario']
    if len(extreme_sc) > 0:
        report.append(f'\n극단 시나리오 ({len(extreme_sc)}개) 통계:')
        report.append(f'  mean_target: {extreme_sc["mean_target"].mean():.2f} ± {extreme_sc["mean_target"].std():.2f}')
        report.append(f'  max_target:  {extreme_sc["max_target"].mean():.2f} ± {extreme_sc["max_target"].std():.2f}')
        report.append(f'  tail_frac:   {extreme_sc["tail_frac"].mean():.3f}')

    # 극단 시나리오의 layout 분포
    extreme_sc_ids = extreme_sc['scenario_id'].values
    extreme_rows = train[train['scenario_id'].isin(extreme_sc_ids)]
    if 'layout_id' in train.columns:
        layout_dist = extreme_rows['layout_id'].value_counts(normalize=True).head(10)
        report.append(f'\n극단 시나리오 layout 분포 (상위 10):')
        for lid, frac in layout_dist.items():
            n_normal = len(train[train['layout_id'] == lid]) - len(extreme_rows[extreme_rows['layout_id'] == lid])
            report.append(f'  layout={lid}: {frac:.3f} (극단), total={len(train[train["layout_id"]==lid])}')

    # fold별 극단 시나리오 분포 확인 (GroupKFold)
    groups = train['scenario_id']
    gkf = GroupKFold(n_splits=5)
    report.append(f'\nGroupKFold 5-fold 극단 시나리오 분포:')
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups)):
        va_sc = train.iloc[va_idx]['scenario_id'].unique()
        n_ext = len(set(va_sc) & set(extreme_sc_ids))
        report.append(f'  Fold {fold+1}: val 시나리오={len(va_sc)}, 극단={n_ext} ({n_ext/len(va_sc)*100:.1f}%)')

    train.drop(columns=['_target', '_is_tail', '_is_extreme'], inplace=True)

    return sc


# ─────────────────────────────────────────────
# 7. 큐잉이론 파생 피처 탐색
# ─────────────────────────────────────────────
def physics_feature_exploration(train, y, report):
    """도메인 공식 후보 탐색: Little's Law, M/M/1 등"""
    report.append('\n' + '=' * 70)
    report.append('§1-7. 큐잉이론 Physics-informed Feature 후보')
    report.append('=' * 70)

    candidates = {}

    # Little's Law: L = λW → W = L/λ (대기시간 = 대기열/도착률)
    if 'charge_queue_length' in train.columns and 'order_inflow_15m' in train.columns:
        lam = train['order_inflow_15m'].clip(lower=1)
        candidates['littles_law_W'] = train['charge_queue_length'] / lam

    # M/M/1 대기시간: W_q = ρ / (μ(1-ρ))  where ρ = λ/μ ≈ utilization
    if 'robot_utilization' in train.columns:
        rho = train['robot_utilization'].clip(0.01, 0.99)
        candidates['mm1_wait'] = rho / (1 - rho)

    # 혼잡도 × 배터리 위기: 복합 스트레스
    if 'congestion_score' in train.columns and 'low_battery_ratio' in train.columns:
        candidates['stress_product'] = train['congestion_score'] * train['low_battery_ratio']

    # 유효 로봇 수: total × (1 - low_battery) × (1 - idle_frac)
    if all(c in train.columns for c in ['robot_total', 'low_battery_ratio', 'robot_idle']):
        effective = train['robot_total'] * (1 - train['low_battery_ratio']) * \
                    (1 - train['robot_idle'] / (train['robot_total'] + 1e-8))
        candidates['effective_robots'] = effective.clip(lower=0.1)
        # demand / effective capacity
        if 'order_inflow_15m' in train.columns:
            candidates['demand_capacity_ratio'] = train['order_inflow_15m'] / candidates['effective_robots']

    # 병목 지표: max(congestion, queue, battery_stress) 의 기하평균
    if all(c in train.columns for c in ['congestion_score', 'charge_queue_length', 'low_battery_ratio']):
        a = train['congestion_score'].clip(lower=0.1)
        b = train['charge_queue_length'].clip(lower=0.1)
        c_val = (train['low_battery_ratio'] * 100).clip(lower=0.1)
        candidates['bottleneck_geometric'] = (a * b * c_val) ** (1/3)

    # 포화도: ρ/(1-ρ) × congestion
    if 'robot_utilization' in train.columns and 'congestion_score' in train.columns:
        rho = train['robot_utilization'].clip(0.01, 0.99)
        candidates['saturated_congestion'] = (rho / (1 - rho)) * train['congestion_score']

    # 각 후보의 tail 구간 상관관계 분석
    report.append(f'\n후보 피처 {len(candidates)}종:')
    report.append(f'{"피처":<35s} {"전체 corr":>10s} {"tail corr":>10s} {"[80+] corr":>10s} {"mean_low":>10s} {"mean_ext":>10s} {"sep_σ":>7s}')
    report.append('-' * 100)

    tail_mask = y >= TAIL_THRESHOLD
    ext_mask = y >= EXTREME_THRESHOLD
    low_mask = y < 5

    for name, vals in candidates.items():
        vals_clean = vals.replace([np.inf, -np.inf], np.nan).fillna(0)
        corr_all = np.corrcoef(vals_clean, y)[0, 1]
        corr_tail = np.corrcoef(vals_clean[tail_mask], y[tail_mask])[0, 1] if tail_mask.sum() > 10 else 0
        corr_ext = np.corrcoef(vals_clean[ext_mask], y[ext_mask])[0, 1] if ext_mask.sum() > 10 else 0
        mean_low = vals_clean[low_mask].mean()
        mean_ext = vals_clean[ext_mask].mean()
        pooled_std = np.sqrt((vals_clean[low_mask].var() + vals_clean[ext_mask].var()) / 2)
        sep = (mean_ext - mean_low) / (pooled_std + 1e-8)
        report.append(f'{name:<35s} {corr_all:10.4f} {corr_tail:10.4f} {corr_ext:10.4f} '
                     f'{mean_low:10.3f} {mean_ext:10.3f} {sep:+7.3f}')

    return candidates


# ─────────────────────────────────────────────
# 8. 종합 판정
# ─────────────────────────────────────────────
def final_verdict(df_ks, df_lift, df_mi, df_shap, combo_results, report):
    """수치 vs 범주 기여도 종합 판정"""
    report.append('\n' + '=' * 70)
    report.append('§1-FINAL. 종합 판정: 수치 vs 범주 기여도')
    report.append('=' * 70)

    # KS-test 기반
    n_sig_ks = len(df_ks[df_ks['ks_stat'] > 0.3])
    report.append(f'\n[KS-test] 유의한 수치 피처: {n_sig_ks}개 (KS > 0.3)')

    # Lift 기반
    n_high_lift = len(df_lift[df_lift['lift'] >= 3.0])
    report.append(f'[Lift] lift >= 3 범주: {n_high_lift}개')

    # MI 기반 상위 10 중 수치/범주 비율
    top10_mi = df_mi.head(10)
    # (numeric_cols is not passed here, use heuristic)
    report.append(f'[MI] 상위 10 피처: {list(top10_mi["feature"])}')

    # SHAP 기반
    top10_shap = df_shap.head(10)
    report.append(f'[SHAP] tail 구간 상위 10: {list(top10_shap["feature"])}')

    # 조합 lift
    report.append(f'[Combo] lift >= 3 조합: {len(combo_results)}개')

    report.append(f'\n판정:')
    if n_sig_ks >= 10 and n_high_lift < 5:
        report.append('  → 수치 피처의 연속적 극값이 tail의 주요 구동 인자')
        report.append('  → SMOGN/연속 오버샘플링 + 비대칭 loss가 적절한 전략')
    elif n_sig_ks < 5 and n_high_lift >= 10:
        report.append('  → 범주형 조합(rare regime)이 tail의 주요 구동 인자')
        report.append('  → regime별 모델 분리 / regime 임베딩 / rare-combo 오버샘플링이 적절')
    else:
        report.append('  → 수치와 범주 모두 기여 (conditional augmentation 적합)')
        report.append(f'  → 수치 기여 강도: KS>0.3 {n_sig_ks}개')
        report.append(f'  → 범주 기여 강도: lift>=3 {n_high_lift}개, combo>=3 {len(combo_results)}개')

    report.append(f'\n→ 다음 단계: §3 Loss ablation + §2 Physics-informed FE 실험 설계')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    report = []
    report.append('=' * 70)
    report.append('§1 Tail Driver Decomposition EDA Report')
    report.append(f'생성: {time.strftime("%Y-%m-%d %H:%M")}')
    report.append('기준: model33 Public 9.8223 / 1위 9.69923 / 갭 0.123')
    report.append('=' * 70)

    print('[데이터 로드]')
    train, test, raw_cols = load_data()
    y = train['avg_delay_minutes_next_30m']

    # 타겟 기본 통계
    report.append(f'\n[타겟 기본 통계]')
    report.append(f'  mean={y.mean():.2f}, median={y.median():.2f}, std={y.std():.2f}')
    report.append(f'  skew={y.skew():.2f}, kurtosis={y.kurtosis():.2f}')
    report.append(f'  min={y.min():.2f}, max={y.max():.2f}')
    for lo, hi in BINS:
        mask = (y >= lo) & (y < hi)
        report.append(f'  [{lo},{hi}): n={mask.sum()} ({mask.sum()/len(y)*100:.1f}%)')

    numeric_cols, cat_cols = identify_col_types(train)
    report.append(f'\n  수치 피처: {len(numeric_cols)}개, 범주 피처: {len(cat_cols)}개')

    # §1-1. KS-test
    print('\n[§1-1] KS-test 분석...')
    df_ks, sig_features = ks_test_analysis(train, y, numeric_cols, report)

    # §1-2. Lift
    print('[§1-2] 범주형 Lift 분석...')
    df_lift = categorical_lift_analysis(train, y, cat_cols, report)

    # §1-3. MI
    print('[§1-3] Mutual Information 계산...')
    df_mi = mutual_info_analysis(train, y, numeric_cols, cat_cols, report)

    # §1-4. SHAP
    print('[§1-4] SHAP 분석...')
    df_shap = shap_tail_analysis(train, y, numeric_cols, cat_cols, report)

    # §1-5. Combo Lift
    print('[§1-5] 범주형 조합 Lift...')
    combo_results = combo_lift_analysis(train, y, cat_cols, report)

    # §1-6. 시나리오 레벨
    print('[§1-6] 시나리오 레벨 분석...')
    sc = scenario_level_analysis(train, y, report)

    # §1-7. Physics-informed
    print('[§1-7] 큐잉이론 피처 탐색...')
    candidates = physics_feature_exploration(train, y, report)

    # §1-FINAL. 종합 판정
    print('[종합 판정]')
    final_verdict(df_ks, df_lift, df_mi, df_shap, combo_results, report)

    # 보고서 저장
    elapsed = (time.time() - t0) / 60
    report.append(f'\n\n총 소요: {elapsed:.1f}분')

    report_path = os.path.join(DOCS_DIR, 'eda_tail_driver_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f'\n보고서 저장: {report_path}')
    print(f'총 소요: {elapsed:.1f}분')

    # 콘솔에도 핵심 출력
    print('\n' + '=' * 70)
    print('핵심 결과 요약')
    print('=' * 70)
    for line in report[-20:]:
        print(line)


if __name__ == '__main__':
    main()
