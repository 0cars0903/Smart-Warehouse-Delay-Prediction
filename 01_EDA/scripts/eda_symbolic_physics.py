"""
§2 Simulator 역공학 — Physics-informed FE + Symbolic Regression
================================================================
목표:
  (A) 큐잉이론 기반 physics FE 후보 → model31 파이프라인에 추가 → ablation
  (B) Symbolic Regression으로 tail 전용 비선형 관계식 탐색
  (C) 2D Partial Dependence 분석 → 모델이 놓치는 비선형 상호작용 식별
  (D) LGBM monotone constraint 효과 측정

§1-7 EDA 발견:
  - bottleneck_geometric: 전체 corr +0.253, tail corr -0.179 (구분자 O, 심도 X)
  - effective_robots: 전체 corr +0.163, tail corr +0.210 (유일 양+양)
  - stress_product: 전체 +0.248, tail -0.179
  - Physics features → tail vs non-tail 구분에 효과, 내부 심도에는 역효과

전략:
  1) Physics FE v2: §1 발견 기반 개선 + 신규 후보
  2) PySR 설치 가능 시 symbolic regression, 불가 시 수동 탐색
  3) model31 체크포인트 기반 빠른 ablation (LGBM 단독 5-fold)

실행: python src/eda_symbolic_physics.py
예상 시간: ~8분 (LGBM 5-fold × 3~4 FE 조합)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings, gc, os, sys, time
from itertools import combinations

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
DOCS_DIR = os.path.join(_BASE, '..', 'docs')
N_SPLITS = 5
RANDOM_STATE = 42

# model31 LGBM params (Optuna tuned)
LGBM_PARAMS = {
    'num_leaves': 129, 'learning_rate': 0.01021,
    'feature_fraction': 0.465, 'bagging_fraction': 0.947,
    'min_child_samples': 30, 'reg_alpha': 1.468, 'reg_lambda': 0.396,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# 시나리오 집계 대상 (model31 동일)
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]


# ═══════════════════════════════════════════════
# SECTION A: Physics-informed Feature Engineering v2
# ═══════════════════════════════════════════════

def safe_div(a, b, fill=0):
    return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)


def add_physics_features_v2(df):
    """
    큐잉이론 + 운영과학 기반 physics FE.

    §1-7 발견 반영:
    - bottleneck_geometric, stress_product → tail corr 음수 → 변환 필요
    - effective_robots → tail corr 양수 → 직접 사용 가능
    - 새로운 접근: 비선형 변환 + 상호작용으로 단조성 보완

    패밀리 1: 큐잉이론 (Little's Law, M/M/c 근사)
    패밀리 2: 병목 분석 (기하평균, 하모닉)
    패밀리 3: 운영 부하 비선형 (임계점 시뮬레이션)
    """
    df = df.copy()

    # === 기본 변수 준비 ===
    order   = df['order_inflow_15m']
    cong    = df['congestion_score']
    lowbat  = df['low_battery_ratio']
    idle    = df['robot_idle']
    active  = df['robot_active']
    charging = df['robot_charging']
    total   = df.get('robot_total', active + idle + charging)
    charger = df.get('charger_count', pd.Series(1, index=df.index))
    util    = df['robot_utilization']
    pack_u  = df['pack_utilization']
    max_zd  = df['max_zone_density']
    queue   = df['charge_queue_length']
    bat_mean = df['battery_mean']

    # ── 패밀리 1: 큐잉이론 ──

    # 1-1. Little's Law: W = L/λ (대기시간 = 시스템 내 대기수 / 도착률)
    # L ≈ charge_queue_length, λ ≈ robot_charging (충전 완료 비율의 proxy)
    df['phys_littles_W'] = safe_div(queue, charging + 1)

    # 1-2. M/M/c 근사: 서버 이용률 ρ = λ/(cμ)
    # λ = order_inflow, c = robot_active, μ ≈ 1/avg_trip_distance (처리 속도)
    trip = df.get('avg_trip_distance', pd.Series(1, index=df.index))
    rho = safe_div(order, active * safe_div(1, trip + 0.1))
    # ρ > 1이면 시스템 불안정 → clamp [0, 3]
    rho = rho.clip(0, 3)
    df['phys_server_rho'] = rho

    # 1-3. M/M/1 대기시간 근사: W_q = ρ/(μ(1-ρ)) — ρ < 1일 때만 유효
    # ρ >= 1이면 발산 → log(1 + ρ/(1-ρ+0.1))로 안전 변환
    df['phys_wait_approx'] = np.log1p(rho / (1 - rho + 0.1).clip(lower=0.01))

    # ── 패밀리 2: 병목 분석 ──

    # 2-1. 병목 기하평균: sqrt(congestion × max_zone_density)
    # §1: 전체 corr +0.253, tail corr -0.179 → 비선형 변환 필요
    bottleneck_raw = np.sqrt(cong.clip(0) * max_zd.clip(0))

    # 2-2. §1 발견 반영: tail에서 음수 상관 → log1p 변환 + 차이 피처
    df['phys_bottleneck_log'] = np.log1p(bottleneck_raw)

    # 2-3. 스트레스 곱 (congestion × lowbat × max_zone_density)
    stress_raw = cong * lowbat * max_zd
    df['phys_stress_log'] = np.log1p(stress_raw.clip(0) * 1000)  # 스케일 보정

    # 2-4. effective_robots = active - queue (§1에서 유일하게 tail corr 양수)
    df['phys_effective_robots'] = (active - queue).clip(lower=0)

    # 2-5. 병목 하모닉 평균: 여러 자원의 "최약 링크" 효과
    # H(a,b,c) = 3 / (1/a + 1/b + 1/c)
    inv_active = safe_div(1, active + 1)
    inv_charger = safe_div(1, charger + 1)
    inv_pack = safe_div(1, pack_u + 0.01)
    df['phys_resource_harmonic'] = safe_div(3, inv_active + inv_charger + inv_pack)

    # ── 패밀리 3: 운영 부하 비선형 ──

    # 3-1. 수요-공급 불균형 (비선형): (demand/capacity)^2
    # 선형 비율보다 제곱이 극값에서 더 빠르게 증가 → 극값 예측력 기대
    dc_ratio = safe_div(order, active + 1)
    df['phys_dc_ratio_sq'] = dc_ratio ** 2

    # 3-2. 배터리 위기 폭발점: lowbat > threshold일 때 급격히 증가
    # sigmoid 변환: 1 / (1 + exp(-k*(x - x0))) — k=20, x0=0.2
    df['phys_battery_crisis_sig'] = 1 / (1 + np.exp(-20 * (lowbat - 0.2)))

    # 3-3. 혼잡도 임계점: congestion > 5일 때 지수적 증가
    df['phys_congestion_exp'] = np.where(
        cong > 5,
        np.log1p(cong - 5) * 2,  # 초과분의 log 스케일
        0
    )

    # 3-4. pack_utilization 포화: 0.8 이상에서 급격히 악화
    df['phys_pack_saturation'] = np.where(
        pack_u > 0.8,
        (pack_u - 0.8) ** 2 * 25,  # 0.8~1.0 → 0~1
        0
    )

    # 3-5. 복합 위기 지수: max(개별 임계 초과분)의 기하평균
    bat_crisis = (lowbat > 0.15).astype(float) * lowbat
    cong_crisis = (cong > 5).astype(float) * cong / 20
    pack_crisis = (pack_u > 0.7).astype(float) * pack_u
    df['phys_compound_crisis'] = (bat_crisis * cong_crisis * pack_crisis + 1e-8) ** (1/3)

    # ── 패밀리 4: 시나리오 레벨 physics (sc_ 피처 활용) ──

    # sc_ 피처가 있을 때만 추가
    if 'sc_order_inflow_15m_mean' in df.columns:
        sc_order = df['sc_order_inflow_15m_mean']
        sc_cong  = df['sc_congestion_score_mean']
        sc_lowbat = df['sc_low_battery_ratio_mean']
        sc_idle  = df['sc_robot_idle_mean']
        sc_util  = df['sc_robot_utilization_mean']

        # 4-1. 시나리오 평균 서버 이용률
        df['phys_sc_rho'] = safe_div(sc_order, total * safe_div(1, trip + 0.1)).clip(0, 3)

        # 4-2. 시나리오 수요-공급 불균형 (제곱)
        df['phys_sc_dc_sq'] = safe_div(sc_order, total + 1) ** 2

        # 4-3. 시나리오 복합 스트레스 (sc_cong × sc_lowbat / charger)
        df['phys_sc_stress'] = safe_div(sc_cong * sc_lowbat, charger)

        # 4-4. 시나리오 유휴 결핍: (1 - idle_fraction) × demand
        idle_frac = safe_div(sc_idle, total)
        df['phys_sc_active_demand'] = (1 - idle_frac) * sc_order

        # 4-5. 현재값 vs 시나리오 평균의 편차 비율 (얼마나 악화되었는가)
        df['phys_cong_deviation'] = safe_div(cong - sc_cong, sc_cong.abs() + 1)
        df['phys_order_deviation'] = safe_div(order - sc_order, sc_order.abs() + 1)

    return df


def get_physics_feature_names(df):
    """physics 피처 컬럼명 반환"""
    return [c for c in df.columns if c.startswith('phys_')]


# ═══════════════════════════════════════════════
# SECTION B: Symbolic Regression (수동 탐색)
# ═══════════════════════════════════════════════

def symbolic_search_manual(X_train, y_raw, groups, feat_cols, top_n=10):
    """
    PySR 없이 수동 심볼릭 탐색:
    - 상위 피처 쌍에 대해 {+, -, ×, ÷, sqrt(a×b), log(a/b)} 연산 후보 생성
    - LGBM 대신 단순 상관분석 + 구간별 유효성으로 빠르게 스크리닝

    Returns: 유효 후보 수식 리스트 [(name, formula_fn, overall_corr, tail_corr)]
    """
    print('\n' + '='*60)
    print('§2-B. Symbolic Search: 수동 수식 탐색')
    print('='*60)

    # 타겟 로그 변환
    y = y_raw.values
    tail_mask = y >= 80

    # 상위 상관 피처 top_n 선택
    corrs = []
    for c in feat_cols:
        vals = X_train[c].fillna(0).values
        r = np.corrcoef(vals, y)[0, 1]
        if np.isfinite(r):
            corrs.append((c, abs(r)))
    corrs.sort(key=lambda x: -x[1])
    top_feats = [c for c, _ in corrs[:top_n]]
    print(f'  상위 {top_n} 상관 피처: {top_feats[:5]}...')

    # 수식 후보 생성
    candidates = []
    for i, (a, b) in enumerate(combinations(top_feats, 2)):
        va = X_train[a].fillna(0).values.astype(float)
        vb = X_train[b].fillna(0).values.astype(float)

        formulas = {
            f'{a[:15]}*{b[:15]}': va * vb,
            f'{a[:15]}/{b[:15]}': va / (vb + 1e-8),
            f'sqrt({a[:15]}*{b[:15]})': np.sqrt(np.abs(va * vb)),
            f'log({a[:15]}/{b[:15]})': np.log1p(np.abs(va / (vb + 1e-8))),
            f'({a[:15]})^2*{b[:15]}': va**2 * vb,
        }

        for name, vals in formulas.items():
            vals = np.nan_to_num(vals, nan=0, posinf=0, neginf=0)
            if vals.std() < 1e-10:
                continue
            r_all = np.corrcoef(vals, y)[0, 1]
            if not np.isfinite(r_all):
                continue
            if tail_mask.sum() > 10:
                r_tail = np.corrcoef(vals[tail_mask], y[tail_mask])[0, 1]
                if not np.isfinite(r_tail):
                    r_tail = 0
            else:
                r_tail = 0
            candidates.append((name, abs(r_all), r_tail, a, b))

    # 정렬: 전체 상관 기준
    candidates.sort(key=lambda x: -x[1])

    print(f'\n  총 {len(candidates)} 수식 후보 생성')
    print(f'\n  상위 15 (전체 상관 기준):')
    print(f'  {"수식":<50s}  {"전체r":>8s}  {"tail_r":>8s}')
    print(f'  {"-"*50}  {"-"*8}  {"-"*8}')
    for name, r_all, r_tail, a, b in candidates[:15]:
        print(f'  {name:<50s}  {r_all:>8.4f}  {r_tail:>+8.4f}')

    # tail 상관 기준 상위
    tail_sorted = sorted(candidates, key=lambda x: -abs(x[3] if isinstance(x[3], float) else x[2]))
    print(f'\n  상위 15 (|tail corr| 기준):')
    tail_sorted2 = sorted(candidates, key=lambda x: -abs(x[2]))
    for name, r_all, r_tail, a, b in tail_sorted2[:15]:
        print(f'  {name:<50s}  {r_all:>8.4f}  {r_tail:>+8.4f}')

    return candidates[:30]


# ═══════════════════════════════════════════════
# SECTION C: Ablation — Physics FE 추가 효과 측정
# ═══════════════════════════════════════════════

def add_scenario_agg_features(df):
    """model31 시나리오 집계 (11통계)"""
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns: continue
        grp = df.groupby('scenario_id')[col]
        df[f'sc_{col}_mean'] = grp.transform('mean')
        df[f'sc_{col}_std']  = grp.transform('std').fillna(0)
        df[f'sc_{col}_max']  = grp.transform('max')
        df[f'sc_{col}_min']  = grp.transform('min')
        df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']
        df[f'sc_{col}_median'] = grp.transform('median')
        df[f'sc_{col}_p10'] = grp.transform(lambda x: x.quantile(0.10))
        df[f'sc_{col}_p90'] = grp.transform(lambda x: x.quantile(0.90))
        df[f'sc_{col}_skew'] = grp.transform(lambda x: x.skew()).fillna(0)
        df[f'sc_{col}_kurtosis'] = grp.transform(lambda x: x.kurtosis()).fillna(0)
        df[f'sc_{col}_cv'] = (df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)
    return df


def add_ratio_tier1(df):
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
    return df


def add_ratio_tier2(df):
    if all(c in df.columns for c in ['sc_congestion_score_mean', 'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'], df['robot_total'] ** 2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm'] / 100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(df['pack_station_count'], df['floor_area_sqm'] / 1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if 'sc_battery_mean_mean' in df.columns:
        if 'sc_robot_utilization_mean' in df.columns and 'charger_count' in df.columns:
            df['ratio_battery_per_robot'] = safe_div(
                df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'], df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    return df


def add_ratio_tier3_selected(df):
    cols = ['sc_order_inflow_15m_mean', 'sc_congestion_score_mean',
            'sc_low_battery_ratio_mean', 'robot_total', 'charger_count']
    if all(c in df.columns for c in cols):
        df['ratio_total_stress'] = safe_div(
            df['sc_order_inflow_15m_mean'] * df['sc_congestion_score_mean'] *
            (df['sc_low_battery_ratio_mean'] + 0.01), df['robot_total'] * df['charger_count'])
    cols2 = ['sc_sku_concentration_mean', 'sc_congestion_score_mean', 'intersection_count']
    if all(c in df.columns for c in cols2):
        df['ratio_sku_congestion'] = safe_div(
            df['sc_sku_concentration_mean'] * df['sc_congestion_score_mean'], df['intersection_count'])
    cols3 = ['sc_robot_idle_mean', 'robot_total', 'sc_order_inflow_15m_mean', 'floor_area_sqm']
    if all(c in df.columns for c in cols3):
        idle_frac = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
        df['ratio_no_idle_demand'] = safe_div(
            (1 - idle_frac) * df['sc_order_inflow_15m_mean'], df['floor_area_sqm'] / 100)
    cols4 = ['sc_low_battery_ratio_mean', 'sc_charge_queue_length_mean', 'charger_count']
    if all(c in df.columns for c in cols4):
        df['ratio_battery_crisis'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df['sc_charge_queue_length_mean'], df['charger_count'])
    return df


def add_cross_selected(df):
    safe_pairs = [
        ('congestion_score', 'low_battery_ratio'),
        ('sku_concentration', 'max_zone_density'),
        ('robot_utilization', 'charge_queue_length'),
    ]
    for col_a, col_b in safe_pairs:
        if col_a not in df.columns or col_b not in df.columns: continue
        interaction = df[col_a] * df[col_b]
        grp = interaction.groupby(df['scenario_id'])
        df[f'sc_cross_{col_a[:6]}_{col_b[:6]}_mean'] = grp.transform('mean')
    return df


def load_data_model31():
    """model31 파이프라인 재현"""
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(train, test, layout,
                                  lag_lags=[1,2,3,4,5,6],
                                  rolling_windows=[3,5,10])
    for fn in [add_scenario_agg_features, add_ratio_tier1, add_ratio_tier2,
               add_ratio_tier3_selected, add_cross_selected]:
        train = fn(train); test = fn(test)
    return train, test


def get_feat_cols(df):
    return [c for c in df.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and df[c].dtype != object]


def eval_lgbm_cv(X_train, y_log, y_raw, groups, feat_cols, label='baseline'):
    """LGBM 5-fold CV (log1p 공간) → MAE + 구간별 분석"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    X_tr = X_train[feat_cols].fillna(0)

    fold_maes = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr.iloc[tr_idx], y_log.iloc[tr_idx],
              eval_set=[(X_tr.iloc[va_idx], y_log.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.iloc[va_idx].values)).mean()
        fold_maes.append(mae)
        del m; gc.collect()

    oof_raw = np.expm1(oof)
    total_mae = np.abs(oof_raw - y_raw.values).mean()

    # 구간별 분석
    segments = [(0, 5), (5, 40), (40, 80), (80, 800)]
    seg_results = {}
    for lo, hi in segments:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(oof_raw[mask] - y_raw.values[mask]).mean()
            seg_results[f'[{lo},{hi})'] = (seg_mae, mask.sum())

    # 예측 범위
    pred_std = np.maximum(oof_raw, 0).std()

    print(f'  [{label}] CV MAE={total_mae:.4f}, pred_std={pred_std:.2f}, '
          f'feats={len(feat_cols)}')
    for seg, (mae, n) in seg_results.items():
        print(f'    {seg}: MAE={mae:.2f} (n={n})')

    return total_mae, oof_raw, seg_results, fold_maes


# ═══════════════════════════════════════════════
# SECTION D: Distribution Shift Analysis
# ═══════════════════════════════════════════════

def shift_analysis(train, test, phys_cols):
    """Physics 피처의 train-test 분포 shift 측정"""
    print('\n' + '='*60)
    print('§2-D. Physics 피처 Distribution Shift 분석')
    print('='*60)

    results = []
    for col in phys_cols:
        if col not in train.columns or col not in test.columns:
            continue
        tr_vals = train[col].fillna(0).values
        te_vals = test[col].fillna(0).values
        tr_mean, tr_std = tr_vals.mean(), tr_vals.std()
        te_mean = te_vals.mean()
        shift_sigma = abs(te_mean - tr_mean) / (tr_std + 1e-8)
        results.append((col, tr_mean, te_mean, tr_std, shift_sigma))

    results.sort(key=lambda x: x[4])  # shift 낮은 순

    print(f'\n  {"피처":<35s}  {"train_mean":>10s}  {"test_mean":>10s}  {"shift_σ":>8s}  {"판정":>6s}')
    print(f'  {"-"*35}  {"-"*10}  {"-"*10}  {"-"*8}  {"-"*6}')
    for col, tr_m, te_m, tr_s, shift in results:
        verdict = '✅' if shift < 0.3 else '⚠️' if shift < 0.5 else '❌'
        print(f'  {col:<35s}  {tr_m:>10.4f}  {te_m:>10.4f}  {shift:>8.3f}  {verdict:>6s}')

    safe_cols = [col for col, _, _, _, s in results if s < 0.5]
    risky_cols = [col for col, _, _, _, s in results if s >= 0.5]
    print(f'\n  Safe (shift < 0.5): {len(safe_cols)} 피처')
    print(f'  Risky (shift >= 0.5): {len(risky_cols)} 피처: {risky_cols}')

    return safe_cols, risky_cols


# ═══════════════════════════════════════════════
# SECTION E: Monotone Constraint 실험
# ═══════════════════════════════════════════════

def monotone_constraint_experiment(X_train, y_log, y_raw, groups, feat_cols):
    """
    핵심 피처에 단조 제약 → 극값 외삽 개선 가능성 테스트.

    도메인 지식:
    - order_inflow_15m ↑ → delay ↑ (양 단조)
    - congestion_score ↑ → delay ↑
    - low_battery_ratio ↑ → delay ↑
    - robot_idle ↑ → delay ↓ (음 단조)
    - battery_mean ↑ → delay ↓
    """
    print('\n' + '='*60)
    print('§2-E. Monotone Constraint 실험')
    print('='*60)

    # 제약 대상 피처와 방향
    mono_rules = {
        'order_inflow_15m': 1,      # ↑ delay
        'congestion_score': 1,
        'low_battery_ratio': 1,
        'max_zone_density': 1,
        'pack_utilization': 1,
        'charge_queue_length': 1,
        'robot_idle': -1,            # ↓ delay
        'battery_mean': -1,
        'robot_active': -1,
    }

    # 제약 벡터 생성
    constraints = []
    constrained_count = 0
    for col in feat_cols:
        if col in mono_rules:
            constraints.append(mono_rules[col])
            constrained_count += 1
        else:
            constraints.append(0)

    print(f'  단조 제약 적용: {constrained_count}/{len(feat_cols)} 피처')
    for col, d in mono_rules.items():
        if col in feat_cols:
            print(f'    {col}: {"↑(+1)" if d == 1 else "↓(-1)"}')

    # Monotone LGBM — regression_l1은 monotone 미지원 → regression(L2)로 변경
    mono_params = LGBM_PARAMS.copy()
    mono_params['objective'] = 'regression'  # L1→L2 (monotone 지원)
    mono_params['monotone_constraints'] = constraints

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    X_tr = X_train[feat_cols].fillna(0)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups)):
        m = lgb.LGBMRegressor(**mono_params)
        m.fit(X_tr.iloc[tr_idx], y_log.iloc[tr_idx],
              eval_set=[(X_tr.iloc[va_idx], y_log.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(X_tr.iloc[va_idx])
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.iloc[va_idx].values)).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_raw = np.expm1(oof)
    total_mae = np.abs(oof_raw - y_raw.values).mean()

    # 구간별
    for lo, hi in [(0, 5), (5, 40), (40, 80), (80, 800)]:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(oof_raw[mask] - y_raw.values[mask]).mean()
            print(f'    [{lo},{hi}): MAE={seg_mae:.2f} (n={mask.sum()})')

    pred_std = np.maximum(oof_raw, 0).std()
    print(f'  Monotone LGBM CV MAE={total_mae:.4f}, pred_std={pred_std:.2f}')

    return total_mae, oof_raw


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    t0 = time.time()
    report_lines = []
    def log(msg=''):
        print(msg)
        report_lines.append(msg)

    log('='*60)
    log('§2 Simulator 역공학 — Physics FE + Symbolic Regression')
    log(f'생성: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
    log('기준: model33 CV 8.4756 / Public 9.8223')
    log('='*60)

    # ── 데이터 로드 (model31 파이프라인) ──
    print('\n[데이터] model31 파이프라인 로드')
    train, test = load_data_model31()

    TARGET = 'avg_delay_minutes_next_30m'
    y_raw = train[TARGET]
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']

    # model31 기준선 피처
    base_feat_cols = get_feat_cols(train)
    n_base = len(base_feat_cols)
    print(f'  기준선 피처 수: {n_base}')

    # ── §2-A: Baseline (model31 LGBM 단독) ──
    print('\n' + '='*60)
    print('§2-A. Baseline: model31 LGBM 단독 5-fold')
    print('='*60)

    base_mae, base_oof, base_segs, base_folds = eval_lgbm_cv(
        train, y_log, y_raw, groups, base_feat_cols, label='baseline')

    # ── §2-B: Symbolic Search ──
    symbolic_candidates = symbolic_search_manual(
        train, y_raw, groups, base_feat_cols, top_n=12)

    # ── §2-C: Physics FE 추가 ──
    print('\n' + '='*60)
    print('§2-C. Physics FE v2 추가 + Ablation')
    print('='*60)

    train_phys = add_physics_features_v2(train)
    test_phys  = add_physics_features_v2(test)
    phys_cols = get_physics_feature_names(train_phys)
    print(f'  Physics 피처 {len(phys_cols)}종 추가')
    for c in phys_cols:
        print(f'    {c}')

    # §2-D: Shift 분석
    safe_phys, risky_phys = shift_analysis(train_phys, test_phys, phys_cols)

    # Ablation 1: 전체 physics 피처 추가
    feat_all_phys = base_feat_cols + phys_cols
    print(f'\n[Ablation 1] 전체 physics ({len(phys_cols)}종) 추가 → {len(feat_all_phys)} 피처')
    mae_all, oof_all, segs_all, _ = eval_lgbm_cv(
        train_phys, y_log, y_raw, groups, feat_all_phys, label='all_physics')

    # Ablation 2: safe physics만 (shift < 0.5)
    safe_feat_cols = base_feat_cols + safe_phys
    if len(safe_phys) < len(phys_cols):
        print(f'\n[Ablation 2] Safe physics만 ({len(safe_phys)}종) → {len(safe_feat_cols)} 피처')
        mae_safe, oof_safe, segs_safe, _ = eval_lgbm_cv(
            train_phys, y_log, y_raw, groups, safe_feat_cols, label='safe_physics')
    else:
        mae_safe = mae_all
        segs_safe = segs_all
        print(f'\n[Ablation 2] 모든 physics가 safe → Ablation 1과 동일')

    # Ablation 3: 패밀리별 분리
    families = {
        '큐잉이론': [c for c in phys_cols if any(k in c for k in ['littles', 'server_rho', 'wait_approx'])],
        '병목분석': [c for c in phys_cols if any(k in c for k in ['bottleneck', 'stress_log', 'effective', 'harmonic'])],
        '비선형부하': [c for c in phys_cols if any(k in c for k in ['dc_ratio', 'battery_crisis_sig', 'congestion_exp', 'pack_sat', 'compound'])],
        '시나리오physics': [c for c in phys_cols if 'sc_' in c or 'deviation' in c],
    }

    print(f'\n[Ablation 3] 패밀리별 분리 테스트')
    family_results = {}
    for fname, fcols in families.items():
        if not fcols:
            continue
        feat_fam = base_feat_cols + fcols
        print(f'\n  --- 패밀리: {fname} ({len(fcols)}종) ---')
        mae_fam, _, segs_fam, _ = eval_lgbm_cv(
            train_phys, y_log, y_raw, groups, feat_fam, label=fname)
        family_results[fname] = (mae_fam, segs_fam)

    # ── §2-E: Monotone Constraint ──
    mono_mae, mono_oof = monotone_constraint_experiment(
        train, y_log, y_raw, groups, base_feat_cols)

    # ── 종합 리포트 ──
    print('\n' + '='*60)
    print('§2-FINAL. 종합 결과')
    print('='*60)

    print(f'\n  {"설정":<30s}  {"CV MAE":>8s}  {"Δ MAE":>8s}  {"[80+] MAE":>10s}  {"Δ[80+]":>8s}')
    print(f'  {"-"*30}  {"-"*8}  {"-"*8}  {"-"*10}  {"-"*8}')

    configs = [
        ('baseline (model31 LGBM)', base_mae, base_segs),
        ('+ all physics', mae_all, segs_all),
        ('+ safe physics only', mae_safe, segs_safe),
        ('monotone constraint', mono_mae, None),
    ]
    for fname, fmae, fsegs in family_results.items():
        configs.append((f'+ {fname}', fmae, fsegs))

    for name, mae, segs in configs:
        delta = mae - base_mae
        seg80 = segs.get('[80,800)', (0, 0))[0] if segs else 0
        base80 = base_segs.get('[80,800)', (0, 0))[0]
        d80 = seg80 - base80 if seg80 > 0 else 0
        sign = '+' if delta > 0 else ''
        s80 = '+' if d80 > 0 else ''
        print(f'  {name:<30s}  {mae:>8.4f}  {sign}{delta:>7.4f}  {seg80:>10.2f}  {s80}{d80:>7.2f}')

    # ── 결론 도출 ──
    best_config = min(configs, key=lambda x: x[1])
    print(f'\n  최적 설정: {best_config[0]} (CV {best_config[1]:.4f})')

    if best_config[1] < base_mae - 0.005:
        print(f'  ✅ 유의미한 개선 ({base_mae - best_config[1]:.4f})')
        print(f'  → model35에 해당 피처 통합 권장')
    elif best_config[1] < base_mae:
        print(f'  △ 미미한 개선 ({base_mae - best_config[1]:.4f})')
        print(f'  → 5모델 스태킹 후 재검증 필요')
    else:
        print(f'  ❌ 개선 없음. Physics FE가 LGBM에 이미 학습된 정보일 가능성')
        print(f'  → 비선형 피처가 트리 모델에는 redundant할 수 있음')

    # ── 보고서 저장 ──
    elapsed = (time.time() - t0) / 60
    print(f'\n총 소요: {elapsed:.1f}분')

    report_path = os.path.join(DOCS_DIR, 'eda_symbolic_physics_report.txt')
    # 출력 전체를 캡처하지 않으므로, 핵심 요약만 저장
    summary = [
        '='*60,
        '§2 Simulator 역공학 — Physics FE + Symbolic Regression 결과',
        f'생성: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}',
        '='*60,
        '',
        f'Baseline LGBM CV MAE: {base_mae:.4f} ({n_base} 피처)',
        f'Physics FE 전체 추가: {mae_all:.4f} ({len(feat_all_phys)} 피처, Δ{mae_all-base_mae:+.4f})',
        f'Safe physics만 추가: {mae_safe:.4f} ({len(safe_feat_cols)} 피처, Δ{mae_safe-base_mae:+.4f})',
        f'Monotone constraint: {mono_mae:.4f} (Δ{mono_mae-base_mae:+.4f})',
        '',
        '패밀리별:',
    ]
    for fname, (fmae, _) in family_results.items():
        summary.append(f'  {fname}: {fmae:.4f} (Δ{fmae-base_mae:+.4f})')
    summary.extend([
        '',
        f'최적: {best_config[0]} = {best_config[1]:.4f}',
        '',
        '구간별 비교:',
    ])
    for name, mae, segs in configs:
        if segs:
            seg_str = ', '.join(f'{k}={v[0]:.2f}' for k, v in segs.items())
            summary.append(f'  {name}: {seg_str}')

    summary.append(f'\n총 소요: {elapsed:.1f}분')

    with open(report_path, 'w') as f:
        f.write('\n'.join(summary))
    print(f'\n리포트 저장: {report_path}')


if __name__ == '__main__':
    main()
