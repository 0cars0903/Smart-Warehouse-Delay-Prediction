"""
feature_engineering.py
======================
스마트 창고 출고 지연 예측 - 피처 엔지니어링 파이프라인

실험 결과 (GroupKFold 5-Fold, LightGBM L1):
  Base(layout+ts) : 9.1790
  + Lag(1-3)      : 9.0793  (-1.09%)
  + Rolling(3/5)  : 9.0052  (-1.89%)
  + Domain        : 9.0010  (-1.94%)  ← 현재 최고

TS0 Broadcast 실험 (추가 EDA 결과 기반):
  TS0 초기 조건이 시나리오 전체 궤적을 결정
  robot_utilization(TS0) r=0.475 vs 전체평균 r=0.211 — 2.3× 신호 강도
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 핵심 피처 (상관관계 + 도메인 중요도 기준)
# ─────────────────────────────────────────────
KEY_COLS = [
    'low_battery_ratio',
    'battery_mean',
    'charge_queue_length',
    'robot_idle',
    'order_inflow_15m',
    'congestion_score',
    'max_zone_density',
    'avg_trip_distance',
]


# ─────────────────────────────────────────────
# 1. Layout merge
# ─────────────────────────────────────────────
def merge_layout(train: pd.DataFrame, test: pd.DataFrame,
                 layout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """layout_info 보조 테이블 merge (14개 컬럼 추가)"""
    train = train.merge(layout, on='layout_id', how='left')
    test  = test.merge(layout,  on='layout_id', how='left')
    return train, test


# ─────────────────────────────────────────────
# 2. 범주형 인코딩
# ─────────────────────────────────────────────
def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame,
                         target: str = 'avg_delay_minutes_next_30m') -> tuple[pd.DataFrame, pd.DataFrame]:
    """train+test 통합 순서형 인코딩 (범주 누락 방지)"""
    exclude = ['ID', 'layout_id', 'scenario_id', target]
    cat_cols = [c for c in train.select_dtypes(include='object').columns
                if c not in exclude]
    for col in cat_cols:
        combined = pd.concat([train[col], test[col]], axis=0)
        mapping  = {v: i for i, v in enumerate(combined.dropna().unique())}
        train[col] = train[col].map(mapping)
        test[col]  = test[col].map(mapping)
    return train, test


# ─────────────────────────────────────────────
# 3. 타임슬롯 피처
# ─────────────────────────────────────────────
def add_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    """시나리오 내 타임슬롯 위치 피처 (4종)"""
    df = df.copy()
    df['ts_idx']   = df.groupby('scenario_id').cumcount()
    df['ts_ratio'] = df['ts_idx'] / 24.0
    df['ts_sin']   = np.sin(2 * np.pi * df['ts_idx'] / 25)
    df['ts_cos']   = np.cos(2 * np.pi * df['ts_idx'] / 25)
    return df


# ─────────────────────────────────────────────
# 4. TS0 Broadcast 피처
# ─────────────────────────────────────────────

# TS0에서 broadcast할 연속형 피처 (EDA A1: 시나리오 결과와 상관 Top-8)
TS0_CONTINUOUS_COLS = [
    'robot_utilization',    # r=0.475 (TS0→outcome) vs 0.211 (overall)
    'order_inflow_15m',     # r=0.462
    'robot_active',         # r=0.398
    'sku_concentration',    # r=0.370
    'max_zone_density',     # r=0.367
    'congestion_score',     # r=0.366
    'robot_idle',           # r=-0.356
    'urgent_order_ratio',   # r=0.339
]

# TS0 이진 이벤트 플래그 (EDA A2: 붕괴 vs 안정 시나리오 배율 Top)
TS0_EVENT_COLS = [
    'blocked_path_15m',     # 붕괴/안정 배율 2946×
    'fault_count_15m',      # 배율 891×
    'avg_recovery_time',    # 배율 569×
]


def add_ts0_features(train: pd.DataFrame, test: pd.DataFrame,
                     use_continuous: bool = True,
                     use_flags: bool = True,
                     use_composite: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    TS0(시나리오 시작 타임슬롯) 초기 조건을 시나리오 내 모든 타임슬롯에 broadcast

    EDA 발견: TS0 초기값이 시나리오 전체 결과를 강하게 결정
    - robot_utilization(TS0) r=0.475 vs 전체 평균 r=0.211 (2.3× 신호 강도)
    - 붕괴 시나리오: TS0에 이미 blocked_path, fault, high_recovery 존재

    ※ ts_idx 컬럼이 반드시 존재해야 함 (add_ts_features 이후 호출)
    ※ lag/rolling 피처보다 먼저 추가해야 lag 계산에 포함됨

    Parameters
    ----------
    use_continuous : TS0 연속형 8종 broadcast
    use_flags      : TS0 이진 이벤트 플래그 3종 (>0 여부)
    use_composite  : 과부하 취약성 지수 (ts0_robot_utilization × ts0_order_inflow_15m)
    """

    def _compute_ts0_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # TS0 행 추출 (ts_idx == 0)
        ts0 = (df[df['ts_idx'] == 0]
               .set_index('scenario_id'))

        new_cols = {}

        # --- 연속형 broadcast ---
        if use_continuous:
            for col in TS0_CONTINUOUS_COLS:
                if col not in ts0.columns:
                    continue
                ts0_vals = ts0[col].rename(f'ts0_{col}')
                df = df.join(ts0_vals, on='scenario_id')
                # NaN 보정: TS0 자체가 없는 시나리오는 현재값으로 fallback
                df[f'ts0_{col}'] = df[f'ts0_{col}'].fillna(df[col])

        # --- 이진 이벤트 플래그 ---
        if use_flags:
            for col in TS0_EVENT_COLS:
                if col not in ts0.columns:
                    continue
                flag_name = f'ts0_{col}_flag'
                ts0_flags = (ts0[col] > 0).astype(int).rename(flag_name)
                df = df.join(ts0_flags, on='scenario_id')
                df[flag_name] = df[flag_name].fillna(0).astype(int)

        # --- 복합 취약성 지수 ---
        if use_composite:
            if 'ts0_robot_utilization' in df.columns and 'ts0_order_inflow_15m' in df.columns:
                df['ts0_overload_risk'] = df['ts0_robot_utilization'] * df['ts0_order_inflow_15m']

        return df

    train = _compute_ts0_features(train)
    test  = _compute_ts0_features(test)
    return train, test


# ─────────────────────────────────────────────
# 5. Lag 피처
# ─────────────────────────────────────────────
def add_lag_features(train: pd.DataFrame, test: pd.DataFrame,
                     key_cols: list = None, lags: list = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    시나리오 내 lag 피처 (groupby + shift → 시나리오 간 리크 없음)
    ※ ts_idx 컬럼이 반드시 존재해야 함 (정렬 기준)
    """
    if key_cols is None:
        key_cols = KEY_COLS
    if lags is None:
        lags = [1, 2, 3]

    train_c = train.copy(); train_c['_split'] = 0
    test_c  = test.copy();  test_c['_split']  = 1

    # 원본 test 순서 보존을 위해 행 번호 저장
    test_c['_orig_order'] = np.arange(len(test_c))

    combined = (pd.concat([train_c, test_c], axis=0, ignore_index=True)
                  .sort_values(['scenario_id', 'ts_idx'])
                  .reset_index(drop=True))

    for col in key_cols:
        if col not in combined.columns:
            continue
        grp = combined.groupby('scenario_id')[col]
        for lag in lags:
            combined[f'{col}_lag{lag}'] = grp.shift(lag)

    tr_out = combined[combined['_split'] == 0].drop(columns=['_split', '_orig_order'], errors='ignore')
    # test는 원본 ID 순서로 복원
    te_sorted = combined[combined['_split'] == 1].sort_values('_orig_order').drop(
        columns=['_split', '_orig_order'])
    return tr_out, te_sorted


# ─────────────────────────────────────────────
# 5. Rolling 피처
# ─────────────────────────────────────────────
def add_rolling_features(train: pd.DataFrame, test: pd.DataFrame,
                          key_cols: list = None,
                          windows: list = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    이동평균/표준편차 피처 (shift(1) 후 rolling → 현재 타임슬롯 리크 없음)
    """
    if key_cols is None:
        key_cols = KEY_COLS
    if windows is None:
        windows = [3, 5]

    train_c = train.copy(); train_c['_split'] = 0
    test_c  = test.copy();  test_c['_split']  = 1

    # 원본 test 순서 보존
    test_c['_orig_order'] = np.arange(len(test_c))

    combined = (pd.concat([train_c, test_c], axis=0, ignore_index=True)
                  .sort_values(['scenario_id', 'ts_idx'])
                  .reset_index(drop=True))

    for col in key_cols:
        if col not in combined.columns:
            continue
        shifted = combined.groupby('scenario_id')[col].shift(1)
        for w in windows:
            combined[f'{col}_roll{w}_mean'] = (
                shifted.groupby(combined['scenario_id'])
                       .transform(lambda x: x.rolling(w, min_periods=1).mean()))
            combined[f'{col}_roll{w}_std'] = (
                shifted.groupby(combined['scenario_id'])
                       .transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0)))

    tr_out = combined[combined['_split'] == 0].drop(columns=['_split', '_orig_order'], errors='ignore')
    # test는 원본 ID 순서로 복원
    te_sorted = combined[combined['_split'] == 1].sort_values('_orig_order').drop(
        columns=['_split', '_orig_order'])
    return tr_out, te_sorted


# ─────────────────────────────────────────────
# 6. Domain 복합 피처
# ─────────────────────────────────────────────
def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    도메인 지식 기반 복합 피처 (8종)
    배터리 위기 cascade: low_battery → charge_queue → robot 부족 → 지연
    """
    df = df.copy()
    charger_cnt = df['charger_count'] if 'charger_count' in df.columns else 1

    # 배터리 위기 지수
    df['battery_crisis']     = df['low_battery_ratio'] * df['charge_queue_length']
    # 사용 가능 로봇 비율
    df['robot_availability'] = (df['robot_active'] /
                                 (df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1))
    # 충전 병목 지수
    df['charge_bottleneck']  = df['charge_queue_length'] / (charger_cnt + 1)
    # 배터리 결핍 지수
    df['battery_deficit']    = (1 - df['battery_mean'] / 100.0) * df['robot_charging']
    # 주문 압박 지수
    df['order_pressure']     = df['order_inflow_15m'] * df['urgent_order_ratio']
    # 로봇 대비 주문량
    df['orders_per_active_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)
    # 혼잡 × 주문 상호작용
    df['congestion_x_order'] = df['congestion_score'] * df['order_inflow_15m']
    # SKU 복잡도
    df['sku_complexity']     = df['unique_sku_15m'] * df['avg_items_per_order']
    return df


# ─────────────────────────────────────────────
# 통합 파이프라인
# ─────────────────────────────────────────────
def build_features(train: pd.DataFrame, test: pd.DataFrame,
                   layout: pd.DataFrame,
                   target: str = 'avg_delay_minutes_next_30m',
                   use_lag: bool = True,
                   use_rolling: bool = True,
                   use_domain: bool = True,
                   use_ts0: bool = False,
                   ts0_continuous: bool = True,
                   ts0_flags: bool = True,
                   ts0_composite: bool = True,
                   lag_lags: list = None,
                   rolling_windows: list = None,
                   verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    전체 피처 엔지니어링 파이프라인

    Parameters
    ----------
    train, test      : 원본 CSV DataFrame
    layout           : layout_info.csv DataFrame
    target           : 타겟 컬럼명
    use_lag          : Lag 피처 포함 여부
    use_rolling      : Rolling 피처 포함 여부
    use_domain       : Domain 복합 피처 포함 여부
    use_ts0          : TS0 broadcast 피처 포함 여부
    ts0_continuous   : TS0 연속형 8종 broadcast
    ts0_flags        : TS0 이진 이벤트 플래그 3종
    ts0_composite    : TS0 복합 취약성 지수 (robot_util × order_inflow)
    lag_lags         : lag 목록 (default=[1,2,3] → 확장 시 [1,2,3,4,5,6])
    rolling_windows  : rolling 윈도우 목록 (default=[3,5] → 확장 시 [3,5,10])

    Returns
    -------
    (train_fe, test_fe) : 피처 엔지니어링 완료된 DataFrame
    """
    if verbose:
        print(f'[build_features] 시작: train={train.shape}, test={test.shape}')

    # Step 1: Layout merge
    train, test = merge_layout(train, test, layout)
    if verbose:
        print(f'  1. layout merge → {train.shape[1]} cols')

    # Step 2: 범주형 인코딩
    train, test = encode_categoricals(train, test, target)
    if verbose:
        print(f'  2. 범주형 인코딩 완료')

    # Step 3: ts 피처
    train = add_ts_features(train)
    test  = add_ts_features(test)
    if verbose:
        print(f'  3. ts 피처 추가 (4종) → {train.shape[1]} cols')

    # Step 3.5: TS0 Broadcast 피처 (ts_idx 계산 이후, lag/rolling 이전)
    if use_ts0:
        train, test = add_ts0_features(
            train, test,
            use_continuous=ts0_continuous,
            use_flags=ts0_flags,
            use_composite=ts0_composite,
        )
        if verbose:
            ts0_cols = [c for c in train.columns if c.startswith('ts0_')]
            print(f'  3.5 TS0 broadcast 피처 추가 ({len(ts0_cols)}종) → {train.shape[1]} cols')

    # Step 4: Lag 피처
    if use_lag:
        lag_kwargs = {} if lag_lags is None else {'lags': lag_lags}
        train, test = add_lag_features(train, test, **lag_kwargs)
        if verbose:
            lag_cols = [c for c in train.columns if '_lag' in c]
            print(f'  4. Lag 피처 추가 ({len(lag_cols)}종) → {train.shape[1]} cols')

    # Step 5: Rolling 피처
    if use_rolling:
        roll_kwargs = {} if rolling_windows is None else {'windows': rolling_windows}
        train, test = add_rolling_features(train, test, **roll_kwargs)
        if verbose:
            roll_cols = [c for c in train.columns if '_roll' in c]
            print(f'  5. Rolling 피처 추가 ({len(roll_cols)}종) → {train.shape[1]} cols')

    # Step 6: Domain 피처
    if use_domain:
        train = add_domain_features(train)
        test  = add_domain_features(test)
        if verbose:
            print(f'  6. Domain 피처 추가 (8종) → {train.shape[1]} cols')

    if verbose:
        excl = ['ID', 'layout_id', 'scenario_id', target]
        feat_cols = [c for c in train.columns if c not in excl
                     and train[c].dtype.name not in ['object', 'category']]
        print(f'[build_features] 완료: 최종 피처 수 = {len(feat_cols)}')

    return train, test


def get_feature_cols(df: pd.DataFrame,
                     target: str = 'avg_delay_minutes_next_30m') -> list:
    """모델 학습에 사용할 피처 컬럼 목록 반환"""
    exclude = ['ID', 'layout_id', 'scenario_id', target]
    return [c for c in df.columns
            if c not in exclude and df[c].dtype.name not in ['object', 'category']]


if __name__ == '__main__':
    import os
    DATA_PATH = '../data/'
    train  = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_PATH, 'layout_info.csv'))

    train_fe, test_fe = build_features(train, test, layout, verbose=True)
    feat_cols = get_feature_cols(train_fe)
    print(f'\n학습 피처 수: {len(feat_cols)}')
    print(f'Train FE shape: {train_fe.shape}')
    print(f'Test  FE shape: {test_fe.shape}')
