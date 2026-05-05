# layout_info.csv 심층 분석 및 결합 전략 리포트

> DACON 스마트 창고 지연 예측 대회 | 분석일: 2026-04-09

---

## 1. layout_info.csv 완전 분석

### 1-1. 기본 구조

| 항목 | 값 |
|------|-----|
| 행 수 | **300** (창고 300개) |
| 열 수 | **15** (키 1 + 범주 1 + 수치 13) |
| 결측치 | **0** (모든 컬럼 완전) |

### 1-2. 전체 컬럼 상세

| 컬럼명 | 타입 | 의미 | 범위 | 지연 예측 관련성 |
|--------|------|------|------|----------------|
| `layout_id` | object | 창고 고유 식별자 | WH_001 ~ WH_300 | **조인 키** |
| `layout_type` | object | 레이아웃 유형 | grid(106), hybrid(98), narrow(50), hub_spoke(46) | 동선 구조 결정 |
| `aisle_width_avg` | float | 평균 통로 너비(m) | 1.50 ~ 4.00 | 좁으면 로봇 혼잡↑ |
| `intersection_count` | float | 교차로 수 | 5 ~ 60 | 많으면 충돌/대기↑ |
| `one_way_ratio` | float | 일방통행 비율 | 0.002 ~ 0.792 | 높으면 우회 경로↑ |
| `pack_station_count` | float | 포장 스테이션 수 | 1 ~ 24 | 적으면 포장 병목↑ |
| `charger_count` | float | 충전기 수 | 1 ~ 18 | 적으면 충전 대기↑ |
| `layout_compactness` | float | 밀집도 (0~1) | 0.31 ~ 1.00 | 높으면 밀집·혼잡↑ |
| `zone_dispersion` | float | 존 분산도 (0~1) | 0.10 ~ 1.00 | 높으면 이동거리↑ |
| `robot_total` | int | 총 로봇 대수 | 8 ~ 100 | 처리 용량 지표 |
| `building_age_years` | int | 건물 노후도(년) | 5 ~ 50 | 오래되면 설비 열화↑ |
| `floor_area_sqm` | int | 바닥 면적(㎡) | 511 ~ 9,967 | 창고 규모 지표 |
| `ceiling_height_m` | float | 천장 높이(m) | 4.0 ~ 14.8 | 수직 활용도 관련 |
| `fire_sprinkler_count` | int | 스프링클러 수 | 10 ~ 100 | 건물 규모 프록시 |
| `emergency_exit_count` | int | 비상구 수 | 2 ~ 10 | 건물 규모 프록시 |

### 1-3. layout_type별 특성 프로파일

| 유형 | 평균 통로 너비 | 일방통행 비율 | 밀집도 | 해석 |
|------|---------------|-------------|--------|------|
| **narrow** | 1.96m | 0.52 | 0.84 | 좁은 통로, 높은 밀집도, 일방통행 많음 |
| **grid** | 3.13m | 0.15 | 0.65 | 넓은 통로, 양방향, 중간 밀집 |
| **hub_spoke** | 3.30m | 0.09 | 0.44 | 가장 넓은 통로, 분산형 |
| **hybrid** | 2.66m | 0.30 | 0.57 | 중간 특성 혼합형 |

### 1-4. 분포 특성

대부분의 수치 컬럼은 **균등 분포에 가까운 형태**(왜도 ≈ 0, 첨도 < 0)를 보입니다. `one_way_ratio`만 양의 왜도(+0.85)를 가지며 우측 꼬리가 긴 분포입니다. 이는 layout_info가 시뮬레이션으로 생성된 합성 데이터임을 시사합니다.

---

## 2. 메인 데이터와의 조인 가능성

### 2-1. 조인 키: `layout_id` (직접 조인 가능!)

```
train.csv  ──┐
             ├── layout_id (공유 키) ──→ layout_info.csv
test.csv   ──┘
```

| 검증 항목 | 결과 |
|-----------|------|
| train의 고유 layout_id | **250개** |
| test의 고유 layout_id | **100개** |
| layout_info 전체 | **300개** |
| train ⊂ layout_info | ✅ 완전 포함 |
| test ⊂ layout_info | ✅ 완전 포함 |
| train ∩ test | **50개** 공유 |
| **test에만 있는 layout_id** | **50개** (train에 없음!) |

### 2-2. 조인 관계: 1 : N

| 항목 | 값 |
|------|-----|
| layout_id당 train 행 수 | 평균 1,000 / 최소 500 / 최대 1,125 |
| layout_id당 시나리오 수 | 평균 40 / 최소 20 / 최대 45 |

### 2-3. 핵심 발견: test에 unseen 창고 50개!

> **test의 layout_id 중 50개는 train에 전혀 등장하지 않습니다.**
> 이 창고들의 특성을 알 수 있는 유일한 소스가 `layout_info.csv`입니다.
> → **layout_info merge는 선택이 아닌 필수!**

---

## 3. 결합 시 기대 효과

### 3-1. 타겟과의 상관관계 (Pearson r)

| layout 피처 | r | 해석 |
|-------------|---|------|
| `pack_station_count` | **-0.186** | 포장 스테이션↑ → 지연↓ (가장 강한 관계) |
| `robot_total` | **-0.111** | 로봇 수↑ → 지연↓ |
| `emergency_exit_count` | -0.044 | 규모↑ → 약간 지연↓ |
| `zone_dispersion` | -0.027 | |
| `layout_compactness` | -0.022 | |
| 나머지 8개 | |r| < 0.02 | 선형 관계 약함 |

참고: 기존 피처 Top은 `low_battery_ratio` (r=+0.37), `battery_mean` (r=-0.36) 수준.

### 3-2. Mutual Information (비선형 정보량)

MI 분석에서는 **선형 상관과 전혀 다른 순위**가 나타납니다:

| layout 피처 | MI | 기존 피처 Top5 MI (비교) |
|-------------|-----|------------------------|
| `floor_area_sqm` | **0.152** | `congestion_score` = 0.352 |
| `layout_compactness` | **0.151** | `low_battery_ratio` = 0.325 |
| `zone_dispersion` | **0.144** | `robot_idle` = 0.296 |
| `one_way_ratio` | **0.139** | `battery_mean` = 0.270 |
| `aisle_width_avg` | **0.110** | `order_inflow_15m` = 0.194 |

→ layout 피처의 MI는 기존 Top 피처의 **40~80% 수준**으로, 비선형적으로 상당한 정보를 담고 있습니다.

### 3-3. layout_type별 평균 지연 (ANOVA)

| layout_type | 평균 지연(분) | 표준편차 | 건수 |
|-------------|-------------|---------|------|
| **hub_spoke** | **22.28** | 30.44 | 43,375 |
| hybrid | 18.41 | 28.28 | 73,125 |
| narrow | 18.36 | 24.27 | 42,250 |
| grid | 18.10 | 26.25 | 91,250 |

**ANOVA F = 260.12, p ≈ 0** → layout_type에 따른 지연 차이가 통계적으로 매우 유의합니다.
hub_spoke 유형이 다른 유형 대비 **약 4분(22%) 높은 지연**을 보입니다.

### 3-4. 기존 피처와의 중복(다중공선성) 체크

| layout 피처 | 가장 높은 상관 기존 피처 | r | 판정 |
|-------------|----------------------|---|------|
| `robot_total` | `robot_idle` | **+0.72** | ⚠️ 높은 중복 |
| `layout_compactness` | `avg_trip_distance` | -0.49 | 중간 중복 |
| `zone_dispersion` | `vertical_utilization` | +0.48 | 중간 중복 |
| `pack_station_count` | `pack_utilization` | -0.39 | 약간 중복 |
| `one_way_ratio` | `cross_dock_ratio` | +0.37 | 약간 중복 |
| 나머지 8개 피처 | — | |r| < 0.27 | **독립적 정보** ✅ |

→ `robot_total`은 `robot_idle`과 높은 상관이 있어 추가 정보량이 제한적이나, **대부분의 layout 피처는 기존 90개 피처와 독립적인 새로운 정보**를 제공합니다.

### 3-5. 교호작용 분석 (layout_type × pack_utilization)

`pack_utilization → delay` 상관관계가 layout_type에 따라 다릅니다:
- grid: r = +0.131, hybrid: r = +0.129 (효과 큼)
- hub_spoke: r = +0.086, narrow: r = +0.056 (효과 작음)

→ layout_type과 운영 피처 간 **교호작용 피처**가 유의미할 수 있습니다.

---

## 4. 구체적 결합 전략 및 코드

### 전략 A: 직접 Merge (필수 기본)

```python
import pandas as pd

layout = pd.read_csv('data/layout_info.csv')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 직접 left merge
train = train.merge(layout, on='layout_id', how='left')
test = test.merge(layout, on='layout_id', how='left')

# layout_type 인코딩
train = pd.get_dummies(train, columns=['layout_type'], prefix='lt')
test = pd.get_dummies(test, columns=['layout_type'], prefix='lt')
```

→ 94열 → **108열** (14개 피처 추가, layout_type 원핫 포함 시 +17)

### 전략 B: 파생 비율 피처 생성

```python
def create_layout_derived_features(df):
    """layout 정보 기반 파생 피처"""
    # 로봇 밀도 (1000㎡당 로봇 수)
    df['robot_per_1000sqm'] = df['robot_total'] / (df['floor_area_sqm'] / 1000)
    
    # 충전기 대비 로봇 수 (충전 경쟁 강도)
    df['charger_per_robot'] = df['charger_count'] / df['robot_total']
    
    # 포장 스테이션 대비 로봇 수 (포장 병목 강도)
    df['pack_per_robot'] = df['pack_station_count'] / df['robot_total']
    
    # 교차로 밀도 (1000㎡당 교차로)
    df['intersection_per_1000sqm'] = df['intersection_count'] / (df['floor_area_sqm'] / 1000)
    
    # 통로너비 × 밀집도 (실효 통행 용이성)
    df['aisle_x_compactness'] = df['aisle_width_avg'] * df['layout_compactness']
    
    # 비상구당 면적 (건물 복잡도 프록시)
    df['sqm_per_exit'] = df['floor_area_sqm'] / df['emergency_exit_count']
    
    return df

train = create_layout_derived_features(train)
test = create_layout_derived_features(test)
```

파생 피처 상관관계:

| 파생 피처 | r (vs 타겟) |
|----------|------------|
| `charger_per_robot` | +0.079 |
| `robot_per_1000sqm` | -0.042 |
| `aisle_x_compactness` | -0.024 |
| `pack_per_robot` | -0.019 |

### 전략 C: 교호작용 피처 (layout × 운영 피처)

```python
def create_interaction_features(df):
    """layout_type별 운영 피처 교호작용"""
    # layout_type × 핵심 운영 피처
    key_ops = ['congestion_score', 'robot_utilization', 'pack_utilization', 
               'order_inflow_15m', 'low_battery_ratio']
    
    for lt_col in [c for c in df.columns if c.startswith('lt_')]:
        for op in key_ops:
            if op in df.columns:
                df[f'{lt_col}_x_{op}'] = df[lt_col] * df[op]
    
    # 통로너비 × 혼잡도 (좁은 통로에서 혼잡이 더 치명적)
    df['aisle_x_congestion'] = df['aisle_width_avg'] * df['congestion_score']
    
    # 밀집도 × max_zone_density
    df['compact_x_zone_density'] = df['layout_compactness'] * df['max_zone_density']
    
    # 충전기/로봇 × 충전대기
    df['charger_ratio_x_charge_wait'] = df['charger_per_robot'] * df['avg_charge_wait']
    
    return df

train = create_interaction_features(train)
test = create_interaction_features(test)
```

### 전략 D: layout_id별 Aggregation 피처 (Target Encoding 변형)

```python
from sklearn.model_selection import KFold

def layout_target_encoding(train, test, target_col, n_folds=5):
    """layout_id별 타겟 통계를 K-Fold로 안전하게 생성"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    train['layout_target_mean'] = np.nan
    train['layout_target_std'] = np.nan
    
    for tr_idx, val_idx in kf.split(train):
        agg = train.iloc[tr_idx].groupby('layout_id')[target_col].agg(['mean', 'std'])
        train.loc[train.index[val_idx], 'layout_target_mean'] = \
            train.iloc[val_idx]['layout_id'].map(agg['mean'])
        train.loc[train.index[val_idx], 'layout_target_std'] = \
            train.iloc[val_idx]['layout_id'].map(agg['std'])
    
    # test: train 전체 기반 통계
    global_agg = train.groupby('layout_id')[target_col].agg(['mean', 'std'])
    test['layout_target_mean'] = test['layout_id'].map(global_agg['mean'])
    test['layout_target_std'] = test['layout_id'].map(global_agg['std'])
    
    # ⚠️ unseen layout_id → layout_info 기반 유사 창고의 평균으로 대체
    global_mean = train[target_col].mean()
    test['layout_target_mean'].fillna(global_mean, inplace=True)
    test['layout_target_std'].fillna(train[target_col].std(), inplace=True)
    
    return train, test
```

> ⚠️ **주의**: test에 train에 없는 layout_id 50개가 있으므로, target encoding에서 이들은 global mean으로 대체됩니다. 이때 layout_info의 직접 피처가 이 unseen 창고들의 유일한 차별화 요소가 됩니다.

### 전략 E: layout 기반 클러스터링 피처

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def create_layout_cluster_features(layout, train, test, n_clusters=5):
    """layout 피처 기반 창고 클러스터링"""
    cluster_features = ['aisle_width_avg', 'intersection_count', 'one_way_ratio',
                       'pack_station_count', 'charger_count', 'layout_compactness',
                       'zone_dispersion', 'robot_total', 'floor_area_sqm']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(layout[cluster_features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    layout['wh_cluster'] = kmeans.fit_predict(X)
    
    # merge cluster info
    cluster_map = layout[['layout_id', 'wh_cluster']]
    train = train.merge(cluster_map, on='layout_id', how='left')
    test = test.merge(cluster_map, on='layout_id', how='left')
    
    return train, test
```

---

## 5. 종합 권장 사항

### 우선순위별 적용 로드맵

| 순위 | 전략 | 추가 피처 수 | 기대 효과 | 난이도 |
|------|------|-------------|----------|--------|
| **1** | **A. 직접 Merge** | +14 | ⭐⭐⭐ 필수 (unseen 창고 대응) | 쉬움 |
| **2** | **B. 파생 비율 피처** | +6 | ⭐⭐ 비선형 관계 포착 | 쉬움 |
| **3** | **D. Target Encoding** | +2 | ⭐⭐⭐ 강력하지만 누출 주의 | 중간 |
| **4** | **C. 교호작용 피처** | +20~30 | ⭐⭐ 트리 모델 성능↑ | 중간 |
| **5** | **E. 클러스터링** | +1 | ⭐ 보조적 | 중간 |

### 핵심 인사이트 요약

1. **layout_info merge는 필수**: test의 50개 unseen 창고에 대한 유일한 정보 소스
2. **pack_station_count**가 타겟과 가장 강한 선형 관계 (r = -0.19)
3. **MI 기준 top 피처** (floor_area_sqm, layout_compactness, zone_dispersion)는 비선형 정보가 풍부 — 트리 기반 모델에서 효과적
4. **layout_type은 통계적으로 매우 유의** (F=260, hub_spoke가 22분으로 가장 높은 지연)
5. 대부분의 layout 피처는 기존 90개 피처와 **독립적** → 새로운 정보 추가
6. `robot_total`은 `robot_idle`과 높은 상관(0.72) → 중복 가능성 있으나 비율 피처로 변환하면 독립 정보 확보
