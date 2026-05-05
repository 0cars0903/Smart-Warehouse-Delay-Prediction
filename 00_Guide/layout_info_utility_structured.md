# layout_info.csv 활용성 구조화 보고서

> DACON 스마트 창고 지연 예측 | 작성일: 2026-04-09  
> 기반 분석: `layout_info_analysis_report.md`

---

## 1. 보조 데이터 개요 및 위상

### 1-1. layout_info.csv가 특별한 이유

일반적인 대회에서 보조 데이터는 "있으면 좋은" 수준이지만, 이 데이터셋에서 `layout_info.csv`는 **구조적으로 필수**입니다.

```
train.csv   → layout_id 250종 포함
test.csv    → layout_id 100종 포함
               └─ 이 중 50종은 train에 등장 ❌
                  → layout_info가 유일한 정보 소스 ✅

layout_info.csv → layout_id 300종 전체 커버 (train ∪ test ⊂ layout_info)
```

**핵심 수치 요약**

| 항목 | 값 |
|------|-----|
| layout_info 창고 수 | 300개 (WH_001 ~ WH_300) |
| layout_info 피처 수 | 15개 (키 1 + 범주 1 + 수치 13) |
| 결측치 | 0% (완전한 데이터) |
| train에서 쓰이는 layout_id | 250개 |
| test에서 쓰이는 layout_id | 100개 |
| **test에만 있는 unseen layout_id** | **50개** |
| layout_id당 train 행 수 | 평균 1,000행 / 최소 500 / 최대 1,125 |

---

## 2. EDA 핵심 인사이트

### 2-1. 타겟과의 관계 (지연 예측력)

#### 선형 상관 (Pearson r)

| 피처 | r | 해석 |
|------|---|------|
| `pack_station_count` | **-0.186** | 포장 스테이션 ↑ → 지연 ↓ (가장 강한 선형 관계) |
| `robot_total` | **-0.111** | 로봇 수 ↑ → 지연 ↓ |
| 나머지 11개 | \|r\| < 0.05 | 선형 관계 약함 → 비선형 분석 필요 |

*참고: 기존 피처 Top인 `low_battery_ratio`(r=+0.37), `battery_mean`(r=-0.36)에 비해 낮지만, 독립적인 정보이므로 앙상블에서 기여 가능.*

#### 비선형 정보량 (Mutual Information)

선형 상관이 낮아도 MI는 상당히 높습니다. **트리 기반 모델에서 효과적**입니다.

| 피처 | MI | 기존 Top 피처 MI (비교) |
|------|----|------------------------|
| `floor_area_sqm` | **0.152** | `congestion_score` = 0.352 |
| `layout_compactness` | **0.151** | `low_battery_ratio` = 0.325 |
| `zone_dispersion` | **0.144** | `robot_idle` = 0.296 |
| `one_way_ratio` | **0.139** | `battery_mean` = 0.270 |
| `aisle_width_avg` | **0.110** | `order_inflow_15m` = 0.194 |

→ layout 피처 MI는 기존 Top 피처의 **40~80% 수준**으로 의미 있는 비선형 정보를 보유합니다.

### 2-2. layout_type별 지연 차이 (ANOVA)

통계적으로 매우 유의한 차이 확인 (F = 260.12, p ≈ 0).

| layout_type | 평균 지연(분) | 표준편차 | 샘플 수 |
|-------------|-------------|---------|---------|
| **hub_spoke** | **22.28** | 30.44 | 43,375 |
| hybrid | 18.41 | 28.28 | 73,125 |
| narrow | 18.36 | 24.27 | 42,250 |
| grid | 18.10 | 26.25 | 91,250 |

> `hub_spoke`는 타 유형 대비 **약 4분(22%) 높은 지연**을 보입니다.  
> 분산도 가장 크므로, 이 유형에서 예측이 어렵다는 점도 중요합니다.

### 2-3. layout_type별 물리 특성 프로파일

| layout_type | 평균 통로 너비 | 일방통행 비율 | 밀집도 | 특징 |
|-------------|---------------|-------------|--------|------|
| **narrow** | 1.96m | 0.52 | 0.84 | 좁은 통로, 높은 밀집, 일방통행 多 |
| **grid** | 3.13m | 0.15 | 0.65 | 넓은 통로, 양방향, 중간 밀집 |
| **hub_spoke** | 3.30m | 0.09 | 0.44 | 가장 넓은 통로, 분산형 구조 |
| **hybrid** | 2.66m | 0.30 | 0.57 | 혼합형, 중간 특성 |

> ⚠️ `hub_spoke`는 통로가 넓고 밀집도가 낮은데도 지연이 가장 큽니다.  
> → 분산형 구조로 인한 **이동 거리 증가**가 지연의 원인으로 추정됩니다.

### 2-4. 기존 피처와의 중복성 체크 (다중공선성)

| layout 피처 | 가장 유사한 기존 피처 | r | 판정 |
|-------------|----------------------|---|------|
| `robot_total` | `robot_idle` | **+0.72** | ⚠️ 높은 중복, 비율 피처 변환 권장 |
| `layout_compactness` | `avg_trip_distance` | -0.49 | 중간 중복 |
| `zone_dispersion` | `vertical_utilization` | +0.48 | 중간 중복 |
| `pack_station_count` | `pack_utilization` | -0.39 | 약간 중복 |
| `one_way_ratio` | `cross_dock_ratio` | +0.37 | 약간 중복 |
| 나머지 8개 피처 | — | \|r\| < 0.27 | **독립 정보** ✅ |

→ `robot_total`만 제외하면 대부분 **기존 90개 피처와 독립적**입니다.

### 2-5. 교호작용 발견: layout_type × 운영 지표

`pack_utilization → delay` 상관이 layout_type에 따라 달라집니다:

| layout_type | r(pack_utilization → delay) | 해석 |
|-------------|----------------------------|------|
| grid | +0.131 | 포장 병목 효과 큼 |
| hybrid | +0.129 | 포장 병목 효과 큼 |
| hub_spoke | +0.086 | 효과 상대적으로 작음 |
| narrow | +0.056 | 효과 가장 작음 |

→ **layout_type과 운영 피처 간의 교호작용 피처**가 추가 성능을 줄 수 있습니다.

---

## 3. 활용 전략 로드맵

총 5가지 전략으로 체계화합니다. 우선순위 순으로 적용하세요.

---

### ★ 전략 A — 직접 Merge (필수, 즉시 적용)

**적용 이유**: test의 unseen 50개 창고에 대한 유일한 정보 소스  
**추가 피처**: +14열 (수치 13 + layout_type 범주 1)  
**난이도**: 쉬움

```python
import pandas as pd

layout = pd.read_csv('data/layout_info.csv')
train  = pd.read_csv('data/train.csv')
test   = pd.read_csv('data/test.csv')

train = train.merge(layout, on='layout_id', how='left')
test  = test.merge(layout, on='layout_id', how='left')

# layout_type 원-핫 인코딩
train = pd.get_dummies(train, columns=['layout_type'], prefix='lt')
test  = pd.get_dummies(test,  columns=['layout_type'], prefix='lt')
# → 총 94열 + 17열 = 111열
```

**주의사항**: test 전용 50개 창고는 layout_info 직접 피처만으로 모델이 구분해야 합니다. 이 단계가 빠지면 이 창고들은 완전히 blind 상태가 됩니다.

---

### ★ 전략 B — 파생 비율 피처 (우선순위 2)

**적용 이유**: 절대 수치 대신 로봇/면적 기준 상대 지표로 비선형 관계 포착  
**추가 피처**: +6열  
**난이도**: 쉬움

```python
def create_layout_derived_features(df):
    # 로봇 밀도 (1,000㎡당 로봇 수) — 면적 대비 자원 지표
    df['robot_per_1000sqm'] = df['robot_total'] / (df['floor_area_sqm'] / 1000)

    # 충전기 경쟁 강도 — 낮을수록 충전 대기 위험↑
    df['charger_per_robot'] = df['charger_count'] / df['robot_total']

    # 포장 병목 강도 — 낮을수록 포장 스테이션 부족
    df['pack_per_robot'] = df['pack_station_count'] / df['robot_total']

    # 교차로 밀도 (1,000㎡당) — 높을수록 충돌·대기 위험
    df['intersection_per_1000sqm'] = df['intersection_count'] / (df['floor_area_sqm'] / 1000)

    # 실효 통행 용이성 (통로너비 × 밀집도 역수)
    df['aisle_x_compactness'] = df['aisle_width_avg'] * df['layout_compactness']

    # 건물 복잡도 프록시 (비상구당 면적)
    df['sqm_per_exit'] = df['floor_area_sqm'] / df['emergency_exit_count']

    return df
```

| 파생 피처 | r vs 타겟 | 주목 이유 |
|----------|-----------|-----------|
| `charger_per_robot` | +0.079 | `robot_total` 단독보다 중복성 낮음 |
| `robot_per_1000sqm` | -0.042 | 면적 정규화로 순수 밀도 표현 |

---

### ★ 전략 C — 교호작용 피처 (우선순위 4)

**적용 이유**: layout_type별로 운영 지표가 지연에 미치는 영향이 다름  
**추가 피처**: +20~30열  
**난이도**: 중간  
**트리 모델 효과**: 높음 (LGBM/XGB/CatBoost)

```python
def create_interaction_features(df):
    key_ops = ['congestion_score', 'robot_utilization', 'pack_utilization',
               'order_inflow_15m', 'low_battery_ratio']

    # layout_type × 핵심 운영 피처
    for lt_col in [c for c in df.columns if c.startswith('lt_')]:
        for op in key_ops:
            if op in df.columns:
                df[f'{lt_col}_x_{op}'] = df[lt_col] * df[op]

    # 물리 × 운영 교호작용
    df['aisle_x_congestion']         = df['aisle_width_avg']    * df['congestion_score']
    df['compact_x_zone_density']     = df['layout_compactness'] * df['max_zone_density']
    df['charger_ratio_x_charge_wait']= df['charger_per_robot']  * df['avg_charge_wait']

    return df
```

> 전략 B의 파생 피처 생성 후 적용하세요 (`charger_per_robot` 등이 선행 필요).

---

### ★ 전략 D — layout_id Target Encoding (우선순위 3)

**적용 이유**: layout_id는 창고 고유 특성을 통째로 흡수하는 강력한 Aggregation 키  
**추가 피처**: +2열 (mean, std)  
**난이도**: 중간  
**주의**: K-Fold 내부에서 반드시 OOF 방식 적용 (누출 방지)

```python
import numpy as np
from sklearn.model_selection import KFold

def layout_target_encoding(train, test, target_col='avg_delay_minutes_next_30m', n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    train['layout_target_mean'] = np.nan
    train['layout_target_std']  = np.nan

    for tr_idx, val_idx in kf.split(train):
        agg = train.iloc[tr_idx].groupby('layout_id')[target_col].agg(['mean', 'std'])
        train.loc[train.index[val_idx], 'layout_target_mean'] = \
            train.iloc[val_idx]['layout_id'].map(agg['mean'])
        train.loc[train.index[val_idx], 'layout_target_std'] = \
            train.iloc[val_idx]['layout_id'].map(agg['std'])

    # test: train 전체 통계로 매핑
    global_agg = train.groupby('layout_id')[target_col].agg(['mean', 'std'])
    test['layout_target_mean'] = test['layout_id'].map(global_agg['mean'])
    test['layout_target_std']  = test['layout_id'].map(global_agg['std'])

    # unseen 50개 창고 → global mean으로 대체
    test['layout_target_mean'].fillna(train[target_col].mean(), inplace=True)
    test['layout_target_std'].fillna(train[target_col].std(),  inplace=True)

    return train, test
```

> ⚠️ **unseen 50개 창고의 target encoding은 global mean으로 fallback**됩니다.  
> 이 창고들에 대해서는 전략 A(직접 피처)가 더 신뢰할 수 있는 정보를 제공합니다.

---

### 전략 E — 클러스터링 피처 (보조, 선택)

**적용 이유**: layout_type 4개 분류가 충분하지 않을 때, 더 세분화된 창고 군집 생성  
**추가 피처**: +1열 (wh_cluster)  
**난이도**: 중간

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def create_layout_cluster_features(layout, train, test, n_clusters=5):
    cluster_features = [
        'aisle_width_avg', 'intersection_count', 'one_way_ratio',
        'pack_station_count', 'charger_count', 'layout_compactness',
        'zone_dispersion', 'robot_total', 'floor_area_sqm'
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(layout[cluster_features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    layout['wh_cluster'] = kmeans.fit_predict(X)

    cluster_map = layout[['layout_id', 'wh_cluster']]
    train = train.merge(cluster_map, on='layout_id', how='left')
    test  = test.merge(cluster_map, on='layout_id', how='left')

    return train, test
```

---

## 4. 전략별 우선순위 종합표

| 순위 | 전략 | 추가 피처 수 | 기대 효과 | 난이도 | 근거 |
|------|------|-------------|----------|--------|------|
| **1** | **A. 직접 Merge** | +14~17 | ⭐⭐⭐ 필수 | 쉬움 | unseen 50 창고 대응 |
| **2** | **B. 파생 비율** | +6 | ⭐⭐ 비선형 포착 | 쉬움 | 중복성 낮고 해석 가능 |
| **3** | **D. Target Encoding** | +2 | ⭐⭐⭐ 강력 | 중간 | layout_id는 강력한 집계 키 |
| **4** | **C. 교호작용** | +20~30 | ⭐⭐ 트리 성능↑ | 중간 | ANOVA로 교호작용 확인 |
| **5** | **E. 클러스터링** | +1 | ⭐ 보조 | 중간 | layout_type 세분화 시 |

---

## 5. 적용 시 주의사항

### 5-1. robot_total 중복 처리

`robot_total`은 `robot_idle`과 r=+0.72로 높은 상관이 있습니다.

- **직접 사용**: 트리 모델은 중복 피처를 자동으로 낮게 평가 → 큰 해가 없음
- **권장**: `robot_per_1000sqm` 등 비율 피처로 변환하면 새로운 독립 정보 확보

### 5-2. 분포 특성 (합성 데이터 주의)

대부분의 수치 피처가 균등 분포(`왜도 ≈ 0, 첨도 < 0`)를 보입니다.  
`one_way_ratio`만 양의 왜도(+0.85)를 가집니다.  
→ **이 데이터는 시뮬레이션 생성된 합성 데이터**로 추정됩니다.  
→ 극단값 처리나 log transform의 필요성이 낮습니다.

### 5-3. unseen 창고 50개 처리 전략

test에만 있는 50개 창고에 대한 정보 우선순위:

```
1순위: layout_info 직접 피처 (A) → 완전한 정보
2순위: 파생 비율 피처 (B)        → 연산 가능
3순위: 교호작용 피처 (C)         → lt_ 원핫으로 계산 가능
4순위: Target Encoding (D)       → global mean fallback (품질 낮음)
```

---

## 6. 현재 모델 대비 예상 개선 효과

| 현재 상태 | layout_info 미적용 (CV MAE: 8.8703 / Public: 10.3349) |
|-----------|------------------------------------------------------|
| A만 적용 후 예상 | CV -0.05 ~ -0.15 수준 개선 가능 |
| A+B+D 적용 후 | CV -0.1 ~ -0.3 수준, Public < 10.2 목표 |
| A+B+C+D 적용 후 | Public < 10.0 목표 (다음 마일스톤) |

> 수치는 추정값입니다. 실험을 통해 검증이 필요합니다.

---

## 7. 즉시 실행 체크리스트

```
□ 전략 A: train/test에 layout_info left merge
□ 전략 A: layout_type 원-핫 인코딩 (lt_grid, lt_hybrid, lt_hub_spoke, lt_narrow)
□ 전략 B: 6개 파생 비율 피처 생성 함수 추가
□ 전략 D: layout_id Target Encoding (OOF 방식)
□ 실험: CV MAE 비교 후 효과 검증
□ 효과 있으면 전략 C 교호작용 피처 추가
□ Approach Log 및 Notion 업데이트
```
