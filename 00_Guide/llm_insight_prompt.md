# LLM 인사이트 수집 프롬프트
### 스마트 창고 출고 지연 예측 AI 경진대회 (Dacon)
> 작성일: 2026-04-04 | 조사 기반: 유사 논문 5편, 유사 대회 4건, Kaggle Grandmaster 기법

---

## 📌 사용 방법

이 파일은 Claude, GPT-4o, Gemini 등 LLM에게 **경진대회 전략 인사이트**를 구체적으로 질문하기 위한 입력 프롬프트입니다.
아래 `[PROMPT START]` ~ `[PROMPT END]` 사이 전체를 LLM에 입력하세요.

---

## 🔎 조사된 유사 사례 요약 (프롬프트 작성 근거)

### 유사 논문

| 논문 | 핵심 기법 | 결과 |
|---|---|---|
| *Uncertainty-Aware Delivery Delay Duration Prediction via Multi-Task DL* (arXiv 2602.20271) | Classification-then-Regression 전략, PLR(Piecewise Linear Representation) 임베딩, 불균형 데이터 처리 | 기존 GBDT 대비 MAE 41~64% 개선 |
| *Predictive methods for delivery delays in supply chains* (ScienceDirect 2025) | GBDT 2-step (분류→회귀), pinball loss 회귀 | LightGBM > Random Forest |
| *Machine learning in smart production logistics* (Taylor & Francis 2024) | DT, RF, SVM, ANN, GB, KNN 비교 | GBDT 계열이 물류 도메인 표준 |
| *Supply delay risk prediction using big data* (Springer 2025) | 공급망 거시지표 + 딥러닝 | Accuracy 99.61% |
| *AMR feasibility prediction using AutoML* (Springer 2024) | AutoML (에너지 소비 예측) | 경로 불확실성 처리 중요 |

### 유사 대회 & 솔루션 패턴

| 대회 | 우승 전략 | 관련성 |
|---|---|---|
| Dacon 물류 유통량 예측 | LightGBM + 집계 피처 + log 변환 | 직접 유사: 물류 + 정형 + MAE |
| Dacon 추석 선물 수요량 예측 (Private 2위) | AutoGluon + Feature 추가 + Distillation | 앙상블 전략 참고 |
| Amex Credit Fraud (Kaggle 2위) | Brute-force FE (집계·차분·비율·lag 수천개) | 대규모 FE 유효성 |
| Kaggle Store Sales Forecasting | LightGBM + lag14/28 + rolling28 | 시계열 시나리오 구조 유사 |

### Kaggle Grandmaster 7대 기법 (NVIDIA 블로그)

1. 심화 EDA (Train vs Test 분포 비교)
2. 다양한 기본 모델 빠른 비교
3. **대규모 피처 엔지니어링** (수천 개)
4. Hill Climbing (Best 모델부터 점진적 앙상블)
5. **Stacking** (다단계 메타 모델)
6. Pseudo-labeling (미레이블 데이터 활용)
7. 추가 학습 (전체 데이터 + 다중 Seed)

---

---

## [PROMPT START]

### 역할 설정

당신은 Kaggle/Dacon에서 수십 회 이상 입상한 경험을 가진 머신러닝 전문가입니다.
아래에 제시된 대회 정보와 현재까지 파악된 데이터 특성을 바탕으로,
**구체적이고 실행 가능한 인사이트**를 제공해 주세요.
추상적인 제안이 아니라, 실제로 코드에 바로 적용할 수 있는 수준으로 답변해 주세요.

---

### 대회 정보

- **대회명**: 스마트 창고 출고 지연 예측 AI 경진대회 (Dacon)
- **평가지표**: MAE (Mean Absolute Error)
- **타겟**: `avg_delay_minutes_next_30m` — 향후 30분간 평균 출고 지연 시간 (분)
- **대회 기간**: 2026.04.01 ~ 2026.05.04

---

### 데이터 구조

```
Train: 250,000행 × 94컬럼
Test:   50,000행 × 93컬럼
Layout:    300행 × 15컬럼

구조: 10,000 시나리오 × 25 타임슬롯 (15분 간격 ≈ 6시간)
      (Test: 2,000 시나리오 × 25 타임슬롯)
```

**피처 그룹 (90개)**:
- 로봇 운영: `robot_active`, `robot_idle`, `robot_charging`, `robot_utilization`, `avg_trip_distance`, `task_reassign_15m`
- 배터리/충전: `battery_mean`, `battery_std`, `low_battery_ratio`, `charge_queue_length`, `avg_charge_wait`, `charge_efficiency_pct`, `battery_cycle_count_avg`
- 주문/SKU: `order_inflow_15m`, `unique_sku_15m`, `avg_items_per_order`, `urgent_order_ratio`, `heavy_item_ratio`, `cold_chain_ratio`, `sku_concentration`
- 혼잡/경로: `congestion_score`, `max_zone_density`, `blocked_path_15m`, `near_collision_15m`, `aisle_traffic_score`, `path_optimization_score`
- 환경/설비: `warehouse_temp_avg`, `humidity_pct`, `floor_vibration_idx`, `co2_level_ppm`, `hvac_power_kw`
- 인력/운영: `staff_on_floor`, `shift_hour`, `worker_avg_tenure_months`, `safety_score_monthly`
- KPI: `kpi_otd_pct`, `backorder_ratio`, `sort_accuracy_pct`, `quality_check_rate`
- 레이아웃(보조): `layout_type` (grid/hybrid/hub_spoke/narrow), `aisle_width_avg`, `charger_count`, `robot_total` 등 14개

---

### 현재까지 파악된 데이터 특성

```
타겟 분포:
  - 평균: 18.96분, 중앙값: 9.03분, 최대: 715.86분
  - 우편향 심각 (skewness = 5.68 → log1p 변환 후 0.08)
  - 0분 비율: 2.7%

결측치:
  - 86개 컬럼에 약 12% 균등 결측 (MCAR 패턴 추정)

타임슬롯 패턴:
  - ts=0: 평균 11.3분 → ts=24: 평균 21.9분 (1.9배 증가)
  - 시나리오 진행에 따라 지연이 선형 증가

타겟 상관관계 Top 5:
  1. low_battery_ratio     (|r| = 0.366)
  2. battery_mean          (|r| = 0.359)  ← 음의 상관
  3. robot_idle            (|r| = 0.349)
  4. order_inflow_15m      (|r| = 0.342)
  5. robot_charging        (|r| = 0.320)

레이아웃 타입별 평균 지연:
  hub_spoke: 22.3분  >  hybrid: 18.4분  ≈  narrow: 18.4분  >  grid: 18.1분

시나리오 간 차이:
  - 평균 지연 범위: 0 ~ 226분 (시나리오마다 매우 다름)
  - 표준편차 범위: 0 ~ 263분

현재 Baseline MAE:
  - LightGBM 5-Fold, 104 features: OOF MAE = 7.3351분
```

---

### 질문 1: 피처 엔지니어링 전략

아래 데이터 구조의 특수성을 고려하여, **구체적인 피처 엔지니어링 코드**를 제시해 주세요.

**특수성**:
- 각 시나리오는 독립적 (시나리오 간 연속성 없음)
- 시나리오 내에서 25개 타임슬롯이 시간 순서대로 연결됨
- 배터리 고갈 → 로봇 충전 → 가동 중단 → 주문 처리 지연의 연쇄 패턴 존재

**요청 사항**:
1. 시나리오 내 Lag 피처 생성 시 데이터 리크를 방지하는 올바른 코드
2. Rolling/Expanding 통계 피처 (어느 피처에, 어느 윈도우 크기를 우선 적용해야 하는지)
3. 배터리-로봇-혼잡도 간의 **도메인 기반 복합 피처** 아이디어 5가지 이상
4. 타임슬롯 순서를 표현하는 피처 (`ts_idx` 외 추가 아이디어)
5. `layout_info` 와 `train` 데이터를 조합하여 만들 수 있는 유용한 파생 피처

```python
# 피처 엔지니어링 코드 예시를 아래에 제시해 주세요
import pandas as pd
import numpy as np

KEY_COLS = ['low_battery_ratio', 'battery_mean', 'order_inflow_15m',
            'congestion_score', 'robot_idle', 'charge_queue_length',
            'max_zone_density', 'robot_charging', 'robot_utilization']

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ← 여기에 코드를 채워주세요
    return df
```

---

### 질문 2: CV 전략 및 데이터 리크 방지

이 대회에서 **올바른 교차검증(CV) 전략**을 선택하고, 그 이유를 설명해 주세요.

**선택지**:
- A) `KFold(n_splits=5, shuffle=True)` — 기본 랜덤 분할
- B) `GroupKFold(groups=scenario_id)` — 시나리오 단위 분할
- C) `TimeSeriesSplit` — 타임슬롯 순서 기반 분할
- D) 기타 (제안 있으면 설명)

**추가 질문**:
- A)와 B)를 둘 다 사용해서 CV MAE를 비교했을 때, 어느 쪽이 Public LB와 더 높은 상관관계를 보일 가능성이 높은가?
- Lag 피처를 포함했을 때 A) 방식이 데이터 리크를 유발하는 구체적 메커니즘을 설명해 주세요.

---

### 질문 3: 타겟 변환 전략

타겟(`avg_delay_minutes_next_30m`)의 skewness가 5.68인 상황에서:

1. `log1p` 변환 외에 시도할 수 있는 타겟 변환 방법과 각각의 장단점
2. MAE 최소화 목적에서 타겟 변환이 실제로 도움이 되는 이론적 근거
3. 역변환 시 발생할 수 있는 bias 문제와 보정 방법
4. 이상치(max 715분, 상위 1% > 120분)를 어떻게 처리할지 실험 설계 제안

---

### 질문 4: 모델 선택 및 앙상블

현재 LightGBM OOF MAE = 7.3351분을 기준으로:

1. 이 대회 구조(시나리오×타임슬롯, 90+ 피처, MAE 최적화)에서 LightGBM 외 추가할 모델 우선순위와 이유
2. XGBoost에서 `objective='reg:absoluteerror'`(MAE 직접 최적화)가 `'reg:squarederror'` 대비 실제로 유리한가? 언제?
3. CatBoost를 이 데이터에 적용할 때 `layout_type` 등 범주형 피처를 어떻게 처리하면 가장 효과적인가?
4. OOF 예측 기반 최적 앙상블 가중치를 구하는 구체적인 방법 (scipy.optimize 코드 포함)
5. 이 크기의 데이터(25만 행, 104피처)에서 TabNet, SAINT 등 딥러닝 모델이 GBDT를 이길 가능성은?

---

### 질문 5: 도메인 지식 기반 인사이트

AMR(자율이동로봇) 기반 스마트 창고 운영 도메인 지식을 활용해:

1. `robot_idle`이 높으면 왜 지연이 증가하는가? (로봇이 많이 쉬는데 왜 지연?) → 이 역설을 피처로 어떻게 활용할 수 있는가?
2. 배터리 고갈 → 충전 → 가용 로봇 감소 → 주문 처리 지연의 **지연 연쇄 사이클**을 피처로 모델링하는 방법
3. `hub_spoke` 레이아웃이 다른 레이아웃 대비 평균 지연이 높은 이유와, 이를 모델에서 어떻게 활용할지
4. `congestion_score`와 `max_zone_density`가 동시에 높아질 때 나타나는 **비선형 상호작용** 효과를 포착하는 피처
5. `shift_hour`와 `staff_on_floor`의 조합이 지연에 미치는 영향 (교대 교번 효과 등)

---

### 질문 6: 유사 대회/논문에서 가져올 수 있는 핵심 기법

아래 유사 사례들의 방법론을 이 대회에 맞게 **구체적으로 어떻게 적용**할 수 있는지 설명해 주세요:

1. **[arXiv 2602.20271]** Uncertainty-Aware Delivery Delay Prediction:
   - Classification-then-Regression 전략 → 이 대회에서 "지연 0분 여부를 먼저 분류(2.7%가 0분)" 후 회귀를 나누는 것이 유효한가?
   - PLR(Piecewise Linear Representation) 임베딩 → 수치형 피처에 적용 시 이점

2. **[Kaggle Grandmaster Playbook]** 대규모 FE:
   - 90개 피처 × lag/rolling/expanding 조합 시 피처가 수천 개가 될 수 있음 → 어떤 기준으로 선택/제거할 것인가?
   - Pseudo-labeling을 이 대회(시뮬레이션 데이터)에 적용하는 것이 의미 있는가?

3. **[Amex 2위 솔루션]** Brute-force FE:
   - 집계(agg), 차분(diff), 비율(ratio), lag 수천 개 생성 전략 → 어디까지 현실적인가?

4. **[물류 지연 예측 일반]** 2-step GBDT:
   - Step 1: "지연 > 30분" 여부 분류기 (이진)
   - Step 2: 지연 기간 회귀기 (연속)
   - 이 전략을 MAE 최적화에 적용 시 유효성과 구현 방법

---

### 질문 7: 함정 및 주의사항

이 대회에서 흔히 저지르는 실수와 **반드시 주의해야 할 함정**을 알려주세요:

1. 시나리오 구조에서 Lag 피처 생성 시 가장 많이 발생하는 리크 패턴 3가지
2. `KFold` CV MAE가 낮은데 Public LB가 나쁜 경우의 원인과 진단 방법
3. 타겟 로그 변환 후 역변환 시 MAE가 오히려 나빠지는 경우
4. 피처 수가 너무 많아졌을 때 LightGBM에서 발생하는 과적합 징후와 대처법
5. 시나리오 단위 집계 피처가 train/test에서 분포 차이를 일으킬 수 있는 상황

---

### 출력 형식 요청

각 질문에 대해:
- **핵심 결론** (2~3줄 요약)
- **구체적인 코드** (Python, pandas/numpy/sklearn 기반)
- **기대 효과** (MAE 개선 폭 추정 또는 근거)
- **우선순위** (즉시 시도 vs. 나중에 시도)

순서로 답변해 주세요.

## [PROMPT END]

---

## 📋 프롬프트 활용 가이드

### 전체 질문을 한 번에 보낼 때
위 `[PROMPT START]` ~ `[PROMPT END]` 를 그대로 복사하여 LLM에 입력합니다.
응답이 길어질 수 있으므로 **질문 1~3을 먼저**, **질문 4~7을 두 번째**로 분리하여 입력하는 것도 좋습니다.

### 특정 질문만 뽑아 쓸 때
`역할 설정` + `대회 정보` + `데이터 구조` + `현재까지 파악된 데이터 특성` + `원하는 질문 번호`
순으로 조합하면 됩니다.

### 실험 후 재질문 방법
실험 결과를 아래 형식으로 추가한 뒤 재질문:
```
[실험 결과 업데이트]
- Baseline MAE: 7.3351
- log1p 변환 후 MAE: X.XXXX  (개선: +X.XX%)
- ts_idx 추가 후 MAE: X.XXXX
- GroupKFold MAE: X.XXXX  (KFold 대비 괴리: X.XX분)
```

---

## 🔗 참고 자료

| 분류 | 제목 | 링크 |
|---|---|---|
| 논문 | Uncertainty-Aware Delivery Delay Prediction | [arXiv 2602.20271](https://arxiv.org/html/2602.20271) |
| 논문 | Predictive methods for delivery delays (review) | [ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S2949863525000305) |
| 논문 | ML in smart production logistics (review) | [Taylor & Francis 2024](https://www.tandfonline.com/doi/full/10.1080/00207543.2024.2381145) |
| 논문 | AMR feasibility prediction using AutoML | [Springer 2024](https://link.springer.com/chapter/10.1007/978-3-031-62684-5_36) |
| 기법 | Kaggle Grandmaster Playbook (NVIDIA) | [NVIDIA Blog](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) |
| 기법 | Grandmaster FE with cuDF (1등 전략) | [NVIDIA Blog](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/) |
| 대회 | Dacon 물류 유통량 예측 | [Dacon](https://dacon.io/competitions/official/235867/overview/description) |
| 대회 | 이 대회 토크보드 | [Dacon](https://dacon.io/competitions/official/236696/talkboard) |
| GitHub | Kaggle regression solutions 모음 | [GitHub](https://github.com/jayinai/kaggle-regression) |
