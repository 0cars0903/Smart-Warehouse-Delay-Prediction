# v3 전략 보고서: 시퀀스 모델 하이브리드 접근 (전략 E 분석)

> 작성일: 2026-04-17 | 기준: v2.0 (모델실험21, CV 8.5097 / Public 9.9550 / 28위)

---

## 1. 전략 E 핵심 가설

시나리오가 진행됨에 따라(ts 0→24) 피처들의 평균·표준편차·기울기가 **점진적으로 변화**하며,
이 궤적(trajectory)의 형태가 최종 출고 지연을 결정한다.
LSTM/1D-CNN 같은 시퀀스 모델이 이 시계열 동태를 자연스럽게 포착할 수 있으며,
기존 GBDT 스태킹과 **직렬 또는 병렬로 결합**하여 다양성을 확보한다.

---

## 2. 정량 분석 결과

### 2.1 시퀀스 모델이 유리한 근거 (긍정 신호)

| 지표 | 값 | 해석 |
|---|---|---|
| **타겟 자기상관 (lag-1)** | **0.8671** | 이전 타임슬롯 타겟이 현재를 강하게 결정 |
| **Within잔차 자기상관** | **0.6246** | 시나리오 평균 제거 후에도 시계열 구조 뚜렷 |
| **시나리오 내 타겟 분산 비율** | **36.6%** (273.56/748.10) | 전체 분산의 1/3 이상이 시나리오 내 시계열 변동 |
| **피처 자기상관 (lag-1)** | 0.86~0.99 (NaN 제거 후) | 피처가 시계열 연속성 강함 — 순서 정보 유효 |
| **비선형 교호작용** | congestion ΔR²=+0.045 | 다항식 기반 비선형 관계 존재 |
| **타겟 비선형성** | y(t-1,t-2)→y(t) ΔR²=+0.014 | 선형 lag 모델 대비 비선형 이득 확인 |
| **Volatile 시나리오 비율** | **73.8%** (7,377/10,000) | 대다수 시나리오가 비정형 변동 → 궤적 패턴 중요 |

**핵심**: Within잔차 자기상관 0.6246은 매우 강한 시계열 신호다.
v2의 시나리오 집계(mean/std/max/min/diff)는 **정적 요약 통계**에 불과하고,
"ts=5에서 congestion이 급등 후 ts=10에서 안정화"와 같은 **동적 궤적**은 포착하지 못한다.

### 2.2 주의해야 할 제약 (부정 신호)

| 지표 | 값 | 해석 |
|---|---|---|
| **독립 시퀀스 수** | 10,000개 (train) | LSTM 학습에 최소 수준 |
| **LSTM 128 파라미터 비율** | 11.2× (112K / 10K) | 과적합 위험 높음 |
| **Expanding 피처 추가 R²** | +0.011 (선형 기준) | 선형 모델에서는 궤적 정보 기여 미미 |
| **앞 N개 ts → 시나리오 평균 타겟** | N=1: R²=0.268 → N=25: R²=0.336 | 시퀀스 길이 증가의 한계 수익 체감 |
| **NaN 비율** | ~12% (주요 피처) | 시퀀스 내 산재 — 결측 처리 전략 필수 |

**핵심 제약**: 선형 기준으로 expanding 피처의 추가 설명력은 +1.1%뿐이다.
그러나 이는 **선형 모델의 한계**이며, 비선형 시퀀스 모델(LSTM/CNN)은
교호작용·전이점(regime change)·누적 효과를 포착할 수 있어 실제 기여는 더 클 수 있다.

### 2.3 타임슬롯별 동태 심층 분석

```
타겟 평균 추세 (ts 0→24):
  ts= 0: mean=11.29, std=10.98, median=8.00
  ts=12: mean=19.59, std=26.52, median=9.45  (평균 ↑73%, median ↑18%)
  ts=24: mean=21.86, std=36.33, median=9.82  (평균 ↑94%, median ↑23%)

→ 평균은 꾸준히 상승하지만 중앙값은 완만 → 소수 시나리오의 극단적 상승이 평균을 끌어올림
→ 시퀀스 모델이 "극단 상승 시나리오"의 궤적 패턴을 학습하면 큰 이득

Within잔차 구조:
  ts= 0: mean=-7.68, std=18.03  (초반은 시나리오 평균 대비 낮은 경향)
  ts=12: mean=+0.63, std=13.03  (중반 안정)
  ts=24: mean=+2.90, std=23.21  (후반 갈수록 편차 확대)

→ 시간에 따른 체계적 잔차 패턴 존재 → 시퀀스 모델이 이 시간 의존성을 포착 가능
```

### 2.4 시나리오 전반부 기울기 → 후반부 타겟

```
slope(robot_utilization) → late_target:  r=-0.3872  ← 강한 음의 상관
slope(robot_idle)        → late_target:  r=-0.1527
changes(robot_util) → target_std:       r=+0.2371  ← 궤적 변동성 → 예측 난이도
```

robot_utilization이 초반에 급등하는 시나리오일수록 후반 지연이 심해지는 강한 패턴.
이런 "초반 궤적 → 후반 결과" 관계는 LSTM의 hidden state가 자연스럽게 축적하는 정보다.

---

## 3. 전략 E 판정: ⚠️ 조건부 유효

**단독 LSTM 예측기로는 비효과적**이지만,
**GBDT 스태킹의 추가 base learner (시퀀스 임베딩 생성기)**로 결합하면 유효하다.

| 접근 방식 | 판정 | 이유 |
|---|---|---|
| LSTM 단독 예측 | ❌ 비추천 | 10K 시퀀스로 112K 파라미터 학습 = 과적합 필연 |
| LSTM 시나리오 임베딩 → GBDT | ⚠️ 위험 중간 | 과적합 제어 어렵고, v1 MLP 실패(11-A/B) 전례 |
| **1D-CNN 경량 임베딩 → GBDT** | **✅ 추천** | 파라미터 2.9× (28K), 위치 불변 필터, 과적합 상대적 안전 |
| **1D-CNN + LSTM 병렬 → GBDT** | **✅ 최종 추천** | 다양성 극대화, 두 모델 모두 경량으로 제한 |

---

## 4. v3 구체 아키텍처: Hybrid Sequence-Tabular Stacking

### 4.1 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT (per scenario)                      │
│  X_tab: (25, ~90) 원본 피처 + 시나리오 집계 (v2 FE)             │
│  X_seq: (25, ~20) 핵심 연속형 피처 (정규화, NaN 보간)            │
└──────────┬───────────────────────┬──────────────────────────────┘
           │                       │
    ┌──────▼──────┐        ┌───────▼────────┐
    │  Tabular    │        │  Sequence      │
    │  Branch     │        │  Branch        │
    │             │        │                │
    │ LGBM (OOF)  │        │  1D-CNN (k=3,5)│ ← 병렬 필터
    │ TW1.8 (OOF) │        │  BiLSTM (h=32) │ ← 경량 순환
    │ CB   (OOF)  │        │                │
    │ ET   (OOF)  │        │  Output:       │
    │ RF   (OOF)  │        │  per-step pred │ ← 각 ts별 예측값
    │             │        │  (25,1)        │
    └──────┬──────┘        └───────┬────────┘
           │                       │
           │  5개 OOF 예측값        │  시퀀스 모델 OOF 예측값
           │  (LGBM,TW,CB,ET,RF)   │  (CNN_pred, LSTM_pred)
           │                       │
    ┌──────▼───────────────────────▼──────┐
    │         Meta Learner (LGBM)          │
    │  Input: 7차원 (5 tabular + 2 seq)    │
    │  GroupKFold, early stopping           │
    └──────────────────┬──────────────────┘
                       │
                  Final Prediction
```

### 4.2 시퀀스 브랜치 상세 설계

#### A. 1D-CNN (주력)

```python
# 구조: 2-layer 1D-CNN, 파라미터 ~30K
Input: (batch, 25, n_feat)  # n_feat ≈ 20 (핵심 피처만)

Conv1D(filters=32, kernel_size=3, padding='same') → ReLU → Dropout(0.3)
Conv1D(filters=32, kernel_size=5, padding='same') → ReLU → Dropout(0.3)
Dense(1)  # per-timestep 출력

# 총 파라미터: ~30K (10K 시퀀스 대비 3.0×)
# 장점: 위치 불변 필터 → 시나리오 간 전이(transfer) 용이
```

#### B. Bidirectional LSTM (보조)

```python
# 구조: 1-layer BiLSTM, 파라미터 ~20K
Input: (batch, 25, n_feat)  # 동일 입력

BiLSTM(hidden=16, return_sequences=True) → Dropout(0.3)
Dense(1)  # per-timestep 출력

# 총 파라미터: ~20K (hidden=16으로 극도 경량화)
# 장점: 양방향 → 미래 정보도 활용 (test에서 25행 전부 관측)
# Bidirectional 근거: test에서 미래 타임슬롯 피처가 모두 제공됨
```

#### C. 입력 피처 선정 (20종)

시퀀스 모델 입력은 **자기상관이 높고 NaN이 적은** 피처만 선별:

```python
SEQ_FEATURES = [
    # 자기상관 > 0.90, NaN 0%
    'robot_utilization', 'robot_idle', 'robot_active',
    # 자기상관 > 0.85, NaN ~12% (보간 필요)
    'order_inflow_15m', 'congestion_score', 'low_battery_ratio',
    'battery_mean', 'charge_queue_length', 'max_zone_density',
    # 시나리오 집계 diff (시점별 이탈도)
    'sc_robot_utilization_diff', 'sc_congestion_score_diff',
    'sc_order_inflow_15m_diff', 'sc_low_battery_ratio_diff',
    'sc_battery_mean_diff',
    # 타임슬롯 위치
    'ts_idx', 'ts_sin', 'ts_cos',
    # Layout 정보 (정적, broadcast)
    'num_robots_total', 'num_chargers', 'total_sku_types',
]
```

#### D. NaN 처리 전략

시퀀스 모델은 NaN을 직접 처리할 수 없으므로:

```python
# 1순위: 시나리오 내 선형 보간 (시계열 연속성 유지)
df[col] = df.groupby('scenario_id')[col].transform(
    lambda x: x.interpolate(method='linear', limit_direction='both'))

# 2순위: 시나리오 평균으로 채움 (보간 불가 시)
df[col] = df.groupby('scenario_id')[col].transform(
    lambda x: x.fillna(x.mean()))

# 3순위: 전역 0 (극히 드문 경우)
```

#### E. 과적합 방지 핵심 설계

v1에서 MLP가 실패한 교훈 (sklearn MLP 11-A: 과소학습, 11-B: 과적합):

| 제어 수단 | 설정 | 이유 |
|---|---|---|
| **Hidden 크기** | LSTM=16, CNN=32 | 파라미터 50K 이내 엄수 |
| **Dropout** | 0.3 (각 레이어) | 시퀀스 모델 표준 |
| **Early stopping** | patience=10 (val MAE) | GroupKFold val 기준 |
| **피처 정규화** | StandardScaler (fold별) | 스케일 민감한 모델 필수 |
| **학습률** | 1e-3 → ReduceOnPlateau | 안정적 수렴 |
| **Epoch 상한** | 100 | MLP 11-B 실패 방지 |
| **L2 정규화** | 1e-4 | 가중치 폭발 방지 |

---

## 5. v3 실행 계획

### Phase 1: 시퀀스 모델 단독 검증 (1일)

**목표**: 1D-CNN / BiLSTM 단독 OOF MAE 확인

```
실험 22-A: 1D-CNN (20 피처, h=32, k=3+5)
  - 예상 OOF MAE: 9.0~9.5 (LGBM 8.62보다 낮을 것)
  - 핵심 확인: 기존 5모델과의 상관 < 0.95 여부

실험 22-B: BiLSTM (20 피처, h=16, bidirectional)
  - 예상 OOF MAE: 9.2~9.8
  - 핵심 확인: CNN과의 상관 < 0.98 (독립성)
```

**성공 기준**: 기존 LGBM OOF와의 상관이 **0.95 이하**이면 스태킹 기여 유망.
(참고: v2에서 LGBM-ET 상관 0.9661이 가장 큰 다양성 기여)

### Phase 2: 하이브리드 스태킹 (1일)

**목표**: 7모델 스태킹 (5 tabular + 2 sequence) CV 확인

```
실험 23: Hybrid Stacking v1
  - Base: LGBM + TW1.8 + CB + ET + RF + CNN + BiLSTM
  - Meta: LGBM (GroupKFold 5-fold)
  - 기대 CV: 8.45~8.50 (v2 대비 -0.05~-0.06)
```

**성공 기준**: CV 8.48 이하 + Public 배율 1.170 이하

### Phase 3: 아키텍처 최적화 (1~2일)

Phase 2 결과에 따라:

```
실험 24-A: 시퀀스 모델 Optuna 튜닝 (hidden, dropout, lr)
실험 24-B: 피처 조합 탐색 (시퀀스 입력 피처 수 10/15/20/30)
실험 24-C: Attention 메커니즘 추가 (Self-Attention → 어떤 타임슬롯이 중요한지 학습)
실험 24-D: 시퀀스 임베딩 추출 → Tabular 피처로 직접 합류 (스태킹 우회)
```

### Phase 4: 최종 앙상블 + 제출 (0.5일)

```
- 최적 하이브리드 모델 확정
- Public LB 제출, 배율 확인
- Private 제출 2개 선택 (v2 최고 + v3 최고)
```

---

## 6. 위험 분석 및 대안

### 6.1 최대 위험: 시퀀스 모델 과적합 → 배율 악화

v1 MLP 실패 전례에서 핵심 교훈:
- GroupKFold에서 시나리오 분리 시, lag/rolling 피처가 시나리오 고유 패턴을 암기
- 트리(ET/RF)는 분포 차이에 강건했지만 MLP는 취약

**대응**: 시퀀스 모델 입력에서 lag/rolling 피처를 **제외**하고 원본 피처만 사용.
시계열 순서 정보는 모델 구조(LSTM/CNN) 자체가 학습하므로 중복 인코딩 불필요.

### 6.2 차선 전략: 시퀀스 모델 실패 시

시퀀스 모델이 상관 0.95 이상(다양성 부족)이면:

**전략 A 복귀**: 시나리오 집계 피처 고도화
- 형상 피처: trend slope, 변곡점 수, peak 위치, Q25/Q75, skewness
- 구간 피처: first5 vs last5 비율, 최대 연속 상승/하락 구간 길이
- 이벤트 피처: 임계값 초과 타임슬롯 수 (e.g., congestion > 0.8인 구간)

이 방향은 v2 아키텍처 그대로 유지하며 피처만 추가하는 안전한 경로.

### 6.3 위험-수익 매트릭스

| 전략 | 기대 CV 개선 | 과적합 위험 | 구현 난이도 | 추천도 |
|---|---|---|---|---|
| **E: 하이브리드 시퀀스** | -0.05~-0.10 | 중간 | 높음 | ⭐⭐⭐⭐ |
| A: 형상 피처 | -0.02~-0.05 | 낮음 | 낮음 | ⭐⭐⭐ |
| C: 스태킹 구조 변경 | -0.01~-0.03 | 낮음 | 중간 | ⭐⭐ |
| D: 배율 공략 | ±0 | 낮음 | 중간 | ⭐⭐ |

---

## 7. 핵심 수치 요약

```
현재 최고:        CV 8.5097 / Public 9.9550 / 28위 (1위 대비 +0.26)
v3 기대 (보수적): CV 8.46   / Public 9.88   → ~20위
v3 기대 (낙관적): CV 8.40   / Public 9.80   → ~15위
1위 도달 필요:    Public 9.70 → CV 8.34 (배율 1.163 가정) 또는 CV 8.29 (배율 1.170 가정)
```

**결론**: 전략 E 하이브리드 시퀀스 접근은 **이론적 근거가 충분하며 실행 가치가 있다.**
다만 단독 LSTM이 아닌 **경량 시퀀스 모델(1D-CNN + BiLSTM) → GBDT 메타 학습기** 구조가 핵심이고,
Phase 1에서 상관 < 0.95 확인 후에만 Phase 2로 진행하는 gate를 두어야 한다.
