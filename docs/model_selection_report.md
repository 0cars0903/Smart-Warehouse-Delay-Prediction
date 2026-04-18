# 모델 선택 검토 보고서

> 작성일: 2026-04-12
> 작성자: Kuma × Claude
> 목적: LSTM·FT-Transformer·창고 도메인 SOTA 검토 → 다음 실험 방향 결정

---

## 배경 및 현황

| 항목 | 값 |
|---|---|
| 현재 Public 최고 | 10.3347 (`ensemble_optuna_all3.csv`) |
| CV 최고 | 8.8649 (`ensemble_ts0_LGBM_CB_XGB.csv`, Public 10.4091로 역전) |
| CV→Public 갭 | **~1.46** (지속) |
| 예측 std 압축 | 예측 ~13.8 vs 실제 ~27.4 (과소예측 문제) |
| 핵심 과제 | 갭 축소 + 극값 과소예측 해소 |

---

## 1. LSTM — 다변량 시퀀스 구조 검토

### 구조 적합성

이 데이터의 자연스러운 구조는 시퀀스입니다:

```
(300,000 rows) → reshape → (12,000 scenarios, 25 steps, n_features)
```

GBDT는 각 행을 독립 샘플로 처리하며 시계열 정보를 lag/rolling으로 수동 인코딩합니다.
LSTM은 이 시간적 의존성을 자동으로 학습합니다.

### 출력 방식 비교

| 방식 | 구조 | 특징 | 권장도 |
|---|---|---|---|
| **seq2seq** | 각 step hidden → 25개 예측 동시 | 전체 시나리오 MAE 최적화, 파라미터 효율적 | ⭐⭐⭐ |
| seq2one | 마지막 hidden → 1개 예측 | 단순하나 정보 손실 | ⭐⭐ |

seq2seq가 권장됩니다. 25 타임슬롯 전체의 오차를 한 번에 역전파하므로 시나리오 내 시간적 패턴을 더 풍부하게 학습하고, 1개 시나리오 = 1개 훈련 샘플이 되어 데이터 효율도 높습니다.

### 현실적 한계

- **12,000 시나리오는 딥러닝 기준 소규모** — Dropout, LayerNorm 등 강한 정규화 필수
- **단독으로는 GBDT 이기기 어려움** — 앙상블 다양성 확보 목적으로 활용하는 것이 현실적
- 구현 복잡도: reshape + GroupKFold를 scenario 단위로 재설계 필요

### LSTM 아키텍처 초안

```python
# 입력: (batch, 25, n_features)
# 출력: (batch, 25)  ← seq2seq

model = nn.Sequential(
    nn.LSTM(input_size=n_features, hidden_size=256,
            num_layers=2, batch_first=True,
            dropout=0.3, bidirectional=True),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 1)  # 각 step마다 적용 → squeeze → (batch, 25)
)
```

**결론**: 단독 모델보다 GBDT 앙상블(0.7) + LSTM(0.3) 블렌딩 후보로 보류.

---

## 2. FT-Transformer 계열 검토

### FT-Transformer (Gorishniy et al., NeurIPS 2021)

각 피처를 Linear 임베딩으로 토크나이징 → self-attention으로 피처 간 상호작용 학습.

```
[CLS, feat_1, feat_2, ..., feat_n] → Transformer Encoder → Regression Head
```

**이 태스크에서의 평가**:
- 300K rows (12K × 25)는 FT-Transformer 작동 범위 내
- 그러나 GBDT가 이미 트리 분할로 피처 상호작용을 효율적으로 포착 중
- 단독 성능은 GBDT와 유사하거나 열위 예상
- **앙상블 멤버로의 가치**: GBDT와 예측 오차 패턴이 달라 블렌딩 효과 기대 가능

### ★ Temporal Fusion Transformer (TFT) — 가장 적합한 구조

이 데이터 구조와 TFT의 입력 구조가 정확히 일치합니다:

| TFT 입력 타입 | 이 데이터 매핑 |
|---|---|
| **Static covariates** | 시나리오별 고정 창고 정보 (layout_type, 창고 ID 등) |
| **Known future inputs** | ts_idx (0~24), 시간대 등 |
| **Observed past** | 과거 타임슬롯의 모든 센서값 (90종) |
| **출력** | 각 ts의 avg_delay + 분위수 예측 |

**분위수 출력이 현재 과소예측 문제를 직접 해결 가능**:
- 현재: 예측 std ~13.8 vs 실제 std ~27.4
- TFT: P10~P90 분위수 출력 → 극값 범위를 명시적으로 학습

**결론**: 신경망 계열을 시도한다면 TFT를 1순위로 권장.

---

## 3. 창고/물류 도메인 SOTA 모델 정리

### 현업 실사용 모델

| 모델 | 주요 사용처 | 특징 |
|---|---|---|
| **LightGBM/XGBoost 앙상블** | 아마존, 쿠팡 배송 예측 | 실시간 추론, 해석 용이, 현재 방향 |
| **Temporal Fusion Transformer** | Google, Tesla 공급망 | 다중 horizon, 분위수 출력 |
| **Prophet + GBDT 잔차 보정** | Meta 내부 수요 예측 | 추세/계절성 분리 후 ML 보정 |
| **GNN (Graph Neural Network)** | 창고 레이아웃 최적화 | 존 간 공간 의존성 모델링 (layout_info 활용 가능) |
| **Simulation-Calibrated ML** | 대형 물류 시뮬레이터 보정 | 이 대회와 구조적으로 가장 유사 |

### 시계열 SOTA (2023~2024)

| 모델 | 발표 | 핵심 아이디어 | 이 태스크 적합도 |
|---|---|---|---|
| **iTransformer** (ICLR 2024) | 시간축/채널축 역전환 attention | 다변량 예측에서 SOTA, 구현 용이 | ⭐⭐⭐ |
| **TimesNet** (ICLR 2023) | 1D 시계열 → 2D 변환 후 CNN | 짧은 시퀀스(25 steps)에 적합 | ⭐⭐⭐ |
| **DLinear / NLinear** (AAAI 2023) | 트렌드/잔차 분해 후 선형 | 경쟁력 있는 단순 베이스라인, 빠른 검증 | ⭐⭐⭐ |
| **PatchTST** (ICLR 2023) | 시계열을 patch로 분할 후 Transformer | 긴 의존성 포착에 유리 | ⭐⭐ |
| **MOIRAI** (Salesforce, 2024) | Foundation model for time series | 제로샷/fine-tune 가능 | ⭐⭐ |

---

## 4. 신경망 도입 로드맵 (우선순위)

현재 일반화 개선(갭 축소)이 최우선입니다. 신경망은 갭이 안정된 이후 단계에서 시도합니다.

```
Phase 1: 일반화 개선 (현재)
  ├─ sqrt 변환 실험 (log1p 대비 Δ0.0120, 과소예측 보정 가능성)
  ├─ 피처 중요도 하위 10% 제거 (296개 → ~260개)
  └─ 앙상블 가중치 재조정 (XGBoost 완전 제외 검토)

Phase 2: 신경망 다양성 (갭 < 1.3 달성 후)
  ├─ [빠른 검증] DLinear / NLinear — 구현 10분, 성능 체크
  ├─ [구조적 접근] iTransformer — 다변량 25-step에 최적
  └─ [최적 블렌딩] GBDT×0.7 + 신경망×0.3

Phase 3: 고도화 (여유 시 / 대회 후반)
  └─ TFT — 분위수 출력으로 극값 과소예측 근본 해결
```

---

## 5. 결론 및 판단

| 모델 | 단독 성능 예상 | 앙상블 기여 | 구현 난이도 | 우선순위 |
|---|---|---|---|---|
| 현재 GBDT 앙상블 | ✅ 최고 | — | — | 현재 집중 |
| DLinear | GBDT 근접 가능 | 높음 (패턴 다름) | 낮음 | Phase 2 선두 |
| iTransformer | GBDT와 유사 | 높음 | 중간 | Phase 2 |
| seq2seq LSTM | GBDT 열위 가능 | 중간 | 높음 | Phase 2 후순위 |
| TFT | GBDT와 유사~우위 | 높음 | 높음 | Phase 3 |

**핵심 판단**: CV→Public 갭 ~1.46이 해소되지 않은 상황에서 복잡한 신경망 도입은 갭을 오히려 키울 위험이 있습니다. 먼저 GBDT 기반 일반화 개선으로 갭을 1.2 이하로 낮춘 뒤, 신경망을 앙상블 다양성 목적으로 추가하는 것이 최적 전략입니다.
