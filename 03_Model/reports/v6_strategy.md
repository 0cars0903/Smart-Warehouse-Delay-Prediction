# v6 공식 전략 — 궤적 형상(Trajectory Shape) 피처

> 작성일: 2026-04-25 | 기준: blend_m33m34_w80 (Public 9.8073, 배율 1.1564)
> 마감: 2026-05-04 (잔여 9일) | 코드+PPT: 2026-05-07

---

## 1. 왜 v6가 필요한가

### 1-1. v5 전패의 공통 원인

v5에서 시도한 6전략(메가블렌드/CB메타/피처선택/Pseudo-label/멀티시드/KNN후처리)은 모두 **동일 파이프라인 내 변형**이었다. 공통 실패 원인은 하나다.

> blend_w80이 현재 피처셋+모델구조 조합에서 **이미 로컬 최적점에 도달**해 있으며,  
> 같은 공간 안에서의 미세 조정으로는 갭 0.108을 돌파할 수 없다.

### 1-2. 현재 피처셋의 결정적 공백

v2 이후 시나리오 집계 피처(sc_agg)가 핵심 역할을 했다. 그러나 현재 sc_agg가 표현하는 것은 **정적 분포 통계**뿐이다.

| 현재 sc_agg (11종 × 18컬럼 = 198피처) | 상태 |
|---|---|
| mean / std / max / min / diff | ✅ 구현됨 |
| median / p10 / p90 | ✅ 구현됨 |
| skew / kurtosis / cv | ✅ 구현됨 |
| **slope (시간에 따른 추세)** | ❌ **미구현** |
| **first5/last5 비율 (성장 방향)** | ❌ **미구현** |
| **peak 위치 (argmax 타임슬롯)** | ❌ **미구현** |
| **임계값 초과 횟수 (이벤트 빈도)** | ❌ **미구현** |
| **단조증가 비율 (추세 일관성)** | ❌ **미구현** |

정적 통계는 "congestion 평균이 얼마인가"를 알려주지만, "언제 급등했고 얼마나 지속됐는가"를 알지 못한다. 바로 이 **시간 축 위의 궤적 정보**가 v6의 탐색 공간이다.

---

## 2. 핵심 가설과 증거

### 2-1. 이미 확인된 증거 (v3 분석, 2026-04-17)

```
slope(robot_utilization) → late_target:  r = -0.387  ★★★ 강한 상관
slope(robot_idle)        → late_target:  r = -0.153
changes(robot_util)      → target_std:   r = +0.237   (궤적 변동성 → 예측 난이도)
```

- robot_utilization이 **초반에 급등하는 시나리오**일수록 후반 지연이 심해진다.
- 이 "초반 기울기 → 후반 결과" 관계는 mean/std로는 절대 포착되지 않는다.
- 평균이 동일해도 **0→10→0** vs **3→5→7** 궤적은 지연 패턴이 전혀 다르다.

### 2-2. 핵심 가설

> 극값 시나리오(target ≥ 50)의 45.1% MAE 기여는 기존 피처가 해당 시나리오를  
> 일반 시나리오와 충분히 구분하지 못하기 때문이다. 궤적 형상 피처는  
> 이 구분 능력을 base learner 레벨에서 향상시킨다.

극값 시나리오 특성(axis3 분석):
- order_inflow_15m: 극값 152 vs 일반 68 (+1.09σ) ← **단순 평균도 차이 크지만 '언제 급등했는가'가 더 중요**
- congestion_score: 극값 19.1 vs 일반 3.1 (+0.82σ)
- robot_idle: 극값 11 vs 일반 34 (-1.03σ)

---

## 3. v6 피처 설계

### 3-1. 대상 컬럼 (TRAJ_COLS)

```python
TRAJ_COLS = [
    'robot_utilization',    # slope r=-0.387 확인
    'order_inflow_15m',     # 극값 시나리오 최강 구분자
    'congestion_score',     # 극값 시나리오 2위
    'low_battery_ratio',    # 극값 시나리오 3위
    'battery_mean',         # 핵심 상관 피처
    'charge_queue_length',  # 배터리 위기와 연동
    'robot_idle',           # 역방향 신호
    'max_zone_density',     # 공간 압박
]
```

### 3-2. 피처 카테고리별 설계

#### Category A: Slope (선형 추세) — 8피처
```python
# scipy 없이 np.polyfit으로 구현 (속도 우선)
slope_feat = sc_agg.apply(
    lambda x: np.polyfit(np.arange(len(x)), x.values, 1)[0]
)
# → sc_{col}_slope: 시나리오 전체 25ts에 걸친 선형 기울기
# 양수: 시나리오 후반으로 갈수록 증가 / 음수: 감소
```

**왜 slope인가**: 피처가 "15분 후 지연"을 예측하므로, 피처 자체가 증가 추세에 있는 시나리오는 지금보다 상황이 나빠지고 있다는 신호다.

#### Category B: First5/Last5 비율 — 8피처
```python
# 시나리오의 초반(ts 0-4) vs 후반(ts 20-24) 평균 비율
first5_mean = df[df['ts_idx'] < 5].groupby('scenario_id')[col].mean()
last5_mean  = df[df['ts_idx'] >= 20].groupby('scenario_id')[col].mean()
fl_ratio    = last5_mean / (first5_mean.abs() + 1e-8)
# → sc_{col}_fl_ratio: 1보다 크면 악화 추세
```

**왜 fl_ratio인가**: slope보다 노이즈에 강건하다. 전역 선형 추세는 중간 spike를 무시하지만, fl_ratio는 "결국 어디로 갔는가"를 직접 포착한다.

#### Category C: Peak Position (극값 위치) — 5피처
```python
# congestion, order_inflow, low_battery, charge_queue, max_zone_density
peak_ts = df.groupby('scenario_id')[col].transform('idxmax')
# idxmax를 ts_idx로 변환 후 /24 정규화
# → sc_{col}_peak_pos: 0=초반 peak, 1=후반 peak
```

**왜 peak_pos인가**: 후반부에 혼잡/주문이 피크를 찍는 시나리오는 예측 시점에 이미 악화 중이므로 더 높은 지연을 유발한다.

#### Category D: Threshold Crossing Count (이벤트 빈도) — 5피처
```python
# 시나리오 내 해당 피처가 시나리오 평균+0.5σ를 초과한 타임슬롯 수
threshold = sc_mean + 0.5 * sc_std
above_count = df.groupby('scenario_id').apply(
    lambda g: (g[col] > threshold.loc[g.name]).sum()
)
# → sc_{col}_above_count: 고부하 이벤트 빈도 (0~25)
```

**왜 threshold count인가**: 한 번의 spike도 같은 count로 잡히는 점이 약점이지만, "지속적으로 높은 부하"를 카운팅하는 것은 mean과 보완 관계다.

#### Category E: Monotonicity Score — 3피처
```python
# 연속 증가 비율 (robot_utilization, congestion_score, order_inflow_15m만)
def monotonicity(x):
    diffs = np.diff(x.values)
    return (diffs > 0).sum() / len(diffs)  # 0~1, 1=완전 단조증가

mono = df.groupby('scenario_id')[col].apply(monotonicity)
# → sc_{col}_mono: 단조증가 비율
```

**왜 monotonicity인가**: 변동이 크더라도(std 높더라도) 방향이 일관되게 나빠지는 시나리오를 포착한다. std는 이 방향성을 무시한다.

### 3-3. 피처 요약

| 카테고리 | 피처 수 | 근거 |
|---|---|---|
| A: slope | 8 | r=-0.387 직접 확인 |
| B: fl_ratio | 8 | slope보다 노이즈 강건 |
| C: peak_pos | 5 | 극값 구간 시점 포착 |
| D: above_count | 5 | 이벤트 빈도 (mean과 보완) |
| E: monotonicity | 3 | 방향 일관성 |
| **합계** | **29** | model31(429) + 29 = **458 피처** |

### 3-4. Shift-Safe 설계 원칙

v6 피처는 모두 **시나리오 전체 25행을 기반으로 계산**하여 각 행에 broadcast한다.  
이는 v2의 sc_agg와 동일한 설계이며, test에서도 시나리오 25행이 전부 제공되므로 **데이터 리크 없음**.

```
학습 시: scenario_id 기준 GroupKFold → val fold의 시나리오 전체 사용
예측 시: test 시나리오도 25행 전부 visible → 동일한 방식으로 계산
```

---

## 4. 실험 계획 (3-Phase, 9일)

### Phase 1 — model41: 궤적 FE 통합 (04.25~04.26)

**목표**: 29개 궤적 피처 + model31 베이스 → CV/Public 확인

```
실행: python src/run_model41_traj_fe.py
예상 시간: ~20분 (ET/RF만 재학습, LGBM/CB/TW ckpt 재사용 시도)
모니터링 지표:
  - CV MAE: 목표 < 8.47 (model31 8.4786 대비 개선)
  - pred_std: 목표 ≥ 15.5 (model31 15.89 유지)
  - 신규 피처 importance 상위권 진입 여부
```

**성공 기준**:
- CV 개선 OR pred_std 증가 중 하나 충족 → 제출
- 둘 다 악화 → Phase 1B (절반 피처만 사용)

**Phase 1B (contingency)**: A/B/C 카테고리만 사용 (16피처, 가장 이론 근거 강한 것만)

### Phase 2 — model42: 선택적 확장 + Optuna (04.27~04.28)

Phase 1 결과에 따라:

| Phase 1 결과 | Phase 2 행동 |
|---|---|
| CV ↓ + Public ↓ (이상적) | Optuna LGBM+CB 재튜닝 (30 trials) |
| CV ↓ + Public ↑ (배율 악화) | 피처 중요도 하위 10개 제거 후 재실험 |
| CV ↑ + Public ↓ (model29A 패턴) | 그대로 제출, 블렌드 시도 |
| CV ↑ + Public ↑ (둘 다 악화) | Phase 2 skip, Phase 3로 이동 |

### Phase 3 — 블렌드 + 최종 제출 (04.29~05.01)

```python
# 최적 블렌드 조합 탐색
# blend_m33m34_w80 (현재 최고) × model41 or model42

candidates = {
    'model41_w20': blend(m41, blend_w80, w=[0.2, 0.8]),
    'model41_w30': blend(m41, blend_w80, w=[0.3, 0.7]),
    'model41_w50': blend(m41, blend_w80, w=[0.5, 0.5]),
}
# OOF 기반 최적 비율 선택 후 제출 1~2회
```

### Phase 4 — 버퍼 + PPT 준비 (05.02~05.04)

```
05.02~03: 최종 제출 2~3개 전략 (모두 다 실패해도 blend_w80 유지)
05.04: 대회 마감 — 최고 Public 제출로 최종 확정
05.05~07: PPT 작성 (코드+PPT 제출 마감)
```

---

## 5. 일별 액션 플랜

| 날짜 | 주요 작업 | 담당 | 완료 기준 |
|---|---|---|---|
| **04.25 (토, 오늘)** | 전략 문서 + model41 스크립트 작성 | Claude | `src/run_model41_traj_fe.py` 저장 |
| **04.26 (일)** | model41 실행 + 결과 분석 + 제출 | USER 실행 | CV/Public 확인 |
| **04.27 (월)** | 팀 병합 마감 (솔로 유지 결정) + Phase 2 방향 확정 | USER | |
| **04.28 (화)** | model42 (Optuna 재튜닝 또는 피처 정제) | Claude+USER | 스크립트 + 실행 |
| **04.29 (수)** | model42 제출 + 블렌드 조합 탐색 | Claude+USER | 제출 1회 |
| **04.30 (목)** | 최적 블렌드 확정 + 예비 제출 | USER | 제출 1~2회 |
| **05.01 (금)** | 최종 모델 후보 3개 결정 | Claude | OOF 기반 순위 |
| **05.02~03** | Private용 최종 2개 선택 (Public 최고 + CV 최고) | USER | 최종 제출 |
| **05.04 (월)** | 대회 마감 확인 | USER | |

---

## 6. 위험 분석 및 대응

### 위험 1: 궤적 피처가 노이즈로 작용 (CV 악화)

- **가능성**: 중간 (model29A 패턴 — CV 악화에도 Public 개선 사례 있음)
- **대응**: 즉시 제출. pred_std ≥ 15.5이면 배율 개선 가능성 있음

### 위험 2: 배율 악화 (pred_std 압축)

- **판단 기준**: pred_std < 15.0이면 배율 악화 확실 → 제출 보류
- **대응**: Phase 1B로 피처 절반 제거 후 재실험

### 위험 3: 전략 자체 실패 (v5처럼 전패)

- **failsafe**: blend_m33m34_w80 (9.8073)은 항상 유지
- **대응**: 05.01 이후는 신규 실험 중단, 기존 최고 제출로 확정

### 위험 4: slope 계산 NaN (ts 수 < 2인 시나리오)

- **실제로는 불가**: 모든 시나리오가 25 ts → 안전
- **방어 코드**: `fillna(0)` 추가

---

## 7. 성공 지표 정의

| 지표 | 기준값 | 목표값 | 달성 시 |
|---|---|---|---|
| CV MAE | 8.4786 (model31) | < 8.47 | 즉시 제출 |
| Public LB | 9.8073 (blend_w80) | < 9.79 | 역대 최고 갱신 |
| pred_std | 15.89 (model31) | ≥ 15.5 | 배율 정상 |
| 1위 갭 | 0.108 | < 0.09 | 순위 상승 |

---

## 8. 최종 결과 (2026-04-25 실험 완료)

| 지표 | 실제값 | 목표값 | 판정 |
|---|---|---|---|
| CV MAE | 8.4851 | < 8.47 | ❌ +0.0065 악화 |
| Public LB | **9.8449** | < 9.79 | ❌ model31(9.8255)보다 악화 |
| pred_std | 15.73 | ≥ 15.5 | ⚠️ 통과이나 압축 (model31 15.89) |
| 1위 갭 | — | < 0.09 | ❌ 미달 |

**실험 수치:**
- LGBM OOF 8.5486, CB OOF 8.6204, TW15 OOF 8.7857, ET OOF 9.0978, RF OOF 9.2612, Asym20 OOF 8.7749
- 가장 낮은 상관: LGBM-RF 0.8850
- Fold: 8.4118 / 8.5326 / 8.0686 / 8.9226 / 8.4900
- 배율: 9.8449 / 8.4851 = **1.1602** (model31 1.1589보다 높음 = 일반화 악화)

**실패 원인 분석:**
- sc_agg(mean/std/max/min 등)가 이미 시나리오 분포를 충분히 표현하고 있어 궤적 피처가 **새로운 정보 없이 중복 차원만 추가**
- CV 악화 + pred_std 압축이 동시 발생 → model29A 패턴(CV 악화에도 배율 개선) 미재현
- 트리 모델은 방향성(slope, mono)보다 수치 통계에 민감 → 궤적 shape 정보가 트리 분기에 효과적으로 활용되지 않음

**v6 방향 최종 종결. 최고 기준 blend_w80 = 9.8073 유지.**

---

## 8. 이론적 배경 요약

```
현재 파이프라인의 표현력 한계:
  sc_agg (11통계) → "시나리오가 평균적으로 얼마나 극단적인가" ✅
  비율 피처 (12종) → "창고 구조 대비 부하가 얼마나 큰가" ✅
  Asym/TW loss  → "극값 방향으로 예측을 편향" ✅
  
  궤적 형상 피처 → "시나리오가 어떤 방향으로 움직이고 있는가" ❌ (미구현)

v6의 주장:
  동일한 mean/std를 가진 두 시나리오가 있을 때,
  "악화 중인 시나리오"와 "안정적인 시나리오"를 구분하는
  유일한 신호가 slope와 fl_ratio와 peak_pos다.
  이 구분이 극값 구간의 45% MAE 기여를 줄이는 핵심 경로다.
```
