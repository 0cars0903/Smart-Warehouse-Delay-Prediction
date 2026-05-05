# 스마트 창고 출고 지연 — 도메인 지식 정리
### 대상: 제조업 생산관리자 3년차 수준 | ML 대회 참가용

> 제조공장 경험을 갖고 있다면 이미 절반은 알고 있다.
> "설비 = 로봇", "생산라인 = 피킹 경로", "재공품 적체 = 혼잡도", "납기 지연 = 출고 지연"으로 치환하면 된다.

---

## 1. 스마트 창고의 구조 — 공장 레이아웃과 비교

### 1-1. AMR(자율이동로봇)이란?

AMR(Autonomous Mobile Robot)은 창고 바닥을 스스로 돌아다니며 물건을 옮기는 로봇이다.
공장으로 치면 **AGV(무인반송차)** 또는 **컨베이어 시스템**에 해당하지만, 고정 경로가 없고 장애물을 피해 스스로 경로를 결정한다.

```
공장 개념          →    창고 AMR 개념
────────────────────────────────────────
AGV / 지게차       →    AMR (robot_active)
설비 고장/점검     →    충전 중 (robot_charging)
설비 대기          →    유휴 (robot_idle)
설비 가동률        →    robot_utilization
이동 거리/사이클   →    avg_trip_distance
```

### 1-2. 창고 레이아웃 4종

이 대회에는 `layout_type` 컬럼에 4가지 레이아웃이 있다.

| 레이아웃 | 구조 | 특징 | 평균 지연 | 공장 비유 |
|---|---|---|---|---|
| **grid** | 격자형 통로 | 경로 선택지 多, 혼잡 분산 | 18.1분 (最低) | 직선형 생산라인 |
| **hybrid** | 혼합형 | 중간 수준 | 18.4분 | 셀 생산 + 라인 혼합 |
| **narrow** | 좁은 통로 | 로봇 교행 불가, 병목 잦음 | 18.4분 | 단일 통로 공정 |
| **hub_spoke** | 중앙 허브 + 방사형 | 허브 병목 발생 시 전체 지연 | 22.3분 (最高) | 중앙 창고 → 각 라인 공급 방식 |

> **핵심**: `hub_spoke`는 허브(중앙 교차점)가 막히면 모든 로봇이 영향을 받는다. 공장에서 중간 창고가 막히면 전체 라인이 서는 것과 같다.

---

## 2. 출고 프로세스 흐름 — 왜 지연이 발생하는가?

```
[주문 접수]
     ↓
[피킹 지시 → WMS(창고관리시스템)에서 AMR에 작업 할당]
     ↓
[AMR이 선반으로 이동 → 물건 픽업]     ← avg_trip_distance 증가 시 지연
     ↓
[포장 스테이션으로 운반]              ← congestion_score 높으면 경로 충돌
     ↓
[포장 → 검수 → 라벨링]               ← pack_utilization, quality_check_rate 관련
     ↓
[출하 도크로 이동]                    ← outbound_truck_wait_min 관련
     ↓
[트럭 상차 → 출고 완료]
```

**지연 발생 시점**: 위 흐름 중 어느 단계에서든 병목이 생기면 `avg_delay_minutes_next_30m`이 증가한다.

### 병목 원인 분류 (공장 품질관리 4M 유사)

| 분류 | 창고 원인 | 관련 피처 |
|---|---|---|
| **Man (인력)** | 교대 교번, 숙련도 부족 | `shift_handover_delay_min`, `worker_avg_tenure_months`, `staff_on_floor` |
| **Machine (설비)** | 로봇 고장·충전, WMS 응답 지연 | `robot_charging`, `fault_count_15m`, `wms_response_time_ms` |
| **Material (물자)** | SKU 다양성, 과다 주문 유입 | `unique_sku_15m`, `order_inflow_15m`, `urgent_order_ratio` |
| **Method (방법)** | 레이아웃 비효율, 경로 최적화 실패 | `layout_type`, `path_optimization_score`, `congestion_score` |

---

## 3. 배터리 — 이 대회의 핵심 메커니즘

배터리 관련 피처가 타겟 상관관계 Top 5를 독점하는 이유를 이해해야 한다.

### 3-1. 배터리 고갈 → 지연의 연쇄 사이클

```
배터리 잔량 감소 (battery_mean ↓)
        ↓
저배터리 로봇 비율 증가 (low_battery_ratio ↑)
        ↓
충전이 필요한 로봇 증가 (robot_charging ↑)
        ↓
충전 대기 큐 형성 (charge_queue_length ↑, avg_charge_wait ↑)
        ↓
가용 로봇 수 감소 (robot_active ↓, robot_idle ↑*)
        ↓
주문 처리 속도 저하 → 출고 지연 증가 (avg_delay ↑)
```

> ⚠️ **`robot_idle`의 역설**: 유휴 로봇이 많은데 왜 지연이 증가할까?
> "대기 중"인 로봇 = 배터리가 너무 낮아 작업을 못 받거나, 충전 순서를 기다리는 로봇.
> 공장에서 설비가 "대기" 상태인데 실제로는 고장·점검 대기인 것과 같다.
> 즉, `robot_idle`이 높다 = 일할 수 있는 로봇이 아니라 일을 못 하는 로봇이 많다.

### 3-2. 배터리 관련 피처 해석

| 피처 | 의미 | 높을수록 |
|---|---|---|
| `battery_mean` | 전체 로봇 평균 배터리 잔량 | 좋음 (지연 ↓) |
| `battery_std` | 배터리 잔량 분산 | 나쁨 (일부 로봇이 곧 방전) |
| `low_battery_ratio` | 저배터리(<20%) 로봇 비율 | 나쁨 (지연 ↑) |
| `charge_queue_length` | 충전 대기 로봇 수 | 나쁨 (병목 신호) |
| `avg_charge_wait` | 평균 충전 대기 시간 (분) | 나쁨 (충전기 부족) |
| `charge_efficiency_pct` | 충전 효율 (%) | 좋음 (높을수록 빠른 충전) |
| `battery_cycle_count_avg` | 평균 배터리 사용 횟수 | 나쁨 (노후 배터리 → 용량 감소) |

### 3-3. 충전기 수 (layout_info)와의 관계

```python
# 충전 병목 지수 (파생 피처 아이디어)
병목 지수 = charge_queue_length / charger_count
# charger_count는 layout_info에서 merge 후 사용 가능
```

충전기 수 대비 충전 대기 로봇이 많을수록 → 충전 병목 → 지연 증가.

---

## 4. 주문 피처 — 수요 충격의 이해

### 4-1. 주문 관련 피처 해석

| 피처 | 의미 | 지연 영향 |
|---|---|---|
| `order_inflow_15m` | 15분간 유입 주문 수 | 많을수록 로봇 수요 급증 → 지연 |
| `unique_sku_15m` | 15분간 주문된 고유 품목 수 | 많을수록 피킹 경로 복잡 → 지연 |
| `avg_items_per_order` | 주문당 평균 품목 수 | 많을수록 AMR 이동 거리 증가 |
| `urgent_order_ratio` | 긴급 주문 비율 | 높을수록 우선순위 충돌 → 경로 재배정 증가 |
| `heavy_item_ratio` | 무거운 품목 비율 | 높을수록 이동 속도 감소 |
| `cold_chain_ratio` | 냉장 품목 비율 | 높을수록 특수 구역 이동 → 경로 증가 |
| `sku_concentration` | 특정 SKU 집중도 | 낮을수록 피킹 분산 → 혼잡 감소 |

### 4-2. 주문 압박 지수 (도메인 파생 피처)

```python
# 로봇 대비 주문 압박 (공장의 설비 부하율 개념)
order_pressure = order_inflow_15m / (robot_active + 1)
# robot_active가 적은데 주문이 많으면 → 지연 필연적

# 긴급 주문 충격
urgent_load = order_inflow_15m * urgent_order_ratio
# 일반 주문보다 자원 우선 할당 → 다른 주문 대기

# SKU 복잡도
sku_complexity = unique_sku_15m * avg_items_per_order
# 다양한 품목을 여러 개씩 → 피킹 경로 최악
```

---

## 5. 혼잡도 — 공장 내 통로 적체와 동일

### 5-1. 혼잡 관련 피처

| 피처 | 의미 | 공장 비유 |
|---|---|---|
| `congestion_score` | 전체 혼잡도 종합 지수 | 라인 적체율 |
| `max_zone_density` | 가장 혼잡한 구역의 밀도 | 병목 공정 WIP |
| `blocked_path_15m` | 15분간 경로 차단 횟수 | 통로 막힘 발생 건수 |
| `near_collision_15m` | 15분간 충돌 직전 상황 횟수 | 안전사고 위험 건수 |
| `aisle_traffic_score` | 통로 교통량 점수 | 통로 혼잡도 |
| `intersection_wait_time_avg` | 교차로 평균 대기 시간 | 병목 포인트 대기 |
| `path_optimization_score` | 경로 최적화 점수 | 물류 동선 효율성 |

### 5-2. 혼잡 × 주문 상호작용

```
주문 급증 (order_inflow ↑) + 혼잡도 높음 (congestion ↑)
         ↓
AMR이 목적지까지 가는 시간 급증
         ↓
하나의 AMR이 더 오래 걸리면 다음 작업 시작이 지연
         ↓
밀린 주문이 쌓임 (backlog 형성)
         ↓
avg_delay 급등 (비선형 증가 — 임계점 넘으면 폭증)
```

> **임계점 효과**: 혼잡도가 일정 수준을 넘으면 지연이 선형이 아니라 **지수적**으로 증가한다.
> 공장에서 생산 속도가 사이클타임을 초과하는 순간 WIP가 폭발하는 것과 같다.
> → 이것이 타겟 분포의 우편향(skewness 5.68)과 최대값 715분의 원인.

---

## 6. 운영 피처 — 인력과 교대

### 6-1. 교대 관련 피처

| 피처 | 의미 | 주의사항 |
|---|---|---|
| `shift_hour` | 현재 교대 시간 (근무 시간대) | 교대 교번 시간 전후로 지연 급증 |
| `shift_handover_delay_min` | 교대 인수인계 지연 시간 | 높으면 인력 공백 |
| `staff_on_floor` | 현장 직원 수 | 적을수록 수작업 처리 지연 |
| `worker_avg_tenure_months` | 평균 근속 개월수 | 낮을수록 숙련도 부족 → 처리 속도 감소 |

> **교대 교번 효과**: `shift_hour`가 교대 시점(예: 0시, 8시, 16시)에 가까울수록 `shift_handover_delay_min`이 높아지고 일시적으로 현장 인력이 줄어든다. 이 구간에서 `avg_delay`가 일시 급등하는 패턴이 있을 수 있다.

---

## 7. KPI 피처 — 창고 성과지표 해석

| 피처 | 전체 이름 | 의미 |
|---|---|---|
| `kpi_otd_pct` | On-Time Delivery % | 납기 준수율 — 낮으면 이미 지연 중 |
| `backorder_ratio` | 백오더 비율 | 재고 부족으로 처리 못 한 주문 비율 |
| `sort_accuracy_pct` | 분류 정확도 | 낮으면 오분류 → 재처리 지연 |
| `quality_check_rate` | 품질 검사 비율 | 높을수록 검사 시간 증가 |
| `return_order_ratio` | 반품 주문 비율 | 반품 처리로 정상 출고 지연 |
| `barcode_read_success_rate` | 바코드 인식 성공률 | 낮으면 수동 입력 → 병목 |
| `scanner_error_rate` | 스캐너 오류율 | 높으면 작업 중단 빈번 |

> **`kpi_otd_pct`가 낮다 = 이미 지연 상태**. 이 값이 낮은 타임슬롯은 다음 30분도 지연될 가능성이 높다. → Lag 피처로 활용 가치가 매우 높다.

---

## 8. 환경 피처 — 왜 포함되었나?

| 피처 | 의미 | 지연 영향 |
|---|---|---|
| `warehouse_temp_avg` | 창고 평균 온도 | 고온 → 배터리 방전 가속 |
| `humidity_pct` | 습도 | 고습 → 바코드 인식 저하 |
| `co2_level_ppm` | CO₂ 농도 | 높으면 작업자 집중력 저하 |
| `hvac_power_kw` | 냉난방 전력 소비 | 간접적 환경 품질 지표 |
| `floor_vibration_idx` | 바닥 진동 지수 | 로봇 이동 안정성 |
| `ambient_noise_db` | 소음 dB | 직원 작업 효율 |

> 환경 피처는 단독으로는 상관관계가 낮지만, 배터리나 혼잡도와 **상호작용**할 때 유효하다.
> 예: 고온(warehouse_temp_avg ↑) × 저배터리(low_battery_ratio ↑) → 지연 더 증가.

---

## 9. 타임슬롯 패턴 — 시간의 흐름이 지연을 키운다

```
시나리오 시작 (ts=0)
  → 주문 유입 시작, 로봇 배터리 충분
  → 평균 지연: 11.3분

시나리오 중반 (ts=12)
  → 주문 누적, 일부 로봇 충전 필요
  → 평균 지연: 19.6분

시나리오 종료 (ts=24)
  → 배터리 고갈 로봇 증가, 혼잡도 누적
  → 평균 지연: 21.9분
```

**이 패턴의 의미**:
- 창고는 **누적 시스템**이다. 초반에 쌓인 병목은 후반에 폭발한다.
- 직전 타임슬롯의 상태(Lag 피처)가 현재 지연을 강하게 예측한다.
- 공장의 WIP(재공품) 누적 효과와 동일.

---

## 10. 피처별 도메인 기반 해석 요약 (빠른 참조표)

### 🔴 지연을 직접 유발하는 피처 (높을수록 나쁨)

| 피처 | 왜 나쁜가 |
|---|---|
| `low_battery_ratio` | 가용 로봇 감소의 선행 지표 |
| `charge_queue_length` | 충전 병목 = 로봇 부족 |
| `robot_charging` | 지금 당장 가용 불가 로봇 수 |
| `robot_idle` | 실질 가동 불능 로봇 (역설적 의미) |
| `order_inflow_15m` | 처리해야 할 일의 양 |
| `congestion_score` | 이동 시간 증가 |
| `max_zone_density` | 특정 구역 과부하 |
| `near_collision_15m` | 로봇 속도 감소 강제 |
| `fault_count_15m` | 고장 → 작업 재배정 지연 |
| `task_reassign_15m` | 재배정 = 기존 작업 지연 |
| `backorder_ratio` | 처리 못한 주문 누적 |
| `shift_handover_delay_min` | 교대 공백 |

### 🟢 지연을 억제하는 피처 (높을수록 좋음)

| 피처 | 왜 좋은가 |
|---|---|
| `battery_mean` | 충전 여유 → 가용 로봇 안정 |
| `robot_active` | 실제 작업 중인 로봇 수 |
| `robot_utilization` | 전체 로봇이 효율적으로 가동 중 |
| `path_optimization_score` | 효율적 경로 = 이동 시간 감소 |
| `kpi_otd_pct` | 현재 납기 준수 상태 양호 |
| `sort_accuracy_pct` | 오분류 없음 → 재처리 없음 |
| `barcode_read_success_rate` | 스캔 순조 → 처리 속도 유지 |
| `charge_efficiency_pct` | 빠른 충전 → 가용 복귀 신속 |

---

## 11. 도메인 지식 기반 파생 피처 아이디어

아래는 도메인 이해를 바탕으로 설계할 수 있는 파생 피처다.

```python
import pandas as pd
import numpy as np

def create_domain_features(df, layout):
    df = df.merge(layout, on='layout_id', how='left')

    # ─── 1. 로봇 가용성 지수 (설비 가동률 개념) ───────────────
    # 전체 로봇 중 실제로 일하는 비율
    df['robot_availability'] = df['robot_active'] / (
        df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1
    )

    # ─── 2. 배터리 위기 복합 지수 ─────────────────────────────
    # 저배터리 × 충전 대기 = 실질적 로봇 부족 심각도
    df['battery_crisis'] = df['low_battery_ratio'] * df['charge_queue_length']

    # 충전기 대비 충전 병목 (layout_info의 charger_count 필요)
    df['charge_bottleneck'] = df['charge_queue_length'] / (df['charger_count'] + 1)

    # 배터리 고갈 위험도 (분산이 크면 일부 로봇이 곧 방전)
    df['battery_risk'] = (1 - df['battery_mean'] / 100) * df['battery_std']

    # ─── 3. 주문 압박 지수 (생산 부하율 개념) ────────────────
    # 가용 로봇 대비 주문량
    df['order_per_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)

    # 긴급 주문이 만드는 실질 충격량
    df['urgent_shock'] = df['order_inflow_15m'] * df['urgent_order_ratio']

    # SKU 복잡도 (다양한 품목을 많이 → 피킹 경로 복잡)
    df['sku_complexity'] = df['unique_sku_15m'] * df['avg_items_per_order']

    # ─── 4. 혼잡-주문 상호작용 (비선형 효과 포착) ────────────
    df['congestion_x_order'] = df['congestion_score'] * df['order_inflow_15m']
    df['density_x_robot_lack'] = df['max_zone_density'] * (1 - df['robot_utilization'])

    # ─── 5. 교대 위험 지수 ────────────────────────────────────
    # 인력 대비 주문 압박
    df['staff_load'] = df['order_inflow_15m'] / (df['staff_on_floor'] + 1)

    # ─── 6. 시스템 신뢰성 지수 ───────────────────────────────
    # 오류·고장이 많을수록 작업 재배정 → 지연
    df['system_reliability'] = (
        df['barcode_read_success_rate'] *
        df['sort_accuracy_pct'] / 100 *
        (1 - df['scanner_error_rate'])
    )

    # ─── 7. 레이아웃 병목 가중 혼잡도 ───────────────────────
    # hub_spoke는 혼잡 시 지수적으로 나빠짐
    layout_penalty = {'hub_spoke': 1.3, 'narrow': 1.1, 'hybrid': 1.0, 'grid': 0.9}
    df['layout_type_num'] = df['layout_type'].map(layout_penalty).fillna(1.0)
    df['weighted_congestion'] = df['congestion_score'] * df['layout_type_num']

    # ─── 8. 타임슬롯 누적 효과 피처 ─────────────────────────
    df['ts_idx'] = df.groupby('scenario_id').cumcount()
    df['ts_ratio'] = df['ts_idx'] / 24  # 0~1 정규화
    # 시나리오 진행률이 높을수록 배터리 고갈·혼잡 누적
    df['ts_x_battery_risk'] = df['ts_ratio'] * df['low_battery_ratio']

    return df
```

---

## 12. 시나리오 구조의 이해 — 이 대회만의 특수성

### 각 시나리오는 독립적인 창고 운영 시뮬레이션이다

```
시나리오 A (SC_00001):  창고 WH_001, 6시간 운영
  [ts=0] → [ts=1] → ... → [ts=24]   ← 15분 간격, 연속

시나리오 B (SC_00002):  창고 WH_001, 다른 날 6시간 운영
  [ts=0] → [ts=1] → ... → [ts=24]   ← A와 완전 독립
```

**중요 규칙**:
- 시나리오 A의 ts=24가 끝나면 시나리오 B의 ts=0과는 아무 관련 없다.
- Lag 피처는 **같은 시나리오 안에서만** 적용해야 한다.
- `groupby('scenario_id')` 후 `shift(1)` 해야 리크 없는 lag 피처.

```python
# ✅ 올바른 방법 — 시나리오 내에서만 lag
df['battery_mean_lag1'] = df.groupby('scenario_id')['battery_mean'].shift(1)

# ❌ 잘못된 방법 — 시나리오 경계를 무시
df['battery_mean_lag1'] = df['battery_mean'].shift(1)  # A의 ts=24 → B의 ts=0 연결됨!
```

---

## 13. 제조업 경험을 ML 피처로 바꾸는 핵심 관점

| 제조 경험 | 창고 적용 | ML 피처 |
|---|---|---|
| 설비 가동률 관리 | 로봇 가용률 | `robot_active / robot_total` |
| OEE (종합설비효율) | 로봇 종합 효율 | `robot_utilization × availability` |
| 재공품(WIP) 누적 | 대기 중인 주문 적체 | `order_inflow_15m - (robot_active × 처리속도)` |
| 사이클타임 초과 | 처리 용량 초과 | `order_per_robot > 임계값` |
| 예방보전 주기 | 배터리 교체 주기 | `battery_cycle_count_avg` |
| 교대 인수인계 | 교대 손실 | `shift_handover_delay_min` |
| 공정 능력 지수(Cpk) | 납기 준수율 | `kpi_otd_pct` |
| 불량률 | 오분류/스캔 오류 | `1 - sort_accuracy_pct` |

---

## 14. 최종 요약: 지연을 예측하기 위해 봐야 할 것

```
지연 = f(
    배터리 고갈 정도,      ← low_battery_ratio, charge_queue_length
    로봇 가용 수,          ← robot_active vs. robot_charging + idle
    주문 압박,             ← order_inflow_15m × urgent_ratio
    혼잡도,                ← congestion_score × layout_penalty
    시나리오 누적 효과,    ← ts_idx, lag/rolling 피처
    시스템 신뢰성,         ← scanner_error_rate, fault_count_15m
    레이아웃 구조          ← layout_type (hub_spoke가 최악)
)
```

> **한 줄 요약**: "로봇이 부족하고(배터리 고갈), 주문은 많고(유입 급증), 길은 막혀있다(혼잡 고조) — 이 세 가지가 겹치면 지연은 폭발한다."

---

## 15. 실험으로 검증된 도메인 인사이트 (04.04 기준)

> 실제 LightGBM GroupKFold 5-Fold 실험 결과를 바탕으로, 도메인 예측과 실제 중요도를 비교한다.

### 15-1. 실제 피처 중요도 Top 10 (Full FE 기준)

| 순위 | 피처 | 유형 | 중요도 | 도메인 해석 |
|---|---|---|---|---|
| 1 | `pack_station_count` | 원본(layout) | 394 | **포장 스테이션 수** — 창고 처리 용량의 상한선. 적으면 병목 |
| 2 | `avg_trip_distance_roll5_mean` | Rolling | 296 | **최근 5슬롯 평균 이동거리** — 혼잡/레이아웃 악화 추세 포착 |
| 3 | `layout_compactness` | 원본(layout) | 291 | **레이아웃 압축도** — 낮을수록 이동 경로 길어짐 |
| 4 | `sku_concentration` | 원본 | 258 | **SKU 집중도** — 높을수록 특정 선반 병목 |
| 5 | `order_inflow_15m_roll5_mean` | Rolling | 226 | **최근 주문 유입 추세** — 주문 급증 선행 포착 |
| 6 | `zone_dispersion` | 원본(layout) | 217 | **구역 분산도** — 창고 내 작업 분산 가능 여부 |
| 7 | `avg_items_per_order` | 원본 | 181 | **주문당 품목 수** — 많을수록 AMR 적재 횟수 증가 |
| 8 | `avg_trip_distance_roll3_mean` | Rolling | 181 | **단기 이동거리 추세** — 3슬롯 단기 악화 신호 |
| 9 | `floor_area_sqm` | 원본(layout) | 181 | **창고 면적** — 클수록 이동거리 기본값 증가 |
| 10 | `pack_utilization` | 원본 | 170 | **포장 가동률** — 높으면 포장 병목 임박 |

### 15-2. 예측과 실제의 차이 — 도메인 지식 보정

| 예측 (EDA 상관관계) | 실제 (모델 중요도) | 해석 |
|---|---|---|
| 배터리 관련 피처가 1~5위 | **layout 구조 피처가 1~3위** | 배터리는 상관관계는 높지만, 모델이 layout에서 더 많은 정보 추출 |
| congestion_score 중요 | rolling avg_trip_distance가 더 중요 | 이동거리 추세가 혼잡보다 직접적 정보 |
| ts_idx 중요 예상 | **90위 (108개 중)** | 모델이 다른 피처에서 시간 흐름을 이미 포착 |
| log1p 변환 큰 효과 예상 | **효과 없음 (±0.005분)** | LightGBM MAE 목적함수가 왜도에 강건함 |

### 15-3. 새로 추가해야 할 도메인 피처 방향

실험 결과, **layout 구조 피처**와 **이동거리 추세**가 핵심임이 확인됐다. 추가 도메인 피처:

```python
# ── layout 구조 × 혼잡 상호작용 ──────────────────────
df['compact_x_congestion'] = df['layout_compactness'] * df['congestion_score']
# layout이 촘촘한데 혼잡하면 → 탈출 경로 없음 → 지연 폭발

# ── 포장 스테이션 처리 압박 ──────────────────────────
df['pack_pressure'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
# 주문 대비 포장 스테이션 부족 → 처리 속도 상한

# ── 포장 가동률 × 주문 압박 ──────────────────────────
df['pack_overload'] = df['pack_utilization'] * df['order_inflow_15m']
# 둘 다 높으면 포장 병목 → 피킹이 아무리 빨라도 출고 지연

# ── 이동거리 증가율 (추세 가속도) ────────────────────
df['trip_dist_delta'] = (
    df['avg_trip_distance_roll3_mean'] - df['avg_trip_distance_roll5_mean']
)
# 양수면 최근 이동거리 가속 증가 → 혼잡 악화 중
```

### 15-4. 핵심 실험 인사이트 (ML 대회 교훈)

| 교훈 | 내용 |
|---|---|
| **GroupKFold 필수** | KFold는 시나리오 내 타임슬롯이 train/val에 동시 등장 → 리크 1.41분 발생 |
| **ID 순서 검증 필수** | Lag/Rolling FE 후 test 정렬 꼬임 → `assert` 검증 코드를 파이프라인에 반드시 추가 |
| **layout 피처 무료 이익** | `layout_info.csv` merge만으로 14개 피처 추가, 전체 Top 3 점령 |
| **Rolling > Lag** | 단순 lag보다 rolling 추세가 모델에 더 많은 정보 제공 |
| **도메인 복합 피처 소폭** | LGBM은 내부적으로 상호작용 포착 가능 → 도메인 피처는 +0.04분 소폭 기여 |
