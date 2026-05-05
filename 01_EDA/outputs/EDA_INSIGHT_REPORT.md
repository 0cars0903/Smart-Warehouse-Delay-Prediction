# DACON 스마트 창고 출고 지연 예측 — EDA 인사이트 리포트

> 분석 일시: 2026-04-09 | 도메인 지식 없는 일반인 관점 접근

---

## 1. 데이터 개요

- **Train**: 250,000행 × 94열 (10,000개 시나리오 × 25 타임슬롯)
- **Test**: 50,000행 × 93열 (타겟 제외)
- **Layout**: 300개 창고 레이아웃 정보 (15개 변수)
- **피처**: 90개 수치형 피처 + ID 3개 (ID, layout_id, scenario_id)
- **타겟**: `avg_delay_minutes_next_30m` (향후 30분 평균 출고 지연, 분 단위)

### 결측치
- **86개 컬럼**에 결측 존재 (전체 90개 피처 중)
- 대부분 11.7~13% 수준으로 균일한 결측률 → 시뮬레이션에서 특정 타임슬롯 또는 조건에서 센서 데이터가 누락되는 구조로 추정
- `avg_recovery_time`(13.0%), `congestion_score`(12.9%), `avg_charge_wait`(12.3%)가 가장 높음

---

## 2. 타겟 변수 분석

| 통계량 | 값 |
|--------|------|
| Mean | 18.96분 |
| Median | 9.03분 |
| Std | 27.35분 |
| Min / Max | 0.00 / 715.86분 |
| Skewness | 5.68 (강한 양의 왜도) |
| Kurtosis | 64.05 (극단적 뾰족한 분포) |

### 핵심 발견
- **극심한 오른쪽 꼬리 분포**: 중앙값(9분)과 평균(19분)의 큰 차이 → 소수의 극단적 지연이 평균을 끌어올림
- **2.7%가 정확히 0** (지연 없음)
- **P99 = 120.8분**, 하지만 최대는 715.8분 → 상위 1%에 극단 이상치 존재
- MAE 평가이므로 **이상치에 덜 민감**하지만, log 변환이나 로버스트 모델 고려 필요

---

## 3. 피처 그룹 분류 (12개 그룹, 90개 피처)

| 그룹 | 피처 수 | 대표 피처 |
|------|---------|----------|
| Order/SKU | 11 | order_inflow_15m, urgent_order_ratio, sku_concentration |
| Robot | 9 | robot_active/idle/charging, robot_utilization, agv_task_success_rate |
| Battery/Charge | 8 | battery_mean, low_battery_ratio, charge_queue_length |
| Congestion/Path | 8 | congestion_score, max_zone_density, blocked_path_15m |
| Environment | 14 | warehouse_temp, humidity, co2, hvac_power |
| Pack/Manual | 9 | pack_utilization, staging_area_util, sort_accuracy |
| Worker/Safety | 6 | staff_on_floor, shift_hour, shift_handover_delay |
| Logistics/Warehouse | 10 | loading_dock_util, conveyor_speed, inventory_turnover |
| System/IT | 4 | wms_response_time, wifi_signal, network_latency |
| KPI/Performance | 4 | kpi_otd_pct, daily_forecast_accuracy |
| Fault/Recovery | 2 | fault_count_15m, avg_recovery_time |
| Other | 5 | cold_chain_ratio, day_of_week, maintenance_schedule_score |

---

## 4. 타겟 상관관계 — Top 20

| 순위 | 피처 | Pearson r | 해석 |
|------|------|-----------|------|
| 1 | **low_battery_ratio** | +0.366 | 배터리 부족 로봇 비율↑ → 지연↑ |
| 2 | **battery_mean** | -0.359 | 평균 배터리↑ → 지연↓ |
| 3 | **robot_idle** | -0.349 | 유휴 로봇↑ → 지연↓ (여유 있음) |
| 4 | **order_inflow_15m** | +0.342 | 15분 주문량↑ → 지연↑ |
| 5 | **robot_charging** | +0.320 | 충전 중 로봇↑ → 가용↓ → 지연↑ |
| 6 | **max_zone_density** | +0.311 | 특정 구역 밀집↑ → 지연↑ |
| 7 | **battery_std** | +0.308 | 배터리 편차↑ → 불균형 → 지연↑ |
| 8 | **congestion_score** | +0.300 | 혼잡도↑ → 지연↑ |
| 9 | **sku_concentration** | +0.292 | SKU 집중도↑ → 특정 물품 병목 |
| 10 | **urgent_order_ratio** | +0.271 | 긴급 주문 비율↑ → 작업 우선순위 혼란 |
| 11 | charge_queue_length | +0.261 | |
| 12 | avg_charge_wait | +0.251 | |
| 13 | near_collision_15m | +0.243 | |
| 14 | unique_sku_15m | +0.229 | |
| 15 | blocked_path_15m | +0.220 | |
| 16 | loading_dock_util | +0.213 | |
| 17 | robot_utilization | +0.211 | |
| 18 | heavy_item_ratio | +0.210 | |
| 19 | fault_count_15m | +0.203 | |
| 20 | maintenance_schedule_score | -0.197 | 정비 점수 높을수록 지연 감소 |

### 핵심 패턴
**배터리/충전** 그룹이 1,2,5,7,11,12위를 차지 → **로봇 배터리 상태가 지연의 가장 강력한 예측 인자**.
**혼잡도** 관련(congestion, density, blocked_path, collision)도 일관되게 상위 → 물리적 병목.
**주문량/복잡도**(order_inflow, sku_concentration, urgent_ratio)도 중요.

---

## 5. 다중공선성

| 피처 쌍 | r |
|---------|-----|
| battery_mean ↔ low_battery_ratio | **-0.934** |
| robot_charging ↔ charge_queue_length | **+0.917** |
| robot_charging ↔ low_battery_ratio | **+0.859** |
| charge_queue_length ↔ avg_charge_wait | **+0.857** |

- 총 4쌍만 |r| > 0.85 → 심각한 다중공선성은 제한적
- 그러나 **배터리 관련 변수들(battery_mean, low_battery_ratio, battery_std, charge_queue_length, robot_charging)** 사이에 강한 상호연관
- 모델링 시 이 그룹에서 대표 피처를 선택하거나, PCA/그룹 변환 고려

---

## 6. 시계열 패턴 (시나리오 내)

- 각 시나리오는 정확히 **25개 타임슬롯** (약 6시간, 15분 간격)
- **타임슬롯이 증가할수록 평균 지연도 증가하는 경향** → 시뮬레이션이 진행될수록 누적 피로/혼잡 효과
- 분산도 후반 타임슬롯에서 증가 → 시나리오 간 divergence
- **row_order (타임슬롯 인덱스)를 피처로 엔지니어링** 가능

---

## 7. Layout 정보

- 4가지 타입: grid(106), hybrid(98), narrow(50), hub_spoke(46)
- **pack_station_count**(r=-0.186)와 **robot_total**(r=-0.111)만 타겟과 의미 있는 상관
  - 패킹 스테이션이 많을수록 지연 감소 (병목 해소)
  - 로봇 총수가 많을수록 지연 감소
- layout_type별 평균 지연에도 차이 존재 → 범주형 피처로 활용 가능
- 나머지 layout 피처(aisle_width, compactness 등)는 상관이 매우 약함

---

## 8. 고빈도 0값 피처 (>50% zero)

| 피처 | 0값 비율 |
|------|---------|
| task_reassign_15m | 85.2% |
| fault_count_15m | 73.2% |
| avg_recovery_time | 72.5% |
| avg_charge_wait | 71.8% |
| charge_queue_length | 71.7% |
| blocked_path_15m | 71.7% |
| near_collision_15m | 70.6% |

이들은 **"이벤트 발생 여부"가 중요**한 변수 → binary flag 피처 생성 고려 (발생=1, 미발생=0)

---

## 9. Train vs Test 분포

- 상위 6개 피처에서 Train과 Test의 분포가 매우 유사 → 데이터 시프트 우려 낮음
- 안전하게 교차검증 기반 모델링 가능

---

## 10. 모델링 제안 (EDA 기반)

1. **피처 엔지니어링 우선순위**
   - 시나리오 내 타임슬롯 인덱스(row_order) 추가
   - 배터리 관련 복합 지표 (예: battery_mean × robot_utilization)
   - 이벤트 피처 이진화 (fault, collision, blocked_path 등)
   - layout_info merge (특히 pack_station_count, robot_total, layout_type)

2. **결측치 전략**
   - 균일한 결측률(~12%) → 센서/시점 기반 MNAR 가능성
   - 결측 자체를 indicator로 쓰거나, LightGBM/XGBoost의 native missing 처리 활용

3. **타겟 변환**
   - 강한 양의 왜도 → log1p 변환 후 학습, 예측 시 expm1 역변환 고려
   - MAE 기준이므로 quantile regression도 유효

4. **모델 선택**
   - 트리 기반 모델(LightGBM, XGBoost, CatBoost) 권장 — 결측 처리, 비선형 포착
   - 피처 간 상호작용이 복잡 → 선형 모델보다 앙상블 우위 예상

---

## 생성된 파일 목록

| 파일명 | 내용 |
|--------|------|
| 01_target_distribution.png | 타겟 분포 (원본/로그/박스플롯/줌) |
| 02_dist_*.png (12개) | 그룹별 피처 분포 히스토그램 |
| 03_top30_correlation_bar.png | 상위 30 상관계수 막대그래프 |
| 04_correlation_heatmap_top20.png | 상위 20 피처 상관 히트맵 |
| 05_timeseries_patterns.png | 시나리오 내 시계열 패턴 |
| 06_multicollinearity_heatmap.png | 다중공선성 히트맵 |
| 07_scatter_top10.png | 상위 10 피처 산점도 |
| 08_hexbin_top4.png | 상위 4 피처 밀도 산점도 |
| 09_layout_analysis.png | 레이아웃 분석 |
| 10_shift_hour_target.png | 시프트 시간별 타겟 |
| 11_dow_target.png | 요일별 타겟 |
| 12_train_test_dist.png | Train/Test 분포 비교 |
| missing_values.csv | 결측치 현황 |
| feature_descriptive_stats.csv | 전체 피처 기술통계 |
| correlation_with_target.csv | 타겟 상관계수 |
| multicollinearity_pairs.csv | 높은 상관 피처 쌍 |
