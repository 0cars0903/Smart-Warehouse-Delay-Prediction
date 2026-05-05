# 추가 EDA 분석 인사이트 (2026-04-11)
## Part A: 시나리오 시작 조건(TS0) 분석
### A1. TS0 초기값 → 시나리오 평균 지연 상관 (Top)
| 피처 | r | 해석 |
|------|---|------|
| `robot_utilization` | 0.475 | TS0 초기값이 시나리오 결과를 결정 |
| `order_inflow_15m` | 0.462 | TS0 초기값이 시나리오 결과를 결정 |
| `robot_active` | 0.398 | TS0 초기값이 시나리오 결과를 결정 |
| `sku_concentration` | 0.370 | TS0 초기값이 시나리오 결과를 결정 |
| `max_zone_density` | 0.367 | TS0 초기값이 시나리오 결과를 결정 |
| `congestion_score` | 0.366 | TS0 초기값이 시나리오 결과를 결정 |
| `robot_idle` | -0.356 | TS0 초기값이 시나리오 결과를 결정 |
| `urgent_order_ratio` | 0.339 | TS0 초기값이 시나리오 결과를 결정 |

### A2. 붕괴 vs 안정 시나리오 초기 조건 비교 (Top)
| 피처 | 붕괴_TS0 | 안정_TS0 | 배율 |
|------|---------|---------|------|
| `blocked_path_15m` | 1.337 | 0.000 | 2946.3× |
| `fault_count_15m` | 1.217 | 0.001 | 890.7× |
| `avg_recovery_time` | 2.690 | 0.005 | 568.6× |
| `congestion_score` | 25.228 | 0.114 | 220.4× |
| `max_zone_density` | 0.179 | 0.001 | 180.4× |
| `order_inflow_15m` | 148.749 | 48.743 | 3.1× |
| `robot_utilization` | 0.654 | 0.215 | 3.0× |
| `urgent_order_ratio` | 0.173 | 0.064 | 2.7× |

### A3. 피처 엔지니어링 제안
- **TS0 broadcast 피처**: 시나리오 내 모든 타임슬롯에 TS0 값을 복사하여 추가
- 특히 `battery_mean(TS0)`, `robot_idle(TS0)`, `order_inflow_15m(TS0)` 유망
- 시나리오 붕괴 취약성 지수: `low_battery_ratio(TS0) × order_inflow_15m(TS0)`

## Part B: 극단 지연 세그먼트 분석

### B1. 상관관계 역전 피처
| 피처 | r_전체 | r_극단(>P90) | 해석 |
|------|--------|-------------|------|
| `low_battery_ratio` | 0.366 | -0.298 | 극단에서 역전 |
| `battery_mean` | -0.359 | 0.254 | 극단에서 역전 |
| `robot_idle` | -0.349 | 0.262 | 극단에서 역전 |
| `max_zone_density` | 0.311 | -0.237 | 극단에서 역전 |
| `battery_std` | 0.308 | -0.217 | 극단에서 역전 |
| `congestion_score` | 0.300 | -0.224 | 극단에서 역전 |
| `charge_queue_length` | 0.261 | -0.178 | 극단에서 역전 |
| `blocked_path_15m` | 0.220 | -0.151 | 극단에서 역전 |
| `fault_count_15m` | 0.203 | -0.140 | 극단에서 역전 |
| `avg_recovery_time` | 0.184 | -0.126 | 극단에서 역전 |
| `staging_area_util` | 0.166 | -0.007 | 극단에서 역전 |

### B2. 2-stage 모델링 타당성
- P90 기준(45.2분) 이상 극단 지연: 전체의 10%
- 극단 구간에서 pack_utilization 상관이 폭발적 상승 → 패킹 병목이 최종 병목
- 배터리 관련 피처 상관 역전 → 극단 구간은 다른 인과 구조
- **결론**: 2-stage 모델(일반/극단 분류 → 각각 회귀) 실험 가치 있음

## Part C: 결측치 패턴 분석

### C1. MNAR 검증 결과 (결측 indicator 유의 피처)
| 피처 | 결측률 | indicator_r | 해석 |
|------|--------|------------|------|
| `avg_recovery_time` | 13.0% | 0.022 | MNAR 약함 |
| `congestion_score` | 12.9% | 0.012 | MNAR 약함 |

### C2. 피처 엔지니어링 제안
- MNAR 강한 피처에 대해 `is_missing` binary indicator 피처 추가
- 결측 자체가 '이벤트 미발생'의 신호일 가능성 → indicator가 타겟과 음/양 상관

## 생성 파일
- `A1_ts0_vs_outcome.png` — TS0 초기값 vs 시나리오 평균 지연 산점도
- `A2_ts0_collapse_vs_stable.png` — 붕괴/안정 시나리오 초기 조건 박스플롯
- `B1_extreme_segment_corr.png` — 구간별 상관계수 역전 시각화
- `B2_extreme_pattern.png` — 극단 지연 발생 타임슬롯·레이아웃 분포
- `C1_missing_pattern.png` — 결측치 타임슬롯별 히트맵 + MNAR 검증
