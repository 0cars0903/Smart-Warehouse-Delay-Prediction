---
name: 배율 수렴 발견 — KEY_COLS 확장이 Public 역전 원인
description: FE v2 계열 전체 배율 1.170 수렴 — KEY_COLS_V2(8→11) 확장이 CV→Public 배율 악화의 근본 원인. Delta/Cumulative/Optuna 무관.
type: project
---

FE v2 기반 모든 실험의 CV→Public 배율이 1.170±0.001로 수렴한다.

| 실험 | 배율 |
|---|---|
| RF 5모델 (FE v1, KEY_COLS 8종) | **1.1627** (기준) |
| FE v2 full (264피처) | 1.1703 |
| FE v2 no-delta (252피처) | 1.1707 |
| FE v3 Cumulative (281피처) | 1.1701 |
| FE v2+Optuna A (263피처) | 1.1710 |

**Why:** KEY_COLS_V2 확장 (battery_std, robot_charging, sku_concentration, urgent_order_ratio 추가)이 train→test 분포 차이를 증폭시킨다. Delta 제거, Optuna 튜닝, Cumulative 추가 모두 배율 개선에 실패 — 피처가 아닌 KEY_COLS 정의가 원인.

**How to apply:** 새 FE 실험 시 반드시 원본 KEY_COLS(8종) 기반으로 시작하고, KEY_COLS 변경 시에는 배율 영향을 최우선 확인한다. FE v1 base + Cumulative가 다음 검증 대상 (`src/run_exp_fe_v1_cumul.py`).
