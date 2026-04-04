# Dacon 스마트 창고 출고 지연 예측 AI 경진대회

> AMR 기반 스마트 물류창고 운영 데이터를 활용한 출고 지연 시간 예측

## Competition Info

| 항목 | 내용 |
|---|---|
| **대회** | [스마트 창고 출고 지연 예측 AI 경진대회](https://dacon.io/competitions/official/236696/overview/description) |
| **유형** | 알고리즘 · 정형 · 회귀 |
| **평가지표** | MAE (Mean Absolute Error) |
| **기간** | 2026.04.01 ~ 2026.05.04 |
| **주최/주관** | 데이콘 |

## Problem

스마트 물류창고 운영 스냅샷 데이터(90개 피처)를 기반으로 **향후 30분간의 평균 출고 지연 시간(분)**을 예측

- 12,000개 독립 시나리오 (시뮬레이션)
- 각 시나리오: ~6시간, 25개 타임슬롯 (15분 간격)
- 피처: 로봇 상태, 주문량, 배터리, 통로 혼잡도 등

## Data

| 파일 | 설명 |
|---|---|
| `train.csv` | 학습 데이터 |
| `test.csv` | 테스트 데이터 |
| `layout_info.csv` | 창고 레이아웃 보조 정보 |
| `sample_submission.csv` | 제출 양식 (타겟: `avg_delay`) |

## Project Structure

```
Smart-Warehouse-Delay-Prediction/
├── data/                # 원본 데이터 (.gitignore)
├── notebooks/           # EDA, 모델링 노트북
│   ├── 01_EDA.ipynb
│   └── 02_Baseline_Model.ipynb
├── src/                 # 유틸리티, 파이프라인 코드
├── models/              # 학습 모델 저장 (.gitignore)
├── submissions/         # 제출 CSV
├── docs/                # 문서, 분석 메모
├── .gitignore
├── requirements.txt
└── README.md
```

## Timeline

| 날짜 | 내용 |
|---|---|
| 03.31 | 참가 신청 시작 |
| 04.01 | 대회 시작 |
| 04.27 | 팀 병합 마감 |
| 05.04 | 대회 종료 |
| 05.07 | 코드 및 PPT 제출 마감 |
| 05.15 | 코드 검증 |
| 05.18 | 최종 수상자 발표 |

## Approach Log

| 날짜 | 실험 | CV MAE | Public LB | 메모 |
|---|---|---|---|---|
| 04.04 | Baseline LightGBM 5-Fold | - | - | 초기 세팅 |

## Setup

```bash
pip install -r requirements.txt
```
