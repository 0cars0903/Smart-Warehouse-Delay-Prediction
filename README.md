# 스마트 창고 출고 지연 예측 AI 경진대회

> AMR 기반 스마트 물류창고 운영 데이터를 활용한 출고 지연 시간 예측  
> [Dacon 공식 대회 페이지](https://dacon.io/competitions/official/236696/overview/description) | 기간: 2026.04.01 ~ 2026.05.04

---

## 최종 결과

| 지표 | 값 |
|---|---|
| **Public LB (최고)** | **9.7901** |
| **1위 점수** | 9.6992 |
| **1위 대비 갭** | 0.0909 |
| **평가 지표** | MAE (Mean Absolute Error) |
| **총 실험 횟수** | 57회 이상 (제출 기준) / 80회+ (CV 전용 포함) |
| **총 실험 기간** | 34일 |

---

## 핵심 발견 (Key Findings)

### 1. 시나리오 간 분산이 전체의 63.4% → sc_agg가 핵심 돌파구
v1 시리즈에서 row 단위 lag/rolling 피처로 한계(Public ~10.22)에 도달한 후, 시나리오 레벨 분산 분석을 통해 전체 분산의 63.4%가 시나리오 간 차이에서 비롯됨을 발견. 시나리오 25행 전체의 mean/std/max/min/diff를 각 행에 broadcast하는 방식으로 전환하자 Public 10.22 → 9.95로 단번에 돌파. **이 발견이 대회 전체의 핵심 전환점.**

### 2. CV 악화 ≠ 반드시 실패 — 피처 노이즈가 정규화로 작용
model29A에서 Tier2 비율 피처 추가 시 CV가 **+0.025 악화**했음에도 Public은 **−0.021 개선**. CV→Public 배율이 1.1626→1.1567로 역대 최저를 기록. 약간의 노이즈성 피처가 암묵적 정규화 역할을 해 일반화를 향상시킬 수 있음을 확인. **이후 pred_std 모니터링이 CV만큼 중요한 지표로 격상.**

### 3. 극값(target ≥ 80)이 전체 MAE의 27.6% 담당, 해결 불가 확인
전체 데이터의 2.6%에 불과한 [80,800) 구간이 MAE 기여의 27.6%를 차지하고, 모든 base learner가 실제의 32%만 예측(pred/actual = 0.32). 후처리/2-stage/raw-target 학습/분류기 보정 등 12가지 개선 시도가 전부 실패. 이 문제가 tree 모델의 외삽(extrapolation) 한계에서 비롯됨을 확정.

### 4. 동일 파이프라인 내 변형의 한계 — 질적 전환이 필요
v5에서 메가블렌드/CB메타/피처선택/pseudo-label/multi-seed/KNN후처리 6전략을 하루에 동시 실험했으나 전패. 최종 기준선이 현재 피처+모델 구조에서 이미 로컬 최적점에 도달해 있었으며, 근본적으로 다른 피처 공간(궤적 형상 피처, v6)이 필요하다는 결론 도달.

### 5. 시퀀스 모델의 함정 — OOF vs Test 분포 불일치
1D-CNN/BiLSTM의 LGBM 상관이 0.9063으로 낮아(다양성 확보 성공) 기대했으나, 7모델 스태킹 제출에서 Public 10.3531로 급등. 원인은 OOF 예측(단일 fold)과 test 예측(5-fold 평균) 간 분포 불일치였음. 시퀀스 모델을 스태킹에 직접 투입하는 방식 폐기.

---

## 최종 모델 아키텍처

```
──────────────────────────────────────────────────────────────
 Feature Engineering Pipeline (429 피처, model31/34 기준)
──────────────────────────────────────────────────────────────
 [원본 피처 ~90종]
   └─ layout_info merge (+14종)
   └─ ts 피처: ts_idx, ts_ratio, ts_sin, ts_cos
   └─ Lag(1~6) × 8 key_cols           = 48종
   └─ Rolling(3/5/10) × 8 key_cols    = 48종
   └─ 시나리오 집계(sc_agg) 11통계 × 18컬럼 = 198종  ← 핵심 돌파구
   └─ Layout-capacity 비율 피처 Tier1+2 = 12종
   └─ Shift-safe cross 피처           =  7종

──────────────────────────────────────────────────────────────
 Base Learners (6종, GroupKFold 5-fold OOF)
──────────────────────────────────────────────────────────────
  ┌─ LGBM (MAE + log1p)             OOF ≈ 8.55
  ├─ CatBoost (MAE + log1p)         OOF ≈ 8.60
  ├─ CatBoost Tweedie (p=1.5)       OOF ≈ 8.82  ← 극값 특화
  ├─ ExtraTrees                     OOF ≈ 8.67
  ├─ RandomForest                   OOF ≈ 8.73
  └─ LGBM Asymmetric (α=1.5/2.0)   OOF ≈ 8.77  ← pred_std 확장

──────────────────────────────────────────────────────────────
 Meta Learner (LGBM-meta, GroupKFold 5-fold)
──────────────────────────────────────────────────────────────
  model33 (α=1.5): CV 8.4756 / Public 9.8223
  model34 (α=2.0): CV 8.4803 / Public 9.8078

──────────────────────────────────────────────────────────────
 최종 제출: Blend
──────────────────────────────────────────────────────────────
  model45c_q7_q95 (model34 6모델 + LGBM Quantile q=0.95, 7모델 스태킹)
  → Public LB: 9.7931  🏆
```

---

## 문제 정의

스마트 물류창고 운영 스냅샷 데이터(90개 피처)를 기반으로 **향후 30분간의 평균 출고 지연 시간(분)**을 예측

- 12,000개 독립 시나리오 (시뮬레이션)
- 각 시나리오: ~6시간, 25개 타임슬롯 (15분 간격)
- 피처: 로봇 상태, 주문량, 배터리, 통로 혼잡도 등
- 타겟 분포: 평균 18.96분, 중앙값 9.03분, max 715분 (강한 우편향)

## 데이터

| 파일 | 크기 | 설명 |
|---|---|---|
| `train.csv` | 250,000행 × 94컬럼 | 10,000 시나리오 × 25 타임슬롯 |
| `test.csv` | 50,000행 × 93컬럼 | 2,000 시나리오 × 25 타임슬롯 |
| `layout_info.csv` | 300행 × 15컬럼 | 250개 창고 레이아웃 정보 |
| `sample_submission.csv` | — | 제출 양식 (타겟: `avg_delay_minutes_next_30m`) |

---

## 프로젝트 구조

> Notion Analysis Board(A-EDA / B-FE / C-Model / D-Submit / G-Guide)와 1:1 매핑되는 5-Track 구조.

```
Smart-Warehouse-Delay-Prediction/
├── data/                              # 원본 데이터 (.gitignore)
├── submissions/                       # 제출 CSV 누적 (대용량, gitignore 권장)
│
├── 00_Guide/                          # G-대회 안내
│   ├── competition_plan.md            # 대회 전략 계획서
│   ├── daily_schedule.md              # 일일 스케줄
│   ├── domain_knowledge.md            # 도메인 지식
│   ├── layout_info_utility_structured.md
│   ├── llm_insight_prompt.md
│   └── github_upload_checklist.md
│
├── 01_EDA/                            # A-EDA 분석 내용
│   ├── 01_EDA.ipynb                   # 탐색적 분석 (시각화 8종)
│   ├── 03_CV_Strategy.ipynb           # GroupKFold vs KFold 검증
│   ├── scripts/
│   │   ├── eda_tail_driver.py         # 극값 시나리오 특성 분석
│   │   ├── eda_loss_ablation.py       # 손실함수별 구간 MAE 비교
│   │   ├── eda_symbolic_physics.py
│   │   ├── analysis_model28A_axis3.py # 극값 구간 정밀 분석 (MAE 기여도/headroom)
│   │   └── run_additional_eda.py
│   ├── reports/                       # EDA_FULL_REPORT.md, eda_*_report.txt, layout_info_analysis_report.md
│   └── outputs/                       # EDA 시각화 PNG (48종)
│
├── 02_FE/                             # B-피처 엔지니어링
│   ├── 04_Log_Transform.ipynb         # log1p 변환 실험
│   ├── 05_TS_Features.ipynb           # 타임슬롯 피처
│   ├── 06_Feature_Engineering.ipynb   # Lag + Rolling + Domain FE
│   ├── feature_engineering.py         # ✅ 공통 FE 파이프라인 (build_features)
│   ├── experiments/                   # FE 확장 ablation (v1~v4)
│   │   └── run_exp_fe_v{1-4}_*.py
│   ├── reports/                       # project_ratio_convergence.md, feature_importance_exp3.csv
│   └── outputs/                       # FE 진행 시각화 PNG
│
├── 03_Model/                          # C-모델링
│   ├── 02_Baseline_Model.ipynb        # LightGBM 베이스라인
│   ├── 07_Ensemble_Optuna.ipynb       # Optuna 3모델 앙상블
│   ├── baseline/                      # 초기 baseline + ablation (run_ensemble*, run_dart, run_2stage 등 17종)
│   ├── v1_stacking/                   # model1~12: Tweedie/Quantile/Stacking/HGB/MLP/Poisson
│   ├── v2_scenario/                   # model21~27: 시나리오 집계 5~7모델 (+ step1 helpers)
│   ├── v3_ratio/                      # model28~30: Layout-capacity 비율 피처
│   ├── v4_extreme/                    # 2-Stage 극값 + 후처리 3종 (전부 실패)
│   ├── v5_loss/                       # model31~45: Asymmetric/TW1.5/k-fold/LDS/Multi-Q (17종)
│   ├── v6_final/                      # model46~48: SC_AGG 확장 + Layout 교호작용 (10종)
│   ├── reports/                       # model_selection_report.md, v3_strategy_report.md, v6_strategy.md
│   └── outputs/                       # cv_strategy_comparison.png
│
├── 04_Submit/                         # D-실험·제출
│   ├── blends/                        # blend_m33_m34, blend_m34_internal, blend_mega, blend_q85_m34bd
│   └── ablation_results/              # ablation/transform/optuna 결과 CSV
│
├── .gitignore
├── requirements.txt
└── README.md
```

**최종 SOTA 경로** (Public 9.8073, Private 20등):
1. `02_FE/feature_engineering.py` → 공통 FE 빌드
2. `03_Model/v5_loss/run_model33_asymmetric.py` (9.8223) + `run_model34_loss_opt.py` (9.8078)
3. `04_Submit/blends/blend_m33_m34.py` → 최종 블렌드 🏆

---

## 대회 일정

| 날짜 | 내용 |
|---|---|
| 04.01 | 대회 시작 |
| 04.27 | 팀 병합 마감 |
| 05.04 | **대회 종료** |
| 05.07 | 코드 및 PPT 제출 마감 |
| 05.15 | 코드 검증 |
| 05.18 | 최종 수상자 발표 |

---

## 실험 기록 (Approach Log)

### v1 시리즈 — 피처 엔지니어링 + GBDT 스태킹 (04.01~04.16)

| # | 날짜 | 실험 | CV MAE | 피처수 | Public LB | 비고 |
|---|---|---|---|---|---|---|
| 1 | 04.01 | Baseline (KFold, 리크 포함) | 7.3351 | 104 | — | KFold 리크 1.41분 확인 |
| 2 | 04.02 | GroupKFold 기준점 | 9.2156 | 104 | — | |
| 3 | 04.02 | + ts 피처 4종 | 9.1790 | 108 | — | −0.40% |
| 4 | 04.04 | Full FE (ID순서버그) | 9.0010 | 172 | 19.8209 ❌ | test 정렬 misalign |
| 5 | 04.04 | Full FE (버그수정) | 9.0010 | 172 | 10.4936 | |
| 6 | 04.05 | Optuna LGBM 단독 | 8.8895 | 284 | 10.3807 | |
| 7 | 04.05 | **Optuna 앙상블 (LGBM+CB+XGB)** | **8.8703** | **284** | **10.3349** | |
| 8 | 04.11 | Transform Ablation | 8.8836 | 284 | — | log1p 최적 확정 |
| 9 | 04.11 | TS0 Broadcast + 앙상블 | 8.8649 | 296 | 10.4091 ❌ | Public 역전 |
| 10 | 04.11 | **3모델 Optuna 전체 튜닝** | **8.8674** | **284** | **10.3347** | |
| 11 | 04.12 | layout_info Ablation (4전략) | 8.8899 | 284~318 | — | ordinal이 최적 확정 |
| 12 | 04.12 | Stacking LGBM-meta (LGBM+CB+ET) | 8.8541 | 212 | 10.3032 | Ridge meta 무효 확인 |
| 13 | 04.12 | Stacking v2 (TW1.8 교체) | 8.8087 | 212 | 10.3118 ⚠️ | CV↑ 배율 악화 |
| 14 | 04.12 | **Stacking v3 4모델 (LGBM+TW1.8+CB+ET)** | **8.7929** | **212** | **10.2264 🏆** | |
| 15 | 04.13 | **Stacking 5모델 RF 추가** | **8.7911** | **212** | **10.2213 🏆** | |
| 16 | 04.14 | FE v2 (KEY_COLS 확장+Delta+비율) | 8.7842 | 264 | 10.2801 ⚠️ | |
| 17 | 04.15 | Ablation: Delta 무효 확정 | 8.7836 | 252 | 10.2829 ⚠️ | |
| 18 | 04.15 | FE v3 Cumulative 피처 | **8.7663** | 281 | 10.2571 ⚠️ | CV 신기록, 배율 악화 |
| 19 | 04.15 | FE v1+Cumul 가설 검증 | 8.7699 | 239 | 10.2517 ⚠️ | Cumulative가 배율 악화 원인 확정 |
| 20 | 04.15~16 | ExtLag A/B/C + HGB + MLP 탐색 | 8.7697~8.7858 | 244~292 | 미제출 | 배율 1.170 수렴 → FE 방향 차단 |

> **v1 Public 최고**: 10.2213 (5모델 RF 스태킹) | **배율**: 1.1627 고정

---

### v2 시리즈 — 시나리오 집계 피처 돌파 (04.17~04.18)

> **전환 계기**: 시나리오 간 분산 63.4% 발견 → 25행 전체 집계를 broadcast

| # | 날짜 | 실험 | CV MAE | 피처수 | Public LB | 비고 |
|---|---|---|---|---|---|---|
| 21 | 04.17 | **sc_agg 5모델 (mean/std/max/min/diff × 18컬럼)** | **8.5097** | 302 | **9.9550 🏆** | 단번에 0.27 개선 |
| 22 | 04.17 | **sc_agg 11통계 확장 (+median/p10/p90/skew/kurt/cv)** | ~8.51 | 302+α | **9.9385 🏆** | 배율 1.168 (최저) |
| 23 | 04.17 | Optuna v2 (LGBM+CB 재튜닝) | **8.5038** | 302 | 9.9522 | CV 최고, 배율 동일 |
| 24 | 04.18 | 메타 피처 강화 (OOF+sc_mean) | 8.5589 | 302+ | 10.0405 ❌ | 정보 중복 → 폐기 |

> **v2 Public 최고**: 9.9385 | **핵심 발견**: 11통계 분포 피처(skew/kurt/cv)가 일반화 개선

---

### v3 시리즈 — 비율 피처 + 손실함수 최적화 (04.18~04.22)

| # | 날짜 | 실험 | CV MAE | 피처수 | Public LB | 비고 |
|---|---|---|---|---|---|---|
| 25 | 04.18 | 시퀀스 모델 Phase1 (1D-CNN+BiLSTM) | CNN 9.13 / LSTM 8.91 | — | — | 상관 0.9063 다양성 ✅ |
| 26 | 04.18 | **7모델 하이브리드 스태킹** | **8.5128** | — | 10.3531 ❌ | OOF-test 분포 불일치 |
| 27 | 04.19 | **비율 피처 Tier1 5종 (415피처)** | **8.4743** | 415 | **9.8525 🏆** | 배율 1.1626 복귀 |
| 28 | 04.19 | 극값 메타 강화 실험 | 8.5384 | 415 | — | 메타 레벨 한계 확인 |
| 29 | 04.19 | **비율 피처 Tier2 7종 추가 (422피처)** | 8.4989 | 422 | **9.8312 🏆** | ⭐ CV 악화에도 Public 개선 |
| 30 | 04.19 | Optuna 재튜닝 (415피처) | **8.4723** | 415 | 9.8356 | |
| 31 | 04.20 | **422피처 + Optuna 파라미터 결합** | **8.4838** | 422 | **9.8279 🏆** | |
| 32 | 04.20 | 2-Stage 극값 전략 v4.0 (441피처) | 8.4977 | 441 | 9.8414 ❌ | log1p 압축 가설 기각 |
| 33 | 04.21 | 후처리 3종 + IF 보정 + BC 분류기 | — | — | 9.8340~9.8458 ❌ | 후처리 방향 완전 종결 |
| 34 | 04.21 | **Shift-safe FE 7종 추가 (429피처)** | **8.4786** | 429 | **9.8255 🏆** | |
| 35 | 04.21 | FE 자동 필터 확장 / blend | ~8.48 | 449 | 9.8246~9.8339 | 순이익 없음 |
| 36 | 04.22 | **6모델 + Asymmetric MAE (α=1.5)** | **8.4756** | 429 | **9.8223 🏆** | |
| 37 | 04.22 | **TW1.5 교체 (TW1.8→TW1.5)** | **8.4720** | 429 | 9.8144 | [80+] MAE 81.14 |
| 38 | 04.22 | **6모델 + Asymmetric MAE (α=2.0)** | 8.4803 | 429 | **9.8078 🏆** | 배율 1.1565 역대 최저 |
| 39 | 04.22 | **blend model33 × model34 (w=0.3:0.7)** | — | — | **9.8073 🏆** | |
| 40 | 04.27 | model34_7full (Asym α=1.5+2.0 동시) | 8.4783 | 9.8058 | — | 미제출분 제출 |
| 41 | 04.27 | blend_m34bd_b70 (Config B×0.7 + D×0.3) | — | 9.8056 | — | |
| 42 | 04.27 | **blend_m34bd_b60 (Config B×0.6 + D×0.4)** | — | **9.8053** | — | |
| 43 | 04.28 | GroupKFold k=3 (6모델 재학습) | 8.4934 | 9.8119 | 1.1553 | 배율 역대 최저 ↑ but CV 열위 |
| 44 | 04.28 | GroupKFold k=5 (6모델 재학습) | 8.4806 | 9.8356 | 1.1598 | k=5 재현 — 기준 대비 열위 |
| 45 | 04.28 | **GroupKFold k=10 (6모델 재학습)** | **8.4554** | 9.8124 | 1.1603 | **CV 역대 최고 🏆** but 배율 최악 |
| 46 | 04.29 | LDS sample_weight 6모델 재학습 (전략A) | 8.5043 | 9.8952 ❌ | 1.1593 | LDS 역효과 — 극값은 피처 외삽 한계 |
| 47 | 04.29 | **7모델 스태킹 + LGBM Quantile q=0.85 (전략C)** | **8.4735** | **9.8048 🏆** | **1.1571** | **✅ CV + Public 동시 최고** |
| 48 | 04.29 | 7모델 스태킹 + Quantile q=0.90 | 8.4740 | — | — | CV q85 대비 +0.0005 |
| 49 | 04.29 | **7모델 스태킹 + Quantile q=0.95** | **8.4684** | **9.7931 🏆** | **1.1565** | **✅ CV + Public 동시 최고 🏆** |

> **v3+v6 Public 최고**: **9.7931** (model45c_q7_q95) | **1위 대비 갭**: **0.0939**
>
> **k-fold 실험 결론**: k↑ → CV↓(좋아짐) but 배율↑(나빠짐). k=10 CV 신기록(8.4554)에도 Public 9.8124로 최고 미달.
> **LDS 실험 결론**: 극값 문제는 불균형 아닌 피처 외삽 한계 — 가중치 부여는 역효과.
> **Quantile 기여**: q85가 [80+] pred/actual을 0.605로 높이는 다양성 제공 → 배율 1.1571(역대 최저급) 달성.

---

### v7 시리즈 — FE 확장 (SC_AGG + Layout 교호작용) (2026-04-30)

> **전환 계기**: 대회 데이터 전수 감사 → SC_AGG가 90컬럼 중 18개만 활용. layout_info와의 교호작용 미탐색. 두 방향을 독립 실험(46a/46b/46c) 후 최적 조합 탐색.

| # | 날짜 | 실험 | CV MAE | 피처수 | Public LB | 비고 |
|---|---|---|---|---|---|---|
| 50 | 04.30 | model46b: KEY_COLS lag/rolling 확장 (8→10종, +robot_charging/battery_std) | 8.4719 ❌ | 446 | — | KEY_COLS 확장 방향 실패 재확인 |
| 50 | 04.30 | model46b: KEY_COLS lag/rolling 확장 (8→10종) | 8.4719 | 446 | 9.8092 ⚠️ | 배율 1.1578 — CV 최악이나 Public 3개 중 최고 (순서 역전) |
| 51 | 04.30 | model46a: SC_AGG 확장 (18→23컬럼) | 8.4647 | 477 | 9.8097 ⚠️ | 배율 1.1590 — model34 6모델(9.8078) 대비 소폭 악화 |
| 52 | 04.30 | **model46c: Layout×운영 교호작용 6종 (lx_*)** | **8.4600** | 439 | 9.8126 ❌ | **CV 역대 최고 🏆 but Public 최악** — pred_std=15.90 압축이 원인 |
| 53 | 04.30 | **model46a(6) + q95 → 7모델 스태킹** | **8.4615** | — | 9.7997 ❌ | CV 역대 최고 🏆 but 배율 1.1611 악화 — 기준(9.7931) 미달 |
| 54 | 04.30 | model46c(6) + q95 → 7모델 스태킹 | 8.4639 | — | 9.7957 ❌ | 기준 대비 Δ+0.0026 근소 차 미달. 배율 1.1592 |
| 55 | 04.30 | **model47: SC_AGG(23) + Layout 교호작용(6종) 동시 적용** | 8.4649 | 483 | **9.8063** ✅ | pred_std=16.14, 배율 1.1584 — model34(9.8078) 돌파! |
| 56 | 04.30 | **model47(6) + q95 → 7모델 스태킹** | **8.4610** | **9.7901 🏆** | **pred_std=16.35 역대 최고 🚀 CV Δ-0.0074 ✅** |
| 57 | 05.01 | model47(6) + q95 + q85 → 8모델 스태킹 | 8.4593 | 9.7995 ❌ | pred_std=16.47 역대 최고 but CV 개선→Public 악화 역전 |
| 58 | 05.01 | model47(6) + q95 + q90 → 8모델 스태킹 | 8.4617 | 9.7993 ❌ | pred_std=16.38 — q90 추가도 Public 악화 |
| 59 | 05.01 | model47(6) + q95 + q90 + q85 → 9모델 스태킹 | 8.4619 | 9.8006 ❌ | pred_std=16.46 — 3 quantile 조합 최악 |
| 60 | 05.02 | model47(6) + q95 + q70 → 8모델 스태킹 | 8.4577 | 9.7991 ❌ | pred_std=16.59 역대 최고나 Public Δ+0.0090 악화. q70도 동일 실패 패턴 확정 |
| 61 | 05.01 | model47(6) + q95 + q80 → 8모델 스태킹 | 8.4579 | 미제출 | pred_std=16.32 (기준보다 낮음, q80 단독 다양성 제한) |
| 62 | 05.01 | model47(6) + q95 + q70 + q85 → 9모델 스태킹 | 8.4606 | 미제출 | pred_std=16.52 — q70+q85 조합, q70 단독보다 열위 |
| 63 | 05.01 | **model47_addon: +XGB → 7모델+q95** | **8.4573** | 미제출 | ✅ CV Δ-0.0037, pred_std=16.34 — XGB 단독 추가 최우수 |
| 64 | 05.01 | model47_addon: +asym15 → 7모델+q95 | 8.4581 | 미제출 | CV Δ-0.0029, pred_std=16.20 — asym15 단독 추가 2위 |
| 65 | 05.01 | model47_addon: +XGB+asym15 → 8모델+q95 | 8.4614 | 미제출 | ❌ CV Δ+0.0004, 조합 시 오히려 열위 |
| 66 | 05.01 | model47_addon: +DART → 7모델+q95 | 8.4640 | 미제출 | ❌ CV Δ+0.0030, DART 방향 열위 |
| 67 | 05.01 | model48: ts_ratio×SC_AGG 교호작용 10종 (483+α피처, 6모델) | 8.4705 | 미제출 | ❌ CV Δ+0.0056 악화 + pred_std=16.02 압축 — 방향 종결 |
| 68 | 05.01 | model48+q95 → 7모델 스태킹 | 8.4662 | 미제출 | ❌ CV Δ+0.0052 악화 + pred_std=16.14 압축 |

> **v7 결론**: 신규 FE(SC_AGG 확장 / Layout 교호작용) 모두 기준(9.7931) 돌파 실패. CV 역대 최고(8.4615)가 Public 개선으로 이어지지 않음 — 배율 악화(1.1565→1.1592~1.1611)가 원인.
> **핵심 관찰**: 나쁜 base일수록 q95 개선폭이 큼 (46c: Δ-0.0169 > 기준: Δ-0.0147). 그러나 시작점이 낮아 최종 Public이 미달.
> **model47 관찰**: 두 FE 결합 시 CV=8.4649 (46a=8.4647과 거의 동일), pred_std=16.14 (46c=15.90보다 회복). 두 FE를 합쳐도 상호 시너지가 없음 — 피처 공간이 중복 (lx_wait_per_charger가 SC_AGG avg_charge_wait를 이미 포함)
> **Multi-Q 최종 결론 (05.02 확정)**: q70 Public=9.7991 (Δ+0.0090 악화) — q85/q90과 동일 패턴. q70 상관(0.9588)이 낮고 메커니즘이 달랐으나 결과는 같음. **q95 단독이 유일 유효 quantile 최종 확정**. pred_std 확장/OOF 다양성이 Public에서 실현 안 되는 구조적 한계 — quantile 추가 방향 완전 종결.
>
> **addon/model48 결론 (05.01)**: XGB 추가 시 CV 8.4573 미세 개선(Δ-0.0037)이나 v7 전체 패턴상 Public 기대 불투명. ts_ratio×SC_AGG 교호작용(model48)은 CV+pred_std 동시 악화로 방향 종결.

---

### v5 — 6전략 동시 실험 (04.24, 전패)

| 전략 | Public | 판정 | 원인 |
|---|---|---|---|
| 메가블렌드 (13모델) | 미제출 | ❌ | 상관 0.994~0.999, 다양성 제로 |
| CB 메타 교체 | 9.8110 | ❌ | 배율 악화 |
| 피처 선택 (top80) | 미제출 | ❌ | CV 8.8061 완전 실패 |
| Pseudo-label | 미제출 | ❌ | TW1.5 OOF 붕괴 |
| Multi-seed 3개 | 9.8097 | △ | 분산 감소이나 Public 무개선 |
| KNN 후처리 (K=10) | 10.1489 | ❌❌ | 대폭 악화 |

---

### v6 — 궤적 형상 피처 (2026-04-25, 완료)

| # | 모델 | CV MAE | Public | 배율 | 비고 |
|---|---|---|---|---|---|
| 41 | model41 (궤적 FE 29종, 458피처, 6모델) | 8.4851 | 9.8449 | 1.1602 | ❌ model31(9.8255) 대비 악화 |

**결과 분석**: CV +0.0065 악화 + pred_std 15.73 압축(model31 15.89) → 두 지표 동시 악화로 배율 1.1602 (model31 1.1589보다 높음). model29A 패턴(CV 악화→배율 개선) 미재현. 궤적 피처가 sc_agg 통계와 정보 중복으로 순노이즈 작용.

**피처 구성**:
| 카테고리 | 피처 수 | 결과 |
|---|---|---|
| slope × 8 | 8 | 개별 기여 확인 불가 |
| fl_ratio × 8 | 8 | 개별 기여 확인 불가 |
| peak_pos × 5 | 5 | 개별 기여 확인 불가 |
| above_cnt × 5 | 5 | 개별 기여 확인 불가 |
| mono × 3 | 3 | 개별 기여 확인 불가 |

**최종 판정**: ❌ 궤적 형상 피처 방향 종결. sc_agg가 이미 동일 정보의 분포 통계를 포함하여 temporal dynamics 추가 표현의 한계 확인. 최고 기준 blend_m34bd_b60 = **9.8053** 으로 갱신.

- 스크립트: `03_Model/v5_loss/run_model41_traj_fe.py`

---

### v6 추가 탐색 — 비트리(Non-Tree) 모델 다양성 & 피처 경량화 (2026-04-26)

#### model43: 비트리 모델 앙상블 가능성 탐색

lag/rolling 피처를 제외(96종)한 362개 피처로 MLP·Ridge·ElasticNet을 GroupKFold 5-fold 평가.

| 모델 | OOF MAE | pred_std(OOF) | LGBM-모델 상관 | 앙상블 판정 |
|---|---|---|---|---|
| MLP (512→256→128) | 9.9715 | 18.61 | **0.79** | ❌ 성능 부족 |
| Ridge (α=100) | 9.5841 | 13.75 | 0.85 | ❌ 성능 부족 |
| ElasticNet | 11.1906 | 6.03 | 0.80 | ❌ |

**관찰**: 다양성은 압도적(MLP-LGBM 0.79 — model27 CNN 0.91보다 낮음)이나 OOF MAE 격차(LGBM 8.55 vs MLP 9.97)가 너무 커서 메타 가중치가 0에 수렴. MLP 실행 중 메모리 스와핑(Fold 2: 9640초)으로 학습 불완전 가능성도 있으나 성능 개선 여지 불충분. model27 교훈(다양성 있어도 OOF-test 분포 불일치 → 배율 폭등) 적용하여 앙상블 시도 중단.

**최종 판정**: ❌ 비트리 모델 앙상블 방향 종결.

---

## 버그 수정 기록

**ID 순서 misalignment (04.04 발견, 즉시 수정)**

- **현상**: lag/rolling FE 내부에서 `sort_values(['scenario_id', 'ts_idx'])`로 test 행 순서 변경 → 예측값이 다른 ID에 할당 → Public LB 19.82 (정상의 2배)
- **원인**: FE 후 test가 scenario_id 기준 정렬, 제출 파일은 원본 ID 기준
- **수정**: FE 전 `_orig_order` 컬럼 저장 → FE 후 복원

---

## 실험 인사이트 요약

| 발견 | 내용 |
|---|---|
| KFold 리크 | KFold 7.80 vs GroupKFold 9.22 → **1.41분 리크** |
| log1p 효과 | LGBM L1의 왜도 강건성으로 변환 효과 미미 (±0.005) |
| layout_info Ablation | one-hot/비율/Target Enc/교호작용 모두 악화 → ordinal 최적 |
| XGBoost 탈락 | 정규화 부재로 과적합 → 앙상블 가중치 0.038로 탈락 |
| 배율 수렴 함정 | Cumulative/Delta/KEY_COLS 확장이 pred_std 압축 → 배율 고정 |
| sc_agg 돌파 | 시나리오 전체 25행 집계 broadcast → 가장 큰 단일 도약 |
| 비율 피처의 정규화 효과 | Tier2 피처가 CV 노이즈이나 test 과적합 방지 역할 |
| 극값 한계 | target≥80 (2.6% 데이터, MAE의 27.6%)는 tree 외삽 한계 |

---

## 재현 방법

```bash
# 환경 설정
pip install -r requirements.txt

# 최종 모델 학습 (model34, ~30분)
python 03_Model/v5_loss/run_model34_loss_opt.py

# 최종 블렌드 제출 파일 생성
python 04_Submit/blends/blend_m33_m34.py

# v6 실험 (궤적 형상 피처)
python 03_Model/v5_loss/run_model41_traj_fe.py
```

> `data/` 디렉토리에 `train.csv`, `test.csv`, `layout_info.csv`, `sample_submission.csv` 배치 필요 (`.gitignore`로 제외됨)
