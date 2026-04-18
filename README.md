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
| `sample_submission.csv` | 제출 양식 (타겟: `avg_delay_minutes_next_30m`) |

## Project Structure

```
Smart-Warehouse-Delay-Prediction/
├── data/                        # 원본 데이터 (.gitignore)
├── notebooks/
│   ├── 01_EDA.ipynb             # EDA (26셀, PNG 8종)
│   ├── 02_Baseline_Model.ipynb  # LightGBM 베이스라인 (KFold MAE=7.3351)
│   ├── 03_CV_Strategy.ipynb     # GroupKFold vs KFold (리크 1.41분 확인)
│   ├── 04_Log_Transform.ipynb   # log1p 변환 실험 (효과 미미, 원본 유지)
│   ├── 05_TS_Features.ipynb     # ts_idx/ratio/sin/cos (−0.40%)
│   └── 06_Feature_Engineering.ipynb  # Lag+Rolling+Domain (−1.94%)
├── src/
│   ├── __init__.py
│   └── feature_engineering.py   # FE 파이프라인 모듈 (build_features)
├── models/                      # 학습 모델 저장 (.gitignore)
├── submissions/                 # 제출 CSV
├── docs/                        # 문서, 분석 PNG
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

| # | 날짜 | 실험 | CV MAE | ΔvsBase | 피처수 | Public LB | 제출 파일 |
|---|---|---|---|---|---|---|---|
| 1 | 04.01 | Baseline (KFold, 리크) | 7.3351 | — | 104 | — | `baseline_lgbm_mae7.3351.csv` |
| 2 | 04.02 | GroupKFold 기준점 | 9.2156 | — | 104 | — | `groupkfold_lgbm_cv.csv` |
| 3 | 04.02 | + ts 피처 (4종) | 9.1790 | −0.40% | 108 | — | `groupkfold_ts_lgbm.csv` |
| 4 | 04.03 | Full FE (ID순서버그) | 9.0010 | −1.94% | 172 | **19.8209** ❌ | `groupkfold_fullFE_lgbm.csv` |
| 5 | 04.04 | Full FE (버그수정) | 9.0010 | −1.94% | 172 | 10.4936 | `groupkfold_fullFE_lgbm_fixed.csv` |
| 6 | 04.05 | Optuna LGBM 단독 | 8.8895 | −3.43% | 284 | 10.3807 | `best_single_lgbm_optuna.csv` |
| 7 | 04.05 | **Optuna 앙상블 (LGBM+CB+XGB)** | **8.8703** | **−3.64%** | **284** | **10.3349** | **`ensemble_lgbm_cb_xgb_optuna.csv`** |
| 8 | 04.11 | layout_info Ablation (A~C 전략) | — | — | — | — | `ablation_results_20260411_1806.csv` (CV 전용) |
| 9 | 04.11 | Transform Ablation: log1p=**8.8836** / sqrt=8.8956 / identity=8.9089 / stretch 후처리 전부 악화 | 8.8836 (log1p) | ➖ 동일 | 284 | — | `transform_ablation_20260411_1832.csv` (CV 전용) |
| 10 | 04.11 | **3모델 Optuna 앙상블 (CB·XGB 신규 튜닝)** | **8.8674** | **−3.67%** | **284** | **10.3347** | **`ensemble_optuna_all3.csv`** |
| 11 | 04.11 | TS0 Broadcast Ablation (연속8+플래그3+복합1) | 8.9529 (Exp3) | −0.0138 vs Exp0 | 184 | — | `ts0_ablation_*.csv` (CV 전용) |
| 11b | 04.11 | **ensemble_ts0** (TS0 12종 + log1p + LGBM·CB·XGB, LGBM 신규 Optuna) | **8.8649** | **✅ −0.0025** | 296 | 10.4091 ❌ | `ensemble_ts0_LGBM_CB_XGB.csv` |
| 11c | 04.11 | **2-Stage P90 콤보** (base×0.75 + 2stage×0.25) | 8.8745 | ➖ +0.0071 | 284 | — | `2stage_combo.csv` |
| 11d | 04.11 | 2-Stage P90 단독 | 8.9503 | ❌ +0.0829 | 284 | — | `2stage_p90.csv` |
| 12 | 04.12 | P_extreme 메타 피처 앙상블 (분류기 AUC=0.875) | 8.9089 | ❌ +0.0415 | 285 | — | CV 전용 (개선 없음) |
| 13 | 04.12 | DART LGBM Optuna (20 trials, 2-fold) | 8.9661 단독 | ❌ +0.0987 | 285 | — | CV 전용 |
| 14 | 04.12 | DART 앙상블 (DART+CB+XGB, 가중치 0.56/0.34/0.10) | 8.9221 | ❌ +0.0547 | 285 | 미제출 권장 | `ensemble_dart_meta.csv` |
| 15 | 04.12 | **[일반화 실험 3]** 피처 중요도 하위 10% 컷 + LGBM+CB (lag[1-6]+roll[3,5,10]) | **8.8871** | ➖ +0.0197 | 189 | 10.3662 | `feat_prune_bot10pct_LGBM_CB.csv` |
| 16 | 04.12 | **[일반화 실험 2]** XGBoost 제외 LGBM+CB 클린 앙상블 (log1p) | 8.8913 | ➖ +0.0239 | 212 | 미제출 | `ensemble_lgbm_cb_clean.csv` |
| 17 | 04.12 | **[일반화 실험 1]** sqrt 단독 앙상블 | 8.8776 | ➖ +0.0102 | 212 | 미제출 | `ensemble_sqrt_lgbm_cb.csv` |
| 18 | 04.12 | **[일반화 실험 1]** sqrt×0.70 + log1p×0.30 블렌드 | **8.8749** | ➖ +0.0075 | 212 | 10.3674 | `ensemble_sqrt_log1p_blend.csv` |
| 19 | 04.12 | **[모델실험1-A]** Tweedie(p=1.5)+CB 앙상블 | **8.8593** | ✅ −0.0081 | 212 | 미제출 | `ensemble_tweedie15_cb.csv` |
| 20 | 04.12 | **[모델실험1-B]** Tweedie p-sweep 4모델 (Tw1.8+CB 실질) | 8.8828 | ➖ +0.0154 | 212 | 미제출 | `ensemble_tweedie_blend.csv` |
| 21 | 04.12 | **[모델실험2-A]** Quantile LGBM×3 블렌드 (q=0.3/0.5/0.7) | 8.8764 | ➖ +0.0090 | 212 | 미제출 | `ensemble_quantile_lgbm3.csv` |
| 22 | 04.12 | **[모델실험2-B]** Quantile 4모델 블렌드 (q×3+CB) | 8.8697 | ➖ +0.0023 | 212 | 미제출 | `ensemble_quantile_4model.csv` |
| 23 | 04.12 | **[모델실험3-A]** Stacking Ridge-meta (LGBM+CB+ET) | 8.9152 | ❌ +0.0478 | 212 | 미제출 | `stacking_ridge_meta.csv` |
| 24 | 04.12 | **[모델실험3-B] Stacking LGBM-meta (LGBM+CB+ET)** | **8.8541** | ✅ −0.0133 | 212 | 10.3032 🏆 | `stacking_lgbm_meta.csv` |
| 25 | 04.12 | **[모델실험4] Stacking v2 (LGBM+TW1.8+ET → LGBM-meta)** | 8.8087 | ✅ −0.0454 (CV) | 212 | 10.3118 ⚠️ | `stacking_v2_lgbm_tw_et.csv` |
| 26 | 04.12 | **[모델실험5] Stacking v3 4모델 (LGBM+TW1.8+CB+ET → LGBM-meta)** | **8.7929** | **✅ −0.0612** | **212** | **10.2264 🏆** | **`stacking_4model_lgbm_meta.csv`** |
| 27 | 04.13 | **[모델실험6] Optuna 메타 LGBM (4모델 체크포인트, N=50)** | 8.7929 | ➖ 동일 | 212 | 10.2273 | `stacking_4model_optuna_meta.csv` |
| 28 | 04.13 | **[모델실험7-A]** Ridge+LGBM 메타 블렌드 (최적: Ridge×0.05+LGBM×0.95) | **8.7927** | ✅ −0.0002 | 212 | 10.2309 ⚠️ | `stacking_4model_ridge_lgbm_blend.csv` |
| 28b | 04.12 | **[모델실험7-B]** Ridge+LGBM 0.5:0.5 블렌드 | 8.8123 | ➖ +0.0194 | 212 | — | `stacking_4model_half_blend.csv` |
| 29 | 04.13 | **[모델실험9] Stacking 5모델+Q05 (LGBM+TW1.8+CB+ET+Q05 → LGBM-meta)** | 8.7938 | ➖ +0.0009 | 212 | 10.2358 ❌ | `stacking_5model_q05_lgbm_meta.csv` |
| 30 | 04.13 | **[모델실험8] Stacking 5모델+RF (LGBM+TW1.8+CB+ET+RF → LGBM-meta)** | **8.7911** | **✅ −0.0018** | 212 | **10.2213 🏆** | **`stacking_5model_rf_lgbm_meta.csv`** |
| 31 | 04.14 | **[FE v2] KEY_COLS 개선+Delta+Layout비율 (264피처, RF 5모델)** | **8.7842** | **✅ −0.0069 (CV)** | **264** | **10.2801 ⚠️** | **`stacking_fe_v2_rf_lgbm_meta.csv`** |
| 32 | 04.15 | **[Ablation] FE v2 no-delta (252피처, RF 5모델)** | **8.7836** | ✅ −0.0006 vs FE v2 | **252** | 10.2829 ⚠️ | **`stacking_fe_v2_nodelta_rf_lgbm_meta.csv`** |
| 33 | 04.15 | **[FE v3] Cumulative 피처 (281피처, RF 5모델)** | **8.7663** | **✅ −0.0248 🏆 CV 신기록** | **281** | **10.2571 ⚠️** | **`stacking_fe_v3_cumul_rf_lgbm_meta.csv`** |
| 34 | 04.15 | **[FE v2+Optuna A] LGBM 재튜닝 (263피처)** | **8.7816** | ✅ −0.0026 vs FE v2 | **263** | 10.2835 ⚠️ | **`stacking_fe_v2_optuna_lgbm_meta.csv`** |
| 35 | 04.15 | **[FE v4] 위치×신호+가속도+모멘텀 (304피처, RF 5모델)** | 8.7963 | ❌ +0.0121 vs FE v2 | 304 | 미제출 | `stacking_fe_v4_interact_rf_lgbm_meta.csv` |
| 36 | 04.15 | **[FE v1+Cumul] 원본 KEY_COLS(8종)+Cumulative (239피처, RF 5모델)** | **8.7699** | ✅ −0.0212 vs FE v1 | 239 | 10.2517 ⚠️ | `stacking_fe_v1_cumul_rf_lgbm_meta.csv` |
| 37 | 04.15 | **[ExtLag A] lag 1-12 확장 (260피처, RF 5모델)** | **8.7697** | ✅ −0.0214 vs FE v1 | 260 | 미제출 | `stacking_fe_v1_extlag_A_lag_ext_rf_lgbm_meta.csv` |
| 37b | 04.15 | [ExtLag B] rolling 3-20 확장 (244피처) | 8.7719 | ✅ −0.0192 | 244 | 미제출 | `stacking_fe_v1_extlag_B_roll_ext_rf_lgbm_meta.csv` |
| 37c | 04.15 | [ExtLag C] lag+rolling 전체 확장 (292피처) | 8.7732 | ✅ −0.0179 | 292 | 미제출 | `stacking_fe_v1_extlag_C_full_ext_rf_lgbm_meta.csv` |
| 38 | 04.15 | **[모델실험10] HistGradientBoosting 6모델 스태킹 (FE v1 기반)** | 8.7858 (6모델) vs 8.7937 (5모델) | ➖ +0.0026 vs RF5 | 212 | 미제출 | `stacking_6model_histgb_lgbm_meta.csv` |
| 39 | 04.15 | **[모델실험11-A] MLP v1 (early_stop=True, iter=31 조기종료)** | 8.7919 (6모델) | ➖ +0.0008 vs RF5 / MLP OOF 9.8659 과소학습 | 212 | 미제출 | `stacking_6model_mlp_lgbm_meta.csv` |
| 39b | 04.15 | **[모델실험11-B] MLP v2 (early_stop=False, iter=300 과적합)** | — (중단) | ❌ OOF MAE=12.7 / pred_std=72.18 — 시나리오 과적합 치명적 | 212 | 중단 | — |
| 40 | 04.16 | **[모델실험12] LGBM Poisson 6모델 스태킹 (FE v1)** | 8.7782 | ➖ (다양성 유효: LGBM-Poi 0.9348) | 212 | 미제출 | `stacking_6model_poisson_lgbm_meta.csv` |
| — | — | **── v2.0 전환: 시나리오 집계 피처 ──** | — | — | — | — | — |
| 41 | 04.17 | **[모델실험21] 시나리오 집계 FE + 5모델 스태킹 (v2.0)** | **8.5097** | **✅ 돌파** | **302** | **9.9550 🏆** | `sc_agg_stacking_5model.csv` |
| 42 | 04.17 | **[모델실험22] 시나리오집계 11통계 확장 (198피처, 5모델)** | **~8.51** | **✅ Public 최고** | **198** | **9.9385 🏆** | `model22_sc_agg_extended.csv` |
| 43 | 04.17 | [모델실험23] Optuna v2 LGBM+CB 재튜닝 (302피처, 5모델) | **8.5038** | ✅ CV 최고 | 302 | 9.9522 | `model23_optuna_v2.csv` |
| 44 | 04.18 | [모델실험24] 메타 피처 강화 (OOF+sc_mean, CB/XGB/LGBM메타) | 8.5589 (best) | ❌ 정보 중복 → 전면 폐기 | 302+ | 10.0405 ❌ | `model24_meta_enhanced_meta_v3.csv` |

> **v1 시리즈 CV 신기록**: `stacking_fe_v3_cumul_rf_lgbm_meta.csv` (CV **8.7663**)
> **v2 시리즈 Public 최고**: `model22_sc_agg_extended.csv` (Public **9.9385** 🏆) / **CV 최고**: model23 (CV **8.5038**)
> **CV→Public 배율**: v1=1.1627, v2: model22 ~1.168(최고) / model21 1.170 / model23 1.170 / model24 1.173(최악)
> **리더보드**: 1위 9.69923 / 갭 0.239 / **핵심 발견**: 11통계 분포 피처(skew/kurt/cv)가 CV 무관하게 일반화 개선
> **v3.0 Phase 1**: 시퀀스 모델(1D-CNN+BiLSTM) 다양성 탐색 — 시나리오 내 시계열 패턴(자기상관 0.62) 포착 → 7모델 하이브리드 스태킹 목표. 코드: `run_exp_model26_seq_hybrid.py`

### ⚠️ 중요 버그 수정 기록 (04.04)

**버그**: `add_lag_features` / `add_rolling_features` 내부에서 `sort_values(['scenario_id', 'ts_idx'])`를 적용해 test 데이터의 행 순서가 원본 ID 순서와 달라짐. 예측값이 완전히 다른 ID에 할당되어 Public LB = 19.82 (정상 예측의 2배 이상)

**원인**: 원본 test.csv는 ID 순서로 정렬, FE 후 test는 scenario_id 기준으로 정렬됨. `sample_submission.csv`는 ID 순서이므로 예측값이 완전 misaligned

**수정**: `_orig_order` 컬럼을 FE 전에 저장하고 FE 후 `sort_values('_orig_order')`로 원본 순서 복원

**영향 없는 파일**: `groupkfold_lgbm_cv.csv`, `groupkfold_ts_lgbm.csv` (lag/rolling FE 없음)

### 주요 발견 사항

| 발견 | 내용 |
|---|---|
| KFold 리크 | KFold 7.80 vs GroupKFold 9.22 → **1.41분 리크** (시나리오 간 타임슬롯 겹침) |
| log1p 효과 | 왜도 5.68→0.08이지만 L1 LGBM 강건 → 효과 없음 (±0.005) |
| ts 피처 | 0.40% 향상. ts_idx(0→24)는 시나리오 진행에 따른 지연 누적 포착 |
| Lag/Rolling | Rolling avg_trip_distance_roll5_mean이 Top 2 피처. +1.89% 향상 |
| Domain 피처 | 소폭 +0.04분. LGBM이 내부적으로 상호작용 이미 포착 |
| **ID 순서 버그** | **lag/rolling FE 후 test 정렬 꼬임 → Public LB 2배 오차. 수정 완료** |
| **layout_info 추가 전략 Ablation (04.11)** | **one-hot/비율피처/Target Enc/교호작용 모두 CV MAE 악화 → 현재 ordinal 파이프라인이 최적** |

### layout_info Ablation 결과 (04.11, LightGBM 5-fold)

| 실험 | 피처수 | CV MAE | vs Baseline | 판정 |
|---|---|---|---|---|
| Baseline (ordinal layout_type) | 284 | 8.8899 | — | ✅ 현재 파이프라인 |
| A: one-hot layout_type | 287 | 8.9063 | +0.0164 ❌ | 악화 |
| A+B: +파생 비율 피처 6종 | 293 | 8.8910 | +0.0011 ➖ | 무의미 |
| A+B+D: +Target Encoding (OOF) | 295 | 8.9235 | +0.0336 ❌ | 악화 |
| A+B+D+C: +교호작용 피처 | 318 | 8.9388 | +0.0489 ❌ | 악화 |

**원인 분석**:
- **One-hot vs ordinal**: hub_spoke 지연(22.3분) > 나머지(18.1~18.4분) 구조가 ordinal에서 자연스럽게 보존됨. LGBM 트리는 단일 수치 피처를 더 효율적으로 분할
- **비율 피처(B)**: LGBM 트리는 내부 분할로 이미 비율 관계 포착. 명시적 추가는 상관 피처 증가만 유발
- **Target Encoding(D)**: unseen 50개 창고가 global mean으로 fallback → 노이즈 신호 삽입. 기존 직접 피처로 이미 창고별 특성 표현 중
- **교호작용(C)**: 피처 수 318개로 팽창 → feature_fraction 효과 희석, 다중공선성 증가

**결론**: layout_info는 현재 파이프라인(merge + ordinal 인코딩)으로 이미 최적 활용 중. 추가 전략 불필요.

### Optuna 3모델 전체 튜닝 결과 (04.11)

| 모델 | OOF MAE | 가중치 | 비고 |
|---|---|---|---|
| LGBM | 8.8836 | 0.608 | 기존 최적 파라미터 재사용 |
| CatBoost | 8.9125 | 0.354 | Optuna 20 trials 신규 튜닝 |
| XGBoost | 9.3713 | **0.038** | ⚠️ 사실상 제외 수준 |
| 균등 앙상블 | 8.9193 | — | |
| **최적 앙상블** | **8.8674** | — | ↓0.003 vs 이전(8.8703) |

### TS0 Broadcast 피처 Ablation 결과 (04.11)

> **배경**: 추가 EDA — robot_utilization(TS0) r=0.475 vs 전체 r=0.211 (2.3× 신호 강도)
> 시나리오 초기 상태가 25 타임슬롯 전체 결과를 결정한다는 가설 검증

| 실험 | 추가 피처 | 피처수 | CV MAE | Δ Baseline | 판정 |
|---|---|---|---|---|---|
| Exp0 Baseline | — | 172 | 8.9667 | — | 기준 |
| Exp1 TS0_Continuous | TS0 연속형 8종 broadcast | 180 | 8.9550 | −0.0117 | ✅ 개선 |
| Exp2 TS0_Cont+Flags | + 이진 플래그 3종 (blocked/fault/recovery>0) | 183 | 8.9536 | −0.0132 | ✅ 개선 |
| Exp3 TS0_Full | + 복합 취약성 지수 (util×order) | 184 | 8.9529 | −0.0139 | ✅ 개선 |

**주요 발견**:
- TS0 broadcast 피처 전 단계에서 일관된 개선 확인 (가설 검증 성공)
- 연속형 8종이 핵심 신호 (−0.0117). 플래그·복합 지수는 소폭 추가 기여
- 한계: 이 ablation은 log1p 미적용 버전. 앙상블과 결합 시 추가 개선 기대
- **다음 단계**: TS0(Exp3) + log1p transform + LGBM-CB 앙상블 → Public < 10.0 목표

**XGBoost 이슈 분석**:
- Optuna가 reg_alpha≈0.0005, reg_lambda≈0.003 (사실상 정규화 없음) 파라미터를 선택
- 정규화 부재 → 훈련 중 빠른 과적합 → early_stopping 100 rounds에서 조기 종료 (~100~200 trees)
- Fold당 11초 완료가 증거 (정상이면 수분 소요)
- Level-wise 성장 방식(XGB)이 이 데이터셋 구조(시나리오×타임슬롯)에 leaf-wise(LGBM)보다 구조적으로 불리
- **결론**: XGBoost는 이 태스크에서 LGBM·CB 대비 근본적으로 열위. 앙상블 제외 검토 필요

### Transform Ablation 상세 결과 (04.11)

| 변환 방식 | OOF MAE | OOF std | stretch 배율 | stretch 후 MAE | 판정 |
|---|---|---|---|---|---|
| **log1p** | **8.8836** | 12.9492 | ×2.112 | 13.2299 | ✅ 최적 |
| sqrt | 8.8956 | 12.9589 | ×2.111 | 13.2829 | ➖ 근소 열위 |
| identity (변환 없음) | 8.9089 | 13.1249 | ×2.084 | 13.2422 | ❌ 최하 |

**핵심 발견**:
- log1p가 0.0120 차이로 sqrt 대비 우세하나 격차는 미미함 (통계적 유의성 불명확)
- stretch 후처리 (예측 std 보정): OOF MAE 13.2~13.3으로 오히려 대폭 악화 → **stretch는 사용 금지**
- 극값 과소예측 문제(예측 std ~13 vs 실제 std ~27)는 후처리보다 피처/아키텍처로 해결해야 함
- **결론**: log1p 변환 유지, sqrt 대안 가능 (Δ0.0120 차이)

### ensemble_ts0 & 2-Stage 실험 요약 (04.11)

#### ensemble_ts0 (TS0 Broadcast + log1p + 3모델 앙상블) ← **CV 신규 최고**

| 모델 | OOF MAE | 가중치 |
|---|---|---|
| **LGBM** (TS0 + Optuna 신규) | **8.8752** | 0.754 |
| CatBoost | 8.9622 | 0.190 |
| XGBoost | 9.4880 | 0.056 |
| 균등 앙상블 | 8.9369 | — |
| **최적 앙상블** | **8.8649** | — ← **CV 최고** |

- **TS0 피처 12종**: ts0_robot_utilization, ts0_order_inflow_15m, ts0_robot_active, ts0_sku_concentration, ts0_max_zone_density, ts0_congestion_score, ts0_robot_idle, ts0_urgent_order_ratio, ts0_blocked/fault/recovery_flag, ts0_overload_risk
- **LGBM 신규 Optuna 파라미터**: num_leaves=183, lr=0.020703 (기존 181, 0.020616 대비 미세 조정)
- **예측 분포**: mean=18.62, std=13.81, max=86.52 (std 높음 → 극값 더 공격적)
- **Public 미제출**: CV 갭 검증 필요. 제출 강력 권장

#### 2-Stage P90 모델

- **설계**: P90(45.2분) 기준 정상/극값 분리 → Stage1 분류기(AUC 0.8754) + Stage2 체제별 회귀
- **2-Stage 단독**: CV 8.9503 (❌ 기준 8.8836 대비 +0.0667) — 극값 구간 MAE 44.97 (기준 45.60보다 개선이나 전체 악화)
- **콤보 (base×0.75 + 2stage×0.25)**: CV **8.8745** (기준 대비 ↓0.0091 개선, 현재 최고 대비 +0.0096)
- **핵심 원인**: 분류기 P(extreme) 평균 0.097, P(>0.5)=1.0% → 극값에 너무 보수적으로 반응

### DART + P_extreme 메타 피처 실험 결과 (04.12)

#### P_extreme 메타 피처 (분류기 앙상블)
- **아이디어**: 극값(지연 큰 샘플) 예측에 특화된 분류기를 훈련 → 그 확률을 피처로 추가
- **분류기 성능**: AUC 0.8754 (우수)
- **앙상블 CV MAE**: 8.9089 (기존 최고 8.8674 대비 ❌ +0.0415 악화)
- **원인 분석**: P_extreme 신호 자체는 강하지만, 이미 GBDT 앙상블이 내부적으로 극값 패턴을 포착 중. 메타 피처 추가는 중복 신호 → 노이즈 효과

#### DART LightGBM
- **Optuna 탐색**: 20 trials, 2-fold | 최적 DART: 9.0015 (best trial)
- **5-fold OOF**: 8.9661 (GBDT 앙상블 8.8703 대비 ❌ +0.0958)
- **DART 앙상블 (가중치 DART:0.561 / CB:0.339 / XGB:0.100)**: 8.9221 ❌
- **원인 분석**:
  - DART의 드롭아웃 정규화는 모델이 특정 트리에 과도하게 의존할 때 유효
  - 이미 잘 튜닝된 GBDT(num_leaves=181, learning_rate=0.02)에서는 추가 정규화 이점 없음
  - DART n_estimators=734 (낮음) → 충분한 학습 라운드 미확보
- **결론**: DART는 이 데이터셋에서 GBDT 대비 우위 없음. 향후 재시도 불필요

**핵심 인사이트 (04.12)**: CV→Public 갭(8.87→10.33, Δ~1.46)이 큰 상황. 더 복잡한 앙상블보다 **타깃 변환(sqrt) + 깔끔한 피처**로 일반화 개선이 우선.

## Setup

```bash
pip install -r requirements.txt
```
