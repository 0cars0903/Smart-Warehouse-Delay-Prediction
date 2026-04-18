# CLAUDE.md — Smart Warehouse Delay Prediction

> 이 파일은 Claude가 이 프로젝트 폴더를 열 때마다 자동으로 읽는 지침서입니다.
> 최종 수정: 2026-04-05

---

## 1. Jupyter 노트북 실행 방식 (변경됨)

### ✅ 신규 방식: 하이브리드 (Claude in Chrome + Python 스크립트)

**토큰 효율 분석**:
- 구 방식(`nbconvert`): 모든 학습 로그(수천 줄)가 컨텍스트로 유입 → 토큰 낭비, OOM/타임아웃 위험
- 신규 방식: USER가 로컬에서 실행, Claude는 Chrome으로 결과만 확인 → **약 70~80% 토큰 절감 추정**

### 작업별 실행 주체

| 작업 유형 | 실행 주체 | 방법 |
|---|---|---|
| **무거운 훈련** (5-fold × 다중 모델, >10분) | **USER 로컬 실행** | Jupyter에서 직접 Run All |
| **Claude 디버깅/결과 확인** | Claude (Chrome) | localhost:8888 접속, 셀 출력 확인 |
| **경량 실험** (<5분, 단순 검증) | Claude (Python 스크립트) | `python3 src/스크립트.py` |
| **피처 엔지니어링 코드 작성** | Claude (Write/Edit 도구) | `.py` 파일 직접 작성 |
| **제출 파일 생성** | Claude (Python 스크립트) | 모델 로드 후 예측, CSV 저장 |

### Chrome 디버깅 플로우

```
1. USER: 로컬 Jupyter에서 노트북 실행 (localhost:8888)
2. USER: "실행 완료, 확인해줘" 메시지
3. Claude: Chrome으로 localhost:8888 접속
4. Claude: 해당 노트북 열기 → 오류 셀 확인 → 결과 셀 파싱
5. Claude: 개선 코드 작성 후 보고
```

### 예외: Claude가 직접 실행 가능한 경우
- 단일 Python 스크립트 (결과 출력 간결함)
- 검증/체크 코드 (assert, 통계 확인 등)
- 제출 파일 생성 (저장만 하면 됨)

---

## 2. 자율 추진 지침

**기본 원칙**: 사용자 지시 없이도 토큰 여유 시 다음 순서로 자율 추진

```
1순위: 당일 스케줄 미완료 항목
2순위: Python 스크립트 실행 (경량 실험)
3순위: 다음 실험 코드 초안 작성
4순위: Notion Test Case DB 업데이트
5순위: README Approach Log 업데이트
```

### 자율 추진 가능 업무

| 업무 | 자율 여부 | 비고 |
|---|---|---|
| 피처 엔지니어링 코드 작성 | ✅ 자율 | |
| Python 스크립트 실행 (<5분) | ✅ 자율 | |
| 제출 CSV 생성 | ✅ 자율 | |
| Notion DB 업데이트 | ✅ 자율 | |
| README Approach Log 업데이트 | ✅ 자율 | |
| Chrome으로 Jupyter 결과 확인 | ✅ 자율 | USER 실행 후 |
| 노트북 실행 (무거운 훈련) | ❌ USER 직접 | 로컬에서 실행 |
| Public LB 제출 | ❌ USER 직접 | |
| 하이퍼파라미터 대규모 탐색 (>30분) | ⚠️ 사전 확인 | |

---

## 3. 실험 기록 규칙

실험 완료 즉시 **두 곳 동시 업데이트**:
1. `README.md` Approach Log
2. Notion Test Case 데이터베이스

### Approach Log 형식
```
| # | 날짜 | 실험명 | CV MAE | Public LB | 제출 파일 | 핵심 변경 |
```

---

## 4. 파일 구조

```
Smart-Warehouse-Delay-Prediction/
├── data/              ← 원본 데이터
├── notebooks/         ← USER가 로컬 실행하는 노트북
├── src/               ← Claude가 작성하는 Python 모듈
│   ├── feature_engineering.py
│   ├── run_ensemble.py    ← v1 앙상블 (Public: 10.3349)
│   └── run_v2.py          ← v2 (OOM 이슈, 경량화 필요)
├── submissions/       ← 제출 CSV
└── docs/              ← 시각화 PNG, 문서
```

---

## 5. 현재 진행 상황 (2026-04-17)

| 버전 | CV MAE | Public Score | 비고 |
|---|---|---|---|
| Full FE + LightGBM | 9.0010 | 10.4936 | 버그 수정 후 제출 |
| Optuna LGBM 단독 | 8.8895 | 10.3807 | |
| Optuna 앙상블 v1 (LGBM+CB+XGB) | 8.8703 | 10.3349 | |
| Optuna 앙상블 v2 (CB·XGB 신규 튜닝) | 8.8674 | **10.3347** | ✅ Public 최고 |
| ensemble_ts0 (TS0 12종 + 앙상블) | 8.8649 | 10.4091 | Public 역전 (갭 1.544, 과적합) |
| [일반화] sqrt+log1p 블렌드 (212피처) | 8.8749 | 10.3674 | sqrt×0.70 + log1p×0.30 |
| [일반화] 피처 10% 컷 LGBM+CB (189피처) | 8.8871 | 10.3662 | |
| [모델실험1] Tweedie(1.5)+CB 앙상블 | 8.8593 | 미제출 | 제출 우선순위 2위 |
| [모델실험2] Quantile 4모델 (q×3+CB) | 8.8697 | 미제출 | |
| [모델실험3] Stacking LGBM-meta (LGBM+CB+ET) | 8.8541 | 10.3032 | |
| [모델실험4] Stacking v2 (LGBM+TW1.8+ET) | 8.8087 | 10.3118 | ⚠️ CV 개선이나 Public 역전 |
| **[모델실험5] Stacking v3 4모델 (LGBM+TW1.8+CB+ET)** | **8.7929** | **10.2264** | **✅ CV + Public 동시 최고 🏆** |
| [모델실험6] Optuna 메타 LGBM (N=50) | 8.7929 | 10.2273 | Optuna = 기준과 동일 (메타 파라미터 이미 최적) |
| [모델실험7-A] Ridge×0.05+LGBM×0.95 블렌드 | **8.7927** | 10.2309 ⚠️ | CV↑나 Public↓ — OOF 과적합 의심 |
| [모델실험7-B] Ridge+LGBM 0.5:0.5 블렌드 | 8.8123 | 미제출 | Ridge 비중 과다 → 성능 하락 |
| [모델실험9] 5모델+Q05 스태킹 (LGBM-meta) | 8.7938 | 10.2358 ❌ | Q05 상관 0.97~0.99 → 다양성 無 |
| **[모델실험8] Stacking 5모델+RF (LGBM+TW1.8+CB+ET+RF)** | **8.7911** | **10.2213** | **✅ CV + Public 동시 최고 🏆** |
| **[FE v2] KEY_COLS 개선+Delta+Layout비율 (264피처, RF 5모델)** | **8.7842** | **10.2801 ⚠️** | CV 개선이나 Public 역전 — Delta 피처 과적합 의심 |
| [Ablation] FE v2 no-delta (252피처, RF 5모델) | 8.7836 | 10.2829 ⚠️ | Delta 제거 시 CV≈FE v2 → Delta 무효 확정 |
| **[FE v3] Cumulative (281피처, RF 5모델)** | **8.7663** | **10.2571 ⚠️** | **✅ CV 신기록 🏆** |
| **[FE v2+Optuna A] LGBM 재튜닝 (263피처)** | **8.7816** | 10.2835 ⚠️ | LGBM OOF 8.9308→8.9023 ✅, 메타 개선 미미 |
| [FE v4] 위치×신호+가속도+모멘텀 (304피처) | 8.7963 | 미제출 | ❌ FE v2보다 악화 — 가속도 NaN 채움 bias |
| **[FE v1+Cumul] 원본 KEY_COLS(8종)+Cumulative (239피처, RF 5모델)** | **8.7699** | 10.2517 ⚠️ | std=21.51 회복이나 배율 1.1700 — Cumulative가 진짜 원인 확정 |
| [ExtLag A] lag 1-12 확장 (260피처, RF 5모델) | 8.7697 | 미제출 | pred_std=13.88 → 배율 1.170 예상 |
| [ExtLag B] rolling 3-20 확장 (244피처, RF 5모델) | 8.7719 | 미제출 | pred_std=13.79 → 배율 1.170 예상 |
| [ExtLag C] lag+rolling 전체 확장 (292피처, RF 5모델) | 8.7732 | 미제출 | pred_std=13.88 → 배율 1.170 예상 |
| **[모델실험10] HGB 6모델 스태킹 (FE v1+HGB → LGBM-meta)** | 8.7858 (6모델) | 미제출 | LGBM-HGB 상관 **0.9862** → 다양성 無, 방향 폐기 |
| [모델실험12] Poisson 6모델 스태킹 (FE v1) | 8.7782 | 미제출 | Poisson 다양성 유효(LGBM-Poi 0.9348)이나 pred_std=14.10 압축 |
| **[모델실험21] 시나리오 집계 FE + 5모델 스태킹 (v2.0)** | **8.5097** | **9.9550** | 배율 1.1699 |
| **[모델실험22] 시나리오집계 11통계 확장 (198피처, 5모델)** | **~8.51** | **9.9385** | **✅ Public 최고 🏆 배율 ~1.168** |
| [모델실험23] Optuna v2 LGBM+CB 재튜닝 (302피처, 5모델) | **8.5038** | 9.9522 | ✅ CV 최고, 배율 1.1703 |
| [모델실험24] 메타 피처 강화 (OOF+sc_mean, CB/XGB/LGBM메타) | 8.5589 (best) | 10.0405 | ❌ 메타에 집계 추가 = 정보 중복 → 전면 폐기 |

**리더보드**: Public 최고 model22 = 9.9385 / 1위 9.69923 / 갭 0.239

**모델실험21 주요 수치 (04.17 — v2.0 시작)**:
- 핵심 변경: 시나리오 집계 피처 90종 (18개 원본 피처 × mean/std/max/min/diff) broadcast
- 원자 분석 발견: 시나리오 간 분산 63.4%, 기존 모델은 모든 구간에서 ~15분 상수 예측
- LGBM OOF 8.6237, TW OOF 8.8209, CB OOF 8.7212, ET OOF 8.7962, RF OOF 8.7985
- LGBM-ET 상관 0.9661 (기존 0.9744 대비 다양성 ↑), TW-ET 0.8994 (최저)
- 가중 앙상블 8.5872 (LGBM=0.672, TW=0.268)
- 메타 CV **8.5097**, Fold: 8.4747 / 8.5946 / 8.0752 / 8.9277 / 8.4762
- pred_std=**14.47** (기존 13.2 대비 확장 — 극값 예측 개선 신호)
- 기대 Public (×1.1627): **9.89**, (×1.1700): **9.96** — 어느 배율이든 현 최고(10.22) 초월

**모델실험23 주요 수치 (04.17 — Optuna v2)**:
- Optuna LGBM: num_leaves=145, lr=0.00866, feat_frac=0.483, 2-fold MAE=8.648
- Optuna CB: depth=9, lr=0.01832, l2_leaf_reg=9.91, 2-fold MAE=8.711
- LGBM OOF 8.6237→**8.6143** (Δ-0.009), CB OOF 8.7212→**8.6564** (Δ-0.065)
- TW OOF 8.8209, ET OOF 8.7962, RF OOF 8.7985 (재사용)
- 가중 앙상블 8.5811, 메타 CV **8.5038** (model21 대비 Δ-0.006)
- pred_std=14.46 (model21과 동일)
- **Public 9.9522 → 배율 1.1703 (model21 1.1699와 동일 수준)**
- CV 최고이나 Public은 model22에 밀림

**모델실험22 주요 수치 (04.18 — Public 제출 확인)**:
- 11통계 확장: mean/std/max/min/diff + median/p10/p90/skew/kurtosis/cv = 198 sc피처
- CV ~8.51 (정확한 재현 필요), Public **9.9385** ← **v2 시리즈 Public 최고 🏆**
- 제출 예측 통계: mean=19.36, **std=15.27**, max=96.49
- model21(std=15.65), model23(std=15.73) 대비 적절한 압축 → 일반화 유리
- **배율 ~1.168** — v2 시리즈 최저(최고) 배율
- **해석**: 추가 분포 통계(skew/kurt/cv)가 CV에는 기여 없지만 test 분포 대응력 ↑

**모델실험24 주요 수치 (04.18 — 메타 피처 강화 실험)**:
- model21 base OOF 재사용 + 시나리오 집계 피처를 메타에 추가
- meta_v1 (LGBM + 18 sc_mean): 8.5729 ❌, meta_v2 (LGBM + 90 sc_*): 8.5801 ❌
- meta_v3 (CB + 18 sc_mean): 8.5589 ❌, meta_v4 (XGB + 18 sc_mean): 8.5688 ❌
- **Public 10.0405** → 배율 1.1731 (v2 시리즈 최악)
- 제출 예측 통계: mean=19.40, **std=15.15, max=73.64** ← 극단적 압축
- **결론**: base learner가 이미 집계 피처 학습 → 메타에 같은 피처 추가 = 정보 중복 + 예측 범위 축소 → 전면 폐기
- **교훈**: 메타에 유의미한 정보를 넣으려면 base와 이질적인 신호(시퀀스 모델 등) 필요

**v2 시리즈 배율 종합 (04.18)**:
- model22: **~1.168** ← 최저(최고) — 11통계 분포 정보가 일반화 기여
- model21: 1.1699
- model23: 1.1703 — Optuna 튜닝은 CV↑ 하지만 일반화 이점 없음
- model24: 1.1731 ← 최악 — 메타 정보 중복으로 과적합

**FE v1+Cumul 주요 수치 (04.15 — 가설 수정)**:
- LGBM OOF 8.9029, TW OOF 8.9101, CB OOF 8.9732, ET OOF 9.3350, RF OOF 9.4205
- LGBM-ET 상관 0.9128, LGBM-RF 0.9226 (다양성 유지)
- 메타 CV **8.7699**, Fold: 8.7245 / 8.8491 / 8.3609 / 9.1074 / 8.8076
- 예측 std=21.51 (FE v3 13.76 대비 회복), 실제 Public: **10.2517** → 배율 **1.1700** ⚠️

**배율 수렴 최종 분석 (04.15 — 가설 완전 수정)**:
- RF 5모델(FE v1, lag/rolling만): 배율 **1.1627** ← 유일한 기준
- FE v2 no-delta(KEY_COLS_V2, cumul 없음): 1.1707
- FE v3(KEY_COLS_V2 + cumul): 1.1701
- **FE v1+Cumul(KEY_COLS_V1 + cumul): 1.1700** ← KEY_COLS_V1도 cumul 추가 시 1.170
- **최종 결론: 배율 악화의 진짜 원인은 Cumulative 피처 자체**
- KEY_COLS_V2 확장은 거기에 더해 std도 13.76으로 압축시키는 이중 악영향
- Huber 메타: CV 8.8437(악화), pred std=12.41(압축), iter=1000(수렴 실패) → 완전 폐기

**Ablation 최종 결론**:
- Delta 피처: 배율 악화 + CV 기여 없음 → 전면 제외 확정
- KEY_COLS_V2 확장(4종): 배율 악화 + std 압축 → 전면 제외 확정
- **Cumulative 피처**: 배율 1.1627→1.1700 악화 → 전면 제외 확정
- **현재 유일 최강 기준**: RF 5모델 (FE v1, lag/rolling만, CV 8.7911, Public 10.2213)

### 알려진 문제 (v1 시리즈, 04.15 기준)
- CV→Public 배율 1.1627 (RF 5모델만 유지, 나머지 모두 1.170)
- FE 확장 방향 **완전 차단**: Delta/KEY_COLS_V2/Cumulative/ExtLag 모두 pred_std ~13, 배율 1.170 수렴
- 모델 다양성 탐색 차단: HGB/Q05/MLP 모두 실패

### v2.0 전환 (04.17 — 원자 분석 기반)
- **핵심 발견**: 시나리오 간 분산 63.4% → 시나리오 레벨 구분이 관건
- **돌파구**: 시나리오 집계 피처(25행 전체의 mean/std/max/min/diff) broadcast
  - test에서도 시나리오 25행 피처를 모두 볼 수 있으므로 리크 아님
  - 단일 LGBM 8.62 → 5모델 스태킹 8.79를 초월 (시나리오 레벨 구분 시작)
- **model21**: 시나리오 집계 90종 + FE v1 212종 = 302피처, 5모델 스태킹 → CV **8.5097**
- **다음 탐색**: Public 제출 후 배율 확인, 시나리오 집계 Optuna 튜닝, 2단계 분리 모델 결합

### 완료된 실험 (04.11~04.12)
- Transform Ablation: log1p 최적 확정 (stretch 후처리 금지)
- TS0 Broadcast: 12종 피처 → Public 역전으로 과적합 판정
- 2-Stage P90 / P_extreme 메타 / DART 앙상블: 전부 무효
- **[일반화]** sqrt > log1p (CV Δ0.0137), 블렌드(8.8749)가 최고; 피처 10% 컷 (Δ0.0052)
- **[모델실험1] Tweedie**: p=1.5+CB CV 8.8593, p=1.8이 단독 최고, p=1.2/1.5는 블렌드서 가중치 0으로 탈락
- **[모델실험2] Quantile**: q×3+CB CV 8.8697, std 개선 없음 (13.5 유지)
- **[모델실험3] Stacking**: LGBM-meta CV **8.8541** / Public **10.3032** 🏆; Ridge-meta는 역효과(8.9152)
  - ET coef ~0.44 (Ridge 기준): GBDT와 독립 오차 패턴 확인 (LGBM-ET 상관 0.9744, CB-ET 0.9685)
  - CB는 LGBM과 너무 유사(0.9788) → 스태킹 기여 제한적
- **[모델실험4] Stacking v2**: CB→Tweedie(1.8) 교체, LGBM-meta CV **8.8087** / Public **10.3118**
  - LGBM-TW 상관 0.9597 (v1 0.9788 대비 독립성 향상), TW-ET 상관 0.9438 (전체 최저)
  - 메타 학습기 iter: 372/268/304/143/67 — pred std=13.65 (예측 범위 확장)
  - **⚠️ CV 개선(−0.0454)이나 Public 역전(+0.0086)**: 배율 v1=1.1637 vs v2=1.1706 → TW1.8 CV 과적합
  - CB가 Public 일반화에 더 유리했던 것으로 해석 → 4모델(LGBM+TW1.8+CB+ET) 검토
- **[모델실험5] Stacking v3 4모델**: LGBM+TW1.8+CB+ET → LGBM-meta, CV **8.7929** / Public **10.2264** 🏆
  - CB-TW 상관 0.9480으로 예상보다 독립적 → 4모델 조합에서 시너지 발생
  - 4모델 가중치 앙상블: LGBM=0.484, CB=0.155, TW1.8=0.339, ET=0.023
  - 메타 iter: 162/112/202/193/69 → v2(372/268/304/143/67)보다 안정적
- **[모델실험6] Optuna 메타 튜닝**: N=50 trials, 메타 CV 8.7929 → 기준과 완전 동일. 메타 파라미터 이미 최적 확인
  - Fold별 iter: 390/276/406/259/130
- **[모델실험7] Ridge+LGBM 메타 블렌드**:
  - Ridge-meta CV: 8.8825 (LGBM 단독 8.7929보다 훨씬 나쁨)
  - 최적 블렌드(Ridge=0.05, LGBM=0.95): CV **8.7927** (Δ−0.0002, 미세 개선)
  - 0.5:0.5 블렌드: CV 8.8123 → Ridge 비중이 높을수록 성능 하락
  - Ridge 기여 사실상 극소 → 실질적 의미 없음
- **[FE v2] KEY_COLS 개선 + Delta + Layout 비율 피처**: CV **8.7842** / Public **10.2801 ⚠️ 역전**
  - 264 피처 (기존 212): avg_trip_distance 제거(r=0.021), robot_charging/battery_std/sku_concentration/urgent_order_ratio 추가
  - Delta 피처(11종): col_diff1 = col - col_lag1 (변화율 포착)
  - Layout 비율 피처(4종): robot_active_ratio, charging_saturation, charger_per_robot, orders_per_robot_total
  - **핵심 변화**: LGBM-ET 상관 0.9744→**0.9142**, LGBM-RF 0.9749→**0.9244** (다양성 폭발)
  - 개별 모델 성능은 소폭 저하(LGBM 8.9308, CB 8.9726)이나 메타 LGBM에서 다양성 활용 → CV 8.7842
  - 가중 앙상블: 8.8589 (TW=0.464, LGBM=0.261, CB=0.235, ET=0.040, RF=0.000)
  - 메타 iter: 146/78/176/156/51 — Fold5 불안정
  - **⚠️ Public 역전**: 배율 1.1627→1.1703. Delta 피처가 노이즈 기반 다양성 유발로 추정
- **[Ablation] FE v2 no-delta (252피처)**: CV **8.7836**
  - Delta(diff1) 11종만 제거, 나머지 KEY_COLS_V2 확장 + Layout 비율 피처 유지
  - **핵심 발견**: LGBM-ET 0.9143 ≈ FE v2 full 0.9142 → 다양성은 Delta가 아닌 KEY_COLS 확장에서 온 것
  - CV 8.7836 ≈ FE v2 full 8.7842 → Delta 피처는 다양성·CV 개선에 기여 없음 확정
  - **Delta 최종 판정**: 이후 모든 실험에서 제외. KEY_COLS 확장(+ Layout 비율)은 유지
- **[FE v3] Cumulative 피처 (281피처)**: CV **8.7663** 🏆 신기록
  - FE v2 no-delta 기반에 cumulative 피처(cummin/cummax/cumsum) 추가: 18종
  - LGBM OOF 8.9155 (FE v2 8.9308 대비 개선), TW OOF 8.8862
  - LGBM-ET 상관 0.9100 (다양성 유지), 가중 앙상블 8.8438 (TW=0.488, LGBM=0.319, CB=0.151)
  - 메타 iter: 140/69/112/192/53
  - 기대 Public: 8.7663 × 1.1627 ≈ **10.191** → 제출 최우선
- **[FE v2+Optuna A] LGBM 파라미터 재튜닝 (263피처)**: CV **8.7816**
  - Optuna 50 trials 2-fold 최적: num_leaves=226, lr=0.01014, feat_frac=0.7993
  - LGBM OOF: 8.9308→**8.9023** (개선 Δ0.0285), CB/TW/ET/RF는 FE v2 체크포인트 재사용
  - LGBM-ET 상관 0.9225 (FE v2 0.9142보다 올라감 — feat_frac 증가로 다양성 감소)
  - 메타 CV 8.7816 (FE v2 8.7842 대비 미미한 개선, FE v3 8.7663보다 나쁨)
  - **결론**: LGBM 단독 재튜닝은 스태킹 메타 성능 개선 효과 제한적 (다양성 상쇄)
- **[FE v4] 위치×신호+가속도+모멘텀 (304피처)**: CV **8.7963** ❌
  - LGBM-ET 상관 0.9178 (FE v3 0.9100보다 높음), CB-ET 0.8947 (최저!) — 다양성 혼재
  - CB-ET 최저 상관에도 불구하고 메타 CV 8.7963 (FE v2 8.7842보다 나쁨)
  - 의심 원인: 가속도/모멘텀 피처의 NaN→0 채움이 ET/RF에 systematic bias 유발
  - **결론**: 해당 FE 방향 중단. NaN 기반 피처는 ET/RF 학습에 적합하지 않음
- **[모델실험9] Q05 5모델 스태킹**: Q05(q=0.5) 추가, OOF 8.9089, 기존 모델과 상관 0.97~0.99
  - 메타 CV 8.7938 (v3 대비 +0.0009 소폭 악화) → 다양성 기여 제한적
  - Q05 단일 OOF 8.9089 (실험2: 8.9084와 동일 수준)
- **[모델실험8] RF 5모델 스태킹**: LGBM+TW1.8+CB+ET+RF → LGBM-meta, CV **8.7911** / Public **10.2213** 🏆
  - RF OOF MAE: 9.0254 (std=12.62), ET(9.0013)보다 단독 성능은 열위
  - RF 상관: RF-LGBM=0.9749, RF-CB=0.9679, RF-TW=0.9439, **RF-ET=0.9960** (ET와 거의 동일)
  - 가중 앙상블에서 RF 가중치=0.000 (단독 성능 열위로 탈락)
  - **메타 LGBM에서 기여 발생**: CV 8.7911 (v3 대비 −0.0018 개선)
  - 메타 iter: 119/137/194/158/70 — 불안정(fold 간 편차 큼)
  - Public LB: 10.2213 (배율 1.1627, 역대 최저)
- **[FE v1+Cumul] 원본 KEY_COLS(8종)+Cumulative 피처 (239피처, RF 5모델)**: CV **8.7699** / Public **10.2517 ⚠️**
  - LGBM OOF 8.9029, TW OOF 8.9101, CB OOF 8.9732, ET OOF 9.3350, RF OOF 9.4205
  - LGBM-ET 상관 0.9128, LGBM-RF 0.9226 (다양성 유지), ET-RF 0.9902
  - 메타 iter: 186/71/171/204/54
  - **예측 std=21.51** (FE v3 13.76 대비 대폭 회복! 실제 27.4에 근접)
  - **핵심 발견**: KEY_COLS_V2 확장이 배율 악화 + std 압축 동시 야기. 원본 KEY_COLS 복원 시 자동 해소
  - 실제 Public 10.2517 → 배율 1.1700 ⚠️ → **Cumulative가 배율 악화의 진짜 원인 최종 확정**
- **[ExtLag A/B/C] FE v1 기반 lag/rolling 확장**: 배율 수렴 가설 검증
  - A(lag 1-12, 260피처) CV 8.7697 / B(roll 3-20, 244피처) CV 8.7719 / C(전체확장, 292피처) CV 8.7732
  - 모두 pred_std ~13.8 (FE v3 13.76과 동일 수준) → **배율 1.170 예상 → FE 확장 방향 완전 종결**
  - LGBM-ET 상관: A=0.9235, B=0.9171, C=0.9140 (FE v1 대비 소폭 개선이나 무의미)
- **[모델실험10] HistGradientBoosting 6모델 스태킹 (FE v1 기반)**: CV 8.7858 / 미제출
  - HGB OOF MAE: 9.0~9.1 수준 (ET/RF와 유사)
  - **LGBM-HGB 상관: 0.9862** — Histogram-based GBDT = LightGBM과 오차 패턴 거의 동일
  - 6모델 meta CV 8.7858 vs 5모델 8.7937 (Δ−0.0079 미미, 배율 추정 1.170)
  - **결론**: HGB 방향 완전 폐기. 트리 기반 모델 다양성 한계 도달
- **[모델실험11-A] sklearn MLP v1 (early_stopping=True)**: CV 8.7919 / 미제출
  - LGBM-MLP 상관: **0.8043** — 트리 대비 획기적 다양성 ✅
  - 그러나 iter=31 조기종료 → MLP OOF MAE=9.8659 (과소학습) → meta 기여 불충분
  - MLP-TW 상관 0.0028: TW expm1 overflow로 인한 수치 오류 (의미 없음)
  - meta CV 8.7919 (RF5 8.7911 대비 +0.0008 소폭 악화)
- **[모델실험11-B] sklearn MLP v2 (early_stopping=False, 300 iter)**: 중단
  - OOF MAE=12.7 (완료 fold 기준), pred_std=72.18 (실제 27.35의 2.6배) — 심각한 과적합
  - **근본 원인**: GroupKFold 시나리오 분리 환경에서 lag/rolling 피처가 시나리오별 고유 패턴 암기
  - 트리(ET/RF)는 분포 차이에 강건, MLP는 취약 → v1(과소학습)↔v2(과적합) 사이 실용적 균형점 없음
  - **결론**: sklearn MLP 방향 완전 폐기

---

## 6. 핵심 하이퍼파라미터 (재사용)

```python
BEST_LGBM_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': 42,
}
```
