# 📅 일별 압축 작업 스케줄

> 대회 기간: 2026.04.04 ~ 05.04 (31일) | 코드·PPT 제출: 05.07

---

## 🔑 핵심 데이터 인사이트 (계획 수립 근거)

| 항목 | 수치 | 시사점 |
|---|---|---|
| 타겟 왜도(skewness) | 5.68 → log1p 후 **0.08** | 로그 변환 필수, 큰 효과 기대 |
| 결측치 패턴 | 타임슬롯 무관하게 ~12% 균등 | MCAR → LightGBM 내장 처리 충분 |
| 시나리오 간 평균 지연 범위 | 0 ~ 226분 | 시나리오 수준 집계 피처가 핵심 |
| 타임슬롯 진행별 지연 | ts=0: 11.3분 → ts=24: 21.9분 | ts_idx 피처 반드시 추가 |
| 배터리 관련 상관계수 | 0.32 ~ 0.37 (1~5위) | 배터리 복합 피처 우선 집중 |

---

## 🗓️ Week 1 (04.04~04.10) — EDA 완성 + 첫 제출

### Day 1 (04.04 토) — 기반 정비 ✅ 완료
- [x] 데이터 구조 파악 완료 (250K×94, 10K 시나리오)
- [x] `02_Baseline_Model.ipynb` 타겟 컬럼명 버그 수정
  - `'avg_delay'` → `'avg_delay_minutes_next_30m'`
  - id_cols 오탐지 버그 수정 (EXCLUDE_COLS 명시적 지정)
  - layout_info merge 추가 (14개 피처, 총 104개)
  - 5-Fold CV MAE = **7.3351** (std=0.049, 안정적)
  - 제출 파일: `submissions/baseline_lgbm_mae7.3351.csv`
- [x] requirements.txt 의존성 확인 및 설치 (lgbm, xgb, catboost, optuna, sklearn)
- [x] EDA 노트북 실행 완료 (24개 셀, 오류 0)
- [x] 도메인 지식 문서 작성 (`docs/domain_knowledge.md`)
- [x] LLM 인사이트 프롬프트 작성 (`docs/llm_insight_prompt.md`)

### Day 2 (04.05 일) — EDA 노트북 완성 ✅ 완료 (04.04 조기 완료)
- [x] `01_EDA.ipynb` 전체 실행 + 인사이트 채우기
  - 타겟 분포 시각화 (원본 vs log1p) → `docs/target_distribution.png`
  - 타임슬롯별 타겟 변화 (선그래프) → `docs/timeslot_delay_pattern.png`
  - layout_type별 박스플롯 → `docs/layout_type_delay_dist.png`
  - 결측치 히트맵 → `docs/missing_value_heatmap.png`
  - 상관관계 Top 20 바차트 → `docs/feature_correlation.png`
- [x] EDA 결과를 `01_EDA.ipynb` 섹션 9에 인사이트로 정리 (26셀, 오류 0)
- [x] 미리 준비: `notebooks/03_CV_Strategy.ipynb` 초안 (GroupKFold vs KFold)
- [x] 미리 준비: `notebooks/04_Log_Transform.ipynb` 초안 (log1p 변환)

### Day 3 (04.06 월) — 첫 제출 ⏳ 사용자 수동 제출 대기
- [x] Baseline 실행 완료 (GroupKFold 5-Fold MAE=9.2156)
- [x] layout_info merge 포함 (14개 컬럼, 총 104 피처)
- [x] `submissions/groupkfold_lgbm_cv.csv` 생성
- [ ] **Public LB 첫 제출** → 기준점 확보 (**사용자가 직접 업로드**)
- [x] README Approach Log 업데이트

### Day 4 (04.07 화) — CV 전략 검증 ✅ 완료 (04.04 조기)
- [x] `GroupKFold(groups=scenario_id)` vs `KFold` 비교 실험 (`03_CV_Strategy.ipynb`)
  - KFold MAE=7.8033, GroupKFold MAE=9.2156 → **리크 차이 1.41분**
- [x] GroupKFold 채택 → 이후 모든 실험에 고정

### Day 5 (04.08 수) — 로그 변환 ✅ 완료 (04.04 조기)
- [x] `log1p(y)` 실험: 원본 9.2154 vs log1p 9.2203 → **차이 미미, 원본 유지**
  - LightGBM L1은 왜도에 강건함 (기대보다 효과 작음)
- [x] `submissions/groupkfold_orig_lgbm.csv` 생성

### Day 6 (04.09 목) — 타임슬롯 피처 ✅ 완료 (04.04 조기)
- [x] `ts_idx`, `ts_ratio`, `ts_sin`, `ts_cos` 추가 (`05_TS_Features.ipynb`)
- [x] 효과: 9.2154 → 9.1790 (−0.40%)
- [x] `submissions/groupkfold_ts_lgbm.csv` 생성

### Day 7 (04.10 금) — 주간 정리 + 계획 보정 ✅ 완료 (04.04 조기)
- [x] Week 1 실험 결과 정리 (README Approach Log 업데이트)
- [x] GroupKFold + ts 피처 확정 → Week 2 전략 실행

---

## 🗓️ Week 2 (04.11~04.17) — 피처 엔지니어링 집중

> **목표**: 단일 피처셋 완성으로 베이스라인 대비 15~25% MAE 감소

### Day 8 (04.11 토) — Lag 피처 ✅ 완료 (04.04 조기)
```python
key_cols = ['low_battery_ratio', 'battery_mean', 'order_inflow_15m',
            'congestion_score', 'robot_idle', 'charge_queue_length',
            'max_zone_density', 'avg_trip_distance']

for col in key_cols:
    df[f'{col}_lag1'] = df.groupby('scenario_id')[col].shift(1)
    df[f'{col}_lag2'] = df.groupby('scenario_id')[col].shift(2)
    df[f'{col}_lag3'] = df.groupby('scenario_id')[col].shift(3)
```
- [x] Lag 피처 추가 후 LightGBM 재학습: 9.1790 → 9.0793 (−1.09%)
- [x] Rolling 피처도 상위 등장 (avg_trip_distance_roll5_mean #2위)

### Day 9 (04.12 일) — Rolling 피처 ✅ 완료 (04.04 조기)
```python
for col in key_cols:
    grp = df.groupby('scenario_id')[col]
    df[f'{col}_roll3_mean'] = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df[f'{col}_roll3_std']  = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())
    df[f'{col}_roll5_mean'] = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
```
- [x] Rolling 피처 추가 후 MAE: 9.0793 → 9.0052 (−0.80%, 누적 −1.89%)
- [x] 추가 피처 수 32개 (8 key_cols × 2 windows × 2 stats)

### Day 10 (04.13 월) — Expanding/누적 피처
```python
for col in key_cols:
    grp = df.groupby('scenario_id')[col]
    df[f'{col}_cummax']  = grp.transform(lambda x: x.shift(1).expanding().max())
    df[f'{col}_cummean'] = grp.transform(lambda x: x.shift(1).expanding().mean())
    df[f'{col}_cumstd']  = grp.transform(lambda x: x.shift(1).expanding().std())
```
- [ ] 시나리오 첫 타임슬롯(ts_idx=0)에서 NaN 처리 방식 결정
- [ ] Expanding 피처 중요도 확인

### Day 11 (04.14 화) — 배터리 복합 피처 (상관관계 1~5위)
```python
# 배터리 위기 지수
df['battery_crisis_score'] = (
    df['low_battery_ratio'] * df['charge_queue_length'] +
    (1 - df['battery_mean'] / 100) * df['robot_charging']
)
# 사용 가능 로봇 비율
df['available_robot_ratio'] = df['robot_active'] / (df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1)
# 충전 병목 지수
df['charge_bottleneck'] = df['charge_queue_length'] / (df['charger_count'] + 1)  # layout merge 후
```
- [ ] 도메인 기반 복합 피처 5~10개 생성
- [ ] 피처 중요도 변화 분석

### Day 12 (04.15 수) — 혼잡도 + 주문 복합 피처
```python
# 주문 압박 지수
df['order_pressure'] = df['order_inflow_15m'] * df['urgent_order_ratio']
# 혼잡 + 주문 상호작용
df['congestion_x_order'] = df['congestion_score'] * df['order_inflow_15m']
# 로봇 대비 주문량
df['orders_per_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)
# SKU 다양성 × 주문량
df['sku_order_complexity'] = df['unique_sku_15m'] * df['avg_items_per_order']
```
- [ ] 상호작용 피처 5~8개 생성 + MAE 확인

### Day 13 (04.16 목) — 피처 선택 + 정리
- [ ] Feature Importance (gain) 기반 하위 10% 피처 제거 실험
- [ ] SHAP 값 분석 (상위 30개 피처 시각화)
- [ ] 최종 피처셋 확정 → `src/feature_engineering.py`로 모듈화

### Day 14 (04.17 금) — 중간 점검
- [ ] Week 2 결과 정리 + README 업데이트
- [ ] 현재 Public LB 순위 확인
- [ ] Week 3 전략 수정 (앙상블 vs 튜닝 중 우선순위 결정)

---

## 🗓️ Week 3 (04.18~04.24) — 모델 다양화 + 검증 강화

> **목표**: 앙상블로 추가 MAE 5~10% 감소

### Day 15 (04.18 토) — XGBoost 추가
```python
import xgboost as xgb
params_xgb = {
    'objective': 'reg:absoluteerror',  # MAE 직접 최적화
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'seed': 42,
}
```
- [ ] XGBoost 5-Fold 학습 + OOF 예측
- [ ] LGB vs XGB 성능 비교

### Day 16 (04.19 일) — CatBoost 추가
```python
from catboost import CatBoostRegressor
# layout_type 등 범주형 자동 처리
cat_features = ['layout_type']
model_cat = CatBoostRegressor(
    loss_function='MAE',
    iterations=2000,
    learning_rate=0.05,
    depth=7,
    random_seed=42,
    verbose=200,
)
```
- [ ] CatBoost 학습 + OOF 예측
- [ ] 3개 모델 OOF MAE 비교

### Day 17 (04.20 월) — 앙상블 최적화
```python
# OOF 기반 최적 가중치 탐색
from scipy.optimize import minimize

def neg_mae(weights):
    w = np.array(weights)
    w = w / w.sum()
    blended = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat
    return mean_absolute_error(y_train, blended)

result = minimize(neg_mae, [1/3, 1/3, 1/3], method='Nelder-Mead')
```
- [ ] 최적 가중치 앙상블 제출
- [ ] 단순 평균 대비 개선 폭 측정

### Day 18 (04.21 화) — Optuna 튜닝 시작 (LightGBM)
```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 63, 511),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': trial.suggest_float('ff', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bf', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-4, 10, log=True),
        'reg_lambda': trial.suggest_float('lambda', 1e-4, 10, log=True),
    }
    # 3-Fold 빠른 CV로 MAE 반환

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=3600)
```
- [ ] 100 trials Optuna 실행 (약 1시간)
- [ ] 최적 파라미터 적용 후 제출

### Day 19 (04.22 수) — Optuna 튜닝 (XGBoost + CatBoost)
- [ ] XGBoost Optuna 튜닝 (50 trials)
- [ ] 튜닝된 모델 3개 재앙상블 + 제출

### Day 20 (04.23 목) — Stacking 실험
```python
# Level 1: LGB, XGB, CatBoost OOF 예측
# Level 2: Ridge 또는 LightGBM (소규모)
from sklearn.linear_model import Ridge

meta_train = np.column_stack([oof_lgb, oof_xgb, oof_cat])
meta_test  = np.column_stack([test_lgb, test_xgb, test_cat])

ridge = Ridge(alpha=1.0)
ridge.fit(meta_train, y_train)
stacking_pred = ridge.predict(meta_test).clip(min=0)
```
- [ ] Stacking vs 가중 평균 비교
- [ ] 더 나은 방식으로 제출

### Day 21 (04.24 금) — 검증 강화 + 이상치 재검토
- [ ] 이상치 처리 실험 (상위 1% cap: ~120분)
- [ ] train/val 분포 shift 재확인
- [ ] 최종 예측값 분포 vs 훈련 타겟 분포 비교
- [ ] Week 3 결과 정리

---

## 🗓️ Week 4 (04.25~05.04) — 마무리 + 최적화

> **목표**: LB 상위권 확정 + 재현 가능한 코드 정리

### Day 22-23 (04.25~26 토~일) — 추가 피처 실험
- [ ] 시나리오 내 타겟 lag 실험 (train only, 리크 주의)
  - `ts_idx >= 1`인 경우 이전 타임슬롯 타겟 사용
- [ ] layout_id 기반 target encoding (GroupKFold 내에서)
- [ ] 추가 아이디어: `fault_count_15m × congestion_score` 등

### Day 24 (04.27 월) — 팀 병합 마감 (참고)
- [ ] 솔로 유지 vs 팀 결성 최종 결정

### Day 25-26 (04.28~29 화~수) — 최종 모델 결정
- [ ] 전체 실험 결과 표 정리
- [ ] Public LB 기반 최고 제출 조합 선택
- [ ] 마지막 실험: Learning rate 감소 + 더 많은 iterations

### Day 27-28 (04.30~05.01 목~금) — 코드 정리
- [ ] `src/feature_engineering.py` 완성
- [ ] `src/train.py` 파이프라인 정리
- [ ] 재현성 검증 (seed 고정 후 동일 결과 확인)
- [ ] README 최종 업데이트

### Day 29-30 (05.02~05.03 토~일) — 여유 제출
- [ ] 최종 앙상블 2~3가지 버전 제출 (Best 선택)
- [ ] 제출 파일 검증 (ID 순서, 음수 없음, 컬럼명 정확)

### Day 31 (05.04 월) — **대회 마감**
- [ ] 최종 제출 확인 (Public LB 기준 최고 점수)
- [ ] 제출 결과 스크린샷 저장

---

## 🗓️ After 대회 (05.05~05.07) — PPT + 코드 제출

### 05.05 (화) — PPT 초안
슬라이드 구성:
1. 대회 소개 & 문제 정의
2. EDA 주요 인사이트 (타겟 분포, 시간 패턴, 배터리 중요성)
3. 피처 엔지니어링 전략
4. 모델 구조 & 앙상블
5. 실험 결과 비교 (MAE 변화 테이블)
6. 최종 성능 & 향후 개선 방향

### 05.06 (수) — PPT 완성 + 코드 최종 검토
- [ ] PPT 시각화 완성
- [ ] 코드 ZIP 패키징
- [ ] 실행 가이드 (README) 최종 확인

### 05.07 (목) — **코드·PPT 제출 마감**
- [ ] 데이콘 제출 페이지에 업로드

---

## 📊 실험 결과 추적표

| 날짜 | 실험명 | 핵심 변경 | CV MAE | Public LB | 비고 |
|---|---|---|---|---|---|
| 04.06 | Baseline v1 | 버그 수정 + layout merge | - | - | 첫 제출 |
| 04.08 | Log Transform | log1p 타겟 변환 | - | - | 기대 효과 큼 |
| 04.09 | TS Features | ts_idx, ts_ratio 추가 | - | - | |
| 04.11 | Lag v1 | lag1~3 (8개 피처) | - | - | |
| 04.12 | Rolling v1 | roll3/5 mean+std | - | - | |
| 04.14 | Battery FE | 배터리 복합 피처 | - | - | |
| 04.20 | Ensemble v1 | LGB+XGB+CAT 가중평균 | - | - | |
| 04.21 | Optuna LGB | 100 trials | - | - | |
| 04.23 | Stacking | Ridge meta-learner | - | - | |

---

## ⚡ 하루 작업 루틴 (권장)

```
① 어제 실험 결과 확인 (5분)
② 오늘 목표 1~2개만 설정 (5분)
③ 코드 작성 + 실행 (1~3시간)
④ README Approach Log 업데이트 (5분)
⑤ 다음 날 할 일 메모 (5분)
```
