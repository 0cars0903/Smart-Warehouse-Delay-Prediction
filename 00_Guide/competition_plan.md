# 스마트 창고 출고 지연 예측 AI 경진대회 — 준비 계획

> 작성일: 2026-04-04 | 마감: 2026-05-04 | 코드·PPT 제출: 2026-05-07

---

## 📊 데이터 현황 요약 (분석 완료)

| 항목 | 내용 |
|---|---|
| Train | 250,000행 × 94컬럼 (10,000 시나리오 × 25 타임슬롯) |
| Test | 50,000행 × 93컬럼 (2,000 시나리오 × 25 타임슬롯) |
| Layout | 300행 × 15컬럼 (250개 창고 레이아웃) |
| 타겟 분포 | 평균 18.96분, 중앙값 9.03분, max 715분 → **강한 우편향** |
| 결측치 | 40개 이상 컬럼에 12~13% 수준 (MCAR 또는 MAR 확인 필요) |

### 핵심 인사이트

**타겟 상관관계 Top 피처**
1. `low_battery_ratio` (0.366) — 배터리 부족 로봇 비율
2. `battery_mean` (0.359) — 평균 배터리 잔량 (음의 방향)
3. `robot_idle` (0.349) — 유휴 로봇 수 (주문 소화 실패 시그널)
4. `order_inflow_15m` (0.342) — 15분 주문 유입량
5. `robot_charging` (0.320) — 충전 중인 로봇 수
6. `max_zone_density` (0.311) — 최대 구역 밀도
7. `congestion_score` (0.300) — 통로 혼잡도

**시간 패턴**: 타임슬롯이 진행될수록 지연 증가 (0번 슬롯 avg 11.3분 → 24번 슬롯 avg 21.9분) → **시계열 피처 엔지니어링 필수**

**레이아웃 타입별 차이**: hub_spoke(22.3분) > hybrid(18.4분) ≈ narrow(18.4분) > grid(18.1분)

### ⚠️ 현재 Baseline 코드 이슈
베이스라인 노트북에서 타겟 컬럼명이 `'avg_delay'`로 하드코딩되어 있으나, 실제 컬럼명은 `'avg_delay_minutes_next_30m'`임 → **실행 전 수정 필요**

---

## 🗓️ 주차별 실행 계획

### Phase 1 — 탐색 & 기반 구축 (04.04 ~ 04.13)

**목표**: 첫 제출 달성 + 데이터 완전 이해

- [ ] **Baseline 노트북 수정·실행 및 첫 제출**
  - 타겟 컬럼명 버그 수정 (`avg_delay` → `avg_delay_minutes_next_30m`)
  - layout_info 조인 추가
  - 첫 Public LB 스코어 확인
- [ ] **EDA 노트북 완성**
  - 결측치 패턴 분석 (MCAR vs MAR: scenario 내 특정 시간대에만 결측?)
  - 타임슬롯별 타겟 변화 시각화
  - layout_type별 분포 차이
  - 이상치 분석 (max 715분 등)
- [ ] **CV 전략 결정**
  - `KFold` vs `GroupKFold(groups=scenario_id)` 비교 실험
  - 주의: KFold는 같은 시나리오 행들이 train/val에 동시 등장 → 리크 가능성
  - GroupKFold가 LB에 더 가까운 CV일 가능성 높음

---

### Phase 2 — 피처 엔지니어링 (04.14 ~ 04.22)

**목표**: 피처 품질 극대화로 MAE 대폭 감소

#### 2-1. 타임슬롯 피처 (핵심)
```python
# 시나리오 내 타임슬롯 순서 (0~24)
df['ts_idx'] = df.groupby('scenario_id').cumcount()

# 타임슬롯 진행률 (0~1)
df['ts_ratio'] = df['ts_idx'] / 24
```

#### 2-2. 시나리오 내 Lag/Rolling 피처
```python
# 핵심 피처들의 이전 타임슬롯 값
key_cols = ['low_battery_ratio', 'order_inflow_15m', 'congestion_score',
            'robot_idle', 'charge_queue_length']

for col in key_cols:
    df[f'{col}_lag1'] = df.groupby('scenario_id')[col].shift(1)
    df[f'{col}_lag2'] = df.groupby('scenario_id')[col].shift(2)
    df[f'{col}_roll3_mean'] = df.groupby('scenario_id')[col].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
```

#### 2-3. 시나리오 집계 통계
```python
# 시나리오별 누적 통계 (미래 리크 방지 위해 expanding 사용)
for col in key_cols:
    df[f'{col}_cummax'] = df.groupby('scenario_id')[col].transform(
        lambda x: x.shift(1).expanding().max()
    )
    df[f'{col}_cummean'] = df.groupby('scenario_id')[col].transform(
        lambda x: x.shift(1).expanding().mean()
    )
```

#### 2-4. Layout 피처 활용
- `layout_info.csv` merge → 창고 구조 피처 14개 추가
- layout_type 인코딩 (hub_spoke > others 패턴 활용)
- `robot_total` / `robot_active` 비율 등 조합 피처

#### 2-5. 타겟 로그 변환 실험
```python
# 우편향 분포 보정
y_log = np.log1p(y)
# 예측 후: np.expm1(pred)
```

---

### Phase 3 — 모델 개선 & 앙상블 (04.23 ~ 05.02)

**목표**: 다양한 모델 조합으로 MAE 최소화

#### 3-1. 모델 확장
| 모델 | 특징 | 우선순위 |
|---|---|---|
| LightGBM | 빠른 속도, 결측치 처리 내장 | ★★★ (현재) |
| XGBoost | 다양한 정규화 옵션 | ★★★ |
| CatBoost | 범주형 자동 처리 (layout_type 등) | ★★ |
| Random Forest | 분산 감소 앙상블 | ★ |

#### 3-2. 하이퍼파라미터 튜닝 (Optuna)
```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
    }
    # CV MAE 반환
```

#### 3-3. 앙상블 전략
```python
# 단순 가중 평균
final = 0.4 * lgb_pred + 0.4 * xgb_pred + 0.2 * cat_pred

# 또는 Stacking (2nd level: Ridge or LinearRegression)
```

#### 3-4. 후처리
```python
# 음수 클리핑 (필수)
preds = preds.clip(lower=0)

# 이상치 타겟 처리 실험 (상위 1%: max 120분으로 cap)
y_capped = y.clip(upper=np.percentile(y, 99))
```

---

### Phase 4 — 마무리 & 제출 (05.03 ~ 05.07)

- [ ] 최종 앙상블 조합 결정 (Public LB 기준)
- [ ] 코드 정리 및 재현성 확인 (random seed 고정)
- [ ] PPT 작성 (접근법, 피처 중요도, 실험 결과)
- [ ] 최종 제출 (05.04 마감)
- [ ] 코드 + PPT 제출 (05.07 마감)

---

## 🔑 실험 트래킹

`README.md`의 Approach Log를 계속 업데이트할 것.

| 날짜 | 실험 | 주요 변경 | CV MAE | Public LB | 메모 |
|---|---|---|---|---|---|
| 04.04 | Baseline | 초기 세팅 | - | - | 실행 전 |
| 04.?? | Baseline v2 | 버그 수정 + layout merge | ? | ? | 첫 제출 |
| 04.?? | FE v1 | ts_idx + lag 피처 | ? | ? | |
| 04.?? | log 타겟 | log1p 변환 | ? | ? | |
| 04.?? | GroupKFold | CV 전략 변경 | ? | ? | |

---

## 📌 주요 리스크 & 대응

| 리스크 | 설명 | 대응 |
|---|---|---|
| CV-LB 불일치 | KFold 리크로 CV가 낙관적일 수 있음 | GroupKFold 병행 실험 |
| 타임 리크 | Lag 피처 계산 시 미래 정보 유입 | shift(1) 이상 사용 필수 |
| 우편향 타겟 | max 715분 등 이상치가 MAE 왜곡 | 로그 변환 + 이상치 cap 실험 |
| 결측치 처리 | ~12% 결측, MCAR 여부 불명 | LightGBM 내장 처리 우선, 별도 imputation 실험 |
