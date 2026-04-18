"""
run_ts0_ablation.py
====================
TS0 Broadcast 피처 효과 분석 (4-stage Ablation)

배경
----
추가 EDA(ADDITIONAL_EDA_REPORT.md) 결과:
  - robot_utilization(TS0) → scenario MAE 상관 r=0.475 (전체 r=0.211의 2.3×)
  - 붕괴 시나리오 TS0: blocked_path_15m 2946×, fault_count_15m 891×, avg_recovery 569× 높음
  - 가설: TS0 초기 상태가 시나리오 전체 궤적을 결정 → broadcast하면 모든 타임슬롯에 신호 전달

실험 설계 (누적 추가)
---------------------
  Exp0 (Baseline) : 기존 파이프라인 (lag+rolling+domain, ts0 없음)
  Exp1            : + TS0 연속형 8종 broadcast
  Exp2            : + TS0 이진 이벤트 플래그 3종 (blocked/fault/recovery > 0)
  Exp3            : + TS0 복합 취약성 지수 (ts0_robot_util × ts0_order_inflow)

평가
----
  5-fold GroupKFold by scenario_id, MAE (L1)
  BEST_LGBM_PARAMS 재사용 (Optuna 최적, CLAUDE.md 기준)

실행
----
  python src/run_ts0_ablation.py
  결과 → submissions/ts0_ablation_YYYYMMDD_HHMM.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# 프로젝트 루트 기준으로 경로 설정
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from feature_engineering import (
    build_features,
    get_feature_cols,
)

try:
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error
except ImportError as e:
    print(f'[ERROR] 필수 패키지 없음: {e}')
    print('  pip install lightgbm scikit-learn')
    sys.exit(1)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_PATH   = os.path.join(ROOT, 'data')
SUBMIT_PATH = os.path.join(ROOT, 'submissions')
TARGET      = 'avg_delay_minutes_next_30m'
N_FOLDS     = 5
SEED        = 42

# Optuna 최적 LGBM 파라미터 (CLAUDE.md 기준)
BEST_LGBM_PARAMS = {
    'num_leaves'       : 181,
    'learning_rate'    : 0.020616,
    'feature_fraction' : 0.5122,
    'bagging_fraction' : 0.9049,
    'min_child_samples': 26,
    'reg_alpha'        : 0.3805,
    'reg_lambda'       : 0.3630,
    'objective'        : 'regression_l1',
    'n_estimators'     : 3000,
    'bagging_freq'     : 1,
    'random_state'     : SEED,
    'verbosity'        : -1,
    'n_jobs'           : -1,
}


# ─────────────────────────────────────────────
# CV 함수
# ─────────────────────────────────────────────
def run_cv(train_fe: pd.DataFrame, label: str) -> float:
    """5-fold GroupKFold CV → OOF MAE 반환"""
    feat_cols = get_feature_cols(train_fe, TARGET)
    X = train_fe[feat_cols].values
    y = train_fe[TARGET].values
    groups = train_fe['scenario_id'].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_preds = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)

    mae = mean_absolute_error(y, oof_preds)
    print(f'  [{label}] OOF MAE = {mae:.4f}  (피처 수: {len(feat_cols)})')
    return mae


# ─────────────────────────────────────────────
# 실험 정의
# ─────────────────────────────────────────────
EXPERIMENTS = [
    {
        'name'          : 'Exp0_Baseline',
        'desc'          : 'TS0 없음 (기존 파이프라인)',
        'use_ts0'       : False,
        'ts0_continuous': False,
        'ts0_flags'     : False,
        'ts0_composite' : False,
    },
    {
        'name'          : 'Exp1_TS0_Continuous',
        'desc'          : '+ TS0 연속형 8종 broadcast (robot_util, order_inflow 등)',
        'use_ts0'       : True,
        'ts0_continuous': True,
        'ts0_flags'     : False,
        'ts0_composite' : False,
    },
    {
        'name'          : 'Exp2_TS0_Cont+Flags',
        'desc'          : '+ TS0 이진 플래그 3종 (blocked/fault/recovery > 0)',
        'use_ts0'       : True,
        'ts0_continuous': True,
        'ts0_flags'     : True,
        'ts0_composite' : False,
    },
    {
        'name'          : 'Exp3_TS0_Full',
        'desc'          : '+ TS0 복합 취약성 지수 (robot_util × order_inflow)',
        'use_ts0'       : True,
        'ts0_continuous': True,
        'ts0_flags'     : True,
        'ts0_composite' : True,
    },
]


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    print('=' * 60)
    print('TS0 Broadcast 피처 Ablation')
    print('=' * 60)

    # 데이터 로드
    print('\n[1] 데이터 로드...')
    train  = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_PATH, 'layout_info.csv'))
    sample = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
    print(f'  train={train.shape}, test={test.shape}, layout={layout.shape}')

    # TS0 연속형 피처 존재 여부 확인
    ts0_cont_available = [c for c in [
        'robot_utilization', 'order_inflow_15m', 'robot_active',
        'sku_concentration', 'max_zone_density', 'congestion_score',
        'robot_idle', 'urgent_order_ratio'
    ] if c in train.columns]
    ts0_flag_available = [c for c in [
        'blocked_path_15m', 'fault_count_15m', 'avg_recovery_time'
    ] if c in train.columns]
    print(f'  TS0 연속형 피처 사용 가능: {ts0_cont_available}')
    print(f'  TS0 플래그 피처 사용 가능: {ts0_flag_available}')

    # 실험 루프
    print('\n[2] 실험 진행...\n')
    results = []
    baseline_mae = None

    for exp in EXPERIMENTS:
        print(f"  ▶ {exp['name']}: {exp['desc']}")

        train_fe, test_fe = build_features(
            train.copy(), test.copy(), layout.copy(),
            target      = TARGET,
            use_lag     = True,
            use_rolling = True,
            use_domain  = True,
            use_ts0     = exp['use_ts0'],
            ts0_continuous = exp['ts0_continuous'],
            ts0_flags      = exp['ts0_flags'],
            ts0_composite  = exp['ts0_composite'],
            verbose     = False,
        )

        mae = run_cv(train_fe, exp['name'])

        delta = None
        if baseline_mae is None:
            baseline_mae = mae
        else:
            delta = mae - baseline_mae

        results.append({
            'name'      : exp['name'],
            'desc'      : exp['desc'],
            'n_features': len(get_feature_cols(train_fe, TARGET)),
            'oof_mae'   : mae,
            'delta'     : delta,
        })
        print()

    # 결과 요약
    print('=' * 60)
    print('결과 요약')
    print('=' * 60)
    print(f"{'실험':<28} {'피처수':>6} {'CV MAE':>8} {'Δ Baseline':>12} {'판정':>6}")
    print('-' * 65)
    for r in results:
        delta_str = f"{r['delta']:+.4f}" if r['delta'] is not None else '   —'
        if r['delta'] is None:
            verdict = '기준'
        elif r['delta'] < -0.01:
            verdict = '✅ 개선'
        elif r['delta'] < 0.01:
            verdict = '➖ 미미'
        else:
            verdict = '❌ 악화'
        print(f"  {r['name']:<26} {r['n_features']:>6} {r['oof_mae']:>8.4f} {delta_str:>12} {verdict}")

    # 결과 저장
    ts   = datetime.now().strftime('%Y%m%d_%H%M')
    path = os.path.join(SUBMIT_PATH, f'ts0_ablation_{ts}.csv')
    os.makedirs(SUBMIT_PATH, exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)
    print(f'\n결과 저장: {path}')

    # 최적 실험이 baseline보다 나으면 제출 파일 생성
    best = min(results, key=lambda r: r['oof_mae'])
    if best['delta'] is not None and best['delta'] < -0.005:
        best_exp = next(e for e in EXPERIMENTS if e['name'] == best['name'])
        print(f"\n[3] 최적 실험 {best['name']} (MAE {best['oof_mae']:.4f}) → 제출 파일 생성...")
        _generate_submission(train, test, layout, sample, best_exp, best['name'], ts)
    else:
        print(f"\n[3] 최대 개선 {best['delta']:+.4f} — 제출 파일 생략 (임계값 미달)")

    return results


def _generate_submission(train, test, layout, sample, exp_cfg, exp_name, ts):
    """전체 훈련 데이터로 모델 학습 후 제출 파일 생성"""
    train_fe, test_fe = build_features(
        train.copy(), test.copy(), layout.copy(),
        target         = TARGET,
        use_lag        = True,
        use_rolling    = True,
        use_domain     = True,
        use_ts0        = exp_cfg['use_ts0'],
        ts0_continuous = exp_cfg['ts0_continuous'],
        ts0_flags      = exp_cfg['ts0_flags'],
        ts0_composite  = exp_cfg['ts0_composite'],
        verbose        = True,
    )

    feat_cols = get_feature_cols(train_fe, TARGET)
    X_train = train_fe[feat_cols].values
    y_train = train_fe[TARGET].values
    X_test  = test_fe[feat_cols].values

    # 5-fold OOF로 best_iteration 추정
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_fe['scenario_id'].values
    best_iters = []

    for tr_idx, val_idx in gkf.split(X_train, y_train, groups):
        m = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
        m.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
        )
        best_iters.append(m.best_iteration_)

    avg_iter = int(np.mean(best_iters))
    print(f'  평균 best_iteration: {avg_iter}')

    params_final = {**BEST_LGBM_PARAMS, 'n_estimators': avg_iter}
    final_model  = lgb.LGBMRegressor(**params_final)
    final_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(-1)])

    preds = final_model.predict(X_test)

    sub = sample.copy()
    sub[TARGET] = preds
    sub_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'submissions',
        f'{exp_name.lower()}_{ts}.csv'
    )
    sub.to_csv(sub_path, index=False)
    print(f'  제출 파일 저장: {sub_path}')


if __name__ == '__main__':
    results = main()
