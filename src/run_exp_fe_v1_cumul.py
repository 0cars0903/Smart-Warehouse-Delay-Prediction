"""
run_exp_fe_v1_cumul.py  —  FE v1 base + Cumulative 피처
===============================================================
결정적 발견 (04.15):
  FE v2 계열(no-delta/FE v3/Optuna A) 전체가 배율 1.170±0.001 수렴.
  원인: KEY_COLS_V2 확장(8→11 KEY_COLS) 자체가 배율 악화의 주범.

검증 가설:
  원본 KEY_COLS(8종) 유지 + Cumulative 피처만 추가하면
  배율 1.1627 (RF 5모델 수준) 유지하면서 CV를 개선할 수 있는가?

FE 구조:
  - 원본 KEY_COLS 8종 (avg_trip_distance 포함, 기존 변경 없음)
  - 원본 lag[1-6] + rolling[3,5,10] → 기존 212피처 수준
  - Cumulative 피처 추가 (KEY_COLS 대상, cummin/cummax/cummean): 8×3=24종
  - Domain 피처 8종
  - 총 예상: ~236피처

비교 기준:
  RF 5모델 (FE v1): CV 8.7911 / Public 10.2213 / 배율 1.1627 ← 목표
  FE v3 (KEY_COLS_V2 기반): CV 8.7663 / Public 10.2571 / 배율 1.1701

기대:
  - CV: 8.78~8.79 (cumulative 신호 추가로 FE v1 대비 개선)
  - Public: 배율 1.1627 유지 시 10.21 수준 → Public 신기록 가능성
  - 배율이 1.170대로 올라가면 → KEY_COLS 확장이 아닌 cumulative 자체가 원인

체크포인트: docs/fe_v1_cumul_ckpt/
예상 시간: ~90분
===============================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import (
    merge_layout, encode_categoricals, add_ts_features,
    add_lag_features, add_rolling_features, add_domain_features,
)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_PATH   = 'data/'
CKPT_DIR    = 'docs/fe_v1_cumul_ckpt'
SUBMIT_PATH = 'submissions/stacking_fe_v1_cumul_rf_lgbm_meta.csv'
TARGET      = 'avg_delay_minutes_next_30m'
N_FOLDS     = 5
RANDOM_SEED = 42

os.makedirs(CKPT_DIR, exist_ok=True)

# !! 핵심: 원본 KEY_COLS 8종 (FE v2 확장 없음) !!
KEY_COLS_V1 = [
    'low_battery_ratio',
    'battery_mean',
    'charge_queue_length',
    'robot_idle',
    'order_inflow_15m',
    'congestion_score',
    'max_zone_density',
    'avg_trip_distance',    # FE v2에서 제거했지만 원본 유지
]

# Cumulative 피처 적용 대상 (avg_trip_distance 제외 — 상관 r=0.021, 누적 의미 없음)
CUMUL_COLS = [
    'low_battery_ratio',
    'battery_mean',
    'charge_queue_length',
    'robot_idle',
    'order_inflow_15m',
    'congestion_score',
    'max_zone_density',
    'fault_count_15m',      # 누적 고장 횟수 (event counter)
    'blocked_path_15m',     # 누적 경로 차단 (event counter)
]
EVENT_COLS = {'fault_count_15m', 'blocked_path_15m'}  # cumsum만 추가

BEST_LGBM_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_SEED,
    'verbose': -1, 'n_jobs': -1,
}
CB_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3,
    'loss_function': 'MAE', 'eval_metric': 'MAE',
    'random_seed': RANDOM_SEED, 'verbose': 0,
    'early_stopping_rounds': 100,
}
TW_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'tweedie', 'tweedie_variance_power': 1.8,
    'metric': 'mae', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_SEED,
    'verbose': -1, 'n_jobs': -1,
}
ET_PARAMS = {'n_estimators': 500, 'n_jobs': -1, 'random_state': RANDOM_SEED, 'min_samples_leaf': 26}
RF_PARAMS = {'n_estimators': 500, 'max_features': 0.33, 'min_samples_leaf': 26,
             'n_jobs': -1, 'random_state': RANDOM_SEED}
META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'n_estimators': 1000, 'objective': 'regression_l1',
    'random_state': RANDOM_SEED, 'verbose': -1, 'n_jobs': -1,
}


# ──────────────────────────────────────────────
# Cumulative 피처 생성
# ──────────────────────────────────────────────
def add_cumulative_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    시나리오 내 누적 min/max/mean (+ event counter는 cumsum)
    shift(1) 적용 → 현재 타임슬롯 리크 방지
    ※ add_ts_features 이후 호출 필요 (scenario_id + ts_idx 정렬)
    """
    train_c = train.copy(); train_c['_split'] = 0
    test_c  = test.copy();  test_c['_split']  = 1
    test_c['_orig_order'] = np.arange(len(test_c))

    combined = (pd.concat([train_c, test_c], axis=0, ignore_index=True)
                  .sort_values(['scenario_id', 'ts_idx'])
                  .reset_index(drop=True))

    added = []
    for col in CUMUL_COLS:
        if col not in combined.columns:
            continue
        shifted = combined.groupby('scenario_id')[col].shift(1)

        combined[f'{col}_cummin'] = (
            shifted.groupby(combined['scenario_id'])
                   .transform(lambda x: x.expanding().min()))
        combined[f'{col}_cummax'] = (
            shifted.groupby(combined['scenario_id'])
                   .transform(lambda x: x.expanding().max()))

        if col in EVENT_COLS:
            combined[f'{col}_cumsum'] = (
                shifted.groupby(combined['scenario_id'])
                       .transform(lambda x: x.expanding().sum()))
            added += [f'{col}_cummin', f'{col}_cummax', f'{col}_cumsum']
        else:
            combined[f'{col}_cummean'] = (
                shifted.groupby(combined['scenario_id'])
                       .transform(lambda x: x.expanding().mean()))
            added += [f'{col}_cummin', f'{col}_cummax', f'{col}_cummean']

    tr_out = combined[combined['_split'] == 0].drop(
        columns=['_split', '_orig_order'], errors='ignore')
    te_sorted = (combined[combined['_split'] == 1]
                 .sort_values('_orig_order')
                 .drop(columns=['_split', '_orig_order']))
    return tr_out, te_sorted, added


# ──────────────────────────────────────────────
# FE v1 + Cumul 파이프라인
# ──────────────────────────────────────────────
def build_features_v1_cumul(train, test, layout, verbose=True):
    """원본 KEY_COLS(8종) + Cumulative 피처 — KEY_COLS 확장 없음"""
    if verbose:
        print(f'[build_features_v1_cumul] 시작: train={train.shape}')

    train, test = merge_layout(train, test, layout)
    train, test = encode_categoricals(train, test, TARGET)
    train = add_ts_features(train)
    test  = add_ts_features(test)

    # 원본 KEY_COLS(8종) 기반 Lag + Rolling
    train, test = add_lag_features(train, test, key_cols=KEY_COLS_V1, lags=[1,2,3,4,5,6])
    train, test = add_rolling_features(train, test, key_cols=KEY_COLS_V1, windows=[3,5,10])

    # Cumulative 피처 추가
    train, test, cumul_added = add_cumulative_features(train, test)

    train = add_domain_features(train)
    test  = add_domain_features(test)

    excl = ['ID', 'layout_id', 'scenario_id', TARGET]
    feat_cols = [c for c in train.columns if c not in excl
                 and train[c].dtype.name not in ['object', 'category']]

    if verbose:
        print(f'  Cumulative 피처 추가: {len(cumul_added)}종')
        print(f'[build_features_v1_cumul] 완료: 최종 피처 수 = {len(feat_cols)}')
    return train, test


def get_feature_cols(df):
    excl = ['ID', 'layout_id', 'scenario_id', TARGET]
    return [c for c in df.columns if c not in excl
            and df[c].dtype.name not in ['object', 'category']]


# ──────────────────────────────────────────────
# 베이스 모델 학습
# ──────────────────────────────────────────────
def train_base_models(X, y, X_test, groups):
    gkf = GroupKFold(n_splits=N_FOLDS)
    model_names = ['lgbm', 'tw', 'cb', 'et', 'rf']

    oof_dict  = {k: np.zeros(len(X))      for k in model_names}
    test_dict = {k: np.zeros(len(X_test)) for k in model_names}
    mae_dict  = {k: []                     for k in model_names}

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        print(f'\n=== Fold {fold_i+1}/{N_FOLDS} ===')

        for name in model_names:
            ckpt     = os.path.join(CKPT_DIR, f'{name}_fold{fold_i}.npy')
            ckpt_tst = os.path.join(CKPT_DIR, f'{name}_fold{fold_i}_test.npy')

            if os.path.exists(ckpt) and os.path.exists(ckpt_tst):
                oof_dict[name][va_idx] = np.load(ckpt)
                test_dict[name]       += np.load(ckpt_tst) / N_FOLDS
                mae_v = mean_absolute_error(y_va, oof_dict[name][va_idx])
                mae_dict[name].append(mae_v)
                print(f'  [{name.upper()}] 체크포인트 로드  MAE={mae_v:.4f}')
                continue

            print(f'  [{name.upper()}] 학습 중...')
            if name == 'lgbm':
                m = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
                oof_p = m.predict(X_va); tst_p = m.predict(X_test)
            elif name == 'tw':
                m = lgb.LGBMRegressor(**TW_PARAMS)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
                oof_p = m.predict(X_va); tst_p = m.predict(X_test)
            elif name == 'cb':
                m = cb.CatBoostRegressor(**CB_PARAMS)
                m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
                oof_p = m.predict(X_va); tst_p = m.predict(X_test)
            elif name == 'et':
                m = ExtraTreesRegressor(**ET_PARAMS)
                m.fit(np.nan_to_num(X_tr, nan=0.0), y_tr)
                oof_p = m.predict(np.nan_to_num(X_va, nan=0.0))
                tst_p = m.predict(np.nan_to_num(X_test, nan=0.0))
            else:  # rf
                m = RandomForestRegressor(**RF_PARAMS)
                m.fit(np.nan_to_num(X_tr, nan=0.0), y_tr)
                oof_p = m.predict(np.nan_to_num(X_va, nan=0.0))
                tst_p = m.predict(np.nan_to_num(X_test, nan=0.0))

            oof_dict[name][va_idx] = oof_p
            test_dict[name]       += tst_p / N_FOLDS
            mae_v = mean_absolute_error(y_va, oof_p)
            mae_dict[name].append(mae_v)
            print(f'  [{name.upper()}] 완료  MAE={mae_v:.4f}')
            np.save(ckpt, oof_p); np.save(ckpt_tst, tst_p)

    print('\n── 베이스 모델 OOF MAE 요약 ──')
    for name in model_names:
        cv_mae = mean_absolute_error(y, oof_dict[name])
        fold_str = ' / '.join(f'{m:.4f}' for m in mae_dict[name])
        print(f'  {name.upper():<6} CV={cv_mae:.4f}  Folds: {fold_str}')

    oofs = pd.DataFrame(oof_dict)
    print('\n── OOF 상관계수 ──')
    corr = oofs.corr()
    for a, b in [('lgbm','et'),('lgbm','rf'),('lgbm','cb'),('et','rf')]:
        print(f'  {a.upper()}-{b.upper()}: {corr.loc[a,b]:.4f}')

    return oof_dict, test_dict


# ──────────────────────────────────────────────
# 메타 학습기
# ──────────────────────────────────────────────
def train_meta(oof_dict, test_dict, y, groups):
    gkf = GroupKFold(n_splits=N_FOLDS)
    names = list(oof_dict.keys())
    X_tr = np.column_stack([oof_dict[k]  for k in names])
    X_te = np.column_stack([test_dict[k] for k in names])

    oof_meta  = np.zeros(len(y))
    test_meta = np.zeros(len(X_te))
    fold_maes = []

    print('\n── 메타 LGBM 학습 ──')
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y, groups)):
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(X_tr[tr_idx], y[tr_idx],
              eval_set=[(X_tr[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[va_idx] = m.predict(X_tr[va_idx])
        test_meta += m.predict(X_te) / N_FOLDS
        mae_v = mean_absolute_error(y[va_idx], oof_meta[va_idx])
        fold_maes.append(mae_v)
        print(f'  Fold {fold_i+1}: MAE={mae_v:.4f}  iter={m.best_iteration_}')

    cv_mae = mean_absolute_error(y, oof_meta)
    std = np.std(oof_meta - y)
    print(f'\n  메타 CV MAE = {cv_mae:.4f} | std={std:.2f}')
    print(f'  Fold MAEs : {" / ".join(f"{m:.4f}" for m in fold_maes)}')
    return test_meta, cv_mae


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    print('=' * 60)
    print('FE v1 + Cumulative 피처 실험')
    print('가설: 원본 KEY_COLS(8종) + Cumulative → 배율 1.163 유지하면서 CV 개선')
    print('=' * 60)

    train  = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_PATH, 'layout_info.csv'))
    print(f'데이터 로드: train={train.shape}, test={test.shape}')

    train_fe, test_fe = build_features_v1_cumul(train, test, layout, verbose=True)

    feat_cols = get_feature_cols(train_fe)
    cumul_feats = [c for c in feat_cols if any(
        c.endswith(s) for s in ('_cummin', '_cummax', '_cummean', '_cumsum'))]
    print(f'  Cumulative 피처: {len(cumul_feats)}종')
    print(f'  KEY_COLS 확장 없음 확인: {[c for c in feat_cols if "battery_std" in c or "robot_charging_lag" in c][:3]}')

    X     = train_fe[feat_cols].values
    y     = train_fe[TARGET].values
    X_te  = test_fe[feat_cols].values
    groups = train_fe['scenario_id'].values

    print(f'학습 피처 수: {len(feat_cols)}')

    oof_dict, test_dict = train_base_models(X, y, X_te, groups)
    meta_test, cv_mae   = train_meta(oof_dict, test_dict, y, groups)

    sub = pd.DataFrame({'ID': test_fe['ID'], TARGET: meta_test})
    sub.to_csv(SUBMIT_PATH, index=False)
    print(f'\n제출 파일 저장: {SUBMIT_PATH}')

    print('\n── 비교표 ──')
    print(f'  RF 5모델 (FE v1):         CV 8.7911 / Public 10.2213 / 배율 1.1627')
    print(f'  FE v3 (KEY_COLS_V2 기반): CV 8.7663 / Public 10.2571 / 배율 1.1701')
    print(f'  FE v1+Cumul (이 실험):    CV {cv_mae:.4f} / Public 미제출 / 배율 미정')
    print(f'  기대 Public (배율 1.1627): {cv_mae * 1.1627:.4f}')
    print(f'  기대 Public (배율 1.1701): {cv_mae * 1.1701:.4f}')
    print('완료!')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
