"""
run_exp_fe_v1_extlag.py  —  FE v1 + 장기 Lag 확장 실험
===============================================================
배경 (04.15 최종 분석):
  배율 1.1627 유지 = FE v1(lag/rolling만) 만이 가능
  추가 피처 탐색 결과:
    - Cumulative(cummin/cummax/cummean) → 배율 1.1700 악화 ❌
    - KEY_COLS_V2 확장(4종) → 배율 악화 + std 압축 ❌
    - Delta(diff1) → 배율 악화 + CV 기여 없음 ❌

가설:
  lag[1-6] + rolling[3,5,10] 범위 "확장"은 cumulative와 달리
  고정 윈도우 통계(fixed window)이므로 train/test 분포 차이가 작다.
  lag[7-12], rolling[15,20]은 더 긴 맥락을 포착하면서도
  배율을 1.1627 수준으로 유지할 수 있다.

FE 구조 (FE v1 기반, 3가지 변형 비교):

  A. lag_ext_only:  lag[1-6,7-12] + rolling[3,5,10]
  B. roll_ext_only: lag[1-6] + rolling[3,5,10,15,20]
  C. full_ext:      lag[1-6,7-12] + rolling[3,5,10,15,20]

검증 방법:
  - 3가지 변형을 모두 실행
  - 각 변형의 CV MAE + LGBM-ET 상관 기록
  - 최저 CV 변형만 제출 (배율 확인 후)

비교 기준:
  FE v1 (현재 최고): CV 8.7911 / Public 10.2213 / 배율 1.1627 / 피처 212

예상 시간: ~3시간 (3변형 × 5-fold, USER 로컬 실행 권장)
체크포인트: docs/fe_v1_extlag_ckpt/
===============================================================
"""

import os, warnings, argparse
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
TARGET      = 'avg_delay_minutes_next_30m'
N_FOLDS     = 5
RANDOM_SEED = 42

KEY_COLS_V1 = [
    'low_battery_ratio', 'battery_mean', 'charge_queue_length',
    'robot_idle', 'order_inflow_15m', 'congestion_score',
    'max_zone_density', 'avg_trip_distance',
]

# 실험 변형 정의
VARIANTS = {
    'A_lag_ext':  {'lags': list(range(1, 13)),   'windows': [3, 5, 10]},
    'B_roll_ext': {'lags': list(range(1, 7)),    'windows': [3, 5, 10, 15, 20]},
    'C_full_ext': {'lags': list(range(1, 13)),   'windows': [3, 5, 10, 15, 20]},
}

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
# FE 파이프라인 (변형별)
# ──────────────────────────────────────────────
def build_features_extlag(train, test, layout, lags, windows, verbose=True):
    if verbose:
        print(f'[FE] lag={lags} / windows={windows}')
    train, test = merge_layout(train, test, layout)
    train, test = encode_categoricals(train, test, TARGET)
    train = add_ts_features(train); test = add_ts_features(test)
    train, test = add_lag_features(train, test, key_cols=KEY_COLS_V1, lags=lags)
    train, test = add_rolling_features(train, test, key_cols=KEY_COLS_V1, windows=windows)
    train = add_domain_features(train); test = add_domain_features(test)
    return train, test


def get_feature_cols(df):
    excl = ['ID', 'layout_id', 'scenario_id', TARGET]
    return [c for c in df.columns if c not in excl
            and df[c].dtype.name not in ['object', 'category']]


# ──────────────────────────────────────────────
# 베이스 모델 학습
# ──────────────────────────────────────────────
def train_base_models(X, y, X_test, groups, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
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
            ckpt     = os.path.join(ckpt_dir, f'{name}_fold{fold_i}.npy')
            ckpt_tst = os.path.join(ckpt_dir, f'{name}_fold{fold_i}_test.npy')

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
    pred_std = np.std(oof_meta)
    print(f'\n  메타 CV MAE = {cv_mae:.4f} | pred std={pred_std:.2f}')
    print(f'  Fold MAEs : {" / ".join(f"{m:.4f}" for m in fold_maes)}')
    return test_meta, cv_mae


# ──────────────────────────────────────────────
# 단일 변형 실행
# ──────────────────────────────────────────────
def run_variant(variant_name, lags, windows, train, test, layout):
    print(f'\n{"="*60}')
    print(f'변형 {variant_name}: lag={lags} / windows={windows}')
    print(f'{"="*60}')

    ckpt_dir    = f'docs/fe_v1_extlag_{variant_name}_ckpt'
    submit_path = f'submissions/stacking_fe_v1_extlag_{variant_name}_rf_lgbm_meta.csv'

    train_fe, test_fe = build_features_extlag(train, test, layout, lags, windows)
    feat_cols = get_feature_cols(train_fe)
    print(f'피처 수: {len(feat_cols)}')

    X      = train_fe[feat_cols].values
    y      = train_fe[TARGET].values
    X_te   = test_fe[feat_cols].values
    groups = train_fe['scenario_id'].values

    oof_dict, test_dict = train_base_models(X, y, X_te, groups, ckpt_dir)
    meta_test, cv_mae   = train_meta(oof_dict, test_dict, y, groups)

    sub = pd.DataFrame({'ID': test_fe['ID'], TARGET: meta_test})
    sub.to_csv(submit_path, index=False)
    print(f'제출 파일 저장: {submit_path}')
    return cv_mae


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='all',
                        help='A_lag_ext | B_roll_ext | C_full_ext | all')
    args = parser.parse_args()

    print('=' * 60)
    print('FE v1 + 장기 Lag 확장 실험')
    print('가설: lag/rolling 범위 확장은 cumulative와 달리 배율 1.163 유지')
    print('=' * 60)

    train  = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_PATH, 'layout_info.csv'))
    print(f'데이터 로드: train={train.shape}, test={test.shape}')

    if args.variant == 'all':
        run_names = list(VARIANTS.keys())
    else:
        run_names = [args.variant]

    results = {}
    for name in run_names:
        cfg = VARIANTS[name]
        cv = run_variant(name, cfg['lags'], cfg['windows'], train, test, layout)
        results[name] = cv

    print('\n\n── 최종 비교표 ──')
    print(f'  FE v1 (기준):  CV 8.7911 / Public 10.2213 / 배율 1.1627 / 피처 212')
    for name, cv in results.items():
        cfg = VARIANTS[name]
        feat_est = 8 * (len(cfg['lags']) + len(cfg['windows'])) + 20  # rough estimate
        print(f'  {name}: CV {cv:.4f} / 기대 Public {cv*1.1627:.4f} (배율 1.163 유지 가정) / 피처 ~{feat_est}')
    print('완료!')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
