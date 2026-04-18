"""
run_exp_fe_v2_nodelta.py  —  Ablation: FE v2에서 Delta 피처만 제거
===============================================================
FE v2 제출 결과: CV 8.7842 → Public 10.2801 (배율 1.1703, 역전)
RF 5모델 기준:   CV 8.7911 → Public 10.2213 (배율 1.1627, 최고)

가설: Delta 피처(col_diff1 = col - col_lag1) 가 주범
  - 1차 차분은 노이즈 증폭: 학습셋 내 노이즈를 "신호"로 학습
  - LGBM-ET 상관 0.9744→0.9142 개선이 실제 신호가 아닌 노이즈 다양성
  - 결과: CV 개선이나 Public 역전 (배율 악화)

이 실험: FE v2에서 Delta 피처만 제거
  FE v2 no-delta = KEY_COLS_V2 확장 + Layout 비율 피처 (delta 없음)
  기존 212피처 vs 새 ~253피처 (264 - 11 delta = 253)

비교표:
  RF 5모델 (FE v1, 212피처): CV 8.7911 / Public 10.2213 / 배율 1.1627
  FE v2 full (264피처):       CV 8.7842 / Public 10.2801 / 배율 1.1703 ⚠️
  FE v2 no-delta (?피처):     CV ?      / Public ?       / 배율 ?

기대 결과:
  - CV는 FE v2 (8.7842)보다 다소 높을 수 있음 (delta 다양성 손실)
  - Public은 RF 5모델(10.2213) 수준 또는 개선되어야 가설 입증
  - 배율이 1.1627 이하면 delta가 과적합 주범임을 확인

체크포인트: docs/fe_v2_nodelta_ckpt/ (전체 재학습 필요)
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
CKPT_DIR    = 'docs/fe_v2_nodelta_ckpt'
SUBMIT_PATH = 'submissions/stacking_fe_v2_nodelta_rf_lgbm_meta.csv'
TARGET      = 'avg_delay_minutes_next_30m'
N_FOLDS     = 5
RANDOM_SEED = 42

os.makedirs(CKPT_DIR, exist_ok=True)

# FE v2 확장 KEY_COLS (delta 없이 사용)
KEY_COLS_V2 = [
    'low_battery_ratio', 'battery_mean', 'battery_std',
    'robot_idle', 'robot_charging', 'order_inflow_15m',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'sku_concentration', 'urgent_order_ratio',
]

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
# FE v2 no-delta 파이프라인
# ──────────────────────────────────────────────
def build_features_v2_nodelta(train, test, layout, verbose=True):
    """
    FE v2 파이프라인 — Delta 피처(col_diff1) 제거
    KEY_COLS_V2 확장 + Layout 비율 피처는 유지
    """
    if verbose:
        print(f'[build_features_v2_nodelta] 시작: train={train.shape}, test={test.shape}')

    train, test = merge_layout(train, test, layout)
    train, test = encode_categoricals(train, test, TARGET)
    train = add_ts_features(train)
    test  = add_ts_features(test)

    # Layout 비율 피처 (FE v2)
    for df in [train, test]:
        df['robot_active_ratio']     = df['robot_active']   / (df['robot_total'] + 1)
        df['charging_saturation']    = df['robot_charging'] / (df['charger_count'] + 1)
        df['charger_per_robot']      = df['charger_count']  / (df['robot_total'] + 1)
        df['orders_per_robot_total'] = df['order_inflow_15m'] / (df['robot_total'] + 1)

    # KEY_COLS_V2 기반 Lag + Rolling (FE v2와 동일)
    train, test = add_lag_features(train, test, key_cols=KEY_COLS_V2, lags=[1,2,3,4,5,6])
    train, test = add_rolling_features(train, test, key_cols=KEY_COLS_V2, windows=[3,5,10])

    # !! Delta 피처 추가 안 함 (ablation 핵심) !!

    train = add_domain_features(train)
    test  = add_domain_features(test)

    excl = ['ID', 'layout_id', 'scenario_id', TARGET]
    feat_cols = [c for c in train.columns if c not in excl
                 and train[c].dtype.name not in ['object', 'category']]
    if verbose:
        print(f'[build_features_v2_nodelta] 완료: 최종 피처 수 = {len(feat_cols)} '
              f'(FE v2 264 - delta 11 = {264-11} 예상)')
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
    models = {
        'lgbm': 'lgbm',
        'tw':   'tw',
        'cb':   'cb',
        'et':   'et',
        'rf':   'rf',
    }

    oof_dict  = {k: np.zeros(len(X))      for k in models}
    test_dict = {k: np.zeros(len(X_test)) for k in models}
    mae_dict  = {k: []                     for k in models}

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        print(f'\n=== Fold {fold_i+1}/{N_FOLDS} ===')

        for name in models:
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
            if name == 'cb':
                m = cb.CatBoostRegressor(**CB_PARAMS)
                m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
                oof_p = m.predict(X_va); tst_p = m.predict(X_test)
            elif name == 'lgbm':
                m = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
                oof_p = m.predict(X_va); tst_p = m.predict(X_test)
            elif name == 'tw':
                m = lgb.LGBMRegressor(**TW_PARAMS)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
                oof_p = m.predict(X_va); tst_p = m.predict(X_test)
            elif name == 'et':
                m = ExtraTreesRegressor(**ET_PARAMS)
                X_tr_f = np.nan_to_num(X_tr, nan=0.0)
                m.fit(X_tr_f, y_tr)
                oof_p = m.predict(np.nan_to_num(X_va, nan=0.0))
                tst_p = m.predict(np.nan_to_num(X_test, nan=0.0))
            else:  # rf
                m = RandomForestRegressor(**RF_PARAMS)
                X_tr_f = np.nan_to_num(X_tr, nan=0.0)
                m.fit(X_tr_f, y_tr)
                oof_p = m.predict(np.nan_to_num(X_va, nan=0.0))
                tst_p = m.predict(np.nan_to_num(X_test, nan=0.0))

            oof_dict[name][va_idx] = oof_p
            test_dict[name]       += tst_p / N_FOLDS
            mae_v = mean_absolute_error(y_va, oof_p)
            mae_dict[name].append(mae_v)
            print(f'  [{name.upper()}] 완료  MAE={mae_v:.4f}')
            np.save(ckpt, oof_p); np.save(ckpt_tst, tst_p)

    print('\n── 베이스 모델 OOF MAE 요약 ──')
    for name in models:
        cv_mae = mean_absolute_error(y, oof_dict[name])
        fold_str = ' / '.join(f'{m:.4f}' for m in mae_dict[name])
        print(f'  {name.upper():<6} CV={cv_mae:.4f}  Folds: {fold_str}')

    # OOF 상관
    oofs = pd.DataFrame(oof_dict)
    print('\n── OOF 상관계수 ──')
    print(oofs.corr().round(4).to_string())

    return oof_dict, test_dict


# ──────────────────────────────────────────────
# 메타 학습기
# ──────────────────────────────────────────────
def train_meta(oof_dict, test_dict, y, groups):
    gkf = GroupKFold(n_splits=N_FOLDS)
    model_names = list(oof_dict.keys())
    X_meta_tr = np.column_stack([oof_dict[k]  for k in model_names])
    X_meta_te = np.column_stack([test_dict[k] for k in model_names])

    oof_meta  = np.zeros(len(y))
    test_meta = np.zeros(len(X_meta_te))
    fold_maes = []

    print('\n── 메타 LGBM 학습 ──')
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_meta_tr, y, groups)):
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(X_meta_tr[tr_idx], y[tr_idx],
              eval_set=[(X_meta_tr[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[va_idx] = m.predict(X_meta_tr[va_idx])
        test_meta += m.predict(X_meta_te) / N_FOLDS
        mae_v = mean_absolute_error(y[va_idx], oof_meta[va_idx])
        fold_maes.append(mae_v)
        print(f'  Fold {fold_i+1}: MAE={mae_v:.4f}  iter={m.best_iteration_}')

    cv_mae = mean_absolute_error(y, oof_meta)
    print(f'\n  메타 CV MAE = {cv_mae:.4f}')
    print(f'  Fold MAEs : {" / ".join(f"{m:.4f}" for m in fold_maes)}')
    return test_meta, cv_mae


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    print('=' * 60)
    print('FE v2 no-delta Ablation 실험')
    print('가설: Delta 피처 제거 시 Public 배율 개선 여부 확인')
    print('=' * 60)

    train  = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_PATH, 'layout_info.csv'))
    print(f'데이터 로드: train={train.shape}, test={test.shape}')

    train_fe, test_fe = build_features_v2_nodelta(train, test, layout, verbose=True)

    feat_cols = get_feature_cols(train_fe)
    # 검증: delta 피처 없어야 함
    delta_cols = [c for c in feat_cols if c.endswith('_diff1')]
    if delta_cols:
        print(f'⚠️ 경고: Delta 피처가 남아있음: {delta_cols}')
    else:
        print(f'✅ Delta 피처 없음 확인 (no-delta 적용됨)')

    X     = train_fe[feat_cols].values
    y     = train_fe[TARGET].values
    X_te  = test_fe[feat_cols].values
    groups = train_fe['scenario_id'].values

    print(f'학습 피처 수: {len(feat_cols)} (FE v2 264 - delta 11 = {264-11} 예상)')

    oof_dict, test_dict = train_base_models(X, y, X_te, groups)
    meta_test, cv_mae   = train_meta(oof_dict, test_dict, y, groups)

    sub = pd.DataFrame({'ID': test_fe['ID'], TARGET: meta_test})
    sub.to_csv(SUBMIT_PATH, index=False)
    print(f'\n제출 파일 저장: {SUBMIT_PATH}')

    # 기준과 비교 출력
    print('\n── 기준 비교 ──')
    print(f'  RF 5모델 (FE v1): CV 8.7911 / Public 10.2213 / 배율 1.1627')
    print(f'  FE v2 full:       CV 8.7842 / Public 10.2801 / 배율 1.1703 ⚠️')
    print(f'  FE v2 no-delta:   CV {cv_mae:.4f} / Public 미제출 / 배율 미정')
    expected_ratio = 1.1627
    print(f'  no-delta 기대 Public (배율 1.1627 가정): {cv_mae * expected_ratio:.4f}')
    print('완료!')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
