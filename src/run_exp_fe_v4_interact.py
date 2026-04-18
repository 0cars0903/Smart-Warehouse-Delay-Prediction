"""
run_exp_fe_v4_interact.py  —  방향 C: 위치×신호 상호작용 + 가속도 + 모멘텀 피처
===============================================================
아이디어: 현재 모델은 ts_idx=5와 ts_idx=20에서 동일한 battery_mean=50을
         동일하게 취급한다. 하지만 시나리오 후반의 낮은 배터리/높은 혼잡은
         초반보다 훨씬 위험하다.

추가 피처 3그룹:

  [A] 위치×신호 상호작용 (8종)
      ts_ratio × [battery_mean, low_battery_ratio, congestion_score,
                  charge_queue_length, robot_idle, order_inflow_15m,
                  charge_bottleneck, order_pressure]
      → "시나리오 후반에서의 의미" 를 모델에게 알려줌
      → ts_ratio는 add_ts_features에서 이미 [0, 1] 범위로 계산됨

  [B] 가속도 피처 (11종, KEY_COLS_V2 대상)
      col_diff2 = col_diff1 - (col_lag1 - col_lag2)
               = col - 2×col_lag1 + col_lag2  (2차 차분)
      → "상황이 얼마나 빠르게 나빠지고 있나?"
      → GBDT와 트리 계열의 inductive bias 차이 극대화

  [C] 모멘텀 피처 (22종, KEY_COLS_V2 대상)
      col_mom_s = col_lag1 - col_lag3  (단기 모멘텀, 최근 2슬롯 변화)
      col_mom_l = col_lag3 - col_lag6  (중기 모멘텀, 3~6슬롯 전 대비)
      → "최근 추세가 계속되면 어디로 가나?"

총 추가 피처: 8 + 11 + 22 = 41종
예상 최종 피처: 264 + 41 = ~305종

체크포인트: docs/fe_v4_interact_ckpt/ (FE v2와 피처 공간 다름 → 전체 재학습)
예상 시간: ~90분 (RF 5모델 × 5-fold)

비교 기준:
  FE v2: CV 8.7842 / RF 5모델 스태킹
  목표: < 8.78 (다양성 추가로 메타 CV 개선)
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
CKPT_DIR    = 'docs/fe_v4_interact_ckpt'
SUBMIT_PATH = 'submissions/stacking_fe_v4_interact_rf_lgbm_meta.csv'
TARGET      = 'avg_delay_minutes_next_30m'
N_FOLDS     = 5
RANDOM_SEED = 42

os.makedirs(CKPT_DIR, exist_ok=True)

KEY_COLS_V2 = [
    'low_battery_ratio', 'battery_mean', 'battery_std',
    'robot_idle', 'robot_charging', 'order_inflow_15m',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'sku_concentration', 'urgent_order_ratio',
]

# 위치×신호 상호작용에 사용할 컬럼 (add_domain_features 이후 존재하는 복합 피처 포함)
POSITION_INTERACT_COLS = [
    'battery_mean',
    'low_battery_ratio',
    'congestion_score',
    'charge_queue_length',
    'robot_idle',
    'order_inflow_15m',
    'charge_bottleneck',   # add_domain_features에서 생성
    'order_pressure',      # add_domain_features에서 생성
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
# FE v4 파이프라인
# ──────────────────────────────────────────────
def add_interact_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [A] 위치×신호 상호작용 (8종)
    ts_ratio × 핵심 신호 → 후반 시나리오에서의 중요도 인코딩
    ※ add_ts_features, add_domain_features 이후에 호출해야 함
    """
    df = df.copy()
    for col in POSITION_INTERACT_COLS:
        if col in df.columns and 'ts_ratio' in df.columns:
            df[f'ts_x_{col}'] = df['ts_ratio'] * df[col]
    return df


def add_acceleration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [B] 가속도 피처 (11종)
    col_diff2 = col - 2*col_lag1 + col_lag2  (2차 차분)
    ※ add_lag_features 이후에 호출해야 함 (lag1, lag2 필요)
    ※ add_interact_features에서 diff1이 이미 있으므로 활용
    """
    df = df.copy()
    for col in KEY_COLS_V2:
        lag1 = f'{col}_lag1'
        lag2 = f'{col}_lag2'
        diff1 = f'{col}_diff1'
        if lag1 in df.columns and lag2 in df.columns:
            if diff1 in df.columns:
                # diff2 = diff1 - (lag1 - lag2)
                df[f'{col}_diff2'] = df[diff1] - (df[lag1] - df[lag2])
            else:
                # fallback: 직접 계산
                df[f'{col}_diff2'] = df[col] - 2 * df[lag1] + df[lag2]
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [C] 모멘텀 피처 (22종)
    col_mom_s = col_lag1 - col_lag3  (단기: 최근 2슬롯 추세)
    col_mom_l = col_lag3 - col_lag6  (중기: 3~6슬롯 전 대비 추세)
    ※ add_lag_features(lags=[1,...,6]) 이후에 호출해야 함
    """
    df = df.copy()
    for col in KEY_COLS_V2:
        lag1 = f'{col}_lag1'
        lag3 = f'{col}_lag3'
        lag6 = f'{col}_lag6'
        if lag1 in df.columns and lag3 in df.columns:
            df[f'{col}_mom_s'] = df[lag1] - df[lag3]
        if lag3 in df.columns and lag6 in df.columns:
            df[f'{col}_mom_l'] = df[lag3] - df[lag6]
    return df


def build_features_v4(train, test, layout, verbose=True):
    """FE v4 파이프라인: FE v2 + 상호작용 + 가속도 + 모멘텀"""
    if verbose:
        print(f'[build_features_v4] 시작: train={train.shape}, test={test.shape}')

    # Step 1-3: FE v2와 동일
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

    # Lag + Rolling (FE v2)
    train, test = add_lag_features(train, test, key_cols=KEY_COLS_V2, lags=[1,2,3,4,5,6])
    train, test = add_rolling_features(train, test, key_cols=KEY_COLS_V2, windows=[3,5,10])

    # Delta 피처 (FE v2)
    for df in [train, test]:
        for col in KEY_COLS_V2:
            if f'{col}_lag1' in df.columns:
                df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']

    # Domain 피처 (FE v2) — POSITION_INTERACT_COLS의 charge_bottleneck/order_pressure 생성
    train = add_domain_features(train)
    test  = add_domain_features(test)

    # ── 신규: [A] 위치×신호 상호작용 ──
    train = add_interact_features(train)
    test  = add_interact_features(test)

    # ── 신규: [B] 가속도 피처 ──
    train = add_acceleration_features(train)
    test  = add_acceleration_features(test)

    # ── 신규: [C] 모멘텀 피처 ──
    train = add_momentum_features(train)
    test  = add_momentum_features(test)

    excl = ['ID', 'layout_id', 'scenario_id', TARGET]
    feat_cols = [c for c in train.columns if c not in excl
                 and train[c].dtype.name not in ['object', 'category']]
    if verbose:
        v2_base = 264
        new_cnt = len(feat_cols) - v2_base
        print(f'[build_features_v4] 완료: 최종 피처 수 = {len(feat_cols)} '
              f'(FE v2 {v2_base} + 신규 {new_cnt})')
    return train, test


def get_feature_cols(df):
    excl = ['ID', 'layout_id', 'scenario_id', TARGET]
    return [c for c in df.columns if c not in excl
            and df[c].dtype.name not in ['object', 'category']]


# ──────────────────────────────────────────────
# 베이스 모델 학습 (체크포인트)
# ──────────────────────────────────────────────
def train_base_models(X, y, X_test, groups, feat_cols):
    gkf = GroupKFold(n_splits=N_FOLDS)
    models = {
        'lgbm': (lgb.LGBMRegressor(**BEST_LGBM_PARAMS), 'lgbm'),
        'tw':   (lgb.LGBMRegressor(**TW_PARAMS),        'tw'),
        'cb':   (None, 'cb'),
        'et':   (ExtraTreesRegressor(**ET_PARAMS),       'et'),
        'rf':   (RandomForestRegressor(**RF_PARAMS),     'rf'),
    }

    oof_dict  = {k: np.zeros(len(X))        for k in models}
    test_dict = {k: np.zeros(len(X_test))   for k in models}
    mae_dict  = {k: []                       for k in models}

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        print(f'\n=== Fold {fold_i+1}/{N_FOLDS} ===')

        for name, (model_proto, ckpt_key) in models.items():
            ckpt = os.path.join(CKPT_DIR, f'{ckpt_key}_fold{fold_i}.npy')
            ckpt_test = os.path.join(CKPT_DIR, f'{ckpt_key}_fold{fold_i}_test.npy')

            if os.path.exists(ckpt) and os.path.exists(ckpt_test):
                oof_dict[name][va_idx] = np.load(ckpt)
                test_dict[name]       += np.load(ckpt_test) / N_FOLDS
                mae_v = mean_absolute_error(y_va, oof_dict[name][va_idx])
                mae_dict[name].append(mae_v)
                print(f'  [{name.upper()}] 체크포인트 로드  MAE={mae_v:.4f}')
                continue

            print(f'  [{name.upper()}] 학습 중...')
            if name == 'cb':
                m = cb.CatBoostRegressor(**CB_PARAMS)
                m.fit(X_tr, y_tr,
                      eval_set=(X_va, y_va),
                      verbose=False)
                oof_p = m.predict(X_va)
                tst_p = m.predict(X_test)
            elif name in ('lgbm', 'tw'):
                m = lgb.LGBMRegressor(**({**BEST_LGBM_PARAMS} if name == 'lgbm' else TW_PARAMS))
                m.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(-1)])
                oof_p = m.predict(X_va)
                tst_p = m.predict(X_test)
            else:
                # ET/RF는 NaN 비허용 → acc/momentum 피처의 초기 NaN을 0으로 채움
                # (LGBM/TW/CB는 NaN을 분기 기준으로 활용하므로 원본 유지)
                X_tr_f = np.nan_to_num(X_tr,    nan=0.0)
                X_va_f = np.nan_to_num(X_va,    nan=0.0)
                X_te_f = np.nan_to_num(X_test,  nan=0.0)
                m = (ExtraTreesRegressor(**ET_PARAMS) if name == 'et'
                     else RandomForestRegressor(**RF_PARAMS))
                m.fit(X_tr_f, y_tr)
                oof_p = m.predict(X_va_f)
                tst_p = m.predict(X_te_f)

            oof_dict[name][va_idx] = oof_p
            test_dict[name]       += tst_p / N_FOLDS
            mae_v = mean_absolute_error(y_va, oof_p)
            mae_dict[name].append(mae_v)
            print(f'  [{name.upper()}] 완료  MAE={mae_v:.4f}')
            np.save(ckpt, oof_p)
            np.save(ckpt_test, tst_p)

    print('\n── 베이스 모델 OOF MAE 요약 ──')
    for name in models:
        cv_mae = mean_absolute_error(y, oof_dict[name])
        fold_str = ' / '.join(f'{m:.4f}' for m in mae_dict[name])
        print(f'  {name.upper():<6} CV={cv_mae:.4f}  Folds: {fold_str}')

    return oof_dict, test_dict


# ──────────────────────────────────────────────
# OOF 상관 분석
# ──────────────────────────────────────────────
def print_oof_correlation(oof_dict, y):
    import pandas as pd
    oofs = pd.DataFrame(oof_dict)
    print('\n── OOF 상관계수 행렬 ──')
    print(oofs.corr().round(4).to_string())


# ──────────────────────────────────────────────
# 메타 학습기 (LGBM-meta, GroupKFold 2단계 스태킹)
# ──────────────────────────────────────────────
def train_meta(oof_dict, test_dict, y, groups):
    gkf = GroupKFold(n_splits=N_FOLDS)
    model_names = list(oof_dict.keys())

    X_meta_tr = np.column_stack([oof_dict[k]  for k in model_names])
    X_meta_te = np.column_stack([test_dict[k] for k in model_names])

    oof_meta = np.zeros(len(y))
    test_meta = np.zeros(len(X_meta_te))
    fold_maes = []

    print('\n── 메타 LGBM 학습 ──')
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_meta_tr, y, groups)):
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(X_meta_tr[tr_idx], y[tr_idx],
              eval_set=[(X_meta_tr[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
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
# 가중 앙상블 (참고용)
# ──────────────────────────────────────────────
def weighted_blend(oof_dict, test_dict, y):
    from scipy.optimize import minimize
    model_names = list(oof_dict.keys())
    X_oof = np.column_stack([oof_dict[k] for k in model_names])
    X_tst = np.column_stack([test_dict[k] for k in model_names])
    n = len(model_names)

    def loss(w):
        w = np.array(w)
        w = np.abs(w) / np.abs(w).sum()
        return mean_absolute_error(y, X_oof @ w)

    res = minimize(loss, [1/n]*n, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-6})
    w_opt = np.abs(res.x) / np.abs(res.x).sum()
    blend_mae = mean_absolute_error(y, X_oof @ w_opt)

    print('\n── 가중 앙상블 ──')
    for name, w in zip(model_names, w_opt):
        print(f'  {name.upper():<6} weight={w:.3f}')
    print(f'  가중 앙상블 CV MAE = {blend_mae:.4f}')

    return X_tst @ w_opt


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    print('=' * 60)
    print('FE v4 (위치×신호 + 가속도 + 모멘텀) 실험')
    print('=' * 60)

    # 데이터 로드
    train  = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_PATH, 'layout_info.csv'))
    print(f'데이터 로드: train={train.shape}, test={test.shape}')

    # FE v4 피처 생성
    train_fe, test_fe = build_features_v4(train, test, layout, verbose=True)

    feat_cols = get_feature_cols(train_fe)
    X     = train_fe[feat_cols].values
    y     = train_fe[TARGET].values
    X_te  = test_fe[feat_cols].values
    groups = train_fe['scenario_id'].values

    print(f'\n학습 피처 수: {len(feat_cols)}')

    # 신규 피처 목록 출력
    interact_feats = [c for c in feat_cols if c.startswith('ts_x_')]
    accel_feats    = [c for c in feat_cols if c.endswith('_diff2')]
    mom_feats      = [c for c in feat_cols if c.endswith('_mom_s') or c.endswith('_mom_l')]
    print(f'  위치×신호 피처: {len(interact_feats)}종 {interact_feats[:3]}...')
    print(f'  가속도 피처:    {len(accel_feats)}종')
    print(f'  모멘텀 피처:    {len(mom_feats)}종')

    # 베이스 모델 학습
    oof_dict, test_dict = train_base_models(X, y, X_te, groups, feat_cols)

    # OOF 상관 분석
    print_oof_correlation(oof_dict, y)

    # 가중 앙상블 (참고)
    blend_test = weighted_blend(oof_dict, test_dict, y)

    # 메타 LGBM 스태킹
    meta_test, cv_mae = train_meta(oof_dict, test_dict, y, groups)

    # 제출 파일 저장
    sub = pd.DataFrame({'ID': test_fe['ID'], TARGET: meta_test})
    sub.to_csv(SUBMIT_PATH, index=False)
    print(f'\n제출 파일 저장: {SUBMIT_PATH}')
    print(f'최종 메타 CV MAE: {cv_mae:.4f}')
    print('완료!')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
