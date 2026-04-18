"""
모델실험23: Optuna 기반 LGBM + CatBoost 재튜닝 (FE v1 + 시나리오 집계, 302피처)
=============================================================================
목표: model21 (CV 8.5097, Public 9.9550) 기반 하이퍼파라미터 최적화

핵심 전략:
  - Phase 1: Optuna LGBM 튜닝 (50 trials, 2-fold GroupKFold)
  - Phase 2: Optuna CatBoost MAE 튜닝 (30 trials, 2-fold)
  - Phase 3: 최적 LGBM + CB params로 전체 5-fold 스태킹

피처셋: FE v1 (lag/rolling) + 시나리오 집계 (mean/std/max/min/diff × 18종 = 90피처)
        = 총 302피처

실행: python src/run_exp_model23_optuna_v2.py
예상 시간: ~60분 (Optuna 50+30 trials + Phase 3)
체크포인트: docs/model23_ckpt/
제출: submissions/model23_optuna_v2.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import optuna
from optuna.pruners import MedianPruner
import warnings
import gc
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features, get_feature_cols

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 경로 및 상수
# ─────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR = os.path.join(_BASE, '..', 'submissions')
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model23_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 시나리오 집계 피처 대상 (model21과 동일)
# ─────────────────────────────────────────────
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

# ─────────────────────────────────────────────
# 고정 하이퍼파라미터 (model21에서 재사용)
# ─────────────────────────────────────────────
TW18_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3.0,
    'loss_function': 'Tweedie:variance_power=1.8',
    'random_seed': RANDOM_STATE, 'verbose': 0,
    'early_stopping_rounds': 50,
}

ET_PARAMS = {
    'n_estimators': 500, 'max_features': 0.5,
    'min_samples_leaf': 26, 'n_jobs': -1,
    'random_state': RANDOM_STATE,
}

RF_PARAMS = {
    'n_estimators': 500, 'max_features': 0.33,
    'min_samples_leaf': 26, 'n_jobs': -1,
    'random_state': RANDOM_STATE,
}

META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────
def save_ckpt(name, oof, test_pred):
    """체크포인트 저장"""
    np.save(os.path.join(CKPT_DIR, f'{name}_oof.npy'), oof)
    np.save(os.path.join(CKPT_DIR, f'{name}_test.npy'), test_pred)


def load_ckpt(name):
    """체크포인트 로드"""
    oof = np.load(os.path.join(CKPT_DIR, f'{name}_oof.npy'))
    test = np.load(os.path.join(CKPT_DIR, f'{name}_test.npy'))
    return oof, test


def ckpt_exists(name):
    """체크포인트 존재 여부 확인"""
    return (os.path.exists(os.path.join(CKPT_DIR, f'{name}_oof.npy'))
            and os.path.exists(os.path.join(CKPT_DIR, f'{name}_test.npy')))


def add_scenario_agg_features(df):
    """시나리오 집계 피처 broadcast (mean/std/max/min/diff × 18종 = 90피처)"""
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]
        df[f'sc_{col}_mean'] = grp.transform('mean')
        df[f'sc_{col}_std'] = grp.transform('std').fillna(0)
        df[f'sc_{col}_max'] = grp.transform('max')
        df[f'sc_{col}_min'] = grp.transform('min')
        df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']
    return df


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
def load_data():
    """FE v1 + 시나리오 집계 피처 적용"""
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    # FE v1 파이프라인
    train, test = build_features(
        train, test, layout,
        lag_lags=[1, 2, 3, 4, 5, 6],
        rolling_windows=[3, 5, 10],
    )

    # 시나리오 집계 피처 추가
    train = add_scenario_agg_features(train)
    test = add_scenario_agg_features(test)

    sc_feats = [c for c in train.columns if c.startswith('sc_')]
    print(f'시나리오 집계 피처: {len(sc_feats)}종 추가')

    return train, test


def get_feat_cols(train):
    """피처 컬럼 추출"""
    return [c for c in train.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and train[c].dtype != object]


# ═════════════════════════════════════════════
# Phase 1: Optuna LGBM 튜닝 (2-fold 고속 검증)
# ═════════════════════════════════════════════
def objective_lgbm(trial, X_train, X_test, y_log, groups, feat_cols):
    """
    LGBM Optuna objective (2-fold GroupKFold for speed)
    목표: MAE in raw space 최소화
    """
    num_leaves = trial.suggest_int('num_leaves', 64, 512)
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.1, log=True)
    feature_fraction = trial.suggest_float('feature_fraction', 0.3, 0.9)
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.7, 1.0)
    min_child_samples = trial.suggest_int('min_child_samples', 10, 60)
    reg_alpha = trial.suggest_float('reg_alpha', 0.01, 10, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 0.01, 10, log=True)

    params = {
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'min_child_samples': min_child_samples,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'objective': 'regression_l1',
        'n_estimators': 3000,
        'bagging_freq': 1,
        'random_state': RANDOM_STATE,
        'verbosity': -1,
        'n_jobs': -1,
    }

    # 2-fold GroupKFold
    gkf = GroupKFold(n_splits=2)
    mae_scores = []
    X_tr_np = X_train[feat_cols].fillna(0)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_tr_np.iloc[tr_idx], y_log.iloc[tr_idx],
            eval_set=[(X_tr_np.iloc[va_idx], y_log.iloc[va_idx])],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        oof_pred = m.predict(X_tr_np.iloc[va_idx])
        mae = np.abs(np.expm1(oof_pred) - np.expm1(y_log.iloc[va_idx].values)).mean()
        mae_scores.append(mae)
        del m
        gc.collect()

    mean_mae = np.mean(mae_scores)
    return mean_mae


def tune_lgbm(train, test, y_log, groups, feat_cols):
    """Optuna LGBM 튜닝 (50 trials)"""
    print('\n' + '─' * 60)
    print('[Phase 1] Optuna LGBM 튜닝 (50 trials, 2-fold)')
    print('─' * 60)

    def objective(trial):
        return objective_lgbm(trial, train, test, y_log, groups, feat_cols)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=3)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='minimize')

    study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)

    best_trial = study.best_trial
    print(f'\n최적 LGBM MAE: {best_trial.value:.4f}')
    print('최적 하이퍼파라미터:')
    for key, val in best_trial.params.items():
        print(f'  {key}: {val}')

    return best_trial.params, study


# ═════════════════════════════════════════════
# Phase 2: Optuna CatBoost 튜닝 (2-fold 고속 검증)
# ═════════════════════════════════════════════
def objective_cb(trial, X_train, X_test, y_log, groups, feat_cols):
    """
    CatBoost MAE Optuna objective (2-fold GroupKFold for speed)
    """
    depth = trial.suggest_int('depth', 4, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1, 10, log=True)
    random_strength = trial.suggest_float('random_strength', 0.1, 10, log=True)
    bagging_temperature = trial.suggest_float('bagging_temperature', 0.0, 1.0)

    params = {
        'iterations': 3000,
        'depth': depth,
        'learning_rate': learning_rate,
        'l2_leaf_reg': l2_leaf_reg,
        'random_strength': random_strength,
        'bagging_temperature': bagging_temperature,
        'loss_function': 'MAE',
        'random_seed': RANDOM_STATE,
        'verbose': 0,
        'early_stopping_rounds': 50,
    }

    # 2-fold GroupKFold
    gkf = GroupKFold(n_splits=2)
    mae_scores = []
    X_tr_np = X_train[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        train_pool = cb.Pool(X_tr_np[tr_idx], y_log.values[tr_idx])
        val_pool = cb.Pool(X_tr_np[va_idx], y_log.values[va_idx])
        m = cb.CatBoostRegressor(**params)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof_pred = m.predict(X_tr_np[va_idx])
        mae = np.abs(np.expm1(oof_pred) - np.expm1(y_log.values[va_idx])).mean()
        mae_scores.append(mae)
        del m
        gc.collect()

    mean_mae = np.mean(mae_scores)
    return mean_mae


def tune_cb(train, test, y_log, groups, feat_cols):
    """Optuna CatBoost 튜닝 (30 trials)"""
    print('\n' + '─' * 60)
    print('[Phase 2] Optuna CatBoost 튜닝 (30 trials, 2-fold)')
    print('─' * 60)

    def objective(trial):
        return objective_cb(trial, train, test, y_log, groups, feat_cols)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='minimize')

    study.optimize(objective, n_trials=30, show_progress_bar=True, n_jobs=1)

    best_trial = study.best_trial
    print(f'\n최적 CatBoost MAE: {best_trial.value:.4f}')
    print('최적 하이퍼파라미터:')
    for key, val in best_trial.params.items():
        print(f'  {key}: {val}')

    return best_trial.params, study


# ═════════════════════════════════════════════
# Phase 3: 5-fold 스태킹 (최적 LGBM + CB)
# ═════════════════════════════════════════════
def train_lgbm_oof_custom(X_train, X_test, y_log, groups, feat_cols, params):
    """LGBM OOF (커스텀 파라미터)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0)
    X_te_np = X_test[feat_cols].fillna(0)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_tr_np.iloc[tr_idx], y_log.iloc[tr_idx],
            eval_set=[(X_tr_np.iloc[va_idx], y_log.iloc[va_idx])],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        oof[va_idx] = m.predict(X_tr_np.iloc[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.iloc[va_idx].values)).mean()
        print(f'  [LGBM] Fold {fold + 1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m
        gc.collect()

    return oof, test_pred


def train_cb_oof_custom(X_train, X_test, y_log, groups, feat_cols, params):
    """CatBoost OOF (커스텀 파라미터)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values
    X_te_np = X_test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        train_pool = cb.Pool(X_tr_np[tr_idx], y_log.values[tr_idx])
        val_pool = cb.Pool(X_tr_np[va_idx], y_log.values[va_idx])
        m = cb.CatBoostRegressor(**params)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [CB] Fold {fold + 1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m
        gc.collect()

    return oof, test_pred


def train_tw18_oof(X_train, X_test, y_raw, groups, feat_cols):
    """TW1.8 OOF (고정 파라미터)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values
    X_te_np = X_test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_raw, groups)):
        train_pool = cb.Pool(X_tr_np[tr_idx], y_raw.values[tr_idx])
        val_pool = cb.Pool(X_tr_np[va_idx], y_raw.values[va_idx])
        m = cb.CatBoostRegressor(**TW18_PARAMS)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.values[va_idx]).mean()
        print(f'  [TW1.8] Fold {fold + 1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m
        gc.collect()

    return oof, test_pred


def train_et_oof(X_train, X_test, y_log, groups, feat_cols):
    """ExtraTrees OOF (고정 파라미터)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values
    X_te_np = X_test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = ExtraTreesRegressor(**ET_PARAMS)
        m.fit(X_tr_np[tr_idx], y_log.values[tr_idx])
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [ET] Fold {fold + 1}  MAE={mae:.4f}')
        del m
        gc.collect()

    return oof, test_pred


def train_rf_oof(X_train, X_test, y_log, groups, feat_cols):
    """RandomForest OOF (고정 파라미터)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values
    X_te_np = X_test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = RandomForestRegressor(**RF_PARAMS)
        m.fit(X_tr_np[tr_idx], y_log.values[tr_idx])
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [RF] Fold {fold + 1}  MAE={mae:.4f}')
        del m
        gc.collect()

    return oof, test_pred


def run_meta_lgbm(meta_train, meta_test, y_raw, groups):
    """메타 LGBM (고정 파라미터)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_meta = np.zeros(len(y_raw))
    test_meta = np.zeros(meta_test.shape[0])

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
        y_tr_log = np.log1p(y_raw.iloc[tr_idx].values)
        y_va_log = np.log1p(y_raw.iloc[va_idx].values)
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(
            X_tr, y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        oof_meta[va_idx] = np.expm1(m.predict(X_va))
        test_meta += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [LGBM-meta] Fold {fold + 1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m
        gc.collect()

    oof_mae = np.abs(oof_meta - y_raw.values).mean()
    pred_std = oof_meta.std()
    print(f'  [LGBM-meta] OOF MAE={oof_mae:.4f} | pred_std={pred_std:.2f}')
    return oof_meta, test_meta, oof_mae


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 70)
    print('모델실험23: Optuna LGBM + CatBoost 재튜닝')
    print('기준: model21 CV 8.5097 / Public 9.9550')
    print('=' * 70)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    # 데이터 로드
    train, test = load_data()
    feat_cols = get_feat_cols(train)
    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'피처 수: {len(feat_cols)}')
    print(f'훈련 샘플: {len(train)}, 테스트 샘플: {len(test)}')

    # ══════════════════════════════════════════
    # Phase 1: Optuna LGBM 튜닝
    # ══════════════════════════════════════════
    best_lgbm_params, lgbm_study = tune_lgbm(train, test, y_log, groups, feat_cols)
    lgbm_params = {
        'num_leaves': best_lgbm_params['num_leaves'],
        'learning_rate': best_lgbm_params['learning_rate'],
        'feature_fraction': best_lgbm_params['feature_fraction'],
        'bagging_fraction': best_lgbm_params['bagging_fraction'],
        'min_child_samples': best_lgbm_params['min_child_samples'],
        'reg_alpha': best_lgbm_params['reg_alpha'],
        'reg_lambda': best_lgbm_params['reg_lambda'],
        'objective': 'regression_l1',
        'n_estimators': 3000,
        'bagging_freq': 1,
        'random_state': RANDOM_STATE,
        'verbosity': -1,
        'n_jobs': -1,
    }

    # ══════════════════════════════════════════
    # Phase 2: Optuna CatBoost 튜닝
    # ══════════════════════════════════════════
    best_cb_params, cb_study = tune_cb(train, test, y_log, groups, feat_cols)
    cb_params = {
        'iterations': 3000,
        'depth': best_cb_params['depth'],
        'learning_rate': best_cb_params['learning_rate'],
        'l2_leaf_reg': best_cb_params['l2_leaf_reg'],
        'random_strength': best_cb_params['random_strength'],
        'bagging_temperature': best_cb_params['bagging_temperature'],
        'loss_function': 'MAE',
        'random_seed': RANDOM_STATE,
        'verbose': 0,
        'early_stopping_rounds': 50,
    }

    # ══════════════════════════════════════════
    # Phase 3: 5-fold 스태킹 (최적 파라미터)
    # ══════════════════════════════════════════
    print('\n' + '─' * 70)
    print('[Phase 3] 5-fold 스태킹 (최적 LGBM + CB)')
    print('─' * 70)

    print('\n[LGBM] 학습 시작...')
    oof_lg, test_lg = train_lgbm_oof_custom(train, test, y_log, groups, feat_cols, lgbm_params)
    mae_lg = np.abs(np.expm1(oof_lg) - y_raw.values).mean()
    print(f'LGBM OOF MAE={mae_lg:.4f}')

    print('\n[CatBoost] 학습 시작...')
    oof_cb, test_cb = train_cb_oof_custom(train, test, y_log, groups, feat_cols, cb_params)
    mae_cb = np.abs(np.expm1(oof_cb) - y_raw.values).mean()
    print(f'CatBoost OOF MAE={mae_cb:.4f}')

    print('\n[TW1.8] 학습 시작...')
    oof_tw, test_tw = train_tw18_oof(train, test, y_raw, groups, feat_cols)
    mae_tw = np.abs(oof_tw - y_raw.values).mean()
    print(f'TW1.8 OOF MAE={mae_tw:.4f}')

    print('\n[ExtraTrees] 학습 시작...')
    oof_et, test_et = train_et_oof(train, test, y_log, groups, feat_cols)
    mae_et = np.abs(np.expm1(oof_et) - y_raw.values).mean()
    print(f'ExtraTrees OOF MAE={mae_et:.4f}')

    print('\n[RandomForest] 학습 시작...')
    oof_rf, test_rf = train_rf_oof(train, test, y_log, groups, feat_cols)
    mae_rf = np.abs(np.expm1(oof_rf) - y_raw.values).mean()
    print(f'RandomForest OOF MAE={mae_rf:.4f}')

    # ══════════════════════════════════════════
    # OOF 다양성 분석
    # ══════════════════════════════════════════
    print('\n' + '─' * 70)
    print('[다양성 분석] OOF 상관관계')
    print('─' * 70)
    oof_raw_lg = np.expm1(oof_lg)
    oof_raw_cb = np.expm1(oof_cb)
    oof_raw_et = np.expm1(oof_et)
    oof_raw_rf = np.expm1(oof_rf)

    models_raw = {
        'LGBM': oof_raw_lg, 'TW': oof_tw, 'CB': oof_raw_cb,
        'ET': oof_raw_et, 'RF': oof_raw_rf
    }
    names = list(models_raw.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = np.corrcoef(models_raw[names[i]], models_raw[names[j]])[0, 1]
            prev = ''
            if names[i] == 'LGBM' and names[j] == 'ET':
                prev = ' (model21 0.9661)'
            elif names[i] == 'LGBM' and names[j] == 'RF':
                prev = ' (model21 0.8994)'
            print(f'  {names[i]:4s}-{names[j]:4s}: {c:.4f}{prev}')

    # ══════════════════════════════════════════
    # 가중 앙상블 (비교용)
    # ══════════════════════════════════════════
    def loss5(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        blend = (w[0] * oof_raw_lg + w[1] * oof_raw_cb +
                 w[2] * oof_tw + w[3] * oof_raw_et + w[4] * oof_raw_rf)
        return np.mean(np.abs(blend - y_raw.values))

    best_loss, best_w = np.inf, np.ones(5) / 5
    for _ in range(500):
        w0 = np.random.dirichlet(np.ones(5))
        res = minimize(loss5, w0, method='Nelder-Mead')
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = np.abs(res.x) / np.abs(res.x).sum()
    print(f'\n  가중 앙상블 CV MAE: {best_loss:.4f}')
    print(f'    LGBM={best_w[0]:.3f}, CB={best_w[1]:.3f}, TW={best_w[2]:.3f}, '
          f'ET={best_w[3]:.3f}, RF={best_w[4]:.3f}')

    # ══════════════════════════════════════════
    # 메타 LGBM
    # ══════════════════════════════════════════
    print('\n' + '─' * 70)
    print('[메타 LGBM] 5모델 스택킹')
    print('─' * 70)

    test_tw_clipped = np.maximum(test_tw, 0)
    meta_train = np.column_stack([oof_lg, oof_cb, np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf])
    meta_test = np.column_stack([test_lg, test_cb, np.log1p(test_tw_clipped), test_et, test_rf])

    oof_meta, test_meta, mae_meta = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    # ══════════════════════════════════════════
    # 제출 파일
    # ══════════════════════════════════════════
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    sub_path = os.path.join(SUB_DIR, 'model23_optuna_v2.csv')
    sample.to_csv(sub_path, index=False)
    print(f'\n제출 파일 저장: {sub_path}')

    # ══════════════════════════════════════════
    # 결과 저장 (JSON)
    # ══════════════════════════════════════════
    result_json = {
        'phase1_lgbm_best_mae': float(lgbm_study.best_value),
        'phase1_lgbm_params': best_lgbm_params,
        'phase2_cb_best_mae': float(cb_study.best_value),
        'phase2_cb_params': best_cb_params,
        'phase3_oof_lgbm_mae': float(mae_lg),
        'phase3_oof_cb_mae': float(mae_cb),
        'phase3_oof_tw_mae': float(mae_tw),
        'phase3_oof_et_mae': float(mae_et),
        'phase3_oof_rf_mae': float(mae_rf),
        'phase3_ensemble_mae': float(best_loss),
        'phase3_meta_mae': float(mae_meta),
        'phase3_meta_pred_std': float(oof_meta.std()),
    }
    with open(os.path.join(CKPT_DIR, 'model23_results.json'), 'w') as f:
        json.dump(result_json, f, indent=2)

    # ══════════════════════════════════════════
    # 타겟 구간별 분석
    # ══════════════════════════════════════════
    print('\n' + '─' * 70)
    print('[분석] 타겟 구간별 MAE')
    print('─' * 70)
    bins = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 800)]
    for lo, hi in bins:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(oof_meta[mask] - y_raw.values[mask]).mean()
            seg_pred = oof_meta[mask].mean()
            print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d}  MAE={seg_mae:6.2f}  pred_mean={seg_pred:6.2f}')

    # ══════════════════════════════════════════
    # 최종 요약
    # ══════════════════════════════════════════
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 70)
    print(f'모델실험23 완료 ({elapsed:.1f}분 소요)')
    print('=' * 70)
    print(f'\n[Phase 1] Optuna LGBM 튜닝')
    print(f'  최적 MAE: {lgbm_study.best_value:.4f}')
    print(f'\n[Phase 2] Optuna CatBoost 튜닝')
    print(f'  최적 MAE: {cb_study.best_value:.4f}')
    print(f'\n[Phase 3] 5-fold 스태킹')
    print(f'  LGBM  OOF MAE : {mae_lg:.4f}')
    print(f'  CB    OOF MAE : {mae_cb:.4f}')
    print(f'  TW1.8 OOF MAE : {mae_tw:.4f}')
    print(f'  ET    OOF MAE : {mae_et:.4f}')
    print(f'  RF    OOF MAE : {mae_rf:.4f}')
    print(f'  가중 앙상블    : {best_loss:.4f}')
    print(f'  메타 LGBM     : {mae_meta:.4f}  pred_std={oof_meta.std():.2f}')
    print(f'\n[기준 비교]')
    print(f'  model21 CV    : 8.5097')
    print(f'  변화          : {mae_meta - 8.5097:+.4f}')
    print(f'  기대 Public (×1.1627): {mae_meta * 1.1627:.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
