"""
모델실험29B: model28A 기반 Optuna 재튜닝
=============================================================
model28A 피처셋(415피처) + 체크포인트 활용.
LGBM + CB만 재튜닝 (TW/ET/RF는 model28A 체크포인트 재사용).

⚠️ 과적합 방지 전략 (LAW-5 "Optuna = CV↑ Public≈ or ↓"):
  1. 2-fold CV → 각 fold가 더 어려움 (일반화 압력 ↑)
  2. Regularization 범위 확대 (reg_alpha/lambda: 0~5, min_child: 20~60)
  3. feature_fraction 하한 0.3 (model28A 0.51 → 더 강한 드롭아웃)
  4. num_leaves 상한 200 (과도한 복잡도 제한)
  5. learning_rate 하한 0.01 (느린 학습 → 일반화 유리)

핵심 교훈 (model23 실패):
  - model23: Optuna LGBM num_leaves=145, lr=0.00866 → CV 8.5038, Public 9.9522
  - model22: 기본 params → CV ~8.51, Public 9.9385
  - model23 배율 1.1703 vs model22 1.168 → Optuna가 오히려 배율 악화
  - 원인: train에 과적합된 파라미터 → test 일반화 저하
  → 이번에는 regularization 강화로 방지

실행: python src/run_exp_v3_model29B_optuna_retune.py
예상 시간: ~120분 (Optuna 50trial × 2fold + 5모델 5fold 재학습)
출력: submissions/model29B_optuna_retune.csv
체크포인트: docs/model29B_ckpt/
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import optuna
import warnings, gc, os, sys, time

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features, get_feature_cols

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_28A = os.path.join(_BASE, '..', 'docs', 'model28A_ckpt')
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model29B_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]


# ─────────────────────────────────────────────
# 피처 구성 (model28A 동일)
# ─────────────────────────────────────────────
def add_scenario_agg_features(df):
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]
        df[f'sc_{col}_mean'] = grp.transform('mean')
        df[f'sc_{col}_std']  = grp.transform('std').fillna(0)
        df[f'sc_{col}_max']  = grp.transform('max')
        df[f'sc_{col}_min']  = grp.transform('min')
        df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']
        df[f'sc_{col}_median'] = grp.transform('median')
        df[f'sc_{col}_p10'] = grp.transform(lambda x: x.quantile(0.10))
        df[f'sc_{col}_p90'] = grp.transform(lambda x: x.quantile(0.90))
        df[f'sc_{col}_skew'] = grp.transform(lambda x: x.skew()).fillna(0)
        df[f'sc_{col}_kurtosis'] = grp.transform(lambda x: x.kurtosis()).fillna(0)
        cv_series = df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)
        df[f'sc_{col}_cv'] = cv_series.fillna(0)
    return df


def add_layout_ratio_features(df):
    """model28A 동일 비율 피처 5종"""
    def safe_div(a, b, fill=0):
        return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = safe_div(
            df['sc_order_inflow_15m_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = safe_div(
            df['sc_congestion_score_mean'], df['intersection_count'])
    if 'sc_low_battery_ratio_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_stress'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df.get('sc_charge_queue_length_mean', 0),
            df['charger_count'])
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = safe_div(
            df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']
    return df


def load_data():
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(train, test, layout,
                                  lag_lags=[1,2,3,4,5,6],
                                  rolling_windows=[3,5,10])
    train = add_scenario_agg_features(train)
    test  = add_scenario_agg_features(test)
    train = add_layout_ratio_features(train)
    test  = add_layout_ratio_features(test)
    return train, test


def get_feat_cols_fn(train):
    return [c for c in train.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and train[c].dtype != object]


# ─────────────────────────────────────────────
# Optuna: LGBM 튜닝 (과적합 방지 강화)
# ─────────────────────────────────────────────
def optuna_lgbm(X, y_log, groups, feat_cols, n_trials=50):
    """
    2-fold GroupKFold로 LGBM 하이퍼파라미터 탐색.
    regularization 범위를 의도적으로 넓혀 과적합 방지.
    """
    X_np = X[feat_cols].fillna(0)

    def objective(trial):
        params = {
            'num_leaves':       trial.suggest_int('num_leaves', 63, 200),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.30, 0.70),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.70, 0.95),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 60),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
            'objective': 'regression_l1',
            'n_estimators': 3000,
            'bagging_freq': 1,
            'random_state': RANDOM_STATE,
            'verbosity': -1,
            'n_jobs': -1,
        }

        gkf = GroupKFold(n_splits=2)  # 2-fold = 더 어려운 검증 = 일반화 압력 ↑
        maes = []
        for tr_idx, va_idx in gkf.split(X_np, y_log, groups):
            m = lgb.LGBMRegressor(**params)
            m.fit(X_np.iloc[tr_idx], y_log.iloc[tr_idx],
                  eval_set=[(X_np.iloc[va_idx], y_log.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            pred = m.predict(X_np.iloc[va_idx])
            mae = np.abs(np.expm1(pred) - np.expm1(y_log.iloc[va_idx].values)).mean()
            maes.append(mae)
            del m; gc.collect()
        return np.mean(maes)

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f'\n[Optuna LGBM] Best 2-fold MAE: {study.best_value:.4f}')
    print(f'  Best params: {study.best_params}')

    # 최적 파라미터 반환
    best = study.best_params.copy()
    best.update({
        'objective': 'regression_l1',
        'n_estimators': 3000,
        'bagging_freq': 1,
        'random_state': RANDOM_STATE,
        'verbosity': -1,
        'n_jobs': -1,
    })
    return best, study.best_value


# ─────────────────────────────────────────────
# Optuna: CB 튜닝 (과적합 방지 강화)
# ─────────────────────────────────────────────
def optuna_cb(X, y_log, groups, feat_cols, n_trials=50):
    """
    2-fold GroupKFold로 CatBoost 하이퍼파라미터 탐색.
    """
    X_np = X[feat_cols].fillna(0).values

    def objective(trial):
        params = {
            'iterations': 3000,
            'learning_rate':  trial.suggest_float('learning_rate', 0.01, 0.10, log=True),
            'depth':          trial.suggest_int('depth', 4, 9),
            'l2_leaf_reg':    trial.suggest_float('l2_leaf_reg', 1.0, 15.0),
            'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 3.0),
            'loss_function': 'MAE',
            'random_seed': RANDOM_STATE,
            'verbose': 0,
            'early_stopping_rounds': 50,
        }

        gkf = GroupKFold(n_splits=2)
        maes = []
        for tr_idx, va_idx in gkf.split(X_np, y_log, groups):
            train_pool = cb.Pool(X_np[tr_idx], y_log.values[tr_idx])
            val_pool   = cb.Pool(X_np[va_idx], y_log.values[va_idx])
            m = cb.CatBoostRegressor(**params)
            m.fit(train_pool, eval_set=val_pool, use_best_model=True)
            pred = m.predict(X_np[va_idx])
            mae = np.abs(np.expm1(pred) - np.expm1(y_log.values[va_idx])).mean()
            maes.append(mae)
            del m; gc.collect()
        return np.mean(maes)

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 1))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f'\n[Optuna CB] Best 2-fold MAE: {study.best_value:.4f}')
    print(f'  Best params: {study.best_params}')

    best = study.best_params.copy()
    best.update({
        'iterations': 3000,
        'loss_function': 'MAE',
        'random_seed': RANDOM_STATE,
        'verbose': 0,
        'early_stopping_rounds': 50,
    })
    return best, study.best_value


# ─────────────────────────────────────────────
# Base Learner 학습 (model28A와 동일 구조)
# ─────────────────────────────────────────────
def save_ckpt(name, oof, test_pred):
    np.save(os.path.join(CKPT_DIR, f'{name}_oof.npy'), oof)
    np.save(os.path.join(CKPT_DIR, f'{name}_test.npy'), test_pred)

def load_ckpt_28A(name):
    return (np.load(os.path.join(CKPT_28A, f'{name}_oof.npy')),
            np.load(os.path.join(CKPT_28A, f'{name}_test.npy')))


def train_lgbm_oof(X_train, X_test, y_log, groups, feat_cols, params):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0); X_te_np = X_test[feat_cols].fillna(0)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr_np.iloc[tr_idx], y_log.iloc[tr_idx],
              eval_set=[(X_tr_np.iloc[va_idx], y_log.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(X_tr_np.iloc[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.iloc[va_idx].values)).mean()
        print(f'  [LGBM-Optuna] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred


def train_cb_oof(X_train, X_test, y_log, groups, feat_cols, params):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train)); test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values; X_te_np = X_test[feat_cols].fillna(0).values
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        train_pool = cb.Pool(X_tr_np[tr_idx], y_log.values[tr_idx])
        val_pool   = cb.Pool(X_tr_np[va_idx], y_log.values[va_idx])
        m = cb.CatBoostRegressor(**params)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [CB-Optuna] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred


# ─────────────────────────────────────────────
# 메타 학습기
# ─────────────────────────────────────────────
META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

def run_meta_lgbm(meta_train, meta_test, y_raw, groups, label='LGBM-meta'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_meta = np.zeros(len(y_raw)); test_meta = np.zeros(meta_test.shape[0])
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(meta_train[tr_idx], np.log1p(y_raw.iloc[tr_idx].values),
              eval_set=[(meta_train[va_idx], np.log1p(y_raw.iloc[va_idx].values))],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[va_idx] = np.expm1(m.predict(meta_train[va_idx]))
        test_meta += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    oof_mae = np.abs(oof_meta - y_raw.values).mean()
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | pred_std={oof_meta.std():.2f}')
    return oof_meta, test_meta, oof_mae


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 60)
    print('모델실험29B: model28A 기반 Optuna 재튜닝')
    print('기준: Model28A CV 8.4743 / Public 9.8525 (배율 1.1626)')
    print('변경: LGBM + CB Optuna 튜닝 (과적합 방지 강화)')
    print('  - 2-fold CV, regularization 범위 확대')
    print('  - TW/ET/RF는 model28A 체크포인트 재사용')
    print('=' * 60)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    # 데이터 로드
    train, test = load_data()
    feat_cols = get_feat_cols_fn(train)
    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'피처 수: {len(feat_cols)}')

    # ══════════════════════════════════════════
    # Phase 1: Optuna 탐색 (LGBM + CB)
    # ══════════════════════════════════════════
    print('\n' + '═' * 60)
    print('[Phase 1] Optuna 하이퍼파라미터 탐색')
    print('═' * 60)

    # model28A 기준 파라미터 (비교용)
    print('\n[기준] model28A LGBM: num_leaves=181, lr=0.0206, feat_frac=0.51, reg_a=0.38, reg_l=0.36')
    print('[기준] model28A CB:   depth=6, lr=0.05, l2_leaf_reg=3.0')

    best_lgbm_params, lgbm_2fold_mae = optuna_lgbm(train, y_log, groups, feat_cols, n_trials=50)
    best_cb_params, cb_2fold_mae = optuna_cb(train, y_log, groups, feat_cols, n_trials=50)

    # ══════════════════════════════════════════
    # Phase 2: 5-fold OOF 학습
    # ══════════════════════════════════════════
    print('\n' + '═' * 60)
    print('[Phase 2] 5-fold OOF 학습 (Optuna LGBM + CB, 나머지 체크포인트)')
    print('═' * 60)

    # LGBM (Optuna 튜닝)
    print('\n[LGBM-Optuna] 학습...')
    oof_lg, test_lg = train_lgbm_oof(train, test, y_log, groups, feat_cols, best_lgbm_params)
    save_ckpt('lgbm_optuna', oof_lg, test_lg)
    mae_lg = np.abs(np.expm1(oof_lg) - y_raw.values).mean()
    print(f'  LGBM-Optuna OOF MAE={mae_lg:.4f}')

    # CB (Optuna 튜닝)
    print('\n[CB-Optuna] 학습...')
    oof_cb, test_cb = train_cb_oof(train, test, y_log, groups, feat_cols, best_cb_params)
    save_ckpt('cb_optuna', oof_cb, test_cb)
    mae_cb = np.abs(np.expm1(oof_cb) - y_raw.values).mean()
    print(f'  CB-Optuna OOF MAE={mae_cb:.4f}')

    # TW/ET/RF: model28A 체크포인트 재사용
    print('\n[TW/ET/RF] model28A 체크포인트 재사용')
    oof_tw, test_tw = load_ckpt_28A('tw18')
    oof_et, test_et = load_ckpt_28A('et')
    oof_rf, test_rf = load_ckpt_28A('rf')
    mae_tw = np.abs(oof_tw - y_raw.values).mean()
    mae_et = np.abs(np.expm1(oof_et) - y_raw.values).mean()
    mae_rf = np.abs(np.expm1(oof_rf) - y_raw.values).mean()
    print(f'  TW1.8 OOF MAE={mae_tw:.4f} (model28A)')
    print(f'  ET    OOF MAE={mae_et:.4f} (model28A)')
    print(f'  RF    OOF MAE={mae_rf:.4f} (model28A)')

    # model28A LGBM/CB 기준 비교
    oof_lg_28A, _ = load_ckpt_28A('lgbm')
    oof_cb_28A, _ = load_ckpt_28A('cb')
    mae_lg_28A = np.abs(np.expm1(oof_lg_28A) - y_raw.values).mean()
    mae_cb_28A = np.abs(np.expm1(oof_cb_28A) - y_raw.values).mean()
    print(f'\n  LGBM 변화: {mae_lg_28A:.4f} → {mae_lg:.4f} (Δ={mae_lg - mae_lg_28A:+.4f})')
    print(f'  CB   변화: {mae_cb_28A:.4f} → {mae_cb:.4f} (Δ={mae_cb - mae_cb_28A:+.4f})')

    # ── 상관관계 ──
    print('\n[다양성] OOF 상관관계')
    oof_raw = {
        'LGBM': np.expm1(oof_lg), 'TW': oof_tw, 'CB': np.expm1(oof_cb),
        'ET': np.expm1(oof_et), 'RF': np.expm1(oof_rf)
    }
    names = list(oof_raw.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            c = np.corrcoef(oof_raw[names[i]], oof_raw[names[j]])[0,1]
            print(f'  {names[i]:4s}-{names[j]:4s}: {c:.4f}')

    # ── 가중 앙상블 ──
    arrs = [oof_raw['LGBM'], oof_raw['CB'], oof_raw['TW'], oof_raw['ET'], oof_raw['RF']]
    def loss5(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        return np.mean(np.abs(sum(w[i]*arrs[i] for i in range(5)) - y_raw.values))
    best_loss, best_w = np.inf, np.ones(5)/5
    for _ in range(500):
        w0 = np.random.dirichlet(np.ones(5))
        res = minimize(loss5, w0, method='Nelder-Mead')
        if res.fun < best_loss:
            best_loss = res.fun; best_w = np.abs(res.x) / np.abs(res.x).sum()
    print(f'\n  가중 앙상블 CV MAE: {best_loss:.4f}')

    # ══════════════════════════════════════════
    # Phase 3: 메타 학습기
    # ══════════════════════════════════════════
    print('\n' + '═' * 60)
    print('[Phase 3] 5모델 LGBM 메타 학습기')
    print('═' * 60)

    test_tw_clipped = np.maximum(test_tw, 0)
    meta_train = np.column_stack([oof_lg, oof_cb,
                                   np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf])
    meta_test  = np.column_stack([test_lg, test_cb,
                                   np.log1p(test_tw_clipped), test_et, test_rf])

    oof_meta, test_meta, mae_meta = run_meta_lgbm(meta_train, meta_test, y_raw, groups)

    # 제출 파일
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    sub_path = os.path.join(SUB_DIR, 'model29B_optuna_retune.csv')
    sample.to_csv(sub_path, index=False)
    print(f'\n제출 파일: {sub_path}')

    # ── 분석 ──
    print('\n' + '─' * 60)
    print('[분석] 타겟 구간별 MAE')
    print('─' * 60)
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    for lo, hi in bins:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            seg_mae = np.abs(oof_meta[mask] - y_raw.values[mask]).mean()
            print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d}  MAE={seg_mae:6.2f}')

    print('\n' + '─' * 60)
    print('[분석] 예측 분포')
    print('─' * 60)
    print(f'  OOF:  mean={oof_meta.mean():.2f}, std={oof_meta.std():.2f}, max={oof_meta.max():.2f}')
    print(f'  test: mean={test_meta.mean():.2f}, std={test_meta.std():.2f}, max={test_meta.max():.2f}')

    # 최종 요약
    elapsed = (time.time() - t0) / 60
    print('\n' + '=' * 60)
    print(f'모델실험29B 결과 ({elapsed:.1f}분 소요)')
    print('=' * 60)
    print(f'  Optuna LGBM 2-fold: {lgbm_2fold_mae:.4f}')
    print(f'  Optuna CB   2-fold: {cb_2fold_mae:.4f}')
    print(f'  LGBM-Optuna OOF   : {mae_lg:.4f} (model28A: {mae_lg_28A:.4f}, Δ={mae_lg - mae_lg_28A:+.4f})')
    print(f'  CB-Optuna   OOF   : {mae_cb:.4f} (model28A: {mae_cb_28A:.4f}, Δ={mae_cb - mae_cb_28A:+.4f})')
    print(f'  가중 앙상블        : {best_loss:.4f}')
    print(f'  메타 LGBM         : {mae_meta:.4f}  pred_std={oof_meta.std():.2f}')
    print(f'  test pred         : mean={test_meta.mean():.2f}, std={test_meta.std():.2f}')
    print(f'  Model28A (기준)   : CV 8.4743 / Public 9.8525 (배율 1.1626)')
    print(f'  Model29B 변화     : {mae_meta - 8.4743:+.4f}')
    print(f'  기대 Public (×1.163): {mae_meta * 1.163:.4f}')
    print(f'  기대 Public (×1.168): {mae_meta * 1.168:.4f}')

    # 과적합 경고
    if test_meta.std() < 15.5:
        print(f'\n  ⚠️ test std={test_meta.std():.2f} < 15.5 — model28A(16.28) 대비 압축 → 배율 악화 위험')
    else:
        print(f'\n  ✅ test std={test_meta.std():.2f} ≥ 15.5 — 분포 유지')

    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
