"""
옵션 B: 메타 학습기 Optuna 튜닝
================================
목적 : v3(LGBM+TW1.8+CB+ET) 4모델 OOF에 대해
       메타 LightGBM 하이퍼파라미터를 Optuna로 최적화

비교 기준:
  v3 기본 메타: CV 8.7929 / Public 10.2264 🏆

체크포인트 재활용 (재학습 없음):
  docs/stacking_ckpt/  → lgbm, cb, et
  docs/stacking_v2_ckpt/ → tw18

예상 시간: ~15분 (50 trials × 5-fold 메타만)
출력: submissions/stacking_4model_optuna_meta.csv
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import warnings, gc, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

_BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_BASE, '..', 'data')
SUB_DIR   = os.path.join(_BASE, '..', 'submissions')
CKPT_V1   = os.path.join(_BASE, '..', 'docs', 'stacking_ckpt')
CKPT_V2   = os.path.join(_BASE, '..', 'docs', 'stacking_v2_ckpt')
N_SPLITS  = 5
RANDOM_STATE = 42
N_TRIALS  = 50  # Optuna 탐색 횟수


def load_data():
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(
        train, test, layout,
        lag_lags=[1,2,3,4,5,6],
        rolling_windows=[3,5,10],
    )
    return train, test


def save_sub(preds, filename):
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(preds, 0)
    sample.to_csv(os.path.join(SUB_DIR, filename), index=False)
    print(f'  → 저장: submissions/{filename}')


def load_ckpts():
    oof_lg, test_lg = (np.load(os.path.join(CKPT_V1, f'lgbm_{s}.npy')) for s in ['oof','test'])
    oof_cb, test_cb = (np.load(os.path.join(CKPT_V1, f'cb_{s}.npy'))   for s in ['oof','test'])
    oof_et, test_et = (np.load(os.path.join(CKPT_V1, f'et_{s}.npy'))   for s in ['oof','test'])
    oof_tw, test_tw = (np.load(os.path.join(CKPT_V2, f'tw18_{s}.npy')) for s in ['oof','test'])
    print('  체크포인트 로드 완료: LGBM / CB / ET / TW1.8')
    return (oof_lg, test_lg, oof_cb, test_cb, oof_et, test_et, oof_tw, test_tw)


def build_meta_features(oof_lg, oof_cb, oof_tw, oof_et,
                        test_lg, test_cb, test_tw, test_et):
    test_tw_clipped = np.maximum(test_tw, 0)
    meta_tr = np.column_stack([oof_lg, oof_cb, np.log1p(oof_tw), oof_et])
    meta_te = np.column_stack([test_lg, test_cb, np.log1p(test_tw_clipped), test_et])
    return meta_tr, meta_te


def cv_meta_lgbm(params, meta_train, y_raw, groups):
    """주어진 파라미터로 5-fold 메타 CV MAE 반환"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y_raw))
    for tr_idx, va_idx in gkf.split(meta_train, y_raw, groups):
        X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
        y_tr_log = np.log1p(y_raw.iloc[tr_idx].values)
        y_va_log = np.log1p(y_raw.iloc[va_idx].values)
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr_log,
              eval_set=[(X_va, y_va_log)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[va_idx] = np.expm1(m.predict(X_va))
        del m; gc.collect()
    return np.mean(np.abs(oof - y_raw.values))


def train_final_meta(params, meta_train, meta_test, y_raw, groups):
    """최적 파라미터로 5-fold 재학습 → 테스트 예측"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y_raw))
    test_pred = np.zeros(meta_test.shape[0])
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
        y_tr_log = np.log1p(y_raw.iloc[tr_idx].values)
        y_va_log = np.log1p(y_raw.iloc[va_idx].values)
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr_log,
              eval_set=[(X_va, y_va_log)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[va_idx] = np.expm1(m.predict(X_va))
        test_pred  += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.mean(np.abs(oof[va_idx] - y_raw.iloc[va_idx].values))
        print(f'  [최적 메타] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    oof_mae = np.mean(np.abs(oof - y_raw.values))
    print(f'  [최적 메타] OOF MAE={oof_mae:.4f} | std={oof.std():.2f}')
    return oof, test_pred, oof_mae


def main():
    print('=' * 60)
    print('옵션 B: 4모델 OOF 기반 메타 LGBM Optuna 튜닝')
    print(f'  N_TRIALS={N_TRIALS} / 비교 기준: v3 CV 8.7929')
    print('=' * 60)

    train, test = load_data()
    y_raw  = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']

    print('\n[체크포인트 로드]')
    oof_lg, test_lg, oof_cb, test_cb, oof_et, test_et, oof_tw, test_tw = load_ckpts()
    meta_train, meta_test = build_meta_features(
        oof_lg, oof_cb, oof_tw, oof_et,
        test_lg, test_cb, test_tw, test_et)

    # ── Optuna 탐색 ──────────────────────────────────────────
    print(f'\n[Optuna 탐색] {N_TRIALS} trials 시작...')

    def objective(trial):
        params = {
            'num_leaves'       : trial.suggest_int('num_leaves', 15, 63),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'feature_fraction' : trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
            'n_estimators'     : 500,
            'bagging_freq'     : 1,
            'objective'        : 'regression_l1',
            'random_state'     : RANDOM_STATE,
            'verbosity'        : -1,
            'n_jobs'           : -1,
        }
        return cv_meta_lgbm(params, meta_train, y_raw, groups)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        'n_estimators': 500,
        'bagging_freq': 1,
        'objective'   : 'regression_l1',
        'random_state': RANDOM_STATE,
        'verbosity'   : -1,
        'n_jobs'      : -1,
    })

    print(f'\n  Optuna 최적 CV MAE: {study.best_value:.4f}')
    print(f'  최적 파라미터:')
    for k, v in study.best_params.items():
        print(f'    {k}: {v}')

    # ── 최적 파라미터로 최종 예측 ────────────────────────────
    print('\n[최적 파라미터로 최종 예측]')
    _, test_pred, final_mae = train_final_meta(
        best_params, meta_train, meta_test, y_raw, groups)

    save_sub(test_pred, 'stacking_4model_optuna_meta.csv')

    print(f'\n{"="*60}')
    print('옵션 B 결과 요약')
    print(f'{"="*60}')
    print(f'  [비교] v3 기본 메타    : CV 8.7929 / Public 10.2264 🏆')
    print(f'  [결과] Optuna 최적 메타: CV {final_mae:.4f}')
    print(f'  Optuna 탐색 중 최고   : {study.best_value:.4f}')


if __name__ == '__main__':
    main()
