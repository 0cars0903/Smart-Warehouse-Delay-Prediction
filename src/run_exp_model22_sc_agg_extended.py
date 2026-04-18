"""
모델실험22: 시나리오 집계 확장 피처 + 5모델 스태킹
=========================================================
모델21 확장 버전: 시나리오 집계 통계를 5종(mean/std/max/min/diff)에서 11종으로 확대

핵심 변경:
  - 기존(model21): mean/std/max/min/diff → 18종 피처 × 5통계 = 90피처
  - 신규(model22): mean/std/max/min/diff/median/p10/p90/skew/kurtosis/cv
    → 18종 피처 × 11통계 = 198피처
  - FE v1 (lag/rolling) + 확장 시나리오 집계 피처
  - 5모델(LGBM+TW1.8+CB+ET+RF) 전체 재학습

예상:
  - 확장된 통계 정보로 시나리오 내 분포 특성 더 세밀 모델링
  - 메타 학습기가 시나리오별 이질성(skew/kurtosis) 포착 가능
  - CV: model21 8.5097 대비 개선 기대

실행: python src/run_exp_model22_sc_agg_extended.py
예상 시간: ~90분 (5모델 × 5fold + 메타)
출력: submissions/model22_sc_agg_extended.csv
체크포인트: docs/model22_ckpt/
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import warnings, gc, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features, get_feature_cols

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model22_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 시나리오 집계 피처 대상 (18종)
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
# 모델 하이퍼파라미터
# ─────────────────────────────────────────────
LGBM_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

TW18_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3.0,
    'loss_function': 'Tweedie:variance_power=1.8',
    'random_seed': RANDOM_STATE, 'verbose': 0,
    'early_stopping_rounds': 50,
}

CB_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05,
    'depth': 6, 'l2_leaf_reg': 3.0,
    'loss_function': 'MAE',
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
    np.save(os.path.join(CKPT_DIR, f'{name}_oof.npy'), oof)
    np.save(os.path.join(CKPT_DIR, f'{name}_test.npy'), test_pred)


def load_ckpt(name):
    oof  = np.load(os.path.join(CKPT_DIR, f'{name}_oof.npy'))
    test = np.load(os.path.join(CKPT_DIR, f'{name}_test.npy'))
    return oof, test


def ckpt_exists(name):
    return (os.path.exists(os.path.join(CKPT_DIR, f'{name}_oof.npy'))
            and os.path.exists(os.path.join(CKPT_DIR, f'{name}_test.npy')))


def add_scenario_agg_features(df):
    """
    시나리오 집계 피처 확장 broadcast
    기존(5통계): mean/std/max/min/diff
    신규(11통계): mean/std/max/min/diff + median/p10/p90/skew/kurtosis/cv
    18종 피처 × 11통계 = 198피처
    """
    df = df.copy()

    for col in SC_AGG_COLS:
        if col not in df.columns:
            continue

        grp = df.groupby('scenario_id')[col]

        # 기본 5가지 통계
        df[f'sc_{col}_mean'] = grp.transform('mean')
        df[f'sc_{col}_std']  = grp.transform('std').fillna(0)
        df[f'sc_{col}_max']  = grp.transform('max')
        df[f'sc_{col}_min']  = grp.transform('min')
        df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']

        # 추가 6가지 통계
        df[f'sc_{col}_median'] = grp.transform('median')

        # percentiles
        df[f'sc_{col}_p10'] = grp.transform(lambda x: x.quantile(0.10))
        df[f'sc_{col}_p90'] = grp.transform(lambda x: x.quantile(0.90))

        # skewness (왜도)
        df[f'sc_{col}_skew'] = grp.transform(lambda x: x.skew()).fillna(0)

        # kurtosis (첨도)
        df[f'sc_{col}_kurtosis'] = grp.transform(lambda x: x.kurtosis()).fillna(0)

        # coefficient of variation (변동 계수) — std/mean, 0 division 방지
        cv_series = df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)
        df[f'sc_{col}_cv'] = cv_series.fillna(0)

    return df


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
def load_data():
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    # FE v1 파이프라인
    train, test = build_features(
        train, test, layout,
        lag_lags=[1,2,3,4,5,6],
        rolling_windows=[3,5,10],
    )

    # 시나리오 집계 피처 추가 (확장)
    train = add_scenario_agg_features(train)
    test  = add_scenario_agg_features(test)

    sc_feats = [c for c in train.columns if c.startswith('sc_')]
    print(f'시나리오 집계 피처(확장): {len(sc_feats)}종 추가')

    return train, test


def get_feat_cols(train):
    return [c for c in train.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and train[c].dtype != object]


# ─────────────────────────────────────────────
# Layer 1: Base Learner OOF 생성
# ─────────────────────────────────────────────
def train_lgbm_oof(X_train, X_test, y_log, groups, feat_cols):
    """LightGBM (log1p 공간)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0)
    X_te_np = X_test[feat_cols].fillna(0)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(X_tr_np.iloc[tr_idx], y_log.iloc[tr_idx],
              eval_set=[(X_tr_np.iloc[va_idx], y_log.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[va_idx] = m.predict(X_tr_np.iloc[va_idx])
        test_pred  += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.iloc[va_idx].values)).mean()
        print(f'  [LGBM] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred


def train_tw18_oof(X_train, X_test, y_raw, groups, feat_cols):
    """CatBoost Tweedie p=1.8 (raw 공간 — expm1 불필요)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values
    X_te_np = X_test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_raw, groups)):
        train_pool = cb.Pool(X_tr_np[tr_idx], y_raw.values[tr_idx])
        val_pool   = cb.Pool(X_tr_np[va_idx], y_raw.values[va_idx])
        m = cb.CatBoostRegressor(**TW18_PARAMS)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred  += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(oof[va_idx] - y_raw.values[va_idx]).mean()
        print(f'  [TW1.8] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred


def train_cb_oof(X_train, X_test, y_log, groups, feat_cols):
    """CatBoost MAE (log1p 공간)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values
    X_te_np = X_test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        train_pool = cb.Pool(X_tr_np[tr_idx], y_log.values[tr_idx])
        val_pool   = cb.Pool(X_tr_np[va_idx], y_log.values[va_idx])
        m = cb.CatBoostRegressor(**CB_PARAMS)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred  += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [CB] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    return oof, test_pred


def train_et_oof(X_train, X_test, y_log, groups, feat_cols):
    """ExtraTrees (log1p 공간)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values
    X_te_np = X_test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = ExtraTreesRegressor(**ET_PARAMS)
        m.fit(X_tr_np[tr_idx], y_log.values[tr_idx])
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred  += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [ET] Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    return oof, test_pred


def train_rf_oof(X_train, X_test, y_log, groups, feat_cols):
    """RandomForest (log1p 공간)"""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    X_tr_np = X_train[feat_cols].fillna(0).values
    X_te_np = X_test[feat_cols].fillna(0).values

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_np, y_log, groups)):
        m = RandomForestRegressor(**RF_PARAMS)
        m.fit(X_tr_np[tr_idx], y_log.values[tr_idx])
        oof[va_idx] = m.predict(X_tr_np[va_idx])
        test_pred  += m.predict(X_te_np) / N_SPLITS
        mae = np.abs(np.expm1(oof[va_idx]) - np.expm1(y_log.values[va_idx])).mean()
        print(f'  [RF] Fold {fold+1}  MAE={mae:.4f}')
        del m; gc.collect()
    return oof, test_pred


# ─────────────────────────────────────────────
# Layer 2: 메타 학습기
# ─────────────────────────────────────────────
def run_meta_lgbm(meta_train, meta_test, y_raw, groups, label='LGBM-meta'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_meta  = np.zeros(len(y_raw))
    test_meta = np.zeros(meta_test.shape[0])

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        X_tr, X_va = meta_train[tr_idx], meta_train[va_idx]
        y_tr_log = np.log1p(y_raw.iloc[tr_idx].values)
        y_va_log = np.log1p(y_raw.iloc[va_idx].values)
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(X_tr, y_tr_log,
              eval_set=[(X_va, y_va_log)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        oof_meta[va_idx] = np.expm1(m.predict(X_va))
        test_meta       += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [{label}] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_mae = np.abs(oof_meta - y_raw.values).mean()
    pred_std = oof_meta.std()
    print(f'  [{label}] OOF MAE={oof_mae:.4f} | pred_std={pred_std:.2f}')
    return oof_meta, test_meta, oof_mae


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 60)
    print('모델실험22: 시나리오 집계 확장')
    print('기준: Model21 CV 8.5097 / Public 9.9550')
    print('변경: 시나리오 집계 통계 5종→11종 (198피처)')
    print('=' * 60)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    # 데이터 로드
    train, test = load_data()
    feat_cols = get_feat_cols(train)
    y_raw  = train['avg_delay_minutes_next_30m']
    y_log  = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'피처 수: {len(feat_cols)}')

    # ══════════════════════════════════════════
    # Layer 1: 5모델 Base Learner OOF
    # ══════════════════════════════════════════
    print('\n' + '─' * 60)
    print('[Layer 1] Base Learner OOF 생성')
    print('─' * 60)

    # ── LGBM ──
    if ckpt_exists('lgbm'):
        print('\n[LGBM] 체크포인트 로드')
        oof_lg, test_lg = load_ckpt('lgbm')
    else:
        print('\n[LGBM] 학습 시작...')
        oof_lg, test_lg = train_lgbm_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('lgbm', oof_lg, test_lg)
    mae_lg = np.abs(np.expm1(oof_lg) - y_raw.values).mean()
    print(f'  LGBM OOF MAE={mae_lg:.4f}')

    # ── TW1.8 ──
    if ckpt_exists('tw18'):
        print('\n[TW1.8] 체크포인트 로드')
        oof_tw, test_tw = load_ckpt('tw18')
    else:
        print('\n[TW1.8] 학습 시작...')
        oof_tw, test_tw = train_tw18_oof(train, test, y_raw, groups, feat_cols)
        save_ckpt('tw18', oof_tw, test_tw)
    mae_tw = np.abs(oof_tw - y_raw.values).mean()
    print(f'  TW1.8 OOF MAE={mae_tw:.4f} (raw 공간)')

    # ── CatBoost ──
    if ckpt_exists('cb'):
        print('\n[CB] 체크포인트 로드')
        oof_cb, test_cb = load_ckpt('cb')
    else:
        print('\n[CB] 학습 시작...')
        oof_cb, test_cb = train_cb_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('cb', oof_cb, test_cb)
    mae_cb = np.abs(np.expm1(oof_cb) - y_raw.values).mean()
    print(f'  CB OOF MAE={mae_cb:.4f}')

    # ── ExtraTrees ──
    if ckpt_exists('et'):
        print('\n[ET] 체크포인트 로드')
        oof_et, test_et = load_ckpt('et')
    else:
        print('\n[ET] 학습 시작...')
        oof_et, test_et = train_et_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('et', oof_et, test_et)
    mae_et = np.abs(np.expm1(oof_et) - y_raw.values).mean()
    print(f'  ET OOF MAE={mae_et:.4f}')

    # ── RandomForest ──
    if ckpt_exists('rf'):
        print('\n[RF] 체크포인트 로드')
        oof_rf, test_rf = load_ckpt('rf')
    else:
        print('\n[RF] 학습 시작...')
        oof_rf, test_rf = train_rf_oof(train, test, y_log, groups, feat_cols)
        save_ckpt('rf', oof_rf, test_rf)
    mae_rf = np.abs(np.expm1(oof_rf) - y_raw.values).mean()
    print(f'  RF OOF MAE={mae_rf:.4f}')

    # ══════════════════════════════════════════
    # OOF 상관관계 (다양성 분석)
    # ══════════════════════════════════════════
    print('\n' + '─' * 60)
    print('[다양성 분석] OOF 상관관계')
    print('─' * 60)
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
        for j in range(i+1, len(names)):
            c = np.corrcoef(models_raw[names[i]], models_raw[names[j]])[0,1]
            print(f'  {names[i]:4s}-{names[j]:4s}: {c:.4f}')

    # ══════════════════════════════════════════
    # 가중치 앙상블 (비교용)
    # ══════════════════════════════════════════
    def loss5(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        blend = (w[0]*oof_raw_lg + w[1]*oof_raw_cb +
                 w[2]*oof_tw + w[3]*oof_raw_et + w[4]*oof_raw_rf)
        return np.mean(np.abs(blend - y_raw.values))

    best_loss, best_w = np.inf, np.ones(5)/5
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
    # Layer 2: LGBM 메타 학습기
    # ══════════════════════════════════════════
    print('\n' + '─' * 60)
    print('[Layer 2] 5모델 LGBM 메타 학습기')
    print('─' * 60)

    # TW는 raw 공간이므로 log1p 변환
    test_tw_clipped = np.maximum(test_tw, 0)
    meta_train = np.column_stack([oof_lg, oof_cb, np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf])
    meta_test  = np.column_stack([test_lg, test_cb, np.log1p(test_tw_clipped), test_et, test_rf])

    oof_meta, test_meta, mae_meta = run_meta_lgbm(
        meta_train, meta_test, y_raw, groups
    )

    # ── 제출 파일 ──
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    sub_path = os.path.join(SUB_DIR, 'model22_sc_agg_extended.csv')
    sample.to_csv(sub_path, index=False)
    print(f'\n제출 파일 저장: {sub_path}')

    # ══════════════════════════════════════════
    # 타겟 구간별 분석
    # ══════════════════════════════════════════
    print('\n' + '─' * 60)
    print('[분석] 타겟 구간별 MAE')
    print('─' * 60)
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
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
    print('\n' + '=' * 60)
    print(f'모델실험22 결과 ({elapsed:.1f}분 소요)')
    print('=' * 60)
    print(f'  LGBM  OOF MAE : {mae_lg:.4f}')
    print(f'  TW1.8 OOF MAE : {mae_tw:.4f}')
    print(f'  CB    OOF MAE : {mae_cb:.4f}')
    print(f'  ET    OOF MAE : {mae_et:.4f}')
    print(f'  RF    OOF MAE : {mae_rf:.4f}')
    print(f'  가중 앙상블    : {best_loss:.4f}')
    print(f'  메타 LGBM     : {mae_meta:.4f}  pred_std={oof_meta.std():.2f}')
    print(f'  ')
    print(f'  Model21 (기준): 8.5097 (Public 9.9550, 배율 1.1627)')
    print(f'  Model22 변화  : {mae_meta - 8.5097:+.4f}')
    print(f'  기대 Public (×1.1627): {mae_meta * 1.1627:.4f}')
    print(f'  기대 Public (×1.1700): {mae_meta * 1.1700:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
