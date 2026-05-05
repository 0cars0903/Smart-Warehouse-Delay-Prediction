"""
model47 + 추가 base learner 탐색
================================================================
model47 기존 6모델 체크포인트를 재사용하고,
신규 base learner(XGBoost MAE / asym15 / LGBM-DART)를 추가해
메타 다양성 확대 가능성을 검증한다.

근거:
  Q1 분석:
    - model47은 이미 역대 성능향상을 이끈 모든 요소 포함
      (tw15, asym20, SC_AGG 23컬럼, ratio Tier1/2, Layout 교호작용)
    - 미사용 요소: asym15(α=1.5), XGBoost(새 FE에서 미시도), DART
  Q3 탐색:
    - XGBoost: 이전 실험(v1 시리즈)에서 가중치 0.038로 탈락
                → model47의 풍부한 SC_AGG FE에서 재시도 (다른 boosting 알고리즘 → 다양성)
    - asym15:   model47은 asym20만 사용. asym15 추가 시 α 다양성 확보
    - LGBM DART: 드롭아웃 기반 정규화 → 일반 LGBM과 이질적 오차 패턴

실행 전제: run_model47_combined.py 완료 (model47_ckpt 존재)
실행: python src/run_model47_addon_base.py
예상 시간: ~20~30분 (신규 3모델만 학습 + 메타 비교)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
import gc, os, sys, time, warnings
warnings.filterwarnings('ignore')
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)

from run_model46_base import (
    DATA_DIR, SUB_DIR, DOCS_DIR, N_SPLITS,
    SC_AGG_BASE, add_scenario_agg, add_ratio_tier1, add_ratio_tier2,
    get_feat_cols, run_meta, segment_report, diversity_report,
    ckpt_exists, load_ckpt, save_ckpt, safe_div,
    LGBM_PARAMS,
    asymmetric_mae_objective, asymmetric_mae_metric, ASYM_LGBM_PARAMS,
)
from feature_engineering import build_features

CKPT_47 = os.path.join(DOCS_DIR, 'model47_ckpt')    # 기존 6모델 체크포인트
CKPT_Q  = os.path.join(DOCS_DIR, 'model45_ckpt', 'strat_c')
RANDOM_STATE = 42

# model47와 동일한 SC_AGG 확장
SC_AGG_EXPAND = SC_AGG_BASE + [
    'avg_charge_wait', 'unique_sku_15m', 'loading_dock_util',
    'maintenance_schedule_score', 'manual_override_ratio',
]


def add_layout_cross_features(df):
    df = df.copy()
    def _safe(a, b): return safe_div(a, b)
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['lx_orders_per_pack_station'] = _safe(df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'robot_charging' in df.columns and 'robot_total' in df.columns:
        df['lx_charging_ratio_abs'] = _safe(df['robot_charging'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['lx_congestion_aisle_amp'] = df['sc_congestion_score_mean'] / (df['aisle_width_avg'] + 0.1)
    if 'floor_area_sqm' in df.columns and 'robot_total' in df.columns:
        df['lx_area_per_robot'] = _safe(df['floor_area_sqm'], df['robot_total'])
    if 'sc_avg_charge_wait_mean' in df.columns and 'charger_count' in df.columns:
        df['lx_wait_per_charger'] = _safe(df['sc_avg_charge_wait_mean'], df['charger_count'])
    if 'layout_compactness' in df.columns and 'sc_max_zone_density_mean' in df.columns:
        df['lx_compact_density'] = df['layout_compactness'] * df['sc_max_zone_density_mean']
    return df


# ── XGBoost MAE 파라미터 ──
XGB_PARAMS = {
    'n_estimators':     3000,
    'learning_rate':    0.02,
    'max_depth':        6,            # num_leaves=63 근사
    'min_child_weight': 26,           # min_child_samples 대응
    'subsample':        0.90,         # bagging_fraction
    'colsample_bytree': 0.50,         # feature_fraction
    'reg_alpha':        0.38,
    'reg_lambda':       0.36,
    'objective':        'reg:absoluteerror',   # L1 (MAE)
    'random_state':     RANDOM_STATE,
    'n_jobs':           -1,
    'verbosity':        0,
    'device':           'cpu',
}

# ── asym15: α=1.5 Asymmetric MAE ──
def asymmetric_mae_obj_15(y_pred, dtrain):
    """α=1.5 비대칭 MAE — 과소예측 1.5배 패널티"""
    y_true = dtrain.get_label()
    alpha  = 1.5
    residual = y_true - y_pred
    grad = np.where(residual > 0, -alpha, 1.0)
    hess = np.ones_like(grad)
    return grad, hess

def asymmetric_mae_metric_15(y_pred, dtrain):
    y_true = dtrain.get_label()
    alpha  = 1.5
    err    = y_true - y_pred
    loss   = np.where(err > 0, alpha * np.abs(err), np.abs(err)).mean()
    return 'asym15_mae', loss, False

ASYM15_PARAMS = {
    **{k: v for k, v in ASYM_LGBM_PARAMS.items()},  # asym20 기반
    # alpha는 objective 함수 내부에서 처리
}

# ── LGBM DART 파라미터 ──
DART_PARAMS = {
    'n_estimators':    1500,     # DART: early stopping 미지원 → 고정 iter
    'learning_rate':   0.04,     # LGBM 0.02보다 높게 (DART 수렴 보상)
    'num_leaves':      127,
    'feature_fraction': 0.51,
    'bagging_fraction': 0.90,
    'bagging_freq':    1,
    'min_child_samples': 26,
    'reg_alpha':       0.38,
    'reg_lambda':      0.36,
    'boosting_type':   'dart',
    'drop_rate':       0.10,     # 10% 트리 드롭
    'skip_drop':       0.50,     # 50% 확률로 드롭 건너뜀
    'objective':       'regression_l1',
    'random_state':    RANDOM_STATE,
    'verbosity':       -1,
    'n_jobs':          -1,
}


def train_xgb(X_tr, X_te, y_log, groups, feat_cols, ckpt_dir, name='xgb'):
    if ckpt_exists(ckpt_dir, name):
        print(f'  [{name.upper()}] 체크포인트 로드'); return load_ckpt(ckpt_dir, name)
    gkf  = GroupKFold(n_splits=N_SPLITS)
    oof  = np.zeros(len(X_tr)); preds = np.zeros(len(X_te))
    Xtr  = X_tr[feat_cols].fillna(0).values
    Xte  = X_te[feat_cols].fillna(0).values
    y_arr = y_log.values
    t0 = time.time()
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, y_arr, groups)):
        m = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
        m.fit(Xtr[ti], y_arr[ti],
              eval_set=[(Xtr[vi], y_arr[vi])],
              verbose=False)
        oof[vi] = m.predict(Xtr[vi])
        preds   += m.predict(Xte) / N_SPLITS
        mae = np.abs(np.expm1(oof[vi]) - np.expm1(y_arr[vi])).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration}  '
              f'({(time.time()-t0)/60:.1f}m)')
        del m; gc.collect()
    save_ckpt(ckpt_dir, name, oof, preds); return oof, preds


def train_asym15(X_tr, X_te, y_log, groups, feat_cols, ckpt_dir, name='asym15'):
    if ckpt_exists(ckpt_dir, name):
        print(f'  [{name.upper()}] 체크포인트 로드'); return load_ckpt(ckpt_dir, name)
    gkf  = GroupKFold(n_splits=N_SPLITS)
    oof  = np.zeros(len(X_tr)); preds = np.zeros(len(X_te))
    Xtr  = X_tr[feat_cols].fillna(0)
    Xte  = X_te[feat_cols].fillna(0)
    y_arr = y_log.values
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, y_arr, groups)):
        params = {k: v for k, v in ASYM15_PARAMS.items() if k != 'n_estimators'}
        params['objective'] = asymmetric_mae_obj_15
        bst = lgb.train(params, lgb.Dataset(Xtr.iloc[ti], label=y_arr[ti]),
                        num_boost_round=ASYM15_PARAMS['n_estimators'],
                        valid_sets=[lgb.Dataset(Xtr.iloc[vi], label=y_arr[vi])],
                        feval=asymmetric_mae_metric_15,
                        callbacks=[lgb.early_stopping(50, verbose=False),
                                   lgb.log_evaluation(0)])
        oof[vi] = bst.predict(Xtr.iloc[vi])
        preds   += bst.predict(Xte) / N_SPLITS
        mae = np.abs(np.expm1(oof[vi]) - np.expm1(y_arr[vi])).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  iter={bst.best_iteration}')
        del bst; gc.collect()
    save_ckpt(ckpt_dir, name, oof, preds); return oof, preds


def train_dart(X_tr, X_te, y_log, groups, feat_cols, ckpt_dir, name='dart'):
    if ckpt_exists(ckpt_dir, name):
        print(f'  [{name.upper()}] 체크포인트 로드'); return load_ckpt(ckpt_dir, name)
    gkf  = GroupKFold(n_splits=N_SPLITS)
    oof  = np.zeros(len(X_tr)); preds = np.zeros(len(X_te))
    Xtr  = X_tr[feat_cols].fillna(0)
    Xte  = X_te[feat_cols].fillna(0)
    y_arr = y_log.values
    print('  ⚠️ DART: early stopping 미지원 → 고정 1500 iter')
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, y_arr, groups)):
        m = lgb.LGBMRegressor(**DART_PARAMS)
        m.fit(Xtr.iloc[ti], y_arr[ti])   # eval_set 없이 고정 iter
        oof[vi] = m.predict(Xtr.iloc[vi])
        preds   += m.predict(Xte) / N_SPLITS
        mae = np.abs(np.expm1(oof[vi]) - np.expm1(y_arr[vi])).mean()
        print(f'    Fold {fold+1}  MAE={mae:.4f}  (fixed 1500 iter)')
        del m; gc.collect()
    save_ckpt(ckpt_dir, name, oof, preds); return oof, preds


def load_model47_ckpts():
    names = ['lgbm', 'cb', 'tw15', 'et', 'rf', 'asym20']
    oof_d, test_d = {}, {}
    for n in names:
        op = os.path.join(CKPT_47, f'{n}_oof.npy')
        tp = os.path.join(CKPT_47, f'{n}_test.npy')
        if os.path.exists(op):
            oof_d[n], test_d[n] = np.load(op), np.load(tp)
        else:
            print(f'  ⚠️ {n} 체크포인트 없음')
    return oof_d, test_d


def main():
    t0 = time.time()
    print('=' * 70)
    print('[model47 Addon] 추가 base learner: XGBoost / asym15 / LGBM-DART')
    print('  기준: model47(6)+q95 CV=8.4610 / Public=9.7901')
    print('=' * 70)

    os.makedirs(CKPT_47, exist_ok=True)

    # ── FE 재구성 (model47 동일) ──
    print('\n[데이터 로드 + FE (model47 동일)]')
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(train, test, layout,
                                 lag_lags=[1,2,3,4,5,6],
                                 rolling_windows=[3,5,10], verbose=True)
    for fn in [lambda df: add_scenario_agg(df, SC_AGG_EXPAND),
               add_ratio_tier1, add_ratio_tier2]:
        train = fn(train); test = fn(test)
    train = add_layout_cross_features(train)
    test  = add_layout_cross_features(test)

    feat_cols = get_feat_cols(train)
    y_raw  = train['avg_delay_minutes_next_30m']
    y_log  = np.log1p(y_raw)
    groups = train['scenario_id']
    print(f'  피처 수: {len(feat_cols)} (model47 기준 483)')

    # ── model47 기존 체크포인트 로드 ──
    print('\n[model47 기존 6모델 체크포인트 로드]')
    oof_47, test_47 = load_model47_ckpts()
    print(f'  로드 완료: {list(oof_47.keys())}')
    for n, oof in oof_47.items():
        mae = np.abs(np.expm1(oof) - y_raw.values).mean() if oof.max() < 20 \
              else np.abs(oof - y_raw.values).mean()
        print(f'  {n} OOF MAE={mae:.4f}')

    # ── q95 로드 ──
    q95_oof  = np.load(os.path.join(CKPT_Q, 'q95_oof.npy'))
    q95_test = np.load(os.path.join(CKPT_Q, 'q95_test.npy'))
    print(f'  q95 OOF MAE={np.abs(q95_oof - y_raw.values).mean():.4f}')

    # ── 신규 base learner 학습 ──
    print(f'\n{"="*60}\n▶ XGBoost MAE (reg:absoluteerror)')
    xgb_oof, xgb_test = train_xgb(train, test, y_log, groups, feat_cols, CKPT_47, 'xgb')
    xgb_mae = np.abs(np.expm1(xgb_oof) - y_raw.values).mean()
    xgb_lgbm_corr = np.corrcoef(xgb_oof, oof_47['lgbm'])[0, 1]
    xgb_q95_corr  = np.corrcoef(np.expm1(xgb_oof), q95_oof)[0, 1]
    print(f'  XGB OOF MAE={xgb_mae:.4f} | LGBM 상관={xgb_lgbm_corr:.4f} | q95 상관={xgb_q95_corr:.4f}')

    print(f'\n{"="*60}\n▶ Asymmetric MAE α=1.5 (asym15)')
    a15_oof, a15_test = train_asym15(train, test, y_log, groups, feat_cols, CKPT_47, 'asym15')
    a15_mae  = np.abs(np.expm1(a15_oof) - y_raw.values).mean()
    a15_lgbm_corr = np.corrcoef(a15_oof, oof_47['lgbm'])[0, 1]
    a15_a20_corr  = np.corrcoef(a15_oof, oof_47['asym20'])[0, 1]
    print(f'  asym15 OOF MAE={a15_mae:.4f} | LGBM 상관={a15_lgbm_corr:.4f} | asym20 상관={a15_a20_corr:.4f}')

    print(f'\n{"="*60}\n▶ LGBM DART (boosting_type=dart, 1500 iter)')
    dart_oof, dart_test = train_dart(train, test, y_log, groups, feat_cols, CKPT_47, 'dart')
    dart_mae  = np.abs(np.expm1(dart_oof) - y_raw.values).mean()
    dart_lgbm_corr = np.corrcoef(dart_oof, oof_47['lgbm'])[0, 1]
    print(f'  DART OOF MAE={dart_mae:.4f} | LGBM 상관={dart_lgbm_corr:.4f}')

    # ── 상관 요약 ──
    print(f'\n{"="*60}')
    print('신규 base learner 다양성 요약 (vs LGBM):')
    print(f'  XGB    상관={xgb_lgbm_corr:.4f}  MAE={xgb_mae:.4f}')
    print(f'  asym15 상관={a15_lgbm_corr:.4f}  MAE={a15_mae:.4f}  (asym20 상관={a15_a20_corr:.4f})')
    print(f'  DART   상관={dart_lgbm_corr:.4f}  MAE={dart_mae:.4f}')
    print('  기준(model47): model47(6)+q95 CV=8.4610')

    results = {}

    def run_config(label, extra):
        od = dict(oof_47); td = dict(test_47)
        od['q95']  = q95_oof; td['q95']  = q95_test
        for key, (o, t) in extra.items():
            od[key] = o; td[key] = t
        print(f'\n── {label} ({len(od)}모델) ──')
        cv, _, tm = run_meta(od, td, y_raw, groups, label=label)
        results[label] = (cv, tm)
        sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
        pred_col = [c for c in sub.columns if c != 'ID'][0]
        sub[pred_col] = np.clip(tm, 0, None)
        fname = f'model47_addon_{label.replace("+","_")}_cv{cv:.4f}.csv'
        sub.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f'  💾 {fname}')

    # 기준 재현 (model47+q95 7모델)
    run_config('q95', {})

    # +XGB
    run_config('q95+xgb', {'xgb': (xgb_oof, xgb_test)})

    # +asym15
    run_config('q95+asym15', {'asym15': (a15_oof, a15_test)})

    # +DART
    run_config('q95+dart', {'dart': (dart_oof, dart_test)})

    # 최우수 조합 (XGB + asym15 중 다양성 높은 쪽)
    run_config('q95+xgb+asym15', {'xgb': (xgb_oof, xgb_test),
                                   'asym15': (a15_oof, a15_test)})

    # ── 결과 요약 ──
    print(f'\n{"="*70}')
    print('결과 요약 (기준: model47+q95 CV=8.4610 / Public=9.7901)')
    print(f'{"="*70}')
    ref_cv = 8.4610
    for k, (cv, tm) in results.items():
        delta = cv - ref_cv
        mark  = '✅' if delta < 0 else ('≈' if abs(delta) < 0.0005 else '❌')
        print(f'  {k:22s}: CV={cv:.4f} (Δ{delta:+.4f}) {mark} | pred_std={tm.std():.2f}')

    elapsed = (time.time() - t0) / 60
    print(f'\n완료 ({elapsed:.1f}분)')


if __name__ == '__main__':
    main()
