"""
§3 Loss 함수 체계적 Ablation
================================================================
Notion 전략문서 "다시 EDA" §3 + §1 EDA 결과 반영:
  - §1 결론: 수치 피처가 tail driver, 모델은 driver를 이미 안다
  - 문제는 MAE(조건부 중앙값)가 예측값을 억제하는 것
  - 해결: loss 감도를 체계적으로 비교하여 최적 조합 도출

실험 설계:
  1) 타겟 변환: raw / log1p / sqrt / Box-Cox / Yeo-Johnson
  2) Loss: MAE / MSE / Huber / Tweedie(1.5~2.0) / Asymmetric(α=1.0~3.0)
  3) 모든 조합 동일 HP로 LGBM 단독 OOF → 구간별 MAE 기여도 매트릭스
  4) model31 피처(429종) 사용, GroupKFold 5-fold

핵심 출력:
  - 구간별 MAE 기여도 매트릭스 (loss × 구간)
  - 최적 loss 후보 → model34에 투입할 base learner 조합 결정

실행: python src/eda_loss_ablation.py
예상 시간: ~20분 (15개 LGBM × 5fold)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scipy.special import boxcox1p, inv_boxcox1p
from scipy.stats import boxcox_normmax
import warnings, os, sys, time, gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
DOCS_DIR = os.path.join(_BASE, '..', 'docs')
N_SPLITS = 5
RANDOM_STATE = 42

# ── 구간 정의 ──
BINS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 800)]

# ── 공통 LGBM 파라미터 (model31 기준, loss만 변경) ──
BASE_PARAMS = {
    'num_leaves': 129,
    'learning_rate': 0.01021,
    'feature_fraction': 0.465,
    'bagging_fraction': 0.947,
    'min_child_samples': 30,
    'reg_alpha': 1.468,
    'reg_lambda': 0.396,
    'n_estimators': 3000,
    'bagging_freq': 1,
    'random_state': RANDOM_STATE,
    'verbosity': -1,
    'n_jobs': -1,
}

# ── Scenario aggregation (model31 동일) ──
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

def add_scenario_agg_features(df):
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns: continue
        grp = df.groupby('scenario_id')[col]
        for stat, fn in [('mean', 'mean'), ('std', 'std'), ('max', 'max'),
                         ('min', 'min'), ('median', 'median')]:
            df[f'sc_{col}_{stat}'] = grp.transform(fn)
            if stat == 'std':
                df[f'sc_{col}_{stat}'] = df[f'sc_{col}_{stat}'].fillna(0)
        df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']
        df[f'sc_{col}_p10'] = grp.transform(lambda x: x.quantile(0.10))
        df[f'sc_{col}_p90'] = grp.transform(lambda x: x.quantile(0.90))
        df[f'sc_{col}_skew'] = grp.transform(lambda x: x.skew()).fillna(0)
        df[f'sc_{col}_kurtosis'] = grp.transform(lambda x: x.kurtosis()).fillna(0)
        df[f'sc_{col}_cv'] = (df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)
    return df

def safe_div(a, b, fill=0):
    return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

def add_ratio_features(df):
    """Tier 1+2+3 비율 피처 (model31 동일)"""
    # Tier 1
    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = safe_div(df['sc_order_inflow_15m_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = safe_div(df['sc_congestion_score_mean'], df['intersection_count'])
    if 'sc_low_battery_ratio_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_stress'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df.get('sc_charge_queue_length_mean', 0), df['charger_count'])
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = safe_div(df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']
    # Tier 2
    if all(c in df.columns for c in ['sc_congestion_score_mean', 'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'], df['robot_total'] ** 2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm'] / 100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(df['pack_station_count'], df['floor_area_sqm'] / 1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if 'sc_battery_mean_mean' in df.columns and 'sc_robot_utilization_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_per_robot'] = safe_div(
            df['sc_battery_mean_mean'] * df['sc_robot_utilization_mean'], df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    # Tier 3 selected
    cols = ['sc_order_inflow_15m_mean', 'sc_congestion_score_mean', 'sc_low_battery_ratio_mean', 'robot_total', 'charger_count']
    if all(c in df.columns for c in cols):
        df['ratio_total_stress'] = safe_div(
            df['sc_order_inflow_15m_mean'] * df['sc_congestion_score_mean'] *
            (df['sc_low_battery_ratio_mean'] + 0.01), df['robot_total'] * df['charger_count'])
    cols2 = ['sc_sku_concentration_mean', 'sc_congestion_score_mean', 'intersection_count']
    if all(c in df.columns for c in cols2):
        df['ratio_sku_congestion'] = safe_div(
            df['sc_sku_concentration_mean'] * df['sc_congestion_score_mean'], df['intersection_count'])
    cols3 = ['sc_robot_idle_mean', 'robot_total', 'sc_order_inflow_15m_mean', 'floor_area_sqm']
    if all(c in df.columns for c in cols3):
        idle_frac = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
        df['ratio_no_idle_demand'] = safe_div(
            (1 - idle_frac) * df['sc_order_inflow_15m_mean'], df['floor_area_sqm'] / 100)
    cols4 = ['sc_low_battery_ratio_mean', 'sc_charge_queue_length_mean', 'charger_count']
    if all(c in df.columns for c in cols4):
        df['ratio_battery_crisis'] = safe_div(
            df['sc_low_battery_ratio_mean'] * df['sc_charge_queue_length_mean'], df['charger_count'])
    # Cross selected
    safe_pairs = [('congestion_score', 'low_battery_ratio'), ('sku_concentration', 'max_zone_density'),
                  ('robot_utilization', 'charge_queue_length')]
    for col_a, col_b in safe_pairs:
        if col_a not in df.columns or col_b not in df.columns: continue
        interaction = df[col_a] * df[col_b]
        grp = interaction.groupby(df['scenario_id'])
        df[f'sc_cross_{col_a[:6]}_{col_b[:6]}_mean'] = grp.transform('mean')
    return df


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    train, test = build_features(train, test, layout, lag_lags=[1,2,3,4,5,6], rolling_windows=[3,5,10])
    for fn in [add_scenario_agg_features, add_ratio_features]:
        train = fn(train); test = fn(test)
    return train, test


def get_feat_cols(df):
    return [c for c in df.columns
            if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
            and df[c].dtype != object]


# ─────────────────────────────────────────────
# Custom Loss Functions
# ─────────────────────────────────────────────
def make_asymmetric_objective(alpha):
    """Asymmetric MAE: under-prediction에 α배 페널티"""
    def objective(y_pred, dtrain):
        y_true = dtrain.get_label()
        residual = y_true - y_pred
        grad = np.where(residual > 0, -alpha, 1.0)
        hess = np.ones_like(y_pred)
        return grad, hess
    return objective

def make_asymmetric_metric(name='asym_mae'):
    def metric(y_pred, dtrain):
        y_true = dtrain.get_label()
        mae = np.abs(y_pred - y_true).mean()
        return name, mae, False
    return metric

def huber_objective(y_pred, dtrain):
    """Huber loss with delta=1.0 (log1p space)"""
    y_true = dtrain.get_label()
    residual = y_true - y_pred
    delta = 1.0
    mask = np.abs(residual) <= delta
    grad = np.where(mask, -residual, -delta * np.sign(residual))
    hess = np.where(mask, 1.0, 0.01)  # small hessian outside delta
    return grad, hess

def huber_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    return 'huber_mae', np.abs(y_pred - y_true).mean(), False

def make_expectile_objective(tau):
    def objective(y_pred, dtrain):
        y_true = dtrain.get_label()
        residual = y_true - y_pred
        weight = np.where(residual >= 0, tau, 1 - tau)
        grad = -2 * weight * residual
        hess = 2 * weight
        return grad, hess
    return objective


# ─────────────────────────────────────────────
# OOF 학습 엔진
# ─────────────────────────────────────────────
def run_oof(X_train, y_target, groups, feat_cols, params, y_raw,
            transform='log1p', custom_obj=None, custom_metric=None, label=''):
    """
    단일 LGBM OOF 학습.
    transform: 'raw', 'log1p', 'sqrt', 'boxcox', 'yeojohnson'
    custom_obj: None이면 params['objective'] 사용, 있으면 custom
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X_train))
    X = X_train[feat_cols].fillna(0)

    # 타겟 변환
    if transform == 'raw':
        y = y_raw.copy()
    elif transform == 'log1p':
        y = np.log1p(y_raw)
    elif transform == 'sqrt':
        y = np.sqrt(y_raw)
    elif transform == 'boxcox':
        lam = boxcox_normmax(y_raw.clip(lower=0.001) + 1)
        y = boxcox1p(y_raw.clip(lower=0), lam)
    elif transform == 'yeojohnson':
        from sklearn.preprocessing import PowerTransformer
        pt = PowerTransformer(method='yeo-johnson')
        y = pd.Series(pt.fit_transform(y_raw.values.reshape(-1, 1)).ravel(), index=y_raw.index)
        lam = pt.lambdas_[0]
    else:
        raise ValueError(f'Unknown transform: {transform}')

    iters = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        if custom_obj is not None:
            # lgb.train API (custom objective)
            dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx].values if hasattr(y, 'iloc') else y[tr_idx])
            dval = lgb.Dataset(X.iloc[va_idx],
                               label=y.iloc[va_idx].values if hasattr(y, 'iloc') else y[va_idx],
                               reference=dtrain)

            p = {k: v for k, v in params.items() if k not in ['n_estimators', 'objective']}
            p['objective'] = custom_obj

            bst = lgb.train(
                p, dtrain,
                num_boost_round=params.get('n_estimators', 3000),
                valid_sets=[dval],
                feval=custom_metric,
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            )
            preds = bst.predict(X.iloc[va_idx])
            iters.append(bst.best_iteration)
            del bst
        else:
            # LGBMRegressor API (built-in objective)
            m = lgb.LGBMRegressor(**params)
            y_tr = y.iloc[tr_idx].values if hasattr(y, 'iloc') else y[tr_idx]
            y_va = y.iloc[va_idx].values if hasattr(y, 'iloc') else y[va_idx]
            m.fit(X.iloc[tr_idx], y_tr,
                  eval_set=[(X.iloc[va_idx], y_va)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            preds = m.predict(X.iloc[va_idx])
            iters.append(m.best_iteration_)
            del m

        # 역변환 → raw 공간으로
        if transform == 'raw':
            oof[va_idx] = preds
        elif transform == 'log1p':
            oof[va_idx] = np.expm1(preds)
        elif transform == 'sqrt':
            oof[va_idx] = np.square(preds)
        elif transform == 'boxcox':
            oof[va_idx] = inv_boxcox1p(preds, lam)
        elif transform == 'yeojohnson':
            oof[va_idx] = pt.inverse_transform(preds.reshape(-1, 1)).ravel()

        gc.collect()

    # raw 공간 MAE
    oof = np.maximum(oof, 0)  # 음수 방지
    overall_mae = np.abs(oof - y_raw.values).mean()
    avg_iter = np.mean(iters)
    print(f'  [{label:30s}] MAE={overall_mae:.4f}, avg_iter={avg_iter:.0f}')

    return oof, overall_mae, avg_iter


def segment_analysis(oof, y_raw, label=''):
    """구간별 MAE 기여도 분석"""
    results = {}
    total_mae = np.abs(oof - y_raw.values).mean()
    for lo, hi in BINS:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() == 0:
            continue
        seg_mae = np.abs(oof[mask] - y_raw.values[mask]).mean()
        n = mask.sum()
        pct = n / len(y_raw) * 100
        contribution = seg_mae * n / len(y_raw)
        pred_actual = oof[mask].mean() / (y_raw.values[mask].mean() + 1e-8)
        results[f'[{lo},{hi})'] = {
            'n': n, 'pct': pct, 'mae': seg_mae,
            'contribution': contribution, 'pred_actual': pred_actual,
        }
    results['total'] = {'mae': total_mae, 'pred_std': oof.std()}
    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 70)
    print('§3 Loss 함수 체계적 Ablation')
    print('=' * 70)

    train, test = load_data()
    feat_cols = get_feat_cols(train)
    y_raw = train['avg_delay_minutes_next_30m']
    groups = train['scenario_id']
    print(f'  피처: {len(feat_cols)}')

    # ── 실험 목록 ──
    experiments = []

    # (A) 타겟 변환 × MAE
    for tf in ['log1p', 'sqrt', 'raw']:
        obj = 'regression_l1' if tf != 'raw' else 'regression_l1'
        p = {**BASE_PARAMS, 'objective': obj}
        experiments.append({
            'label': f'MAE+{tf}',
            'params': p,
            'transform': tf,
            'custom_obj': None,
            'custom_metric': None,
        })

    # (B) MSE (log1p)
    p_mse = {**BASE_PARAMS, 'objective': 'regression_l2'}
    experiments.append({
        'label': 'MSE+log1p',
        'params': p_mse,
        'transform': 'log1p',
        'custom_obj': None,
        'custom_metric': None,
    })

    # (C) Huber (log1p)
    experiments.append({
        'label': 'Huber+log1p',
        'params': BASE_PARAMS,
        'transform': 'log1p',
        'custom_obj': huber_objective,
        'custom_metric': huber_metric,
    })

    # (D) Asymmetric MAE α 탐색 (log1p)
    for alpha in [1.2, 1.5, 2.0, 2.5, 3.0]:
        experiments.append({
            'label': f'Asym(α={alpha})+log1p',
            'params': BASE_PARAMS,
            'transform': 'log1p',
            'custom_obj': make_asymmetric_objective(alpha),
            'custom_metric': make_asymmetric_metric(f'asym{alpha}_mae'),
        })

    # (E) Expectile τ 탐색 (log1p)
    for tau in [0.6, 0.7, 0.8]:
        experiments.append({
            'label': f'Expectile(τ={tau})+log1p',
            'params': BASE_PARAMS,
            'transform': 'log1p',
            'custom_obj': make_expectile_objective(tau),
            'custom_metric': make_asymmetric_metric(f'exp{tau}_mae'),
        })

    # (F) Tweedie (raw space, CatBoost가 아닌 LGBM)
    for p_val in [1.5, 1.8, 2.0]:
        p_tw = {**BASE_PARAMS, 'objective': 'tweedie', 'tweedie_variance_power': p_val}
        experiments.append({
            'label': f'Tweedie(p={p_val})+raw',
            'params': p_tw,
            'transform': 'raw',
            'custom_obj': None,
            'custom_metric': None,
        })

    # (G) Gamma (raw space)
    p_gamma = {**BASE_PARAMS, 'objective': 'gamma'}
    experiments.append({
        'label': 'Gamma+raw',
        'params': p_gamma,
        'transform': 'raw',
        'custom_obj': None,
        'custom_metric': None,
    })

    print(f'\n총 {len(experiments)}개 실험\n')

    # ── 실행 ──
    all_results = {}
    all_oof = {}

    for i, exp in enumerate(experiments):
        print(f'[{i+1}/{len(experiments)}]', end='')
        try:
            oof, mae, avg_iter = run_oof(
                train, None, groups, feat_cols,
                exp['params'], y_raw,
                transform=exp['transform'],
                custom_obj=exp['custom_obj'],
                custom_metric=exp['custom_metric'],
                label=exp['label'],
            )
            seg = segment_analysis(oof, y_raw, exp['label'])
            all_results[exp['label']] = seg
            all_oof[exp['label']] = oof
        except Exception as e:
            print(f'  [{exp["label"]:30s}] ERROR: {e}')
            all_results[exp['label']] = {'error': str(e)}

    # ── 결과 매트릭스 생성 ──
    print('\n' + '=' * 70)
    print('구간별 MAE 기여도 매트릭스')
    print('=' * 70)

    # 헤더
    seg_names = [f'[{lo},{hi})' for lo, hi in BINS]
    header = f'{"Loss":<30s} {"전체MAE":>8s}'
    for s in seg_names:
        header += f' {s:>10s}'
    header += f' {"pred_std":>9s}'
    print(header)
    print('-' * (30 + 8 + 10 * len(seg_names) + 9 + len(seg_names) + 2))

    report_lines = [header, '-' * 130]

    # 데이터
    sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('total', {}).get('mae', 999))
    for label, seg in sorted_results:
        if 'error' in seg:
            line = f'{label:<30s} ERROR: {seg["error"]}'
        else:
            total = seg['total']['mae']
            pred_std = seg['total'].get('pred_std', 0)
            line = f'{label:<30s} {total:8.4f}'
            for s in seg_names:
                if s in seg:
                    line += f' {seg[s]["mae"]:10.2f}'
                else:
                    line += f' {"N/A":>10s}'
            line += f' {pred_std:9.2f}'
        print(line)
        report_lines.append(line)

    # ── 극값 구간 특화 분석 ──
    print('\n' + '=' * 70)
    print('[80,800) 구간 pred/actual 비율 (높을수록 극값 예측 ↑)')
    print('=' * 70)

    pa_lines = []
    header2 = f'{"Loss":<30s} {"[80+]MAE":>9s} {"pred/act":>9s} {"[50-80]MAE":>10s} {"[0-5]MAE":>9s}'
    print(header2)
    pa_lines.append(header2)

    for label, seg in sorted_results:
        if 'error' in seg:
            continue
        ext = seg.get('[80,800)', {})
        high = seg.get('[50,80)', {})
        low = seg.get('[0,5)', {})
        line = f'{label:<30s} {ext.get("mae", 0):9.2f} {ext.get("pred_actual", 0):9.3f} '
        line += f'{high.get("mae", 0):10.2f} {low.get("mae", 0):9.2f}'
        print(line)
        pa_lines.append(line)

    # ── 상관관계 매트릭스 (다양성 분석) ──
    print('\n' + '=' * 70)
    print('OOF 상관관계 (주요 실험만)')
    print('=' * 70)

    key_labels = [l for l in all_oof.keys() if 'MAE+log1p' in l or 'Asym' in l or 'MSE' in l
                  or 'Tweedie' in l or 'Expectile' in l or 'Huber' in l]
    key_labels = key_labels[:10]  # 최대 10개

    for i in range(len(key_labels)):
        for j in range(i + 1, len(key_labels)):
            c = np.corrcoef(all_oof[key_labels[i]], all_oof[key_labels[j]])[0, 1]
            marker = '✅' if c < 0.95 else ('⚠️' if c < 0.98 else '❌')
            print(f'  {key_labels[i]:25s} - {key_labels[j]:25s}: {c:.4f} {marker}')

    # ── 보고서 저장 ──
    elapsed = (time.time() - t0) / 60
    report_path = os.path.join(DOCS_DIR, 'eda_loss_ablation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('§3 Loss Ablation Report\n')
        f.write(f'생성: {time.strftime("%Y-%m-%d %H:%M")}\n')
        f.write(f'총 소요: {elapsed:.1f}분\n\n')
        f.write('[구간별 MAE 매트릭스]\n')
        f.write('\n'.join(report_lines))
        f.write('\n\n[극값 pred/actual]\n')
        f.write('\n'.join(pa_lines))

    # ── 최종 판정 ──
    print(f'\n{"=" * 70}')
    print(f'최종 판정 (총 {elapsed:.1f}분 소요)')
    print(f'{"=" * 70}')

    if sorted_results:
        best_label, best_seg = sorted_results[0]
        baseline = all_results.get('MAE+log1p', {}).get('total', {}).get('mae', 999)
        best_mae = best_seg.get('total', {}).get('mae', 999)
        print(f'  기준 (MAE+log1p): {baseline:.4f}')
        print(f'  최적: {best_label} = {best_mae:.4f} (Δ={best_mae - baseline:+.4f})')

        # 극값 최적
        ext_best = min(
            [(l, s.get('[80,800)', {}).get('mae', 999)) for l, s in sorted_results if 'error' not in s],
            key=lambda x: x[1]
        )
        print(f'  극값[80+] 최적: {ext_best[0]} = {ext_best[1]:.2f}')

        # 추천 조합
        print(f'\n  → model34 base learner 후보:')
        print(f'    1. {best_label} (전체 MAE 최적)')
        print(f'    2. {ext_best[0]} (극값 최적)')
        print(f'    3. 기존 model31 5모델 유지 + 위 후보 추가 스태킹')

    print(f'\n  보고서: {report_path}')


if __name__ == '__main__':
    main()
