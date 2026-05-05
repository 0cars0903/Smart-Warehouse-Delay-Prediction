"""
model45c 확장 — q90 / q95 7모델 스태킹 실험
================================================================
배경:
  model45c_q7_q85 (q=0.85 사용) → Public 9.8048 신기록 달성
  q85 선택 기준: OOF MAE 최소 → 11.61로 q85가 최적

  그러나 LGBM 상관은 q90(0.7644), q95(0.7669)도 q85(0.7804)보다 낮아
  다양성 측면에서 더 유리할 수 있음.

  이 스크립트는 q90, q95를 각각 사용한 7모델 스태킹을 실험한다.
  quantile 체크포인트는 strat_c에서 이미 존재하므로 즉시 실행 가능.

실행: python src/run_model45c_q_stack.py
예상 시간: ~5분 (메타 학습만, quantile 재학습 없음)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import os, sys, warnings

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_BASE, '..', 'data')
SUB_DIR   = os.path.join(_BASE, '..', 'submissions')
DOCS_DIR  = os.path.join(_BASE, '..', 'docs')
CKPT_C    = os.path.join(DOCS_DIR, 'model45_ckpt', 'strat_c')
RANDOM_STATE = 42
N_SPLITS     = 5

META_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]


def add_scenario_agg(df):
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns: continue
        grp = df.groupby('scenario_id')[col]
        df[f'sc_{col}_mean']     = grp.transform('mean')
        df[f'sc_{col}_std']      = grp.transform('std').fillna(0)
        df[f'sc_{col}_max']      = grp.transform('max')
        df[f'sc_{col}_min']      = grp.transform('min')
        df[f'sc_{col}_diff']     = df[col] - df[f'sc_{col}_mean']
        df[f'sc_{col}_median']   = grp.transform('median')
        df[f'sc_{col}_p10']      = grp.transform(lambda x: x.quantile(0.10))
        df[f'sc_{col}_p90']      = grp.transform(lambda x: x.quantile(0.90))
        df[f'sc_{col}_skew']     = grp.transform(lambda x: x.skew()).fillna(0)
        df[f'sc_{col}_kurtosis'] = grp.transform(lambda x: x.kurtosis()).fillna(0)
        df[f'sc_{col}_cv']       = (df[f'sc_{col}_std'] /
                                    (df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)
    return df


def safe_div(a, b, fill=0):
    return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)


def add_ratio_features(df):
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
    if all(c in df.columns for c in ['sc_congestion_score_mean', 'sc_order_inflow_15m_mean', 'robot_total']):
        df['ratio_cross_stress'] = safe_div(
            df['sc_congestion_score_mean'] * df['sc_order_inflow_15m_mean'], df['robot_total'] ** 2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm'] / 100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(df['pack_station_count'], df['floor_area_sqm'] / 1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns:
        df['ss_congestion_x_battery'] = (
            df.get('sc_congestion_score_mean', 0) * df.get('sc_low_battery_ratio_mean', 0))
        df['ss_order_x_util'] = (
            df.get('sc_order_inflow_15m_mean', 0) * df.get('sc_robot_utilization_mean', 0))
        df['ss_demand_x_congestion'] = (
            df.get('ratio_demand_per_robot', 0) * df.get('ratio_congestion_per_intersection', 0))
        df['ss_stress_x_pressure'] = (
            df.get('ratio_battery_stress', 0) * df.get('ratio_packing_pressure', 0))
        df['ss_idle_x_demand'] = (
            df.get('ratio_idle_fraction', 0) * df.get('ratio_demand_per_robot', 0))
    return df


def load_model34_config_b():
    m31 = os.path.join(DOCS_DIR, 'model31_ckpt')
    m34 = os.path.join(DOCS_DIR, 'model34_ckpt')
    mapping = {
        'lgbm': (m31, 'lgbm'),
        'cb':   (m31, 'cb'),
        'et':   (m31, 'et'),
        'rf':   (m31, 'rf'),
        'tw15': (m34, 'tw15'),
        'asym': (m34, 'asym20'),
    }
    oof_dict, test_dict = {}, {}
    for key, (d, fname) in mapping.items():
        oof_p  = os.path.join(d, f'{fname}_oof.npy')
        test_p = os.path.join(d, f'{fname}_test.npy')
        if os.path.exists(oof_p) and os.path.exists(test_p):
            oof_dict[key]  = np.load(oof_p)
            test_dict[key] = np.load(test_p)
        else:
            print(f"  ⚠️  없음: {fname} in {os.path.basename(d)}")
    print(f"  model34 Config B 로드: {list(oof_dict.keys())}")
    return oof_dict, test_dict


def train_meta(oof_dict, test_dict, y_tr, grp, label=""):
    names = list(oof_dict.keys())
    X_meta_tr = np.column_stack([oof_dict[n] for n in names])
    X_meta_te = np.column_stack([test_dict[n] for n in names])
    y_log = np.log1p(y_tr)
    oof_meta   = np.zeros(len(y_tr))
    test_preds = []
    kf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_meta_tr, y_log, grp)):
        m = lgb.LGBMRegressor(**META_PARAMS)
        m.fit(X_meta_tr[tr_idx], y_log[tr_idx],
              eval_set=[(X_meta_tr[va_idx], y_log[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        oof_meta[va_idx] = np.expm1(m.predict(X_meta_tr[va_idx]))
        test_preds.append(np.expm1(m.predict(X_meta_te)))
    test_meta = np.mean(test_preds, axis=0)
    cv_mae = np.abs(oof_meta - y_tr).mean()
    print(f"  [{label}] CV MAE = {cv_mae:.4f} | pred_std = {test_meta.std():.2f} | "
          f"test_mean = {test_meta.mean():.2f}")
    # 구간별 MAE
    for lo, hi in [(0,5),(5,20),(20,50),(50,80),(80,800)]:
        mask = (y_tr >= lo) & (y_tr < hi)
        if mask.sum() == 0: continue
        mae = np.abs(oof_meta[mask] - y_tr[mask]).mean()
        pr  = oof_meta[mask].mean() / (y_tr[mask].mean() + 1e-8)
        print(f"    [{lo:3d},{hi:3d}) n={mask.sum():5d}  MAE={mae:.2f}  pred/actual={pr:.3f}")
    return cv_mae, oof_meta, test_meta


def main():
    print("데이터 로드 중...")
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    train_fe, test_fe = build_features(train, test, layout, verbose=True)
    train_fe = add_scenario_agg(train_fe)
    test_fe  = add_scenario_agg(test_fe)
    train_fe = add_ratio_features(train_fe)
    test_fe  = add_ratio_features(test_fe)

    drop_cols = {'id', 'ID', 'target', 'scenario_id', 'timestamp',
                 'layout_id', 'avg_delay_minutes_next_30m'}
    target_col = ('avg_delay_minutes_next_30m'
                  if 'avg_delay_minutes_next_30m' in train_fe.columns else 'target')
    y_tr = train_fe[target_col].values.astype(np.float32)
    grp  = train_fe['scenario_id'].values

    # model34 Config B base learners 로드
    base_oof, base_test = load_model34_config_b()

    # strat_c quantile 체크포인트 로드
    quantile_loaded = {}
    for qname in ['q85', 'q90', 'q95']:
        oof_p  = os.path.join(CKPT_C, f'{qname}_oof.npy')
        test_p = os.path.join(CKPT_C, f'{qname}_test.npy')
        if os.path.exists(oof_p) and os.path.exists(test_p):
            quantile_loaded[qname] = {
                'oof':  np.load(oof_p),
                'test': np.load(test_p),
            }
            mae = np.abs(quantile_loaded[qname]['oof'] - y_tr).mean()
            tail = quantile_loaded[qname]['oof'][y_tr >= 80]
            pr80 = tail.mean() / (y_tr[y_tr >= 80].mean() + 1e-8)
            lgbm_corr = np.corrcoef(quantile_loaded[qname]['oof'], base_oof['lgbm'])[0, 1]
            print(f"  {qname}: OOF={mae:.4f} | [80+] pred/actual={pr80:.3f} | LGBM-corr={lgbm_corr:.4f}")
        else:
            print(f"  ⚠️  {qname} 체크포인트 없음 (먼저 --strategy C 실행 필요)")

    if not quantile_loaded:
        print("\n체크포인트가 없습니다. 먼저 아래 명령을 실행하세요:")
        print("  python src/run_model45_lds_extreme.py --strategy C")
        return

    print(f"\n{'='*60}")
    print(f"  q90 / q95 7모델 스태킹 실험")
    print(f"  기준: model45c_q85 → CV 8.4735 | Public 9.8048")
    print(f"{'='*60}")

    results = {}
    pred_col = [c for c in sample.columns if c != 'ID'][0]

    for qname in sorted(quantile_loaded.keys()):
        print(f"\n  ── {qname} 7모델 스태킹 ──")
        oof_dict  = dict(base_oof)
        test_dict = dict(base_test)
        oof_dict[qname]  = quantile_loaded[qname]['oof']
        test_dict[qname] = quantile_loaded[qname]['test']

        cv_mae, oof_meta, test_meta = train_meta(
            oof_dict, test_dict, y_tr, grp, label=f"C-Q7({qname})")

        sub = sample.copy()
        sub[pred_col] = np.clip(test_meta, 0, None)
        sub_path = os.path.join(SUB_DIR, f'model45c_q7_{qname}_cv{cv_mae:.4f}.csv')
        sub.to_csv(sub_path, index=False)
        print(f"  저장: {os.path.basename(sub_path)}")
        results[qname] = cv_mae

    # 요약
    print(f"\n{'='*60}")
    print(f"  결과 요약")
    print(f"{'='*60}")
    print(f"  기준 model45c_q85: CV=8.4735 | Public=9.8048")
    for qname, cv in sorted(results.items(), key=lambda x: x[1]):
        diff = cv - 8.4735
        mark = "✅" if diff < 0 else ("≈" if abs(diff) < 0.001 else "❌")
        print(f"  {mark} {qname}: CV={cv:.4f} (Δ{diff:+.4f})")
    print(f"\n  제출 파일: submissions/model45c_q7_q9[0/5]_*.csv")


if __name__ == '__main__':
    main()
