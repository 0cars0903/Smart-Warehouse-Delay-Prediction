"""
model45c 확장 — 다중 Quantile 8모델 / 9모델 스태킹 + q95 블렌드
================================================================
배경:
  q85 (7모델): CV 8.4735 | Public 9.8048
  q95 (7모델): CV 8.4684 | Public 9.7931 ← 현재 최고

  q85와 q95를 동시에 메타에 투입하면 두 꼬리 분위수의 시너지 가능성이 있다.
  LGBM 상관: q85=0.7804, q95=0.7669 — 모두 충분히 이질적.
  q85-q95 상관은 높을 것이나(동일 피처셋) 예측 패턴이 다르므로 실험 가치 있음.

실험 목록:
  (A) 8모델: model34_6 + q95 + q85
  (B) 8모델: model34_6 + q95 + q90
  (C) 9모델: model34_6 + q85 + q90 + q95

추가: q95 기반 CSV 블렌드
  (D) q95 × blend_m34bd_b60 (w=0.3~0.7)

실행: python src/run_model45c_multi_q.py
예상 시간: ~8분 (메타 학습만)
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

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
DOCS_DIR = os.path.join(_BASE, '..', 'docs')
CKPT_C   = os.path.join(DOCS_DIR, 'model45_ckpt', 'strat_c')
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
        'lgbm': (m31, 'lgbm'), 'cb': (m31, 'cb'),
        'et':   (m31, 'et'),   'rf': (m31, 'rf'),
        'tw15': (m34, 'tw15'), 'asym': (m34, 'asym20'),
    }
    oof_d, test_d = {}, {}
    for key, (d, fname) in mapping.items():
        op = os.path.join(d, f'{fname}_oof.npy')
        tp = os.path.join(d, f'{fname}_test.npy')
        if os.path.exists(op) and os.path.exists(tp):
            oof_d[key], test_d[key] = np.load(op), np.load(tp)
    print(f"  model34 Config B: {list(oof_d.keys())}")
    return oof_d, test_d


def load_q(qname):
    op = os.path.join(CKPT_C, f'{qname}_oof.npy')
    tp = os.path.join(CKPT_C, f'{qname}_test.npy')
    if os.path.exists(op) and os.path.exists(tp):
        return np.load(op), np.load(tp)
    print(f"  ⚠️ {qname} 체크포인트 없음")
    return None, None


def train_meta(oof_dict, test_dict, y_tr, grp, label):
    names = list(oof_dict.keys())
    Xm_tr = np.column_stack([oof_dict[n] for n in names])
    Xm_te = np.column_stack([test_dict[n] for n in names])
    y_log = np.log1p(y_tr)
    oof_meta, test_preds = np.zeros(len(y_tr)), []
    kf = GroupKFold(n_splits=N_SPLITS)
    for fold, (ti, vi) in enumerate(kf.split(Xm_tr, y_log, grp)):
        m = lgb.LGBMRegressor(**META_PARAMS)
        m.fit(Xm_tr[ti], y_log[ti],
              eval_set=[(Xm_tr[vi], y_log[vi])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[vi] = np.expm1(m.predict(Xm_tr[vi]))
        test_preds.append(np.expm1(m.predict(Xm_te)))
    test_meta = np.mean(test_preds, axis=0)
    cv = np.abs(oof_meta - y_tr).mean()
    print(f"  [{label}] CV={cv:.4f} | pred_std={test_meta.std():.2f} | "
          f"test_mean={test_meta.mean():.2f}")
    for lo, hi in [(0,5),(5,20),(20,50),(50,80),(80,800)]:
        mask = (y_tr >= lo) & (y_tr < hi)
        if not mask.any(): continue
        mae = np.abs(oof_meta[mask] - y_tr[mask]).mean()
        pr  = oof_meta[mask].mean() / (y_tr[mask].mean() + 1e-8)
        print(f"    [{lo:3d},{hi:3d}) n={mask.sum():5d}  MAE={mae:.2f}  pred/actual={pr:.3f}")
    return cv, oof_meta, test_meta


def main():
    print("데이터 로드 중...")
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    train_fe, test_fe = build_features(train, test, layout, verbose=False)
    train_fe = add_scenario_agg(train_fe)
    test_fe  = add_scenario_agg(test_fe)
    train_fe = add_ratio_features(train_fe)
    test_fe  = add_ratio_features(test_fe)

    target_col = ('avg_delay_minutes_next_30m'
                  if 'avg_delay_minutes_next_30m' in train_fe.columns else 'target')
    y_tr = train_fe[target_col].values.astype(np.float32)
    grp  = train_fe['scenario_id'].values
    pred_col = [c for c in sample.columns if c != 'ID'][0]

    base_oof, base_test = load_model34_config_b()

    # quantile OOF 로드
    q_data = {}
    for qn in ['q85', 'q90', 'q95']:
        o, t = load_q(qn)
        if o is not None:
            q_data[qn] = (o, t)
            corr = np.corrcoef(o, base_oof['lgbm'])[0,1]
            print(f"  {qn}: LGBM-corr={corr:.4f}")

    # q85-q95 상관 확인
    if 'q85' in q_data and 'q95' in q_data:
        c = np.corrcoef(q_data['q85'][0], q_data['q95'][0])[0,1]
        print(f"  q85-q95 상관: {c:.4f}")
    if 'q90' in q_data and 'q95' in q_data:
        c = np.corrcoef(q_data['q90'][0], q_data['q95'][0])[0,1]
        print(f"  q90-q95 상관: {c:.4f}")

    print(f"\n{'='*60}")
    print(f"  다중 Quantile 스태킹 실험")
    print(f"  기준: q95(7모델) CV=8.4684 | Public=9.7931")
    print(f"{'='*60}")

    results = {}

    # ── (A) 8모델: 6 + q95 + q85 ──────────────────────────────────
    if 'q85' in q_data and 'q95' in q_data:
        print("\n  ── (A) 8모델: model34_6 + q95 + q85 ──")
        od = dict(base_oof); td = dict(base_test)
        od['q95'], td['q95'] = q_data['q95']
        od['q85'], td['q85'] = q_data['q85']
        cv, _, test_meta = train_meta(od, td, y_tr, grp, "A-8m(q95+q85)")
        sub = sample.copy()
        sub[pred_col] = np.clip(test_meta, 0, None)
        path = os.path.join(SUB_DIR, f'model45c_8m_q95q85_cv{cv:.4f}.csv')
        sub.to_csv(path, index=False)
        print(f"  저장: {os.path.basename(path)}")
        results['A'] = cv

    # ── (B) 8모델: 6 + q95 + q90 ──────────────────────────────────
    if 'q90' in q_data and 'q95' in q_data:
        print("\n  ── (B) 8모델: model34_6 + q95 + q90 ──")
        od = dict(base_oof); td = dict(base_test)
        od['q95'], td['q95'] = q_data['q95']
        od['q90'], td['q90'] = q_data['q90']
        cv, _, test_meta = train_meta(od, td, y_tr, grp, "B-8m(q95+q90)")
        sub = sample.copy()
        sub[pred_col] = np.clip(test_meta, 0, None)
        path = os.path.join(SUB_DIR, f'model45c_8m_q95q90_cv{cv:.4f}.csv')
        sub.to_csv(path, index=False)
        print(f"  저장: {os.path.basename(path)}")
        results['B'] = cv

    # ── (C) 9모델: 6 + q85 + q90 + q95 ───────────────────────────
    if all(k in q_data for k in ['q85', 'q90', 'q95']):
        print("\n  ── (C) 9모델: model34_6 + q85 + q90 + q95 ──")
        od = dict(base_oof); td = dict(base_test)
        for qn in ['q85', 'q90', 'q95']:
            od[qn], td[qn] = q_data[qn]
        cv, _, test_meta = train_meta(od, td, y_tr, grp, "C-9m(all-q)")
        sub = sample.copy()
        sub[pred_col] = np.clip(test_meta, 0, None)
        path = os.path.join(SUB_DIR, f'model45c_9m_allq_cv{cv:.4f}.csv')
        sub.to_csv(path, index=False)
        print(f"  저장: {os.path.basename(path)}")
        results['C'] = cv

    # ── (D) q95 × blend_m34bd_b60 블렌드 ─────────────────────────
    print("\n  ── (D) q95 × blend_m34bd_b60 블렌드 ──")
    file_q95  = os.path.join(SUB_DIR, 'model45c_q7_q95_cv8.4684.csv')
    file_m34bd = os.path.join(SUB_DIR, 'blend_m34bd_b60.csv')
    if os.path.exists(file_q95) and os.path.exists(file_m34bd):
        dfq = pd.read_csv(file_q95)
        dfm = pd.read_csv(file_m34bd)
        col = [c for c in dfq.columns if c != 'ID'][0]
        corr = np.corrcoef(dfq[col].values, dfm[col].values)[0,1]
        print(f"  q95-m34bd 상관: {corr:.4f}")
        if corr < 0.999:
            for wa in [0.6, 0.7, 0.8]:
                out = dfq.copy()
                out[col] = np.clip(wa*dfq[col].values + (1-wa)*dfm[col].values, 0, None)
                p = os.path.join(SUB_DIR, f'blend_q95_m34bd_a{int(wa*10)}b{int((1-wa)*10)}.csv')
                out.to_csv(p, index=False)
                print(f"  w_q95={wa:.1f}: std={out[col].std():.2f} → {os.path.basename(p)}")
        else:
            print(f"  상관 {corr:.4f} ≥ 0.999 → 블렌드 효과 없음, 파일 미생성")
    else:
        print("  파일 없음 — 스킵")

    # ── 요약 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  결과 요약")
    print(f"{'='*60}")
    print(f"  기준 q95(7모델): CV=8.4684 | Public=9.7931")
    labels = {'A': '8m(q95+q85)', 'B': '8m(q95+q90)', 'C': '9m(q85+q90+q95)'}
    for k, cv in sorted(results.items(), key=lambda x: x[1]):
        diff = cv - 8.4684
        mark = "✅" if diff < 0 else ("≈" if abs(diff) < 0.001 else "❌")
        print(f"  {mark} ({k}) {labels[k]}: CV={cv:.4f} (Δ{diff:+.4f})")


if __name__ == '__main__':
    main()
