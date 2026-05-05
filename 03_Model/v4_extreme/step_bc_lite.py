"""
BC 후처리 경량 버전: 체크포인트 OOF/test 로드 → 메타 스태킹 → 분류기 → 2D 보정
중간 피클 캐시 없이 한 번에 실행. 디스크 최소 사용.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
import warnings, gc, os, sys, time

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model30_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42
EXTREME_THRESHOLD = 40

META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': RANDOM_STATE,
    'verbosity': -1, 'n_jobs': -1,
}

CLF_PARAMS = {
    'objective': 'binary', 'metric': 'auc',
    'num_leaves': 63, 'learning_rate': 0.03,
    'feature_fraction': 0.7, 'bagging_fraction': 0.8,
    'min_child_samples': 30, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
    'n_estimators': 1000, 'bagging_freq': 1,
    'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
}

SC_AGG_COLS = [
    'robot_utilization','order_inflow_15m','low_battery_ratio','congestion_score',
    'max_zone_density','charge_queue_length','battery_mean','battery_std',
    'robot_idle','robot_active','robot_charging','near_collision_15m',
    'fault_count_15m','avg_recovery_time','blocked_path_15m','sku_concentration',
    'urgent_order_ratio','pack_utilization',
]


def add_scenario_agg_fast(df):
    """시나리오 집계: 빠른 built-in만 사용, lambda 최소화"""
    new_cols = {}
    for col in SC_AGG_COLS:
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]
        new_cols[f'sc_{col}_mean'] = grp.transform('mean')
        new_cols[f'sc_{col}_std']  = grp.transform('std').fillna(0)
        new_cols[f'sc_{col}_max']  = grp.transform('max')
        new_cols[f'sc_{col}_min']  = grp.transform('min')
        new_cols[f'sc_{col}_median'] = grp.transform('median')
        # agg → map
        sc_agg = grp.agg(['skew'])
        sc_agg.columns = ['skew']
        sc_agg['p10'] = grp.quantile(0.10)
        sc_agg['p90'] = grp.quantile(0.90)
        sc_agg['kurtosis'] = grp.apply(lambda x: x.kurtosis())
        sc_agg = sc_agg.fillna(0)
        sid = df['scenario_id']
        new_cols[f'sc_{col}_p10'] = sid.map(sc_agg['p10'])
        new_cols[f'sc_{col}_p90'] = sid.map(sc_agg['p90'])
        new_cols[f'sc_{col}_skew'] = sid.map(sc_agg['skew']).fillna(0)
        new_cols[f'sc_{col}_kurtosis'] = sid.map(sc_agg['kurtosis']).fillna(0)
    # bulk assign
    new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    # diff and cv
    for col in SC_AGG_COLS:
        if col not in df.columns:
            continue
        new_df[f'sc_{col}_diff'] = new_df[col] - new_df[f'sc_{col}_mean']
        new_df[f'sc_{col}_cv'] = (new_df[f'sc_{col}_std'] /
                                   (new_df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)
    return new_df


def add_ratio_features(df):
    def safe_div(a, b, fill=0):
        return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)
    # Tier 1
    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = safe_div(df['sc_order_inflow_15m_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = safe_div(df['sc_congestion_score_mean'], df['intersection_count'])
    if 'sc_low_battery_ratio_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_stress'] = safe_div(df['sc_low_battery_ratio_mean'] * df.get('sc_charge_queue_length_mean', 0), df['charger_count'])
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = safe_div(df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']
    # Tier 2
    if all(c in df.columns for c in ['sc_congestion_score_mean','sc_order_inflow_15m_mean','robot_total']):
        df['ratio_cross_stress'] = safe_div(df['sc_congestion_score_mean']*df['sc_order_inflow_15m_mean'], df['robot_total']**2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = safe_div(df['robot_total'], df['floor_area_sqm']/100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = safe_div(df['pack_station_count'], df['floor_area_sqm']/1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if 'sc_battery_mean_mean' in df.columns and 'sc_robot_utilization_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_battery_per_robot'] = safe_div(df['sc_battery_mean_mean']*df['sc_robot_utilization_mean'], df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = safe_div(df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    return df


def segment_analysis(pred, actual, label=''):
    bins = [(0,5), (5,10), (10,20), (20,30), (30,50), (50,80), (80,800)]
    total_mae = np.abs(pred - actual).mean()
    print(f'\n[구간 분석] {label} (전체 MAE={total_mae:.4f})')
    for lo, hi in bins:
        mask = (actual >= lo) & (actual < hi)
        if mask.sum() == 0: continue
        seg_mae = np.abs(pred[mask] - actual[mask]).mean()
        contrib = seg_mae * mask.sum() / len(actual)
        pct = mask.sum() / len(actual) * 100
        pr = pred[mask].mean() / (actual[mask].mean() + 1e-8)
        print(f'  [{lo:3d},{hi:3d}): n={mask.sum():6d} ({pct:5.1f}%) '
              f'MAE={seg_mae:7.2f}  contrib={contrib:5.3f}  pred/actual={pr:.3f}')
    return total_mae


def main():
    t0 = time.time()
    print('=' * 70)
    print('v4.1B LITE: 체크포인트 → 메타 → 분류기 → 2D 보정')
    print('=' * 70)
    os.makedirs(SUB_DIR, exist_ok=True)

    # ── Step 1: 데이터 + 피처 (메모리 내) ──
    print('\n[Step 1] 데이터 로드 + 피처 생성')
    train_raw = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_raw  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout    = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    train, test = build_features(train_raw, test_raw, layout,
                                  lag_lags=[1,2,3,4,5,6],
                                  rolling_windows=[3,5,10])
    del train_raw, test_raw, layout; gc.collect()

    print('  시나리오 집계...')
    train = add_scenario_agg_fast(train)
    test  = add_scenario_agg_fast(test)
    print('  비율 피처...')
    train = add_ratio_features(train)
    test  = add_ratio_features(test)

    feat_cols = [c for c in train.columns
                 if c not in {'ID','scenario_id','layout_id','avg_delay_minutes_next_30m'}
                 and train[c].dtype != object]
    print(f'  피처 수: {len(feat_cols)}, 소요: {time.time()-t0:.1f}s')

    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']

    # ── Step 2: 체크포인트 로드 ──
    print('\n[Step 2] 체크포인트 로드')
    def load_ckpt(name):
        return (np.load(os.path.join(CKPT_DIR, f'{name}_oof.npy')),
                np.load(os.path.join(CKPT_DIR, f'{name}_test.npy')))

    oof_lg, test_lg = load_ckpt('lgbm')
    oof_tw, test_tw = load_ckpt('tw18')
    oof_cb, test_cb = load_ckpt('cb')
    oof_et, test_et = load_ckpt('et')
    oof_rf, test_rf = load_ckpt('rf')

    print(f'  LGBM OOF MAE={np.abs(np.expm1(oof_lg) - y_raw.values).mean():.4f}')
    print(f'  TW1.8 OOF MAE={np.abs(oof_tw - y_raw.values).mean():.4f}')
    print(f'  CB OOF MAE={np.abs(np.expm1(oof_cb) - y_raw.values).mean():.4f}')
    print(f'  ET OOF MAE={np.abs(np.expm1(oof_et) - y_raw.values).mean():.4f}')
    print(f'  RF OOF MAE={np.abs(np.expm1(oof_rf) - y_raw.values).mean():.4f}')

    # ── Step 3: 메타 스태킹 ──
    print('\n[Step 3] 5모델 LGBM 메타 스태킹')
    meta_train = np.column_stack([oof_lg, oof_cb,
                                   np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf])
    meta_test  = np.column_stack([test_lg, test_cb,
                                   np.log1p(np.maximum(test_tw, 0)), test_et, test_rf])

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_meta = np.zeros(len(y_raw)); test_meta = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(meta_train, y_raw, groups)):
        m = lgb.LGBMRegressor(**META_LGBM_PARAMS)
        m.fit(meta_train[tr_idx], np.log1p(y_raw.iloc[tr_idx].values),
              eval_set=[(meta_train[va_idx], np.log1p(y_raw.iloc[va_idx].values))],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[va_idx] = np.expm1(m.predict(meta_train[va_idx]))
        test_meta += np.expm1(m.predict(meta_test)) / N_SPLITS
        mae = np.abs(oof_meta[va_idx] - y_raw.iloc[va_idx].values).mean()
        print(f'  [META] Fold {fold+1}  MAE={mae:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()
    mae_baseline = np.abs(oof_meta - y_raw.values).mean()
    print(f'  META OOF MAE={mae_baseline:.4f} | pred_std={oof_meta.std():.2f}')

    segment_analysis(oof_meta, y_raw.values, label='model30 기준선')

    # ── Step 4: 시나리오 분류기 → extreme_prob ──
    print('\n[Step 4] 시나리오 분류기 → extreme_prob')
    sc_mean = train.groupby('scenario_id')['avg_delay_minutes_next_30m'].mean()
    sc_label = (sc_mean >= EXTREME_THRESHOLD).astype(int)
    row_label = train['scenario_id'].map(sc_label).values
    n_ext = sc_label.sum(); n_tot = len(sc_label)
    print(f'  극값 시나리오: {n_ext}/{n_tot} ({n_ext/n_tot*100:.1f}%)')

    clf_feat_cols = [c for c in feat_cols if c.startswith('sc_') or c.startswith('ratio_')]
    print(f'  분류기 피처: {len(clf_feat_cols)}종')

    X_tr_clf = train[clf_feat_cols].fillna(0)
    X_te_clf = test[clf_feat_cols].fillna(0)

    oof_prob = np.zeros(len(train))
    test_prob = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_clf, row_label, groups)):
        m = lgb.LGBMClassifier(**CLF_PARAMS)
        m.fit(X_tr_clf.iloc[tr_idx], row_label[tr_idx],
              eval_set=[(X_tr_clf.iloc[va_idx], row_label[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_prob[va_idx] = m.predict_proba(X_tr_clf.iloc[va_idx])[:, 1]
        test_prob += m.predict_proba(X_te_clf)[:, 1] / N_SPLITS
        auc = roc_auc_score(row_label[va_idx], oof_prob[va_idx])
        print(f'  [CLF] Fold {fold+1}  AUC={auc:.4f}  iter={m.best_iteration_}')
        del m; gc.collect()

    oof_auc = roc_auc_score(row_label, oof_prob)
    oof_f1 = f1_score(row_label, (oof_prob >= 0.5).astype(int))
    print(f'  OOF AUC={oof_auc:.4f}, F1(0.5)={oof_f1:.4f}')
    print(f'  train prob: mean={oof_prob.mean():.4f}, std={oof_prob.std():.4f}')
    print(f'  test  prob: mean={test_prob.mean():.4f}, std={test_prob.std():.4f}')

    # 타겟 구간별 prob
    for lo, hi in [(0,5),(5,20),(20,50),(50,80),(80,800)]:
        mask = (y_raw.values >= lo) & (y_raw.values < hi)
        if mask.sum() > 0:
            print(f'    [{lo:3d},{hi:3d}): mean_prob={oof_prob[mask].mean():.4f}, n={mask.sum()}')

    # ── Step 5: 2D 보정 테이블 ──
    print('\n[Step 5] 2D 보정 테이블 구축')
    pred_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
    prob_bins = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.01]

    pred_bin_idx = np.clip(np.digitize(oof_meta, pred_bins) - 1, 0, len(pred_bins) - 2)
    prob_bin_idx = np.clip(np.digitize(oof_prob, prob_bins) - 1, 0, len(prob_bins) - 2)

    table = {}
    for pi in range(len(pred_bins) - 1):
        for pbi in range(len(prob_bins) - 1):
            mask = (pred_bin_idx == pi) & (prob_bin_idx == pbi)
            n = mask.sum()
            if n < 5:
                table[(pi, pbi)] = 1.0; continue
            mp = oof_meta[mask].mean(); ma = y_raw.values[mask].mean()
            raw = ma / mp if mp > 1.0 else 1.0
            clipped = np.clip(raw, 0.8, 3.0)
            if prob_bins[pbi + 1] <= 0.20:
                clipped = 1.0 + (clipped - 1.0) * 0.2
            table[(pi, pbi)] = clipped
            flag = 'UP' if clipped > 1.05 else ('DN' if clipped < 0.95 else '  ')
            print(f'  [{pred_bins[pi]:3.0f},{pred_bins[pi+1]:3.0f}) x [{prob_bins[pbi]:.2f},{prob_bins[pbi+1]:.2f}): '
                  f'n={n:5d} pred={mp:6.2f} actual={ma:6.2f} corr={clipped:.4f} {flag}')

    # ── Step 6: OOF 보정 + 평가 ──
    print('\n[Step 6] OOF 보정 적용')

    def apply_2d(predictions, probs, tbl, pbins, prbins):
        pbi = np.clip(np.digitize(predictions, pbins) - 1, 0, len(pbins) - 2)
        prbi = np.clip(np.digitize(probs, prbins) - 1, 0, len(prbins) - 2)
        out = predictions.copy()
        for i in range(len(predictions)):
            out[i] *= tbl.get((pbi[i], prbi[i]), 1.0)
        return np.maximum(out, 0)

    oof_corrected = apply_2d(oof_meta, oof_prob, table, pred_bins, prob_bins)
    mae_corrected = np.abs(oof_corrected - y_raw.values).mean()
    print(f'  기준선 MAE: {mae_baseline:.4f}')
    print(f'  B+C 보정 MAE: {mae_corrected:.4f} (delta={mae_corrected - mae_baseline:+.4f})')

    segment_analysis(oof_corrected, y_raw.values, label='B+C 보정 후')

    # ── Step 7: α 최적화 ──
    print('\n[Step 7] alpha 최적화')
    best_alpha, best_mae = 1.0, mae_corrected
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        bl = np.maximum(oof_meta * (1 - alpha) + oof_corrected * alpha, 0)
        mae_a = np.abs(bl - y_raw.values).mean()
        marker = ' <-- best' if mae_a < best_mae else ''
        print(f'  a={alpha:.1f}: MAE={mae_a:.4f}{marker}')
        if mae_a < best_mae:
            best_mae = mae_a; best_alpha = alpha

    print(f'\n  최적 alpha={best_alpha:.1f}, MAE={best_mae:.4f} (기준 {mae_baseline:.4f})')

    # ── Step 8: test 보정 + 제출 ──
    print('\n[Step 8] test 보정 + 제출')
    test_corrected = apply_2d(test_meta, test_prob, table, pred_bins, prob_bins)
    if 0 < best_alpha < 1:
        test_final = np.maximum(test_meta * (1 - best_alpha) + test_corrected * best_alpha, 0)
    elif best_alpha >= 1:
        test_final = test_corrected
    else:
        test_final = np.maximum(test_meta, 0)

    print(f'  기준 test: mean={test_meta.mean():.2f}, std={test_meta.std():.2f}, max={test_meta.max():.2f}')
    print(f'  B+C  test: mean={test_final.mean():.2f}, std={test_final.std():.2f}, max={test_final.max():.2f}')

    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    sub_base = os.path.join(SUB_DIR, 'v4_BC_baseline.csv')
    sample.to_csv(sub_base, index=False)

    sample['avg_delay_minutes_next_30m'] = test_final
    sub_bc = os.path.join(SUB_DIR, 'v4_postprocess_BC.csv')
    sample.to_csv(sub_bc, index=False)
    print(f'  기준선: {sub_base}')
    print(f'  B+C:    {sub_bc}')

    # ── 요약 ──
    elapsed = (time.time() - t0) / 60
    print(f'\n{"=" * 70}')
    print(f'v4.1B LITE 결과 ({elapsed:.1f}분)')
    print(f'{"=" * 70}')
    print(f'  분류기 OOF AUC:      {oof_auc:.4f}')
    print(f'  model30 기준 CV:     {mae_baseline:.4f}')
    print(f'  B+C 보정 CV:         {mae_corrected:.4f} (d={mae_corrected-mae_baseline:+.4f})')
    print(f'  최적 a={best_alpha:.1f} CV:      {best_mae:.4f} (d={best_mae-mae_baseline:+.4f})')
    print(f'  test pred std:       {test_final.std():.2f} (기준 {test_meta.std():.2f})')
    if best_mae < mae_baseline:
        print(f'  => B+C 유효! v4_postprocess_BC.csv 제출 추천')
    else:
        print(f'  => B+C 무효. 기준선 유지')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
