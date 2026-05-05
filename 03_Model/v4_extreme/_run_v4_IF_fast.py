"""
v4.1A 빠른 단계별 실행 래퍼
- 각 단계 완료 시 pickle 체크포인트 저장
- STEP 환경변수로 실행할 단계 지정
- STEP=1: build_features
- STEP=2: 시나리오 집계 (벡터화 최적화)
- STEP=3: 비율 피처 + 캐시 저장
- STEP=4: 메타 스태킹 + IF + 보정 + 제출
"""
import numpy as np
import pandas as pd
import sys, os, time, pickle, gc

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

CACHE_DIR = 'docs'
STEP = int(os.environ.get('STEP', '1'))

def cache_path(name):
    return os.path.join(CACHE_DIR, f'_v4_cache_{name}.pkl')

def save_cache(name, obj):
    with open(cache_path(name), 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def load_cache(name):
    with open(cache_path(name), 'rb') as f:
        return pickle.load(f)

# ── 최적화된 시나리오 집계 (lambda 제거) ──
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

def add_scenario_agg_fast(df):
    """벡터화된 시나리오 집계 - lambda transform 제거"""
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]
        # 기본 5종 (빠름)
        df[f'sc_{col}_mean'] = grp.transform('mean')
        df[f'sc_{col}_std']  = grp.transform('std').fillna(0)
        df[f'sc_{col}_max']  = grp.transform('max')
        df[f'sc_{col}_min']  = grp.transform('min')
        df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']
        df[f'sc_{col}_median'] = grp.transform('median')

        # p10, p90, skew, kurtosis → 시나리오 레벨 계산 후 merge (lambda 회피)
        sc_stats = grp.agg(
            p10=lambda x: x.quantile(0.10),
            p90=lambda x: x.quantile(0.90),
            skew='skew',
            kurtosis=lambda x: x.kurtosis()
        ).fillna(0)
        sc_stats.columns = [f'sc_{col}_{s}' for s in ['p10','p90','skew','kurtosis']]
        sc_stats['scenario_id'] = sc_stats.index
        # merge
        for s in ['p10','p90','skew','kurtosis']:
            cname = f'sc_{col}_{s}'
            mapping = sc_stats.set_index('scenario_id')[cname]
            df[cname] = df['scenario_id'].map(mapping).fillna(0)

        # cv
        df[f'sc_{col}_cv'] = (df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)

    return df


if STEP == 1:
    t0 = time.time()
    print('[STEP 1] build_features')
    from feature_engineering import build_features
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    layout = pd.read_csv('data/layout_info.csv')
    train, test = build_features(train, test, layout,
                                  lag_lags=[1,2,3,4,5,6],
                                  rolling_windows=[3,5,10])
    save_cache('step1', {'train': train, 'test': test})
    print(f'  완료: {time.time()-t0:.1f}s, train={train.shape}')

elif STEP == 2:
    t0 = time.time()
    print('[STEP 2] 시나리오 집계 (최적화)')
    d = load_cache('step1')
    train, test = d['train'], d['test']
    del d; gc.collect()
    print(f'  캐시 로드: {time.time()-t0:.1f}s')

    t1 = time.time()
    train = add_scenario_agg_fast(train)
    print(f'  train 집계: {time.time()-t1:.1f}s, shape={train.shape}')
    t2 = time.time()
    test = add_scenario_agg_fast(test)
    print(f'  test 집계: {time.time()-t2:.1f}s, shape={test.shape}')

    save_cache('step2', {'train': train, 'test': test})
    print(f'  완료: {time.time()-t0:.1f}s')

elif STEP == 3:
    t0 = time.time()
    print('[STEP 3] 비율 피처 + 최종 캐시')
    from run_v4_postprocess_IF import (add_layout_ratio_features_tier1,
                                        add_layout_ratio_features_tier2,
                                        get_feat_cols)
    d = load_cache('step2')
    train, test = d['train'], d['test']
    del d; gc.collect()
    print(f'  캐시 로드: {time.time()-t0:.1f}s')

    train = add_layout_ratio_features_tier1(train)
    test  = add_layout_ratio_features_tier1(test)
    train = add_layout_ratio_features_tier2(train)
    test  = add_layout_ratio_features_tier2(test)
    feat_cols = get_feat_cols(train)
    print(f'  피처 수: {len(feat_cols)}, train={train.shape}')

    save_cache('step3', {'train': train, 'test': test, 'feat_cols': feat_cols})
    print(f'  완료: {time.time()-t0:.1f}s')

elif STEP == 4:
    t0 = time.time()
    print('[STEP 4] 메타 스태킹 + IF + 2D 보정')
    import lightgbm as lgb
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import GroupKFold
    from run_v4_postprocess_IF import (
        LGBM_PARAMS, META_LGBM_PARAMS, CKPT_DIR,
        IF_SCENARIO_COLS,
        compute_isolation_forest_scores,
        build_2d_calibration_table,
        apply_2d_calibration,
        segment_analysis,
        run_meta_lgbm,
    )

    d = load_cache('step3')
    train, test, feat_cols = d['train'], d['test'], d['feat_cols']
    del d; gc.collect()
    print(f'  캐시 로드: {time.time()-t0:.1f}s, 피처={len(feat_cols)}')

    y_raw = train['avg_delay_minutes_next_30m']
    y_log = np.log1p(y_raw)
    groups = train['scenario_id']

    # 체크포인트 로드
    oof_lg = np.load(os.path.join(CKPT_DIR, 'lgbm_oof.npy'))
    test_lg = np.load(os.path.join(CKPT_DIR, 'lgbm_test.npy'))
    oof_tw = np.load(os.path.join(CKPT_DIR, 'tw18_oof.npy'))
    test_tw = np.load(os.path.join(CKPT_DIR, 'tw18_test.npy'))
    oof_cb = np.load(os.path.join(CKPT_DIR, 'cb_oof.npy'))
    test_cb = np.load(os.path.join(CKPT_DIR, 'cb_test.npy'))
    oof_et = np.load(os.path.join(CKPT_DIR, 'et_oof.npy'))
    test_et = np.load(os.path.join(CKPT_DIR, 'et_test.npy'))
    oof_rf = np.load(os.path.join(CKPT_DIR, 'rf_oof.npy'))
    test_rf = np.load(os.path.join(CKPT_DIR, 'rf_test.npy'))
    print('  5모델 체크포인트 로드 완료')

    print(f'  LGBM OOF MAE={np.abs(np.expm1(oof_lg) - y_raw.values).mean():.4f}')
    print(f'  TW1.8 OOF MAE={np.abs(oof_tw - y_raw.values).mean():.4f}')
    print(f'  CB OOF MAE={np.abs(np.expm1(oof_cb) - y_raw.values).mean():.4f}')
    print(f'  ET OOF MAE={np.abs(np.expm1(oof_et) - y_raw.values).mean():.4f}')
    print(f'  RF OOF MAE={np.abs(np.expm1(oof_rf) - y_raw.values).mean():.4f}')

    # 메타 스태킹
    test_tw_clipped = np.maximum(test_tw, 0)
    meta_train = np.column_stack([oof_lg, oof_cb,
                                   np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf])
    meta_test  = np.column_stack([test_lg, test_cb,
                                   np.log1p(test_tw_clipped), test_et, test_rf])

    oof_meta, test_meta, mae_baseline = run_meta_lgbm(meta_train, meta_test, y_raw, groups)
    segment_analysis(oof_meta, y_raw.values, label='model30 기준선')

    # IF 점수
    train_if_scores, test_if_scores = compute_isolation_forest_scores(train, test)

    # 2D 보정 테이블
    pred_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
    score_pcts = [0, 30, 50, 70, 85, 92, 97, 100]
    score_bins = np.percentile(train_if_scores, score_pcts)
    score_bins = np.unique(np.round(score_bins, 6))

    table, p_bins, s_bins = build_2d_calibration_table(
        oof_meta, y_raw.values, train_if_scores,
        pred_bins=pred_bins, score_bins=score_bins
    )

    # OOF 보정
    oof_corrected = apply_2d_calibration(oof_meta, train_if_scores, table, p_bins, s_bins)
    mae_corrected = np.abs(oof_corrected - y_raw.values).mean()
    print(f'\n  기준선 MAE: {mae_baseline:.4f}')
    print(f'  보정 후 MAE: {mae_corrected:.4f}')
    print(f'  변화:       {mae_corrected - mae_baseline:+.4f}')
    segment_analysis(oof_corrected, y_raw.values, label='IF 2D 보정 후')

    # α 탐색
    best_alpha, best_mae_alpha = 1.0, mae_corrected
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blended = oof_meta * (1 - alpha) + oof_corrected * alpha
        blended = np.maximum(blended, 0)
        mae_a = np.abs(blended - y_raw.values).mean()
        marker = ' ✅' if mae_a < best_mae_alpha else ''
        print(f'  α={alpha:.1f}: MAE={mae_a:.4f}{marker}')
        if mae_a < best_mae_alpha:
            best_mae_alpha = mae_a
            best_alpha = alpha

    print(f'\n  최적 α={best_alpha:.1f}, MAE={best_mae_alpha:.4f} (기준 {mae_baseline:.4f})')

    # test 보정 + 제출
    test_corrected = apply_2d_calibration(test_meta, test_if_scores, table, p_bins, s_bins)
    if best_alpha > 0 and best_alpha < 1.0:
        test_final = test_meta * (1 - best_alpha) + test_corrected * best_alpha
    elif best_alpha >= 1.0:
        test_final = test_corrected
    else:
        test_final = test_meta.copy()
    test_final = np.maximum(test_final, 0)

    print(f'\n  기준 test: mean={test_meta.mean():.2f}, std={test_meta.std():.2f}, max={test_meta.max():.2f}')
    print(f'  보정 test: mean={test_final.mean():.2f}, std={test_final.std():.2f}, max={test_final.max():.2f}')

    # 제출 CSV
    os.makedirs('submissions', exist_ok=True)
    sample = pd.read_csv('data/sample_submission.csv')

    sample['avg_delay_minutes_next_30m'] = np.maximum(test_meta, 0)
    sample.to_csv('submissions/v4_IF_baseline.csv', index=False)

    sample['avg_delay_minutes_next_30m'] = test_final
    sample.to_csv('submissions/v4_postprocess_IF.csv', index=False)
    print(f'\n  제출 파일: submissions/v4_IF_baseline.csv, submissions/v4_postprocess_IF.csv')

    # 최종 요약
    elapsed = (time.time() - t0) / 60
    print(f'\n{"="*70}')
    print(f'v4.1A 결과 ({elapsed:.1f}분)')
    print(f'{"="*70}')
    print(f'  model30 기준선 CV:   {mae_baseline:.4f}')
    print(f'  IF 2D 보정 CV:       {mae_corrected:.4f} (Δ={mae_corrected - mae_baseline:+.4f})')
    print(f'  최적 α={best_alpha:.1f} CV:      {best_mae_alpha:.4f} (Δ={best_mae_alpha - mae_baseline:+.4f})')
    print(f'  test pred std:       {test_final.std():.2f} (기준 {test_meta.std():.2f})')
    ratio_est = best_mae_alpha * 1.158 if best_mae_alpha < mae_baseline else mae_baseline * 1.158
    print(f'  기대 Public (×1.158): {ratio_est:.4f}')
    if best_mae_alpha < mae_baseline:
        print(f'\n  ✅ IF 후처리 유효! → v4_postprocess_IF.csv 제출 추천')
    else:
        print(f'\n  ⚠️ IF 후처리 무효 — 기준선 유지')
    print(f'{"="*70}')

else:
    print(f'Unknown STEP={STEP}')
