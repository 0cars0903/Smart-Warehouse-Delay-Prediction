"""
model43: 비트리(Non-Tree) 모델 다양성 탐색 + 앙상블 가능성 판정
================================================================
동기:
  - 현재 파이프라인(LGBM+CB+TW15+ET+RF+Asym)은 모두 GBDT 계열
  - model11에서 sklearn MLP(LGBM-MLP 상관 0.8043!)로 다양성 확인했으나
    lag/rolling 피처가 GroupKFold에서 시나리오 패턴을 암기 → OOF 12.7 참패
  - 핵심 변경: sc_agg + ratio + shift-safe 피처만 사용 (lag/rolling 완전 제외)
    → sc_agg는 시나리오 전체 통계의 broadcast → 암기 불가, 일반화 강건

탐색 모델:
  1. sklearn MLPRegressor — 비선형 tabular, 다양성 기대 높음
  2. Ridge regression     — 선형 베이스라인, 고차원 희소 신호 포착
  3. ElasticNet           — L1+L2 정규화, Ridge보다 피처 선택적

앙상블 판정 기준:
  - LGBM-모델 상관 < 0.90  → 다양성 충분, 앙상블 시도 가치 있음
  - LGBM-모델 상관 0.90~0.95 → 조건부 (OOF MAE < 9.0인 경우만)
  - LGBM-모델 상관 > 0.95  → 다양성 없음, 앙상블 불필요

사용 피처 (lag/rolling 제외):
  - sc_agg     (198종): 시나리오 집계 11통계 × 18컬럼
  - ratio      (~12종): capacity 비율
  - shift-safe (  7종): cross-ratio 피처
  - trajectory ( 29종): 궤적 형상 피처 (시나리오 레벨 broadcast)
  - base_raw   (~18종): 원본 연속형 피처
  - ts_idx, layout    : 타임스탬프, 창고 구조
  → 총 ~280 피처 (lag 48종 + rolling 48종 제외)

실행: python src/run_model43_nontree.py
예상 시간: MLP ~30분 / Ridge ~5분 / 앙상블 평가 ~5분
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings, gc, os, sys, time

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import build_features

warnings.filterwarnings('ignore')

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
SUB_DIR  = os.path.join(_BASE, '..', 'submissions')
CKPT_41  = os.path.join(_BASE, '..', 'docs', 'model41_ckpt')  # 기존 6모델 OOF
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model43_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42

# 앙상블 판단 기준 (LGBM-비트리 상관)
DIVERSITY_THRESHOLD = 0.92
MAE_THRESHOLD = 9.5  # 이 이상이면 앙상블 기여 없음

# ── 메타 파라미터 ──
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
TRAJ_COLS  = ['robot_utilization','order_inflow_15m','congestion_score',
              'low_battery_ratio','battery_mean','charge_queue_length',
              'robot_idle','max_zone_density']
PEAK_COLS  = ['order_inflow_15m','congestion_score','low_battery_ratio',
              'charge_queue_length','max_zone_density']
MONO_COLS  = ['robot_utilization','congestion_score','order_inflow_15m']


# ═══════════════════════════════════════════════════════
# Feature Engineering (model41과 동일)
# ═══════════════════════════════════════════════════════

def _safe_div(a, b, fill=0):
    return (a / (b + 1e-8)).fillna(fill).replace([np.inf, -np.inf], fill)

def add_scenario_agg_features(df):
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
        df[f'sc_{col}_cv']       = (
            df[f'sc_{col}_std'] / (df[f'sc_{col}_mean'].abs() + 1e-8)).fillna(0)
    return df

def add_ratio_features(df):
    if 'sc_order_inflow_15m_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_demand_per_robot'] = _safe_div(df['sc_order_inflow_15m_mean'], df['robot_total'])
    if 'sc_congestion_score_mean' in df.columns and 'intersection_count' in df.columns:
        df['ratio_congestion_per_intersection'] = _safe_div(df['sc_congestion_score_mean'], df['intersection_count'])
    if all(c in df.columns for c in ['sc_low_battery_ratio_mean','sc_charge_queue_length_mean','charger_count']):
        df['ratio_battery_stress'] = _safe_div(
            df['sc_low_battery_ratio_mean']*df['sc_charge_queue_length_mean'], df['charger_count'])
    if 'sc_order_inflow_15m_mean' in df.columns and 'pack_station_count' in df.columns:
        df['ratio_packing_pressure'] = _safe_div(df['sc_order_inflow_15m_mean'], df['pack_station_count'])
    if 'sc_robot_utilization_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_active_capacity'] = df['sc_robot_utilization_mean'] * df['robot_total']
    if all(c in df.columns for c in ['sc_congestion_score_mean','sc_order_inflow_15m_mean','robot_total']):
        df['ratio_cross_stress'] = _safe_div(
            df['sc_congestion_score_mean']*df['sc_order_inflow_15m_mean'], df['robot_total']**2)
    if 'robot_total' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_robot_density'] = _safe_div(df['robot_total'], df['floor_area_sqm']/100)
    if 'pack_station_count' in df.columns and 'floor_area_sqm' in df.columns:
        df['ratio_pack_density'] = _safe_div(df['pack_station_count'], df['floor_area_sqm']/1000)
    if 'sc_robot_charging_mean' in df.columns and 'charger_count' in df.columns:
        df['ratio_charge_competition'] = _safe_div(df['sc_robot_charging_mean'], df['charger_count'])
    if all(c in df.columns for c in ['sc_battery_mean_mean','sc_robot_utilization_mean','charger_count']):
        df['ratio_battery_per_robot'] = _safe_div(
            df['sc_battery_mean_mean']*df['sc_robot_utilization_mean'], df['charger_count'])
    if 'sc_congestion_score_mean' in df.columns and 'aisle_width_avg' in df.columns:
        df['ratio_congestion_per_aisle'] = _safe_div(df['sc_congestion_score_mean'], df['aisle_width_avg'])
    if 'sc_robot_idle_mean' in df.columns and 'robot_total' in df.columns:
        df['ratio_idle_fraction'] = _safe_div(df['sc_robot_idle_mean'], df['robot_total'])
    return df

def add_shift_safe_fe(df):
    if 'robot_utilization' in df.columns and 'order_inflow_15m' in df.columns:
        df['feat_util_x_order']    = df['robot_utilization'] * df['order_inflow_15m']
    if 'low_battery_ratio' in df.columns and 'congestion_score' in df.columns:
        df['feat_batt_x_cong']     = df['low_battery_ratio'] * df['congestion_score']
    if 'charge_queue_length' in df.columns and 'charger_count' in df.columns:
        df['feat_queue_per_charger'] = _safe_div(df['charge_queue_length'].fillna(0), df['charger_count'])
    if 'robot_idle' in df.columns and 'robot_utilization' in df.columns:
        df['feat_idle_util_ratio'] = _safe_div(df['robot_idle'].fillna(0),
                                                df['robot_utilization'].fillna(0)+1e-8)
    if 'order_inflow_15m' in df.columns and 'congestion_score' in df.columns:
        df['feat_order_cong']      = df['order_inflow_15m'].fillna(0) * df['congestion_score'].fillna(0)
    if 'battery_mean' in df.columns and 'low_battery_ratio' in df.columns:
        df['feat_batt_risk']       = (100-df['battery_mean'].fillna(100)) * df['low_battery_ratio'].fillna(0)
    if 'max_zone_density' in df.columns and 'order_inflow_15m' in df.columns:
        df['feat_density_order']   = df['max_zone_density'].fillna(0) * df['order_inflow_15m'].fillna(0)
    return df

def add_trajectory_features(df):
    df = df.copy()
    if 'ts_idx' not in df.columns:
        df['ts_idx'] = df.groupby('scenario_id').cumcount()
    ts_arr = np.arange(25, dtype=np.float64)
    for col in TRAJ_COLS:
        if col not in df.columns: continue
        slope_map = (df.groupby('scenario_id')[col]
                     .apply(lambda x: np.polyfit(ts_arr[:len(x)], x.fillna(x.mean()).values, 1)[0]
                            if len(x)>1 else 0.0).fillna(0))
        df[f'sc_{col}_slope'] = df['scenario_id'].map(slope_map)
    for col in TRAJ_COLS:
        if col not in df.columns: continue
        f5 = df[df['ts_idx']<5].groupby('scenario_id')[col].mean()
        l5 = df[df['ts_idx']>=20].groupby('scenario_id')[col].mean()
        fl = (l5/(f5.abs()+1e-8)).fillna(1.0).replace([np.inf,-np.inf],1.0)
        df[f'sc_{col}_fl_ratio'] = df['scenario_id'].map(fl)
    for col in PEAK_COLS:
        if col not in df.columns: continue
        peak_map = (df.groupby('scenario_id')
                    .apply(lambda g: g.loc[g[col].fillna(-np.inf).idxmax(),'ts_idx']/24.0
                           if col in g.columns else 0.5).fillna(0.5))
        df[f'sc_{col}_peak_pos'] = df['scenario_id'].map(peak_map)
    for col in PEAK_COLS:
        if col not in df.columns: continue
        sm = f'sc_{col}_mean'; ss = f'sc_{col}_std'
        if sm not in df.columns or ss not in df.columns: continue
        above_map = ((df[col].fillna(0) > df[sm]+0.5*df[ss]).astype(int)
                     .groupby(df['scenario_id']).sum())
        df[f'sc_{col}_above_cnt'] = df['scenario_id'].map(above_map).fillna(0)
    for col in MONO_COLS:
        if col not in df.columns: continue
        def _mono(x):
            v = x.fillna(x.mean()).values
            return float((np.diff(v)>0).sum())/len(np.diff(v)) if len(v)>1 else 0.5
        mono_map = df.groupby('scenario_id')[col].apply(_mono).fillna(0.5)
        df[f'sc_{col}_mono'] = df['scenario_id'].map(mono_map)
    return df

def load_and_prepare_data():
    t0 = time.time()
    print('데이터 로드 중...')
    train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
    print('기본 FE (lag+rolling 포함, MLP용 필터링은 이후)...')
    train, test = build_features(train, test, layout,
                                 lag_lags=[1,2,3,4,5,6],
                                 rolling_windows=[3,5,10])
    train = add_scenario_agg_features(train)
    test  = add_scenario_agg_features(test)
    train = add_ratio_features(train)
    test  = add_ratio_features(test)
    train = add_shift_safe_fe(train)
    test  = add_shift_safe_fe(test)
    train = add_trajectory_features(train)
    test  = add_trajectory_features(test)
    elapsed = time.time() - t0
    print(f'FE 완료: {train.shape}, {elapsed:.1f}s')
    return train, test

def get_all_feat_cols(df):
    return [c for c in df.columns
            if c not in {'ID','scenario_id','layout_id','avg_delay_minutes_next_30m'}
            and df[c].dtype != object]

def get_nontree_feat_cols(df, all_feat_cols):
    """
    MLP / Ridge용 피처: lag/rolling 제외
    근거: model11 실패 원인 — GroupKFold에서 lag/rolling이 시나리오 패턴 암기
    sc_agg / ratio / shift_safe / trajectory / base_raw / ts_idx / layout 유지
    """
    excluded = [c for c in all_feat_cols if 'lag' in c or 'roll' in c]
    included = [c for c in all_feat_cols if c not in excluded]
    print(f'\n[피처 구성] 전체 {len(all_feat_cols)}개 → lag/rolling 제외 {len(excluded)}개 → MLP용 {len(included)}개')
    return included, excluded


# ═══════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════

def save_ckpt(name, oof, test_pred):
    os.makedirs(CKPT_DIR, exist_ok=True)
    np.save(os.path.join(CKPT_DIR, f'{name}_oof.npy'),  oof)
    np.save(os.path.join(CKPT_DIR, f'{name}_test.npy'), test_pred)

def load_ckpt(name):
    return (np.load(os.path.join(CKPT_DIR, f'{name}_oof.npy')),
            np.load(os.path.join(CKPT_DIR, f'{name}_test.npy')))

def ckpt_exists(name):
    return (os.path.exists(os.path.join(CKPT_DIR, f'{name}_oof.npy')) and
            os.path.exists(os.path.join(CKPT_DIR, f'{name}_test.npy')))


# ═══════════════════════════════════════════════════════
# 1. MLP (sklearn MLPRegressor)
# ═══════════════════════════════════════════════════════

def train_mlp_oof(X_tr, X_te, y_raw, groups, feat_cols, name='mlp'):
    """
    GroupKFold 5-fold MLP
    - 타겟: log1p(y_raw) → 예측 expm1으로 복원
    - StandardScaler: fold 내 fit → 리크 방지
    - early_stopping=True: 내부 10% val로 조기종료 (GroupKFold val과 무관)
    - n_iter_no_change=30, tol=1e-5: model11 31-iter 조기종료 방지
    """
    if ckpt_exists(name):
        print(f'  [{name}] 체크포인트 로드')
        return load_ckpt(name)

    print(f'\n[MLP 설정]')
    print(f'  피처수: {len(feat_cols)}  타겟: log1p(y)  아키텍처: {len(feat_cols)}→512→256→128→1')
    print(f'  early_stopping=True  n_iter_no_change=30  max_iter=500')
    print(f'  주의: model11 실패는 lag/rolling 피처 때문 → 이번엔 완전 제외')

    gkf = GroupKFold(n_splits=N_SPLITS)
    y_log = np.log1p(y_raw)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt_all = X_tr[feat_cols].fillna(0)
    Xte_all = X_te[feat_cols].fillna(0)
    fold_maes = []; fold_iters = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt_all, y_log, groups)):
        t_fold = time.time()
        Xtr = Xt_all.iloc[tr_i].values
        Xva = Xt_all.iloc[va_i].values
        ytr = y_log.iloc[tr_i].values
        yva = y_log.iloc[va_i].values

        # fold 내 스케일링
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xte_all.values)

        mlp = MLPRegressor(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            tol=1e-5,
            random_state=RANDOM_STATE + fold,
            verbose=False,
        )
        mlp.fit(Xtr_s, ytr)

        oof[va_i]  = np.expm1(mlp.predict(Xva_s))
        test_pred += np.expm1(mlp.predict(Xte_s)) / N_SPLITS
        mae = np.abs(oof[va_i] - np.expm1(yva)).mean()
        fold_maes.append(mae)
        fold_iters.append(mlp.n_iter_)
        elapsed = time.time() - t_fold
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}  iter={mlp.n_iter_}  ({elapsed:.0f}s)')
        del mlp, scaler; gc.collect()

    oof_mae = np.mean(fold_maes)
    print(f'  [{name}] OOF MAE = {oof_mae:.4f}  avg_iter={np.mean(fold_iters):.0f}')
    print(f'  [{name}] pred_std={np.std(test_pred):.2f}  pred_mean={np.mean(test_pred):.2f}')
    save_ckpt(name, oof, test_pred)
    return oof, test_pred


# ═══════════════════════════════════════════════════════
# 2. Ridge Regression
# ═══════════════════════════════════════════════════════

def train_ridge_oof(X_tr, X_te, y_raw, groups, feat_cols, alpha=100.0, name='ridge'):
    """
    Ridge regression — 선형 베이스라인
    - 고차원 선형 패턴 포착 (GBDT가 놓치는 장거리 선형 관계)
    - 타겟: log1p(y), 예측: expm1 복원
    """
    if ckpt_exists(name):
        print(f'  [{name}] 체크포인트 로드')
        return load_ckpt(name)

    print(f'\n[Ridge alpha={alpha}]  피처수: {len(feat_cols)}')
    gkf = GroupKFold(n_splits=N_SPLITS)
    y_log = np.log1p(y_raw)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt_all = X_tr[feat_cols].fillna(0)
    Xte_all = X_te[feat_cols].fillna(0)
    fold_maes = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt_all, y_log.values, groups)):
        Xtr = Xt_all.iloc[tr_i].values
        Xva = Xt_all.iloc[va_i].values
        ytr = y_log.iloc[tr_i].values
        yva = y_log.iloc[va_i].values

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xte_all.values)

        m = Ridge(alpha=alpha)
        m.fit(Xtr_s, ytr)
        oof[va_i]  = np.expm1(m.predict(Xva_s))
        test_pred += np.expm1(m.predict(Xte_s)) / N_SPLITS
        mae = np.abs(oof[va_i] - np.expm1(yva)).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}')
        del m, scaler; gc.collect()

    oof_mae = np.mean(fold_maes)
    print(f'  [{name}] OOF MAE = {oof_mae:.4f}')
    save_ckpt(name, oof, test_pred)
    return oof, test_pred


# ═══════════════════════════════════════════════════════
# 3. ElasticNet
# ═══════════════════════════════════════════════════════

def train_elasticnet_oof(X_tr, X_te, y_raw, groups, feat_cols,
                          alpha=1.0, l1_ratio=0.5, name='elasticnet'):
    if ckpt_exists(name):
        print(f'  [{name}] 체크포인트 로드')
        return load_ckpt(name)

    print(f'\n[ElasticNet alpha={alpha} l1_ratio={l1_ratio}]  피처수: {len(feat_cols)}')
    gkf = GroupKFold(n_splits=N_SPLITS)
    y_log = np.log1p(y_raw)
    oof = np.zeros(len(X_tr)); test_pred = np.zeros(len(X_te))
    Xt_all = X_tr[feat_cols].fillna(0)
    Xte_all = X_te[feat_cols].fillna(0)
    fold_maes = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(Xt_all, y_log.values, groups)):
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xt_all.iloc[tr_i].values)
        Xva_s = scaler.transform(Xt_all.iloc[va_i].values)
        Xte_s = scaler.transform(Xte_all.values)

        m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=RANDOM_STATE)
        m.fit(Xtr_s, y_log.iloc[tr_i].values)
        oof[va_i]  = np.expm1(m.predict(Xva_s))
        test_pred += np.expm1(m.predict(Xte_s)) / N_SPLITS
        mae = np.abs(oof[va_i] - np.expm1(y_log.iloc[va_i].values)).mean()
        fold_maes.append(mae)
        print(f'  [{name}] Fold {fold+1}  MAE={mae:.4f}')
        del m, scaler; gc.collect()

    oof_mae = np.mean(fold_maes)
    print(f'  [{name}] OOF MAE = {oof_mae:.4f}')
    save_ckpt(name, oof, test_pred)
    return oof, test_pred


# ═══════════════════════════════════════════════════════
# 다양성 분석 & 앙상블 평가
# ═══════════════════════════════════════════════════════

def diversity_report(new_oof, new_name, ref_oofs, ref_names, y_raw_arr):
    """신규 모델의 다양성 및 앙상블 잠재력 평가"""
    new_mae = np.abs(new_oof - y_raw_arr).mean()
    print(f'\n[다양성 분석] {new_name}  OOF MAE={new_mae:.4f}')
    print(f'  pred_std={np.std(new_oof):.2f}  pred_mean={np.mean(new_oof):.2f}')

    corrs = {}
    for ref_oof, ref_name in zip(ref_oofs, ref_names):
        c = np.corrcoef(new_oof, ref_oof)[0, 1]
        corrs[ref_name] = c
        diversity_flag = '✅ 다양성 충분' if c < DIVERSITY_THRESHOLD else '❌ 다양성 부족'
        print(f'  {new_name}-{ref_name} 상관: {c:.4f}  {diversity_flag}')

    # 앙상블 잠재력 판정
    min_corr = min(corrs.values())
    if min_corr < DIVERSITY_THRESHOLD and new_mae < MAE_THRESHOLD:
        verdict = f'✅ 앙상블 가치 있음 (최저 상관 {min_corr:.4f} < {DIVERSITY_THRESHOLD}, MAE {new_mae:.4f} < {MAE_THRESHOLD})'
        ensemble_ok = True
    elif new_mae >= MAE_THRESHOLD:
        verdict = f'❌ OOF MAE={new_mae:.4f} ≥ {MAE_THRESHOLD} → 성능 불충분'
        ensemble_ok = False
    else:
        verdict = f'⚠️  상관 {min_corr:.4f} > {DIVERSITY_THRESHOLD} → 기존 모델과 중복 신호'
        ensemble_ok = False

    print(f'\n  판정: {verdict}')
    return ensemble_ok, new_mae, corrs


def test_ensemble_7model(new_oof, new_test, new_name,
                          ref_oofs, ref_tests, ref_names, y_raw_arr, groups):
    """기존 6모델 OOF + 신규 1모델 → 7모델 메타 스태킹 CV"""
    all_oofs  = ref_oofs  + [new_oof]
    all_tests = ref_tests + [new_test]
    all_names = ref_names + [new_name]

    print(f'\n[7모델 앙상블 테스트] {" + ".join(all_names)}')
    meta_X = np.column_stack(all_oofs)
    meta_oof = np.zeros(len(y_raw_arr))
    meta_test_preds = []
    fold_maes = []

    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr_i, va_i) in enumerate(gkf.split(meta_X, y_raw_arr, groups)):
        m = lgb.LGBMRegressor(**META_PARAMS)
        m.fit(meta_X[tr_i], y_raw_arr[tr_i],
              eval_set=[(meta_X[va_i], y_raw_arr[va_i])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        meta_oof[va_i] = m.predict(meta_X[va_i])
        meta_test_preds.append(m.predict(np.column_stack(all_tests)))
        mae = np.abs(meta_oof[va_i] - y_raw_arr[va_i]).mean()
        fold_maes.append(mae)
        del m; gc.collect()

    cv_mae  = np.mean(fold_maes)
    meta_te = np.mean(meta_test_preds, axis=0)
    print(f'  7모델 메타 CV MAE: {cv_mae:.4f}  (model41 기준: 8.4851)')
    print(f'  pred_std: {np.std(meta_te):.2f}  pred_mean: {np.mean(meta_te):.2f}')
    return cv_mae, meta_te


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    t_total = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    # ── 데이터 로드 & FE ──
    train, test = load_and_prepare_data()
    all_feat_cols = get_all_feat_cols(train)
    nontree_feats, lag_roll_feats = get_nontree_feat_cols(train, all_feat_cols)

    y_raw   = train['avg_delay_minutes_next_30m']
    y_raw_arr = y_raw.values
    groups  = train['scenario_id'].values

    print(f'\n[피처 확인]')
    print(f'  전체: {len(all_feat_cols)}  MLP/Ridge용: {len(nontree_feats)}  lag/rolling 제외: {len(lag_roll_feats)}')

    # ── 기존 6모델 OOF 로드 (model41 체크포인트) ──
    print(f'\n[기존 6모델 OOF 로드 시도 (model41_ckpt)]')
    ref_oofs = []; ref_tests = []; ref_names = []
    model41_models = ['lgbm', 'cb', 'tw15', 'et', 'rf', 'asym20']
    for mname in model41_models:
        oof_path  = os.path.join(CKPT_41, f'{mname}_oof.npy')
        test_path = os.path.join(CKPT_41, f'{mname}_test.npy')
        if os.path.exists(oof_path) and os.path.exists(test_path):
            ref_oofs.append(np.load(oof_path))
            ref_tests.append(np.load(test_path))
            ref_names.append(mname)
            print(f'  ✅ {mname} OOF 로드 완료')
        else:
            print(f'  ⚠️  {mname} 체크포인트 없음 — 앙상블 테스트에서 제외')

    if len(ref_oofs) == 0:
        print('  ❌ model41 체크포인트가 없습니다.')
        print('  → 다양성 분석은 수행하지만 앙상블 CV 테스트는 생략됩니다.')
        print('  → model41을 먼저 실행하세요: python src/run_model41_traj_fe.py')

    # ════════════════════════════════════════════
    # 모델 1: MLP
    # ════════════════════════════════════════════
    print(f'\n{"="*60}')
    print('[Model 1] sklearn MLP  (lag/rolling 제외 피처)')
    print(f'{"="*60}')
    mlp_oof, mlp_test = train_mlp_oof(train, test, y_raw, groups, nontree_feats, 'mlp')
    mlp_ok, mlp_mae, mlp_corrs = diversity_report(
        mlp_oof, 'MLP', ref_oofs, ref_names, y_raw_arr)

    # ════════════════════════════════════════════
    # 모델 2: Ridge
    # ════════════════════════════════════════════
    print(f'\n{"="*60}')
    print('[Model 2] Ridge Regression  (lag/rolling 제외 피처)')
    print(f'{"="*60}')
    ridge_oof, ridge_test = train_ridge_oof(
        train, test, y_raw, groups, nontree_feats, alpha=100.0, name='ridge')
    ridge_ok, ridge_mae, ridge_corrs = diversity_report(
        ridge_oof, 'Ridge', ref_oofs, ref_names, y_raw_arr)

    # ════════════════════════════════════════════
    # 모델 3: ElasticNet
    # ════════════════════════════════════════════
    print(f'\n{"="*60}')
    print('[Model 3] ElasticNet  (lag/rolling 제외 피처)')
    print(f'{"="*60}')
    en_oof, en_test = train_elasticnet_oof(
        train, test, y_raw, groups, nontree_feats,
        alpha=0.5, l1_ratio=0.5, name='elasticnet')
    en_ok, en_mae, en_corrs = diversity_report(
        en_oof, 'ElasticNet', ref_oofs, ref_names, y_raw_arr)

    # ════════════════════════════════════════════
    # 앙상블 테스트 (유망 모델만)
    # ════════════════════════════════════════════
    print(f'\n{"="*60}')
    print('[앙상블 가능성 판정 종합]')
    print(f'{"="*60}')

    candidates = []
    for name, ok, mae, oof, test_p in [
        ('MLP',        mlp_ok,   mlp_mae,   mlp_oof,   mlp_test),
        ('Ridge',      ridge_ok, ridge_mae, ridge_oof, ridge_test),
        ('ElasticNet', en_ok,    en_mae,    en_oof,    en_test),
    ]:
        status = '✅ 앙상블 후보' if ok else '❌ 제외'
        print(f'  {name:12s}: MAE={mae:.4f}  {status}')
        if ok:
            candidates.append((name, oof, test_p))

    if len(candidates) == 0:
        print('\n  → 앙상블 가능한 비트리 모델 없음. blend_w80 기준 유지.')
    elif len(ref_oofs) > 0:
        for cname, c_oof, c_test in candidates:
            print(f'\n  [{cname} 7모델 앙상블 테스트]')
            cv_7m, pred_7m = test_ensemble_7model(
                c_oof, c_test, cname.lower(),
                ref_oofs, ref_tests, ref_names, y_raw_arr, groups)

            # 제출 파일 생성 (개선된 경우만)
            if cv_7m < 8.4851:  # model41 CV 기준
                sub_name = f'model43_{cname.lower()}_7model_cv{cv_7m:.4f}.csv'
                sub_path = os.path.join(SUB_DIR, sub_name)
                sub = pd.DataFrame({'ID': test['ID'],
                                    'avg_delay_minutes_next_30m': pred_7m})
                sub.to_csv(sub_path, index=False)
                print(f'  → 제출 파일 생성: {sub_name}')
                print(f'  → pred_std={np.std(pred_7m):.2f} 확인 후 제출 판단')
            else:
                print(f'  → CV {cv_7m:.4f} ≥ 8.4851 (model41 기준) — 개선 없음')
    else:
        print('\n  → model41 체크포인트 없어 7모델 앙상블 테스트 생략')
        print('  → model41 실행 후 재시도: python src/run_model43_nontree.py')

    # ── 최종 요약 ──
    elapsed_total = (time.time() - t_total) / 60
    print(f'\n{"="*60}')
    print('[model43 최종 요약]')
    print(f'{"="*60}')
    print(f'  {"모델":12s}  {"OOF MAE":10s}  {"pred_std":10s}  {"앙상블"}')
    for name, ok, mae, oof_v, _ in [
        ('MLP',        mlp_ok,   mlp_mae,   mlp_oof,   None),
        ('Ridge',      ridge_ok, ridge_mae, ridge_oof, None),
        ('ElasticNet', en_ok,    en_mae,    en_oof,    None),
    ]:
        flag = '✅' if ok else '❌'
        print(f'  {name:12s}  {mae:.4f}      {np.std(oof_v):.2f}        {flag}')
    print(f'\n  기준: LGBM-모델 상관 < {DIVERSITY_THRESHOLD}  AND  MAE < {MAE_THRESHOLD}')
    print(f'  총 소요 시간: {elapsed_total:.1f}분')

    # ── model11 비교 ──
    print(f'\n[model11 대비 MLP 비교]')
    print(f'  model11 MLP v1 (lag/roll 포함, early_stop): iter=31, OOF=9.8659')
    print(f'  model11 MLP v2 (lag/roll 포함, 300iter):    OOF=12.7, std=72.18 ← 참패')
    print(f'  model43 MLP    (lag/roll 제외):             OOF={mlp_mae:.4f}, std={np.std(mlp_oof):.2f}')
    corr_str = ', '.join([f'{k}:{v:.4f}' for k,v in mlp_corrs.items()])
    print(f'  model43 MLP 상관: {corr_str}')
