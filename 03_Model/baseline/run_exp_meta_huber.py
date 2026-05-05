"""
run_exp_meta_huber.py  —  FE v1+Cumul 체크포인트 + Huber 메타 LGBM
===============================================================
배경:
  FE v1+Cumul (04.15): CV 8.7699, 예측 std=21.51 (FE v3 13.76 대비 대폭 개선)
  그러나 실제 std=27.4 대비 아직 6점 부족 → 메타 레벨에서 극값 예측 추가 개선 여지

가설:
  MAE(L1) 대신 Huber Loss를 메타 학습기에 사용하면
  - 극값 방향으로 학습 신호가 커져 std 압축이 완화될 수 있음
  - Huber는 delta 이하 오차는 MAE처럼, 이상은 MSE처럼 페널티 → 극값 반응성↑

구현:
  - 베이스 모델 OOF/test 예측: FE v1+Cumul 체크포인트 그대로 로드 (재학습 없음)
  - 메타 LGBM만 objective='huber' + alpha=0.9 로 교체 (delta 탐색: 5, 10, 15, 20)
  - delta가 크면 MAE에 가까워지고, 작으면 MSE에 가까워짐

비교 기준:
  FE v1+Cumul MAE 메타:  CV 8.7699 / std=21.51 / 기대 Public ~10.197
  (이 실험): CV ? / std=? / 배율 미정

체크포인트: docs/fe_v1_cumul_ckpt/  (재사용)
예상 시간: ~5분 (메타만 재학습)
===============================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

import sys
# [auto-patched] sys.path: feature_engineering 모듈은 02_FE/ 에 위치
_HERE = os.path.dirname(os.path.abspath(__file__))
_FE_DIR = os.path.abspath(os.path.join(_HERE, '../../02_FE'))
sys.path.insert(0, _FE_DIR)
from feature_engineering import (
    merge_layout, encode_categoricals, add_ts_features,
    add_lag_features, add_rolling_features, add_domain_features,
)

# ──────────────────────────────────────────────
# 설정 (FE v1+Cumul과 동일)
# ──────────────────────────────────────────────
DATA_PATH   = 'data/'
CKPT_DIR    = 'docs/fe_v1_cumul_ckpt'   # 기존 체크포인트 재사용
TARGET      = 'avg_delay_minutes_next_30m'
N_FOLDS     = 5
RANDOM_SEED = 42

KEY_COLS_V1 = [
    'low_battery_ratio', 'battery_mean', 'charge_queue_length',
    'robot_idle', 'order_inflow_15m', 'congestion_score',
    'max_zone_density', 'avg_trip_distance',
]
CUMUL_COLS = [
    'low_battery_ratio', 'battery_mean', 'charge_queue_length',
    'robot_idle', 'order_inflow_15m', 'congestion_score',
    'max_zone_density', 'fault_count_15m', 'blocked_path_15m',
]
EVENT_COLS = {'fault_count_15m', 'blocked_path_15m'}

META_MODEL_NAMES = ['lgbm', 'tw', 'cb', 'et', 'rf']


# ──────────────────────────────────────────────
# (FE v1+Cumul과 동일) Cumulative 피처 생성
# ──────────────────────────────────────────────
def add_cumulative_features(train, test):
    train_c = train.copy(); train_c['_split'] = 0
    test_c  = test.copy();  test_c['_split']  = 1
    test_c['_orig_order'] = np.arange(len(test_c))
    combined = (pd.concat([train_c, test_c], axis=0, ignore_index=True)
                  .sort_values(['scenario_id', 'ts_idx']).reset_index(drop=True))
    for col in CUMUL_COLS:
        if col not in combined.columns: continue
        shifted = combined.groupby('scenario_id')[col].shift(1)
        combined[f'{col}_cummin'] = (shifted.groupby(combined['scenario_id'])
                                            .transform(lambda x: x.expanding().min()))
        combined[f'{col}_cummax'] = (shifted.groupby(combined['scenario_id'])
                                            .transform(lambda x: x.expanding().max()))
        if col in EVENT_COLS:
            combined[f'{col}_cumsum'] = (shifted.groupby(combined['scenario_id'])
                                                .transform(lambda x: x.expanding().sum()))
        else:
            combined[f'{col}_cummean'] = (shifted.groupby(combined['scenario_id'])
                                                  .transform(lambda x: x.expanding().mean()))
    tr_out = combined[combined['_split'] == 0].drop(columns=['_split', '_orig_order'], errors='ignore')
    te_out = (combined[combined['_split'] == 1]
              .sort_values('_orig_order').drop(columns=['_split', '_orig_order']))
    return tr_out, te_out


def build_features_v1_cumul(train, test, layout):
    train, test = merge_layout(train, test, layout)
    train, test = encode_categoricals(train, test, TARGET)
    train = add_ts_features(train); test = add_ts_features(test)
    train, test = add_lag_features(train, test, key_cols=KEY_COLS_V1, lags=[1,2,3,4,5,6])
    train, test = add_rolling_features(train, test, key_cols=KEY_COLS_V1, windows=[3,5,10])
    train, test = add_cumulative_features(train, test)
    train = add_domain_features(train); test = add_domain_features(test)
    return train, test


def get_feature_cols(df):
    excl = ['ID', 'layout_id', 'scenario_id', TARGET]
    return [c for c in df.columns if c not in excl and df[c].dtype.name not in ['object', 'category']]


# ──────────────────────────────────────────────
# 체크포인트에서 OOF/test 로드
# ──────────────────────────────────────────────
def load_checkpoints(train_len, test_len):
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_dict  = {k: np.zeros(train_len)  for k in META_MODEL_NAMES}
    test_dict = {k: np.zeros(test_len)   for k in META_MODEL_NAMES}

    # 각 fold의 val 인덱스를 재구성하려면 그룹 정보 필요
    # → 체크포인트 파일에서 fold별 OOF를 직접 합산
    # 방법: fold별 oof 파일은 va_idx 예측만 저장돼 있음 → 전체 OOF는 재조립 불가
    # 대신: 각 fold 체크포인트를 그냥 로드해서 test_dict는 평균으로, oof_dict는 full 로드
    # → 실제로는 모든 fold가 완료된 경우 전체 OOF = concat of fold OOFs
    # 여기서는 간단히: fold별 OOF + test 파일 존재 확인
    all_present = True
    for name in META_MODEL_NAMES:
        for fold_i in range(N_FOLDS):
            if not os.path.exists(os.path.join(CKPT_DIR, f'{name}_fold{fold_i}.npy')):
                all_present = False
                print(f'  ⚠️  체크포인트 없음: {name}_fold{fold_i}.npy')
    if not all_present:
        raise FileNotFoundError(f'체크포인트가 완전하지 않음: {CKPT_DIR}')

    # 전체 OOF는 베이스 모델 재학습 없이 재조립하려면 fold별 val 인덱스 필요
    # → 데이터 그룹 정보를 읽어서 GroupKFold 재현
    return oof_dict, test_dict, True


# ──────────────────────────────────────────────
# 베이스 모델 (체크포인트 only) — OOF 재조립
# ──────────────────────────────────────────────
def load_base_oof_from_ckpt(y, groups):
    """체크포인트에서 fold별 OOF 재조립 (재학습 없음)"""
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_dict  = {k: np.zeros(len(y))    for k in META_MODEL_NAMES}
    test_dict = {k: np.zeros(1)         for k in META_MODEL_NAMES}  # dummy
    test_preds = {k: []                 for k in META_MODEL_NAMES}

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(np.zeros(len(y)), y, groups)):
        for name in META_MODEL_NAMES:
            ckpt     = os.path.join(CKPT_DIR, f'{name}_fold{fold_i}.npy')
            ckpt_tst = os.path.join(CKPT_DIR, f'{name}_fold{fold_i}_test.npy')
            oof_dict[name][va_idx] = np.load(ckpt)
            test_preds[name].append(np.load(ckpt_tst))

    for name in META_MODEL_NAMES:
        test_dict[name] = np.mean(test_preds[name], axis=0)
        cv_mae = mean_absolute_error(y, oof_dict[name])
        print(f'  {name.upper():<6} CV={cv_mae:.4f}')

    return oof_dict, test_dict


# ──────────────────────────────────────────────
# Huber 메타 학습기 (delta 탐색)
# ──────────────────────────────────────────────
def train_meta_huber(oof_dict, test_dict, y, groups, delta=10.0, alpha=0.9):
    gkf = GroupKFold(n_splits=N_FOLDS)
    names = list(oof_dict.keys())
    X_tr = np.column_stack([oof_dict[k]  for k in names])
    X_te = np.column_stack([test_dict[k] for k in names])

    meta_params = {
        'num_leaves': 31, 'learning_rate': 0.05,
        'n_estimators': 1000,
        'objective': 'huber',
        'alpha': alpha,          # Huber delta (LightGBM에서는 alpha 파라미터)
        'random_state': RANDOM_SEED, 'verbose': -1, 'n_jobs': -1,
    }

    oof_meta  = np.zeros(len(y))
    test_meta = np.zeros(len(X_te))
    fold_maes = []

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y, groups)):
        m = lgb.LGBMRegressor(**meta_params)
        m.fit(X_tr[tr_idx], y[tr_idx],
              eval_set=[(X_tr[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        oof_meta[va_idx] = m.predict(X_tr[va_idx])
        test_meta += m.predict(X_te) / N_FOLDS
        mae_v = mean_absolute_error(y[va_idx], oof_meta[va_idx])
        fold_maes.append(mae_v)
        print(f'  Fold {fold_i+1}: MAE={mae_v:.4f}  iter={m.best_iteration_}')

    cv_mae = mean_absolute_error(y, oof_meta)
    pred_std = np.std(oof_meta - y)
    print(f'  메타 CV MAE = {cv_mae:.4f} | residual std={pred_std:.2f} | pred std={np.std(oof_meta):.2f}')
    print(f'  Fold MAEs : {" / ".join(f"{m:.4f}" for m in fold_maes)}')
    return test_meta, cv_mae, np.std(oof_meta)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    print('=' * 60)
    print('FE v1+Cumul 체크포인트 + Huber 메타 LGBM 실험')
    print('목표: std 21.51 (MAE 메타) → 더 높이 개선')
    print('=' * 60)

    train  = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test   = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_PATH, 'layout_info.csv'))

    print('[피처 엔지니어링] FE v1+Cumul 재구성...')
    train_fe, test_fe = build_features_v1_cumul(train, test, layout)
    feat_cols = get_feature_cols(train_fe)
    y         = train_fe[TARGET].values
    groups    = train_fe['scenario_id'].values

    print(f'  피처 수: {len(feat_cols)} (체크포인트 재사용)')
    print(f'  실제 타겟 std: {np.std(y):.2f}')

    print('\n[베이스 모델] 체크포인트 로드...')
    oof_dict, test_dict = load_base_oof_from_ckpt(y, groups)

    # Huber delta 탐색
    deltas = [0.9, 0.7, 0.5]  # LightGBM Huber의 alpha는 quantile — 0.5=MAE, 0.9=Huber(default)
    # 참고: LightGBM objective='huber'에서 alpha는 quantile 파라미터가 아닌
    #        huber의 delta 역할. 기본값 alpha=0.9
    # alpha가 클수록 더 많은 샘플을 "큰 잔차"로 취급 → 극값에 더 강하게 반응

    best_result = {'cv': 999, 'std': 0, 'alpha': None, 'test': None}

    for alpha in deltas:
        print(f'\n── Huber 메타 alpha={alpha} ──')
        test_meta, cv_mae, pred_std = train_meta_huber(
            oof_dict, test_dict, y, groups, alpha=alpha
        )
        print(f'  → CV={cv_mae:.4f}, pred std={pred_std:.2f}')

        if cv_mae < best_result['cv']:
            best_result.update({'cv': cv_mae, 'std': pred_std, 'alpha': alpha, 'test': test_meta.copy()})

    # MAE 기준선 (비교용)
    print('\n── 기준선: MAE 메타 (원래 FE v1+Cumul 결과) ──')
    print('  CV=8.7699, pred std=21.51')

    print(f'\n── 최적 Huber 설정: alpha={best_result["alpha"]} ──')
    print(f'  CV={best_result["cv"]:.4f} | pred std={best_result["std"]:.2f}')
    print(f'  기대 Public (배율 1.1627): {best_result["cv"] * 1.1627:.4f}')

    submit_path = f'submissions/stacking_fe_v1_cumul_huber_a{str(best_result["alpha"]).replace(".", "")}_meta.csv'
    sub = pd.DataFrame({'ID': test_fe['ID'], TARGET: best_result['test']})
    sub.to_csv(submit_path, index=False)
    print(f'\n제출 파일 저장: {submit_path}')

    print('\n── 비교표 ──')
    print(f'  FE v1+Cumul MAE 메타:  CV 8.7699 / std=21.51 / 기대 Public ~10.197')
    print(f'  FE v1+Cumul Huber 최적: CV {best_result["cv"]:.4f} / std={best_result["std"]:.2f} / 기대 Public {best_result["cv"]*1.1627:.4f}')
    print('완료!')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
