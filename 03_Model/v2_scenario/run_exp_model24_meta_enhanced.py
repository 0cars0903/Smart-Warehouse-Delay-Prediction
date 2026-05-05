"""
Model24: Enhanced Meta Learner with Scenario Aggregate Features

This experiment augments model21's meta learner by adding scenario-level aggregate features
as additional meta inputs (beyond just the 5 OOF predictions). This allows the meta learner
to learn scenario-specific model weights.

Key Ideas:
- Reuse model21 base learner checkpoints (no retraining)
- Add scenario aggregate features (18 sc_*_mean or all 90 sc_* features)
- Compare 4 meta variants: LGBM(sc_mean), LGBM(all_sc), CatBoost(sc_mean), XGBoost(sc_mean)
- Baseline: model21 meta CV 8.5097

Architecture:
- Load OOF/test from: docs/model21_ckpt/
- Save meta results: docs/model24_ckpt/
- GroupKFold(5) by scenario_id
- Best variant → submissions/model24_meta_enhanced.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DOCS_DIR = PROJECT_ROOT / 'docs'
CKPT_DIR_MODEL21 = DOCS_DIR / 'model21_ckpt'
CKPT_DIR_MODEL24 = DOCS_DIR / 'model24_ckpt'
SUBMISSION_DIR = PROJECT_ROOT / 'submissions'

CKPT_DIR_MODEL24.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from feature_engineering import build_features, get_feature_cols

print(f"[MODEL24] Project root: {PROJECT_ROOT}")
print(f"[MODEL24] Model21 checkpoint dir: {CKPT_DIR_MODEL21}")
print(f"[MODEL24] Model24 checkpoint dir: {CKPT_DIR_MODEL24}")

# ============================================================================
# Scenario Aggregate Features (same as model21)
# ============================================================================
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

def add_scenario_agg_features(df):
    """Add scenario-level aggregate features (mean, std, max, min, diff)."""
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]
        df[f'sc_{col}_mean'] = grp.transform('mean')
        df[f'sc_{col}_std'] = grp.transform('std').fillna(0)
        df[f'sc_{col}_max'] = grp.transform('max')
        df[f'sc_{col}_min'] = grp.transform('min')
        df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']
    return df

# ============================================================================
# Load Data
# ============================================================================
print("\n[MODEL24] Loading data...")
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
layout = pd.read_csv(DATA_DIR / 'layout_info.csv')

# Build features (FE v1 pipeline)
print("[MODEL24] Building features...")
train, test = build_features(train, test, layout,
                              lag_lags=[1,2,3,4,5,6],
                              rolling_windows=[3,5,10])

# Add scenario aggregates
print("[MODEL24] Adding scenario aggregate features...")
train = add_scenario_agg_features(train)
test = add_scenario_agg_features(test)

print(f"[MODEL24] Train shape: {train.shape}, Test shape: {test.shape}")
print(f"[MODEL24] Scenario aggregate columns created: {[c for c in train.columns if c.startswith('sc_')][:5]}... (total {len([c for c in train.columns if c.startswith('sc_')])})")

# ============================================================================
# Load Model21 Checkpoints
# ============================================================================
print("\n[MODEL24] Loading model21 checkpoints...")

# Load OOF predictions
# model21 saves as {name}_oof.npy / {name}_test.npy
oof_lgbm = np.load(CKPT_DIR_MODEL21 / 'lgbm_oof.npy')
oof_tw = np.load(CKPT_DIR_MODEL21 / 'tw18_oof.npy')
oof_cb = np.load(CKPT_DIR_MODEL21 / 'cb_oof.npy')
oof_et = np.load(CKPT_DIR_MODEL21 / 'et_oof.npy')
oof_rf = np.load(CKPT_DIR_MODEL21 / 'rf_oof.npy')

# Load test predictions
test_lgbm = np.load(CKPT_DIR_MODEL21 / 'lgbm_test.npy')
test_tw = np.load(CKPT_DIR_MODEL21 / 'tw18_test.npy')
test_cb = np.load(CKPT_DIR_MODEL21 / 'cb_test.npy')
test_et = np.load(CKPT_DIR_MODEL21 / 'et_test.npy')
test_rf = np.load(CKPT_DIR_MODEL21 / 'rf_test.npy')

print(f"[MODEL24] OOF shapes: LGBM={oof_lgbm.shape}, TW={oof_tw.shape}, CB={oof_cb.shape}, ET={oof_et.shape}, RF={oof_rf.shape}")
print(f"[MODEL24] Test shapes: LGBM={test_lgbm.shape}, TW={test_tw.shape}, CB={test_cb.shape}, ET={test_et.shape}, RF={test_rf.shape}")

# Target
y_raw = train['avg_delay_minutes_next_30m'].values
y_train = y_raw  # raw space for MAE computation
y_test = None  # For submission

# ============================================================================
# Prepare Scenario Aggregate Features
# ============================================================================
sc_mean_cols = [f'sc_{col}_mean' for col in SC_AGG_COLS]
sc_all_cols = sorted([c for c in train.columns if c.startswith('sc_')])

print(f"\n[MODEL24] Scenario aggregate features:")
print(f"  - sc_mean columns (18): {len(sc_mean_cols)}")
print(f"  - all sc_* columns (all variants): {len(sc_all_cols)}")

# Ensure columns exist
for col in sc_mean_cols:
    if col not in train.columns:
        print(f"[WARNING] {col} not in train, filling with 0")
        train[col] = 0
        test[col] = 0

# ============================================================================
# GroupKFold Setup
# ============================================================================
gkf = GroupKFold(n_splits=5)
groups = train['scenario_id'].values

print(f"\n[MODEL24] GroupKFold with {gkf.n_splits} splits by scenario_id")

# ============================================================================
# Meta Learner Hyperparameters
# ============================================================================
META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
}

META_CB_PARAMS = {
    'iterations': 500, 'learning_rate': 0.05,
    'depth': 4, 'l2_leaf_reg': 3.0,
    'loss_function': 'MAE',
    'random_seed': 42, 'verbose': False,
}

META_XGB_PARAMS = {
    'n_estimators': 500, 'learning_rate': 0.05,
    'max_depth': 4, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'reg:absoluteerror',
    'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
}

# ============================================================================
# Train Meta Variants in 5-Fold CV
# ============================================================================

variants = {
    'meta_v1': {
        'desc': '5 OOF + 18 sc_mean features (LGBM meta)',
        'sc_cols': sc_mean_cols,
        'meta_model': 'lgbm',
        'results': {'oof': [], 'test': [], 'maes': []},
    },
    'meta_v2': {
        'desc': '5 OOF + 90 sc_* features (LGBM meta)',
        'sc_cols': sc_all_cols,
        'meta_model': 'lgbm',
        'results': {'oof': [], 'test': [], 'maes': []},
    },
    'meta_v3': {
        'desc': '5 OOF + 18 sc_mean features (CatBoost meta)',
        'sc_cols': sc_mean_cols,
        'meta_model': 'cb',
        'results': {'oof': [], 'test': [], 'maes': []},
    },
    'meta_v4': {
        'desc': '5 OOF + 18 sc_mean features (XGBoost meta)',
        'sc_cols': sc_mean_cols,
        'meta_model': 'xgb',
        'results': {'oof': [], 'test': [], 'maes': []},
    },
}

print("\n[MODEL24] Training 4 meta variants in 5-fold CV...")

for variant_name, variant_config in variants.items():
    print(f"\n{'='*70}")
    print(f"[MODEL24] {variant_name}: {variant_config['desc']}")
    print(f"{'='*70}")

    sc_cols = variant_config['sc_cols']
    meta_model_type = variant_config['meta_model']

    # Meta input features
    sc_train_values = train[sc_cols].fillna(0).values
    sc_test_values = test[sc_cols].fillna(0).values

    meta_train_oof = np.column_stack([
        oof_lgbm, oof_cb, np.log1p(np.maximum(oof_tw, 0)), oof_et, oof_rf
    ])
    meta_train_all = np.column_stack([meta_train_oof, sc_train_values])

    meta_test_oof = np.column_stack([
        test_lgbm, test_cb, np.log1p(np.maximum(test_tw, 0)), test_et, test_rf
    ])
    meta_test_all = np.column_stack([meta_test_oof, sc_test_values])

    print(f"[{variant_name}] Meta train input shape: {meta_train_all.shape}")
    print(f"[{variant_name}] Meta test input shape: {meta_test_all.shape}")
    print(f"[{variant_name}] Using {len(sc_cols)} scenario aggregate features")

    # CV loop
    oof_meta_pred = np.zeros(len(train))
    test_meta_pred_list = []
    fold_maes = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train, groups=groups)):
        print(f"\n  Fold {fold_idx + 1}/5:")

        # Split meta train data
        meta_train_fold = meta_train_all[train_idx]
        meta_val_fold = meta_train_all[val_idx]
        # Meta trains in log1p space (same as model21)
        y_tr_log = np.log1p(y_raw[train_idx])
        y_va_log = np.log1p(y_raw[val_idx])

        # Train meta model
        if meta_model_type == 'lgbm':
            meta_model = lgb.LGBMRegressor(**META_LGBM_PARAMS)
            meta_model.fit(meta_train_fold, y_tr_log,
                          eval_set=[(meta_val_fold, y_va_log)],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(-1)])
            val_pred = meta_model.predict(meta_val_fold)
            test_pred = meta_model.predict(meta_test_all)
            print(f"    iter={meta_model.best_iteration_}")

        elif meta_model_type == 'cb':
            meta_model = cb.CatBoostRegressor(**META_CB_PARAMS)
            train_pool = cb.Pool(meta_train_fold, y_tr_log)
            val_pool = cb.Pool(meta_val_fold, y_va_log)
            meta_model.fit(train_pool, eval_set=val_pool,
                          use_best_model=True, verbose=False)
            val_pred = meta_model.predict(meta_val_fold)
            test_pred = meta_model.predict(meta_test_all)
            print(f"    iter={meta_model.best_iteration_}")

        elif meta_model_type == 'xgb':
            meta_model = xgb.XGBRegressor(**META_XGB_PARAMS)
            meta_model.fit(meta_train_fold, y_tr_log,
                          eval_set=[(meta_val_fold, y_va_log)],
                          early_stopping_rounds=50,
                          verbose=False)
            val_pred = meta_model.predict(meta_val_fold)
            test_pred = meta_model.predict(meta_test_all)
            print(f"    iter={meta_model.best_iteration}")

        # Metrics (expm1 back to raw space for MAE)
        val_pred_raw = np.expm1(val_pred)
        y_val_raw = y_raw[val_idx]
        fold_mae = np.mean(np.abs(val_pred_raw - y_val_raw))
        fold_maes.append(fold_mae)

        print(f"    Fold MAE: {fold_mae:.4f}")

        # Store results
        oof_meta_pred[val_idx] = val_pred
        test_meta_pred_list.append(test_pred)

    # Aggregate results
    cv_mae = np.mean(fold_maes)
    cv_std = np.std(fold_maes)
    test_meta_pred = np.mean(test_meta_pred_list, axis=0)
    test_meta_pred_expm1 = np.expm1(test_meta_pred)

    variant_config['results']['oof'] = oof_meta_pred
    variant_config['results']['test'] = test_meta_pred_expm1
    variant_config['results']['maes'] = fold_maes

    print(f"\n  [{variant_name}] CV MAE: {cv_mae:.4f} ± {cv_std:.4f}")
    print(f"  Fold MAEs: {[f'{m:.4f}' for m in fold_maes]}")

# ============================================================================
# Compare Variants
# ============================================================================
print("\n" + "="*70)
print("[MODEL24] SUMMARY: All Variants")
print("="*70)

variant_summary = []
for variant_name, variant_config in variants.items():
    cv_mae = np.mean(variant_config['results']['maes'])
    cv_std = np.std(variant_config['results']['maes'])
    variant_summary.append({
        'variant': variant_name,
        'desc': variant_config['desc'],
        'cv_mae': cv_mae,
        'cv_std': cv_std,
    })

summary_df = pd.DataFrame(variant_summary)
summary_df = summary_df.sort_values('cv_mae')

print("\n" + summary_df.to_string(index=False))

best_variant_name = summary_df.iloc[0]['variant']
best_variant_config = variants[best_variant_name]
best_cv_mae = summary_df.iloc[0]['cv_mae']

print(f"\n[MODEL24] BEST VARIANT: {best_variant_name}")
print(f"[MODEL24] Best CV MAE: {best_cv_mae:.4f}")
print(f"[MODEL24] Baseline (model21): 8.5097")
print(f"[MODEL24] Improvement: {8.5097 - best_cv_mae:.4f}")

# ============================================================================
# Save Results
# ============================================================================
print("\n[MODEL24] Saving results...")

# Save variant summary
summary_df.to_csv(CKPT_DIR_MODEL24 / 'variant_summary.csv', index=False)

# Save best variant OOF and test
np.save(CKPT_DIR_MODEL24 / f'{best_variant_name}_oof.npy', best_variant_config['results']['oof'])
np.save(CKPT_DIR_MODEL24 / f'{best_variant_name}_test.npy', best_variant_config['results']['test'])

# Save all variants for comparison
for variant_name, variant_config in variants.items():
    np.save(CKPT_DIR_MODEL24 / f'{variant_name}_oof.npy', variant_config['results']['oof'])
    np.save(CKPT_DIR_MODEL24 / f'{variant_name}_test.npy', variant_config['results']['test'])

print(f"[MODEL24] Saved results to {CKPT_DIR_MODEL24}/")

# ============================================================================
# Generate Submission
# ============================================================================
print("\n[MODEL24] Generating submission...")

test_pred_best = best_variant_config['results']['test']

sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
sample['avg_delay_minutes_next_30m'] = np.maximum(test_pred_best, 0)

submission_path = SUBMISSION_DIR / f'model24_meta_enhanced_{best_variant_name}.csv'
sample.to_csv(submission_path, index=False)

print(f"[MODEL24] Submission saved to {submission_path}")
print(f"[MODEL24] Submission shape: {sample.shape}")
print(f"[MODEL24] Prediction stats: mean={sample['avg_delay_minutes_next_30m'].mean():.4f}, std={sample['avg_delay_minutes_next_30m'].std():.4f}")
print(f"[MODEL24] Submission preview:")
print(sample.head(10))

# ============================================================================
# Final Report
# ============================================================================
print("\n" + "="*70)
print("[MODEL24] FINAL REPORT")
print("="*70)
print(f"""
Experiment: Enhanced Meta Learner with Scenario Aggregate Features
Best Variant: {best_variant_name}
Best CV MAE: {best_cv_mae:.4f}

Baseline (model21 meta only): 8.5097
Improvement: {8.5097 - best_cv_mae:.4f}

Variants Tested:
  1. meta_v1: 5 OOF + 18 sc_mean (LGBM) → CV MAE {np.mean(variants['meta_v1']['results']['maes']):.4f}
  2. meta_v2: 5 OOF + 90 sc_* (LGBM) → CV MAE {np.mean(variants['meta_v2']['results']['maes']):.4f}
  3. meta_v3: 5 OOF + 18 sc_mean (CatBoost) → CV MAE {np.mean(variants['meta_v3']['results']['maes']):.4f}
  4. meta_v4: 5 OOF + 18 sc_mean (XGBoost) → CV MAE {np.mean(variants['meta_v4']['results']['maes']):.4f}

Submission: {submission_path}

Next Steps:
- Review Public LB result to assess generalization
- If meta_v1/v3/v4 improve, scenario aggregates help meta learn scenario-specific weights
- If meta_v2 (all 90 features) is best, overfitting may occur on Public LB
""")

print("[MODEL24] DONE!")
