"""
Experiment: Two-Stage Scenario Clustering + Specialized Stacking
Date: 2026-04-17

Key idea: Separate scenarios into clusters based on scenario-level aggregates,
then train specialized 5-model stacking pipelines per cluster.
Atomic analysis found 63.4% of variance is between-scenario.

Clustering strategies:
1. KMeans with K=3, 4, 5 on scenario-level features
2. Binary median split (high delay vs low delay scenarios)

For each strategy, train per-cluster stacking (LGBM + TW1.8 + CB + ET + RF → LGBM meta).
Merge predictions and compute overall OOF MAE.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features, get_feature_cols

import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = '/Users/junhee/Desktop/DACON_Monthly_Hackerton/Smart-Warehouse-Delay-Prediction/data'
CKPT_DIR = '/Users/junhee/Desktop/DACON_Monthly_Hackerton/Smart-Warehouse-Delay-Prediction/docs/model25_ckpt'
SUBMISSION_DIR = '/Users/junhee/Desktop/DACON_Monthly_Hackerton/Smart-Warehouse-Delay-Prediction/submissions'

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Scenario aggregation columns (18 features)
SC_AGG_COLS = [
    'robot_utilization', 'order_inflow_15m', 'low_battery_ratio',
    'congestion_score', 'max_zone_density', 'charge_queue_length',
    'battery_mean', 'battery_std', 'robot_idle', 'robot_active',
    'robot_charging', 'near_collision_15m', 'fault_count_15m',
    'avg_recovery_time', 'blocked_path_15m', 'sku_concentration',
    'urgent_order_ratio', 'pack_utilization',
]

# Model parameters (same as model21)
LGBM_PARAMS = {
    'num_leaves': 181, 'learning_rate': 0.020616,
    'feature_fraction': 0.5122, 'bagging_fraction': 0.9049,
    'min_child_samples': 26, 'reg_alpha': 0.3805, 'reg_lambda': 0.3630,
    'objective': 'regression_l1', 'n_estimators': 3000,
    'bagging_freq': 1, 'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
}

TW18_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3.0,
    'loss_function': 'Tweedie:variance_power=1.8',
    'random_seed': 42, 'verbose': 0, 'early_stopping_rounds': 50,
}

CB_PARAMS = {
    'iterations': 3000, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3.0,
    'loss_function': 'MAE', 'random_seed': 42, 'verbose': 0, 'early_stopping_rounds': 50,
}

ET_PARAMS = {
    'n_estimators': 500, 'max_features': 0.5, 'min_samples_leaf': 26,
    'n_jobs': -1, 'random_state': 42
}

RF_PARAMS = {
    'n_estimators': 500, 'max_features': 0.33, 'min_samples_leaf': 26,
    'n_jobs': -1, 'random_state': 42
}

META_LGBM_PARAMS = {
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'objective': 'regression_l1', 'n_estimators': 500,
    'bagging_freq': 1, 'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
}

MIN_CLUSTER_ROWS = 500  # Merge clusters with fewer rows

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_data():
    """Load train and test data with FE v1 pipeline + scenario aggregates."""
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))

    # FE v1 파이프라인
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from feature_engineering import build_features
    train, test = build_features(train, test, layout,
                                  lag_lags=[1,2,3,4,5,6],
                                  rolling_windows=[3,5,10])

    # 시나리오 집계 피처
    train = add_scenario_agg_features(train)
    test = add_scenario_agg_features(test)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test

def add_scenario_agg_features(df):
    """Add scenario-level aggregation features to the dataframe."""
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

def build_scenario_feature_matrix(df, target_col='avg_delay_minutes_next_30m'):
    """
    For each scenario_id in df, compute one-row aggregation from SC_AGG_COLS.
    Returns scenario_feature_matrix (n_scenarios, len(SC_AGG_COLS)),
            scenario_ids (list of scenario_id),
            scenario_targets (mean delay per scenario for binary split, None if target absent)
    """
    agg_cols = [c for c in SC_AGG_COLS if c in df.columns]
    if target_col in df.columns:
        scenario_agg = df.groupby('scenario_id')[agg_cols + [target_col]].agg('mean')
        scenario_targets = scenario_agg[target_col].values
    else:
        scenario_agg = df.groupby('scenario_id')[agg_cols].agg('mean')
        scenario_targets = None
    scenario_ids = scenario_agg.index.tolist()
    scenario_features = scenario_agg[agg_cols].fillna(0).values
    return scenario_features, scenario_ids, scenario_targets

def scenario_id_to_cluster_mapping(train_df, test_df, cluster_labels, scenario_ids):
    """
    Given cluster labels for train scenarios, return dict mapping scenario_id → cluster.
    For test scenarios, we'll assign them using the same KMeans model.
    """
    cluster_map = {sid: label for sid, label in zip(scenario_ids, cluster_labels)}
    return cluster_map

def assign_scenarios_to_clusters(df, cluster_map):
    """Assign each row in df to a cluster based on its scenario_id."""
    return df['scenario_id'].map(cluster_map).fillna(-1).astype(int)

def merge_small_clusters(df, cluster_col, min_size=MIN_CLUSTER_ROWS):
    """
    If a cluster has fewer than min_size rows, merge it with the nearest cluster.
    Returns updated df with adjusted cluster assignments.
    """
    df = df.copy()
    cluster_sizes = df[cluster_col].value_counts()
    small_clusters = cluster_sizes[cluster_sizes < min_size].index.tolist()

    if small_clusters:
        print(f"[Merge] Small clusters (< {min_size}): {small_clusters}")
        for small_c in small_clusters:
            # Find nearest cluster (arbitrary: merge with smallest nearby)
            large_clusters = [c for c in df[cluster_col].unique() if c not in small_clusters]
            if large_clusters:
                nearest = large_clusters[0]
                df.loc[df[cluster_col] == small_c, cluster_col] = nearest
                print(f"  Merged cluster {small_c} → {nearest}")

    return df

# ============================================================================
# 5-MODEL STACKING
# ============================================================================

def train_5model_stacking(X_train, y_train, X_test, groups, feature_names, cluster_id=0):
    """
    Train 5-model stacking (LGBM, TW1.8, CB, ET, RF → LGBM meta).
    Returns (oof_meta, test_pred_meta, models_dict, meta_model, fold_scores)
    """
    print(f"\n[Cluster {cluster_id}] Training 5-model stacking on {len(X_train)} rows")

    n_rows = len(X_train)
    n_models = 5

    # Initialize OOF arrays
    oof_lgbm = np.zeros(n_rows)
    oof_tw = np.zeros(n_rows)
    oof_cb = np.zeros(n_rows)
    oof_et = np.zeros(n_rows)
    oof_rf = np.zeros(n_rows)

    # Test predictions (accumulated across folds)
    test_lgbm = np.zeros(len(X_test))
    test_tw = np.zeros(len(X_test))
    test_cb = np.zeros(len(X_test))
    test_et = np.zeros(len(X_test))
    test_rf = np.zeros(len(X_test))

    models_dict = {
        'lgbm': [], 'tw': [], 'cb': [], 'et': [], 'rf': []
    }

    gkf = GroupKFold(n_splits=5)
    fold_count = 0

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
        print(f"  Fold {fold+1}/5...")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # ---- LGBM (log1p space) ----
        y_tr_lg = np.log1p(y_tr)
        lgbm_model = lgb.LGBMRegressor(**LGBM_PARAMS)
        lgbm_model.fit(X_tr, y_tr_lg, eval_set=[(X_val, np.log1p(y_val))],
                       eval_metric='mae', callbacks=[lgb.early_stopping(50)])
        oof_lgbm[val_idx] = np.expm1(lgbm_model.predict(X_val))
        test_lgbm += np.expm1(lgbm_model.predict(X_test)) / 5
        models_dict['lgbm'].append(lgbm_model)

        # ---- Tweedie 1.8 (raw space) ----
        tw_model = cb.CatBoostRegressor(**TW18_PARAMS)
        tw_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_tw[val_idx] = tw_model.predict(X_val)
        test_tw += tw_model.predict(X_test) / 5
        models_dict['tw'].append(tw_model)

        # ---- CatBoost (log1p space) ----
        y_tr_lg = np.log1p(y_tr)
        cb_model = cb.CatBoostRegressor(**CB_PARAMS)
        cb_model.fit(X_tr, y_tr_lg, eval_set=[(X_val, np.log1p(y_val))], verbose=False)
        oof_cb[val_idx] = np.expm1(cb_model.predict(X_val))
        test_cb += np.expm1(cb_model.predict(X_test)) / 5
        models_dict['cb'].append(cb_model)

        # ---- ExtraTrees (log1p space) ----
        y_tr_lg = np.log1p(y_tr)
        et_model = ExtraTreesRegressor(**ET_PARAMS)
        et_model.fit(X_tr, y_tr_lg)
        oof_et[val_idx] = np.expm1(et_model.predict(X_val))
        test_et += np.expm1(et_model.predict(X_test)) / 5
        models_dict['et'].append(et_model)

        # ---- RandomForest (log1p space) ----
        y_tr_lg = np.log1p(y_tr)
        rf_model = RandomForestRegressor(**RF_PARAMS)
        rf_model.fit(X_tr, y_tr_lg)
        oof_rf[val_idx] = np.expm1(rf_model.predict(X_val))
        test_rf += np.expm1(rf_model.predict(X_test)) / 5
        models_dict['rf'].append(rf_model)

        fold_count += 1

    # ---- Meta-model: LGBM ----
    meta_X_train = np.column_stack([
        oof_lgbm,
        oof_cb,
        np.log1p(np.maximum(oof_tw, 0)),
        oof_et,
        oof_rf
    ])
    meta_X_test = np.column_stack([
        test_lgbm,
        test_cb,
        np.log1p(np.maximum(test_tw, 0)),
        test_et,
        test_rf
    ])

    # Train meta-model on full training OOF
    y_meta_lg = np.log1p(y_train)
    meta_model = lgb.LGBMRegressor(**META_LGBM_PARAMS)
    meta_model.fit(meta_X_train, y_meta_lg)

    oof_meta = np.expm1(meta_model.predict(meta_X_train))
    test_meta = np.expm1(meta_model.predict(meta_X_test))

    oof_mae = mean_absolute_error(y_train, oof_meta)
    print(f"  [Cluster {cluster_id}] OOF MAE: {oof_mae:.4f}")

    return oof_meta, test_meta, models_dict, meta_model

# ============================================================================
# CLUSTERING STRATEGIES
# ============================================================================

def strategy_kmeans(train_df, test_df, k=3):
    """KMeans clustering with k clusters."""
    print(f"\n[Strategy] KMeans K={k}")

    # Build scenario feature matrix from train data
    train_sc_features, train_scenario_ids, train_scenario_targets = build_scenario_feature_matrix(train_df)
    test_sc_features, test_scenario_ids, test_scenario_targets = build_scenario_feature_matrix(test_df)

    # Fit KMeans on train scenario features
    scaler = StandardScaler()
    train_sc_features_scaled = scaler.fit_transform(train_sc_features)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    train_cluster_labels = kmeans.fit_predict(train_sc_features_scaled)

    # Predict clusters for test scenarios
    test_sc_features_scaled = scaler.transform(test_sc_features)
    test_cluster_labels = kmeans.predict(test_sc_features_scaled)

    # Build scenario → cluster mapping
    train_cluster_map = {sid: label for sid, label in zip(train_scenario_ids, train_cluster_labels)}
    test_cluster_map = {sid: label for sid, label in zip(test_scenario_ids, test_cluster_labels)}

    return train_cluster_map, test_cluster_map, kmeans, scaler

def strategy_median_split(train_df, test_df):
    """Binary split: high delay vs low delay scenarios.
    Uses LGBM classifier to assign test scenarios (no target leak)."""
    from sklearn.preprocessing import StandardScaler as SS2
    print(f"\n[Strategy] Binary Median Split")

    # Build scenario feature matrix
    train_sc_features, train_scenario_ids, train_scenario_targets = build_scenario_feature_matrix(train_df)
    test_sc_features, test_scenario_ids, _ = build_scenario_feature_matrix(test_df)

    # Compute median delay on train scenarios
    median_delay = np.median(train_scenario_targets)
    print(f"  Median scenario delay: {median_delay:.2f}")

    # Assign train clusters by target
    train_labels = np.array([0 if t <= median_delay else 1 for t in train_scenario_targets])
    train_cluster_map = {
        sid: label for sid, label in zip(train_scenario_ids, train_labels)
    }

    # For test: train LGBM classifier on train scenario features → predict test clusters
    scaler = SS2()
    X_train_sc = scaler.fit_transform(train_sc_features)
    X_test_sc = scaler.transform(test_sc_features)

    clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, random_state=42, verbosity=-1)
    clf.fit(X_train_sc, train_labels)
    test_labels = clf.predict(X_test_sc)
    print(f"  Test cluster distribution: {np.bincount(test_labels)}")

    test_cluster_map = {
        sid: int(label) for sid, label in zip(test_scenario_ids, test_labels)
    }

    return train_cluster_map, test_cluster_map, clf, scaler

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Main experiment: test all clustering strategies."""
    print("="*80)
    print("EXPERIMENT: Two-Stage Scenario Clustering + Specialized Stacking")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load data (includes FE v1 + scenario aggregates)
    train_df, test_df = load_data()

    feature_cols = [c for c in train_df.columns
                    if c not in {'ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m'}
                    and train_df[c].dtype != object]
    print(f"Total features: {len(feature_cols)}")

    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)
    y_train = train_df['avg_delay_minutes_next_30m']

    results = {}

    # ========== Test KMeans K=3, 4, 5 ==========
    for k in [3, 4, 5]:
        strategy_name = f"kmeans_k{k}"
        print(f"\n{'='*80}")
        print(f"Testing strategy: {strategy_name}")
        print(f"{'='*80}")

        train_cluster_map, test_cluster_map, kmeans_model, scaler = strategy_kmeans(train_df, test_df, k=k)

        # Assign clusters
        train_df_tmp = train_df.copy()
        test_df_tmp = test_df.copy()
        train_df_tmp['cluster'] = assign_scenarios_to_clusters(train_df_tmp, train_cluster_map)
        test_df_tmp['cluster'] = assign_scenarios_to_clusters(test_df_tmp, test_cluster_map)

        # Merge small clusters
        train_df_tmp = merge_small_clusters(train_df_tmp, 'cluster')
        test_df_tmp = merge_small_clusters(test_df_tmp, 'cluster')

        # Print cluster sizes
        cluster_sizes_train = train_df_tmp['cluster'].value_counts().sort_index()
        cluster_sizes_test = test_df_tmp['cluster'].value_counts().sort_index()
        print(f"Train cluster sizes: {dict(cluster_sizes_train)}")
        print(f"Test cluster sizes: {dict(cluster_sizes_test)}")

        # Per-cluster stacking
        oof_predictions = np.zeros(len(train_df))
        test_predictions = np.zeros(len(test_df))

        for cluster_id in sorted(train_df_tmp['cluster'].unique()):
            train_mask = (train_df_tmp['cluster'] == cluster_id).values
            test_mask = (test_df_tmp['cluster'] == cluster_id).values

            X_train_c = X_train[train_mask]
            X_test_c = X_test[test_mask]
            y_train_c = y_train[train_mask]
            groups_c = train_df_tmp[train_mask]['scenario_id'].values

            oof_c, test_c, _, _ = train_5model_stacking(
                X_train_c, y_train_c, X_test_c, groups_c, feature_cols, cluster_id=cluster_id
            )

            oof_predictions[train_mask] = oof_c
            test_predictions[test_mask] = test_c

        overall_oof_mae = mean_absolute_error(y_train, oof_predictions)
        print(f"\n[{strategy_name}] Overall OOF MAE: {overall_oof_mae:.4f}")

        results[strategy_name] = {
            'oof_mae': overall_oof_mae,
            'oof_predictions': oof_predictions,
            'test_predictions': test_predictions,
            'train_cluster_map': train_cluster_map,
            'test_cluster_map': test_cluster_map,
        }

    # ========== Test Median Split ==========
    strategy_name = "median_split"
    print(f"\n{'='*80}")
    print(f"Testing strategy: {strategy_name}")
    print(f"{'='*80}")

    train_cluster_map, test_cluster_map, _, _ = strategy_median_split(train_df, test_df)

    train_df_tmp = train_df.copy()
    test_df_tmp = test_df.copy()
    train_df_tmp['cluster'] = assign_scenarios_to_clusters(train_df_tmp, train_cluster_map)
    test_df_tmp['cluster'] = assign_scenarios_to_clusters(test_df_tmp, test_cluster_map)

    train_df_tmp = merge_small_clusters(train_df_tmp, 'cluster')
    test_df_tmp = merge_small_clusters(test_df_tmp, 'cluster')

    cluster_sizes_train = train_df_tmp['cluster'].value_counts().sort_index()
    cluster_sizes_test = test_df_tmp['cluster'].value_counts().sort_index()
    print(f"Train cluster sizes: {dict(cluster_sizes_train)}")
    print(f"Test cluster sizes: {dict(cluster_sizes_test)}")

    oof_predictions = np.zeros(len(train_df))
    test_predictions = np.zeros(len(test_df))

    for cluster_id in sorted(train_df_tmp['cluster'].unique()):
        train_mask = (train_df_tmp['cluster'] == cluster_id).values
        test_mask = (test_df_tmp['cluster'] == cluster_id).values

        X_train_c = X_train[train_mask]
        X_test_c = X_test[test_mask]
        y_train_c = y_train[train_mask]
        groups_c = train_df_tmp[train_mask]['scenario_id'].values

        oof_c, test_c, _, _ = train_5model_stacking(
            X_train_c, y_train_c, X_test_c, groups_c, feature_cols, cluster_id=cluster_id
        )

        oof_predictions[train_mask] = oof_c
        test_predictions[test_mask] = test_c

    overall_oof_mae = mean_absolute_error(y_train, oof_predictions)
    print(f"\n[{strategy_name}] Overall OOF MAE: {overall_oof_mae:.4f}")

    results[strategy_name] = {
        'oof_mae': overall_oof_mae,
        'oof_predictions': oof_predictions,
        'test_predictions': test_predictions,
        'train_cluster_map': train_cluster_map,
        'test_cluster_map': test_cluster_map,
    }

    # ========== Summary ==========
    print(f"\n{'='*80}")
    print("SUMMARY - All Strategies")
    print(f"{'='*80}")

    results_summary = sorted(results.items(), key=lambda x: x[1]['oof_mae'])
    for i, (strategy, metrics) in enumerate(results_summary):
        print(f"{i+1}. {strategy:20s} → OOF MAE: {metrics['oof_mae']:.4f}")

    best_strategy = results_summary[0][0]
    best_result = results_summary[0][1]
    best_oof_mae = best_result['oof_mae']

    print(f"\nBEST STRATEGY: {best_strategy} (OOF MAE: {best_oof_mae:.4f})")

    # ========== Generate Submission ==========
    print(f"\nGenerating submission for best strategy: {best_strategy}")
    test_predictions_final = best_result['test_predictions']

    sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sample['avg_delay_minutes_next_30m'] = np.maximum(test_predictions_final, 0)

    submission_path = os.path.join(SUBMISSION_DIR, f'model25_{best_strategy}.csv')
    sample.to_csv(submission_path, index=False)
    print(f"Submission saved: {submission_path}")

    # ========== Save artifacts ==========
    print(f"\nSaving artifacts to {CKPT_DIR}")

    artifacts = {
        'results': results,
        'best_strategy': best_strategy,
        'best_oof_mae': best_oof_mae,
        'timestamp': datetime.now().isoformat(),
    }

    with open(os.path.join(CKPT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump(artifacts, f)

    # Save summary CSV
    summary_df = pd.DataFrame([
        {'strategy': strategy, 'oof_mae': metrics['oof_mae']}
        for strategy, metrics in results.items()
    ]).sort_values('oof_mae')
    summary_df.to_csv(os.path.join(CKPT_DIR, 'summary.csv'), index=False)
    print(f"Summary saved to {CKPT_DIR}/summary.csv")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")

if __name__ == '__main__':
    run_experiment()
