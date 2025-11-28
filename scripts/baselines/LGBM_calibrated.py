# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 09:57:51 2025

@author: T
"""

"""
Script: LGBM_calibrated.py
Location: scripts/baselines/LGBM_calibrated.py

Description:
    Implements the LightGBM baseline with Isotonic Calibration.
    
    Pipeline:
    1. Preprocessing: Parsing logs and extracting rolling behavioral features.
    2. Splitting: Strict User-level split (60% Train, 20% Val, 20% Test).
    3. Training: Fits LightGBM on the Training set (with early stopping on Val).
    4. Calibration: Fits Isotonic Regression on the Validation set probabilities.
    5. Evaluation: Runs 3 independent trials and reports Mean ± Std for Table 1 metrics.

Usage:
    Run from the project root:
    python scripts/baselines/LGBM_calibrated.py
"""

import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb

# --- Configuration & Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "slam.listen.train")

COLUMN_NAMES = [
    'user', 'exercise_id', 'instance_id', 'token', 'client',
    'countries', 'session', 'days', 'time', 'label'
]

# --- Helper Functions ---

def compute_ece(y_true, y_prob, n_bins=10):
    """Calculates Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_weight = np.sum(mask) / total_samples
            ece += np.abs(bin_acc - bin_conf) * bin_weight
            
    return ece

def parse_countries(x):
    """Parses list string to extract country code."""
    try:
        arr = ast.literal_eval(x)
        if isinstance(arr, list) and len(arr) > 0:
            return arr[0]
        else:
            return 'Unknown'
    except Exception:
        return 'Unknown'

def run_experiment(run_id):
    """Runs a single experiment iteration (Train -> Calibrate -> Test)."""
    print(f"\n--- Starting Run {run_id+1}/3 ---")
    
    # 1. Load Data
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data not found at: {RAW_DATA_PATH}")
    
    df = pd.read_csv(RAW_DATA_PATH, header=None, names=COLUMN_NAMES, sep='\t', dtype=str)

    # 2. Preprocessing
    df['days'] = pd.to_numeric(df['days'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['country'] = df['countries'].apply(parse_countries)

    df['token_sequence_order'] = (
        df['instance_id'].str[-2:]
        .apply(lambda s: pd.to_numeric(s, errors='coerce'))
        .fillna(0).astype(int)
    )
    
    df['sentence_length'] = df.groupby('exercise_id')['token'].transform('count')

    # Rolling History Accuracy
    df = df.sort_values(by=['user', 'days', 'time', 'token_sequence_order']).reset_index(drop=True)
    df['label_float'] = pd.to_numeric(df['label'], errors='coerce').fillna(0.0)
    
    df['cum_correct'] = df.groupby('user')['label_float'].cumsum()
    df['cum_total'] = df.groupby('user').cumcount() + 1
    df['user_history_accuracy'] = (df['cum_correct'] - df['label_float']) / (df['cum_total'] - 1)
    df['user_history_accuracy'] = df['user_history_accuracy'].fillna(0.5)

    # Fill Missing Values
    for col in ['days', 'time']:
        if df[col].isnull().any(): df[col] = df[col].fillna(df[col].median())
    for col in ['client', 'country', 'session']:
        df[col] = df[col].fillna('Unknown')

    # 3. Splitting (Dynamic Seed)
    unique_users = df['user'].unique()
    current_seed = 42 + run_id
    np.random.seed(current_seed)
    np.random.shuffle(unique_users)

    n_users = len(unique_users)
    n_train = int(n_users * 0.6)
    n_val = int(n_users * 0.2)
    
    train_users = unique_users[:n_train]
    val_users = unique_users[n_train : n_train + n_val]
    test_users = unique_users[n_train + n_val:]

    train_df = df[df['user'].isin(train_users)].reset_index(drop=True)
    val_df = df[df['user'].isin(val_users)].reset_index(drop=True)
    test_df = df[df['user'].isin(test_users)].reset_index(drop=True)

    # 4. Feature Prep
    numeric_features = ['days', 'time', 'token_sequence_order', 'sentence_length', 'user_history_accuracy']
    categorical_features = ['client', 'country', 'session']
    
    y_train = train_df['label_float'].values
    y_val = val_df['label_float'].values
    y_test = test_df['label_float'].values

    # Convert categoricals to 'category' dtype for LGBM
    for col in categorical_features:
        train_df[col] = train_df[col].astype('category')
        val_df[col] = val_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

    # Scale numerics
    scaler = StandardScaler()
    train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
    val_df[numeric_features] = scaler.transform(val_df[numeric_features])
    test_df[numeric_features] = scaler.transform(test_df[numeric_features])

    X_train = train_df[numeric_features + categorical_features]
    X_val = val_df[numeric_features + categorical_features]
    X_test = test_df[numeric_features + categorical_features]

    # 5. Train LightGBM
    # Calculate scale_pos_weight for class imbalance
    pos_count = np.sum(y_train == 1)
    neg_count = len(y_train) - pos_count
    scale_weight = neg_count / pos_count if pos_count > 0 else 1

    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=current_seed,
        scale_pos_weight=scale_weight,
        n_jobs=-1
    )
    
    # Use Validation set for Early Stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)
    ]
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=callbacks,
        categorical_feature=categorical_features
    )

    # 6. Train Calibration Model
    # Get uncalibrated probabilities on Validation set
    prob_val_raw = lgb_model.predict_proba(X_val)[:, 1]
    
    iso_calibrator = IsotonicRegression(out_of_bounds='clip')
    iso_calibrator.fit(prob_val_raw, y_val)

    # 7. Final Evaluation
    prob_test_raw = lgb_model.predict_proba(X_test)[:, 1]
    prob_test_cal = iso_calibrator.predict(prob_test_raw)
    y_pred_raw = (prob_test_raw >= 0.5).astype(int)

    metrics = {
        'AUC': roc_auc_score(y_test, prob_test_raw),
        'F1 Score': f1_score(y_test, y_pred_raw),
        'ECE (Raw)': compute_ece(y_test, prob_test_raw),
        'ECE (Calibrated)': compute_ece(y_test, prob_test_cal)
    }
    
    print(f"Run {run_id+1} Complete: AUC={metrics['AUC']:.4f}, ECE(Cal)={metrics['ECE (Calibrated)']:.4f}")
    return metrics

def main():
    N_RUNS = 3
    all_results = []
    
    print(f"Running LightGBM Baseline for {N_RUNS} independent runs...")
    
    for i in range(N_RUNS):
        res = run_experiment(i)
        all_results.append(res)

    # Output formatted for Table 1
    target_columns = ['AUC', 'F1 Score', 'ECE (Raw)', 'ECE (Calibrated)']
    
    print("\n" + "="*80)
    print("   LightGBM Results (Formatted for Table 1)   ")
    print("="*80)
    print(f"{'Metric':<20} | {'Mean':<10} | {'Std':<10} | {'Format: Mean ± Std'}")
    print("-" * 80)
    
    for col in target_columns:
        values = [r[col] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        formatted_str = f"{mean_val:.3f} ± {std_val:.3f}"
        print(f"{col:<20} | {mean_val:.4f}     | {std_val:.4f}     | {formatted_str}")
    print("="*80)

if __name__ == "__main__":
    main()