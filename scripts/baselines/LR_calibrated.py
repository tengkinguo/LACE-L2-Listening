# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 09:44:16 2025

@author: T
"""
# -*- coding: utf-8 -*-
"""
Script: LR_calibrated.py
Location: scripts/baselines/LR_calibrated.py

Description:
    Implements the Logistic Regression baseline with Isotonic Calibration.
    
    This script is designed to reproduce the baseline metrics for Table 1 of the paper.
    It performs the following steps:
    1. Preprocessing: Parsing raw logs and extracting behavioral features (e.g., rolling accuracy).
    2. Splitting: Strict User-level split (60% Train, 20% Val, 20% Test) to prevent data leakage.
    3. Training: Fits Logistic Regression on the Training set.
    4. Calibration: Fits Isotonic Regression on the Validation set to correct probability estimates.
    5. Evaluation: Runs the experiment 3 times (with different random seeds) and reports 
       the Mean ± Std for the four required metrics: AUC, F1 Score, ECE (Raw), and ECE (Calibrated).

Usage:
    Run from the project root:
    python scripts/baselines/LR_calibrated.py
"""

import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, f1_score

# --- Configuration & Path Setup ---
# Calculate paths relative to this script location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "slam.listen.train")

# Define columns for raw TSV data (no header in original file)
COLUMN_NAMES = [
    'user', 'exercise_id', 'instance_id', 'token', 'client',
    'countries', 'session', 'days', 'time', 'label'
]

# --- Helper Functions ---

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE).
    Formula: ECE = sum( (bin_count / total_count) * abs(bin_acc - bin_conf) )
    """
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
    """Parses string representation of lists (e.g., "['CO']") to extract the country code."""
    try:
        arr = ast.literal_eval(x)
        if isinstance(arr, list) and len(arr) > 0:
            return arr[0]
        else:
            return 'Unknown'
    except Exception:
        return 'Unknown'

def run_experiment(run_id):
    """
    Runs a single complete experiment iteration.
    Returns: Dictionary containing the 4 key metrics for the Test set.
    """
    print(f"\n--- Starting Run {run_id+1}/3 ---")
    
    # 1. Load Data
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data not found at: {RAW_DATA_PATH}")
    
    # Load as string first to avoid type inference errors
    df = pd.read_csv(RAW_DATA_PATH, header=None, names=COLUMN_NAMES, sep='\t', dtype=str)

    # 2. Preprocessing & Feature Engineering
    df['days'] = pd.to_numeric(df['days'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['country'] = df['countries'].apply(parse_countries)

    # Extract token order (last 2 digits of instance_id)
    df['token_sequence_order'] = (
        df['instance_id']
        .str[-2:]
        .apply(lambda s: pd.to_numeric(s, errors='coerce'))
        .fillna(0)
        .astype(int)
    )

    # Calculate sentence length
    df['sentence_length'] = df.groupby('exercise_id')['token'].transform('count')

    # Calculate Rolling History Accuracy (Crucial: Avoid data leakage)
    # Sort by user and time to ensure past events come first
    df = df.sort_values(by=['user', 'days', 'time', 'token_sequence_order']).reset_index(drop=True)
    df['label_float'] = pd.to_numeric(df['label'], errors='coerce').fillna(0.0)
    
    # Cumulative stats
    df['cum_correct'] = df.groupby('user')['label_float'].cumsum()
    df['cum_total'] = df.groupby('user').cumcount() + 1
    
    # Accuracy up to the *previous* attempt
    df['user_history_accuracy'] = (df['cum_correct'] - df['label_float']) / (df['cum_total'] - 1)
    df['user_history_accuracy'] = df['user_history_accuracy'].fillna(0.5) # Cold start

    # Handle missing values
    for col in ['days', 'time']:
        if df[col].isnull().any(): df[col] = df[col].fillna(df[col].median())
    for col in ['client', 'country', 'session']:
        df[col] = df[col].fillna('Unknown')

    # 3. User-Level Data Splitting
    unique_users = df['user'].unique()
    
    # Dynamic seed based on run_id ensures different splits for each run
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

    # 4. Feature Preparation
    numeric_features = ['days', 'time', 'token_sequence_order', 'sentence_length', 'user_history_accuracy']
    categorical_features = ['client', 'country', 'session']
    features = numeric_features + categorical_features

    y_train = train_df['label_float'].values
    y_val = val_df['label_float'].values
    y_test = test_df['label_float'].values

    X_train = train_df[features]
    X_val = val_df[features]
    X_test = test_df[features]

    # Pipeline: Scale numerics, One-Hot encode categoricals
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
    
    # Fit on Train, transform others
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # 5. Train Logistic Regression
    lr_model = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        random_state=current_seed,
        max_iter=1000
    )
    lr_model.fit(X_train_processed, y_train)

    # 6. Fit Calibration Model (Isotonic)
    # Use Validation set probabilities to fit the calibrator
    prob_val_raw = lr_model.predict_proba(X_val_processed)[:, 1]
    iso_calibrator = IsotonicRegression(out_of_bounds='clip')
    iso_calibrator.fit(prob_val_raw, y_val)

    # 7. Final Evaluation on Test Set
    prob_test_raw = lr_model.predict_proba(X_test_processed)[:, 1]
    prob_test_cal = iso_calibrator.predict(prob_test_raw)
    
    # For F1, use default threshold of 0.5
    y_pred_raw = (prob_test_raw >= 0.5).astype(int)
    
    # Return the exact 4 metrics required for Table 1
    metrics = {
        'AUC': roc_auc_score(y_test, prob_test_raw),
        'F1 Score': f1_score(y_test, y_pred_raw),
        'ECE (Raw)': compute_ece(y_test, prob_test_raw),
        'ECE (Calibrated)': compute_ece(y_test, prob_test_cal)
    }
    
    print(f"Run {run_id+1} Complete: AUC={metrics['AUC']:.4f}, ECE(Cal)={metrics['ECE (Calibrated)']:.4f}")
    return metrics

def main():
    # Number of runs to calculate Mean +/- Std
    N_RUNS = 3
    all_results = []
    
    print(f"Running Logistic Regression Baseline for {N_RUNS} independent runs...")
    
    # Execute runs
    for i in range(N_RUNS):
        res = run_experiment(i)
        all_results.append(res)

    # Calculate Mean and Std for Table 1
    target_columns = ['AUC', 'F1 Score', 'ECE (Raw)', 'ECE (Calibrated)']
    
    print("\n" + "="*80)
    print("   Logistic Regression Results (Formatted for Table 1)   ")
    print("="*80)
    print(f"{'Metric':<20} | {'Mean':<10} | {'Std':<10} | {'Format: Mean ± Std'}")
    print("-" * 80)
    
    for col in target_columns:
        values = [r[col] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Determine strict formatting based on the paper's table style
        formatted_str = f"{mean_val:.3f} ± {std_val:.3f}"
        
        print(f"{col:<20} | {mean_val:.4f}     | {std_val:.4f}     | {formatted_str}")
    print("="*80)

if __name__ == "__main__":
    main()