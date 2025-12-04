# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 16:43:09 2025

@author: T
"""

# -*- coding: utf-8 -*-
"""
Script: LR_calibrated.py
Location: scripts/baselines/LR_calibrated.py

Description:
    Implements the Logistic Regression baseline with Isotonic Calibration.
    
    CRITICAL UPDATE:
    To ensure a fair comparison with the LACE model, this script now loads 
    the feature-engineered data ('fe_data.parquet') instead of raw logs.
    This grants the baseline access to the exact same lexico-phonological 
    features (word frequency, phoneme complexity, etc.) used by LACE.

    Pipeline:
    1. Load Preprocessed Data: Reads 'fe_data.parquet'.
    2. Split: User-level split (70% Train, 15% Val, 15% Test) matching LACE.
    3. Train: Logistic Regression on Train set.
    4. Calibrate: Isotonic Regression on Validation set probabilities.
    5. Evaluate: Reports AUC, F1, ECE (Raw), ECE (Calibrated) over 3 runs.

Usage:
    python scripts/baselines/LR_calibrated.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, f1_score

# --- Configuration & Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
# Using the exact same feature file as LACE
FEAT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "fe_data.parquet")

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

def run_experiment(run_id, df):
    """
    Runs a single complete experiment iteration.
    Args:
        run_id: Int, seed offset.
        df: DataFrame, loaded feature data.
    """
    print(f"\n--- Starting Run {run_id+1}/3 ---")
    
    # 1. User-Level Data Splitting (Match LACE logic: 70/15/15)
    unique_users = df['user'].unique()
    
    # Dynamic seed ensures we test robustness across different splits
    # Note: To exactly match LACE's specific run, you would use seed=42 for run 0.
    current_seed = 42 + run_id 
    np.random.seed(current_seed)
    np.random.shuffle(unique_users)

    n_users = len(unique_users)
    n_train = int(n_users * 0.70)
    n_val = int(n_users * 0.15)
    
    train_users = set(unique_users[:n_train])
    val_users = set(unique_users[n_train : n_train + n_val])
    test_users = set(unique_users[n_train + n_val:])

    # Boolean indexing is faster
    train_mask = df['user'].isin(train_users)
    val_mask = df['user'].isin(val_users)
    test_mask = df['user'].isin(test_users)

    # 2. Feature Selection
    # Using the pre-scaled features generated in 1_feature_engineering.py
    # These match the inputs to LACE's MLPs.
    feature_cols = [
        # Lexical Features
        'word_freq_scaled',
        'word_length_scaled',
        
        # Phonological Features
        'num_syllables_scaled',
        'is_challenging_phoneme', # Binary, no scaling needed but safe to include if 0/1
        
        # Behavioral / Context Features
        'user_history_accuracy_scaled',
        'days_scaled',
        'time_scaled'
    ]
    
    target_col = 'label'

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, target_col].values.astype(int)
    
    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, target_col].values.astype(int)
    
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, target_col].values.astype(int)

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # 3. Train Logistic Regression
    # class_weight='balanced' handles the 17.5% error rate imbalance
    lr_model = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        random_state=current_seed,
        max_iter=1000
    )
    lr_model.fit(X_train, y_train)

    # 4. Fit Calibration Model (Isotonic)
    # Use Validation set probabilities to fit the calibrator to avoid overfitting
    prob_val_raw = lr_model.predict_proba(X_val)[:, 1]
    
    iso_calibrator = IsotonicRegression(out_of_bounds='clip')
    iso_calibrator.fit(prob_val_raw, y_val)

    # 5. Final Evaluation on Test Set
    prob_test_raw = lr_model.predict_proba(X_test)[:, 1]
    prob_test_cal = iso_calibrator.predict(prob_test_raw)
    
    # For F1, use default threshold of 0.5
    y_pred_raw = (prob_test_raw >= 0.5).astype(int)
    
    metrics = {
        'AUC': roc_auc_score(y_test, prob_test_raw),
        'F1 Score': f1_score(y_test, y_pred_raw),
        'ECE (Raw)': compute_ece(y_test, prob_test_raw),
        'ECE (Calibrated)': compute_ece(y_test, prob_test_cal)
    }
    
    print(f"Result: AUC={metrics['AUC']:.4f}, ECE(Cal)={metrics['ECE (Calibrated)']:.4f}")
    return metrics

def main():
    # Load Data Once
    if not os.path.exists(FEAT_DATA_PATH):
        raise FileNotFoundError(
            f"Feature file not found at: {FEAT_DATA_PATH}\n"
            "Please run 'python scripts/1_feature_engineering.py' first."
        )
    
    print("Loading feature data (parquet)...")
    df = pd.read_parquet(FEAT_DATA_PATH)
    
    # Run Experiments
    N_RUNS = 3
    all_results = []
    
    print(f"Running Logistic Regression Baseline (Fair Comparison Mode) for {N_RUNS} runs...")
    
    for i in range(N_RUNS):
        res = run_experiment(i, df)
        all_results.append(res)

    # Output Table
    target_columns = ['AUC', 'F1 Score', 'ECE (Raw)', 'ECE (Calibrated)']
    
    print("\n" + "="*80)
    print("   Logistic Regression Results (Fair Comparison / LACE Features)   ")
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