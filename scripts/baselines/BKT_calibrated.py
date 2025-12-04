# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 16:49:35 2025

@author: T
"""

# -*- coding: utf-8 -*-
"""
Script: BKT_calibrated.py
Location: scripts/baselines/BKT_calibrated.py

Description:
    Implements the standard Bayesian Knowledge Tracing (BKT) baseline.
    
    UPDATE (Consistency):
    This script now loads 'fe_data.parquet' to ensure the exact same 
    Data Splitting (Users/Rows) as the LACE, LR, and LGBM models.
    
    Note on Features:
    Unlike LR/LGBM/LACE, standard BKT relies solely on Skill IDs (Tokens) 
    and correctness sequences. It does not natively utilize continuous 
    features like word frequency. Its inclusion serves as a baseline for 
    "ID-based Sequence Modeling" vs "Feature-based Sequence Modeling" (LACE).

    Pipeline:
    1. Load Preprocessed Data ('fe_data.parquet').
    2. Split: User-level split (70% Train, 15% Val, 15% Test).
    3. Train: standard pyBKT Model.
    4. Calibrate: Isotonic Regression on Validation set.
    5. Evaluate: AUC, F1, ECE.

Usage:
    python scripts/baselines/BKT_calibrated.py
"""

import os
import numpy as np
import pandas as pd
from pyBKT.models import Model
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, f1_score

# --- Configuration & Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
# Ensure we use the exact same dataset file as other models
FEAT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "fe_data.parquet")

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

def run_experiment(run_id, df):
    """
    Runs a single complete experiment iteration.
    """
    print(f"\n--- Starting Run {run_id+1}/3 ---")
    
    # 1. User-Level Data Splitting (Match LACE logic: 70/15/15)
    unique_users = df['user'].unique()
    
    # Dynamic seed for robustness
    current_seed = 42 + run_id 
    np.random.seed(current_seed)
    np.random.shuffle(unique_users)

    n_users = len(unique_users)
    n_train = int(n_users * 0.70)
    n_val = int(n_users * 0.15)
    
    train_users = set(unique_users[:n_train])
    val_users = set(unique_users[n_train : n_train + n_val])
    test_users = set(unique_users[n_train + n_val:])

    # Boolean indexing
    train_mask = df['user'].isin(train_users)
    val_mask = df['user'].isin(val_users)
    test_mask = df['user'].isin(test_users)

    # 2. Prepare Data for pyBKT
    # BKT needs: user_id, skill_name, correct
    # We map 'token' (word) to 'skill_name'
    
    def prepare_bkt_df(mask):
        subset = df.loc[mask].copy()
        # pyBKT expects specific column names for auto-detection or we specify them
        # We'll create a standardized DF
        return pd.DataFrame({
            'user_id': subset['user'],
            'skill_name': subset['token'], 
            'correct': subset['label'].astype(int)
        })

    bkt_train = prepare_bkt_df(train_mask)
    bkt_val = prepare_bkt_df(val_mask)
    bkt_test = prepare_bkt_df(test_mask)

    print(f"Train size: {len(bkt_train)}, Val size: {len(bkt_val)}, Test size: {len(bkt_test)}")

    # 3. Handle Unknown Skills (Cold Start for Items)
    # BKT cannot predict for skills it never saw in training.
    train_skills = set(bkt_train['skill_name'].unique())
    
    # We must filter out unknown skills from Val/Test to avoid crashes or NaNs
    # (This is standard practice for ID-based KT evaluation)
    bkt_val = bkt_val[bkt_val['skill_name'].isin(train_skills)]
    bkt_test = bkt_test[bkt_test['skill_name'].isin(train_skills)]

    # 4. Train BKT
    # Using 'multiguess' or 'multislip' can be slow for many skills, so we use standard defaults
    # defaults={'prior': 0.5} sets the initial prior probability
    model = Model(seed=current_seed, num_fits=1)
    model.fit(data=bkt_train, defaults={'prior': 0.5})

    # 5. Calibration
    # Get raw predictions on Validation Set
    val_preds = model.predict(data=bkt_val)
    # pyBKT returns a DF with 'correct_predictions' column
    prob_val_raw = val_preds['correct_predictions'].values
    y_val = bkt_val['correct'].values
    
    iso_calibrator = IsotonicRegression(out_of_bounds='clip')
    iso_calibrator.fit(prob_val_raw, y_val)

    # 6. Final Evaluation
    test_preds = model.predict(data=bkt_test)
    prob_test_raw = test_preds['correct_predictions'].values
    y_test = bkt_test['correct'].values
    
    # Apply Calibration
    prob_test_cal = iso_calibrator.predict(prob_test_raw)
    
    y_pred_raw = (prob_test_raw >= 0.5).astype(int)
    
    metrics = {
        'AUC': roc_auc_score(y_test, prob_test_raw),
        'F1 Score': f1_score(y_test, y_pred_raw),
        'ECE (Raw)': compute_ece(y_test, prob_test_raw),
        'ECE (Calibrated)': compute_ece(y_test, prob_test_cal)
    }
    
    print(f"Run {run_id+1} Result: AUC={metrics['AUC']:.4f}, ECE(Cal)={metrics['ECE (Calibrated)']:.4f}")
    return metrics

def main():
    # Load Data Once
    if not os.path.exists(FEAT_DATA_PATH):
        raise FileNotFoundError(
            f"Feature file not found at: {FEAT_DATA_PATH}\n"
            "Please run 'python scripts/1_feature_engineering.py' first."
        )
    
    print("Loading feature data (parquet)...")
    # Load minimal columns needed for BKT to save memory
    df = pd.read_parquet(FEAT_DATA_PATH, columns=['user', 'token', 'label'])
    
    # Run Experiments
    N_RUNS = 3
    all_results = []
    
    print(f"Running BKT Baseline (Standard ID-based) for {N_RUNS} runs...")
    
    for i in range(N_RUNS):
        res = run_experiment(i, df)
        all_results.append(res)

    # Output Table
    target_columns = ['AUC', 'F1 Score', 'ECE (Raw)', 'ECE (Calibrated)']
    
    print("\n" + "="*80)
    print("   BKT Results (Standard ID-based)   ")
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