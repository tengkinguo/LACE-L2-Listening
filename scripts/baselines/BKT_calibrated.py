# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 10:07:45 2025

@author: T
"""

"""
Script: BKT_calibrated.py
Location: scripts/baselines/BKT_calibrated.py

Description:
    Implements the standard Bayesian Knowledge Tracing (BKT) baseline.
    Added Isotonic Calibration to allow fair comparison with LACE regarding ECE.
    
    Metrics: AUC, F1, ECE (Raw), ECE (Calibrated).
"""

import os
import pickle
import numpy as np
import pandas as pd
from pyBKT.models import Model
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.isotonic import IsotonicRegression

# --- Configuration & Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
TRAIN_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "slam.listen.train")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def main():
    # 1. Load Data
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"File not found: {TRAIN_FILE}")
    
    COLUMNS = ['user', 'exercise_id', 'instance_id', 'token', 'client',
               'countries', 'session', 'days', 'time', 'label']
    
    # Load only necessary columns to save memory
    df = pd.read_csv(
        TRAIN_FILE, sep='\t', header=None, names=COLUMNS,
        usecols=['user', 'instance_id', 'token', 'label'],
        dtype={'user': str, 'instance_id': str, 'token': str, 'label': float}
    )
    print(f"Loaded {len(df)} rows.")

    # 2. Preprocessing
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    # Filter non-binary labels if any
    df = df[df['label'].isin([0, 1])]
    
    # Sort chronologically
    df = df.sort_values(by=['user', 'instance_id']).reset_index(drop=True)

    # 3. User Split (70/15/15)
    users = df['user'].unique()
    np.random.seed(42)
    np.random.shuffle(users)
    
    n_train = int(0.70 * len(users))
    n_val = int(0.15 * len(users))

    train_users = set(users[:n_train])
    val_users = set(users[n_train : n_train + n_val])
    test_users = set(users[n_train + n_val:])

    train_df = df[df['user'].isin(train_users)].reset_index(drop=True)
    val_df = df[df['user'].isin(val_users)].reset_index(drop=True)
    test_df = df[df['user'].isin(test_users)].reset_index(drop=True)

    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 4. Prepare BKT Format
    def to_bkt_format(d):
        return pd.DataFrame({
            'user_id': d['user'],
            'skill_name': d['token'],
            'correct': d['label']
        })

    bkt_train = to_bkt_format(train_df)
    bkt_val = to_bkt_format(val_df)
    bkt_test = to_bkt_format(test_df)

    # 5. Handle Unknown Skills
    # BKT cannot predict for skills (tokens) it hasn't seen in training.
    train_skills = set(bkt_train['skill_name'].unique())
    print(f"Unique skills in training: {len(train_skills)}")
    
    # Filter validation/test sets to remove unseen skills
    bkt_val = bkt_val[bkt_val['skill_name'].isin(train_skills)]
    bkt_test = bkt_test[bkt_test['skill_name'].isin(train_skills)]

    # 6. Train BKT Model
    print("Training BKT Model (standard)...")
    # Defaults: guess=0.3, slip=0.1 is a common starting point
    defaults = {'prior': 0.5} 
    model = Model(seed=42, num_fits=1)
    model.fit(data=bkt_train, defaults=defaults)
    print("Training complete.")

    # 7. Validation & Calibration
    print("Predicting on Validation Set...")
    val_preds_df = model.predict(data=bkt_val)
    # pyBKT returns a dataframe with 'correct_predictions' column
    val_prob_raw = val_preds_df['correct_predictions'].values
    val_y = bkt_val['correct'].values

    # Fit Isotonic Calibrator
    print("Fitting Isotonic Calibrator...")
    iso_calibrator = IsotonicRegression(out_of_bounds='clip')
    iso_calibrator.fit(val_prob_raw, val_y)

    # 8. Test Evaluation
    print("Predicting on Test Set...")
    test_preds_df = model.predict(data=bkt_test)
    test_prob_raw = test_preds_df['correct_predictions'].values
    test_y = bkt_test['correct'].values
    
    # Apply Calibration
    test_prob_cal = iso_calibrator.predict(test_prob_raw)
    
    # Calculate Metrics
    y_pred_binary = (test_prob_raw >= 0.5).astype(int)
    
    metrics = {
        'AUC': roc_auc_score(test_y, test_prob_raw),
        'F1 Score': f1_score(test_y, y_pred_binary),
        'ECE (Raw)': compute_ece(test_y, test_prob_raw),
        'ECE (Calibrated)': compute_ece(test_y, test_prob_cal)
    }

    # 9. Output Table 1 Format
    print("\n" + "="*60)
    print("   BKT Baseline Results (Formatted for Table 1)   ")
    print("="*60)
    # Since BKT training is deterministic given a seed, Std is 0.000
    print(f"{'Metric':<20} | {'Value':<10} | {'Table 1 String'}")
    print("-" * 60)
    print(f"{'AUC':<20} | {metrics['AUC']:.4f}     | {metrics['AUC']:.3f} ± 0.000")
    print(f"{'F1 Score':<20} | {metrics['F1 Score']:.4f}     | {metrics['F1 Score']:.3f} ± 0.000")
    print(f"{'ECE (Raw)':<20} | {metrics['ECE (Raw)']:.4f}     | {metrics['ECE (Raw)']:.3f} ± 0.000")
    print(f"{'ECE (Calibrated)':<20} | {metrics['ECE (Calibrated)']:.4f}     | {metrics['ECE (Calibrated)']:.3f} ± 0.000")
    print("="*60)

if __name__ == "__main__":
    main()