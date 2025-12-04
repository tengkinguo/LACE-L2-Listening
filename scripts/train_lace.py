# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 17:09:22 2025

@author: T
"""

# scripts/3_train_lace.py
# Train LACE model with multiple configurations and isotonic calibration
# Input : data/processed/train.npz, val.npz, test.npz + encoders.pkl
# Output: results/models/, results/predictions/, results/figures/
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Utils
from utils.compute_ece import compute_ece

# Import Model Architecture
# Try importing from scripts package (if running from root) or local file (if running from scripts dir)
try:
    from scripts.model_arch import LACEModel
except ImportError:
    from model_arch import LACEModel

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ENCODERS_PATH = os.path.join(DATA_DIR, "encoders.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "predictions"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HPARAMS = {
    "max_seq_len": 100,
    "d_model": 128,
    "n_head": 4,
    "num_layers": 2,
    "ff_dim": 512,
    "dropout_rate": 0.2,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "max_epochs": 50,
    "early_stop_patience": 5,
}

# --- Dataset ---
class UnifiedSAKTDataset(Dataset):
    def __init__(self, split):
        path = os.path.join(DATA_DIR, f"{split}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data split not found: {path}")
            
        with np.load(path, allow_pickle=True) as data:
            # Load all arrays except metadata
            self.data = {k: torch.from_numpy(v) for k, v in data.items() if not k.startswith("meta")}

    def __len__(self):
        return len(self.data["label"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

# --- Helper Functions ---
def load_vocab_sizes():
    """
    Loads all encoder classes to determine embedding layer sizes.
    Includes robust checks for context features (User, Client, etc.)
    """
    if not os.path.exists(ENCODERS_PATH):
        raise FileNotFoundError(f"Encoders file not found at {ENCODERS_PATH}. Please run feature engineering.")

    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
        
    sizes = {
        "num_questions": len(encoders["token"].classes_),
        "num_pos_tags": len(encoders["pos_tag"].classes_),
        "num_phonemes": len(encoders["phoneme"].classes_) + 1,
    }
    
    # Add context sizes if available (required for E2_Full)
    # Check if 'user' key exists, if not, use default or throw error if context is needed
    if "user" in encoders:
        sizes["num_users"] = len(encoders["user"].classes_)
        sizes["num_clients"] = len(encoders["client"].classes_)
        sizes["num_countries"] = len(encoders["countries"].classes_)
        sizes["num_sessions"] = len(encoders["session"].classes_)
        sizes["num_exercises"] = len(encoders["exercise_id"].classes_)
    else:
        # Fallback to avoid crashes if using models that DON'T need context
        # but warn if a context model is attempted
        sizes.update({
            "num_users": 1, "num_clients": 1, "num_countries": 1, 
            "num_sessions": 1, "num_exercises": 1
        })
        
    return sizes

def train_and_evaluate(config_name, config, n_run=0):
    print(f"\n=== Starting Run {n_run} for {config_name} ===")
    sizes = load_vocab_sizes()
    
    # Load Data
    train_set = UnifiedSAKTDataset("train")
    val_set = UnifiedSAKTDataset("val")
    test_set = UnifiedSAKTDataset("test")
    
    train_loader = DataLoader(train_set, batch_size=HPARAMS["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=HPARAMS["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=HPARAMS["batch_size"], shuffle=False)

    # Initialize Model
    model = LACEModel(sizes, HPARAMS, config).to(DEVICE)
    
    # Loss & Optimizer
    # Handle Class Imbalance
    pos_count = train_set.data["label"].sum().item()
    neg_count = len(train_set) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], device=DEVICE) if pos_count > 0 else None
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HPARAMS["learning_rate"], weight_decay=HPARAMS["weight_decay"])

    # --- Training Loop ---
    best_auc = 0.0
    patience = 0
    model_save_path = os.path.join(RESULTS_DIR, "models", f"{config_name}_run{n_run}.pt")

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            for k in batch: batch[k] = batch[k].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch["label"].float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                for k in batch: batch[k] = batch[k].to(DEVICE)
                logits = model(batch)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(batch["label"].cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_save_path)
            patience = 0
        else:
            patience += 1
            if patience >= HPARAMS["early_stop_patience"]:
                print("Early stopping triggered.")
                break

    # --- Calibration & Testing ---
    print("Loading best model for calibration...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Get Val scores for calibration fitting
    val_probs, val_y = [], []
    with torch.no_grad():
        for batch in val_loader:
            for k in batch: batch[k] = batch[k].to(DEVICE)
            val_probs.extend(torch.sigmoid(model(batch)).cpu().numpy())
            val_y.extend(batch["label"].cpu().numpy())
            
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(val_probs, val_y)

    # Testing
    test_probs_raw, test_y = [], []
    shap_data = {"lex": [], "pho": [], "prev": [], "score": []}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            for k in batch: batch[k] = batch[k].to(DEVICE)
            
            logits, feats = model(batch, return_features=True)
            probs = torch.sigmoid(logits)
            
            test_probs_raw.extend(probs.cpu().numpy())
            test_y.extend(batch["label"].cpu().numpy())
            
            # Collect SHAP inputs (Optional: limit size for memory efficiency)
            if len(shap_data["lex"]) < 2000:
                if "lex" in feats: shap_data["lex"].append(feats["lex"].cpu().numpy())
                if "pho" in feats: shap_data["pho"].append(feats["pho"].cpu().numpy())
                if "prev" in feats: shap_data["prev"].append(feats["prev"].cpu().numpy())

    test_probs_raw = np.array(test_probs_raw)
    test_probs_cal = iso_reg.predict(test_probs_raw)
    
    # Calculate Metrics
    results = {
        "model": config_name,
        "run": n_run,
        "auc": roc_auc_score(test_y, test_probs_raw),
        "f1": f1_score(test_y, (test_probs_raw >= 0.5).astype(int)),
        "ece_raw": compute_ece(test_y, test_probs_raw),
        "ece_cal": compute_ece(test_y, test_probs_cal)
    }
    
    print(f"Results: AUC={results['auc']:.4f}, ECE(Raw)={results['ece_raw']:.4f}, ECE(Cal)={results['ece_cal']:.4f}")
    
    # Save Predictions
    np.savez(
        os.path.join(RESULTS_DIR, "predictions", f"{config_name}_run{n_run}.npz"),
        y_true=test_y,
        y_raw=test_probs_raw,
        y_cal=test_probs_cal,
        shap_lex=np.concatenate(shap_data["lex"]) if shap_data["lex"] else [],
        shap_pho=np.concatenate(shap_data["pho"]) if shap_data["pho"] else [],
        shap_prev=np.concatenate(shap_data["prev"]) if shap_data["prev"] else []
    )
    
    return results

if __name__ == "__main__":
    # Define Configurations
    CONFIGS = {
        "A0_SeqOnly": {"use_hist_pos_phoneme": False, "use_lexical": False, "use_phoneme": False, "use_context": False, "use_prev_interaction": False},
        "A1_Paradox": {"use_hist_pos_phoneme": True,  "use_lexical": False, "use_phoneme": False, "use_context": False, "use_prev_interaction": False},
        "C1_LexPho":  {"use_hist_pos_phoneme": False,  "use_lexical": True,  "use_phoneme": True,  "use_context": False, "use_prev_interaction": False},
        
        # E1: LACE (Proposed in paper) - No User/Country Context
        "E1_LACE":    {"use_hist_pos_phoneme": False,  "use_lexical": True,  "use_phoneme": True,  "use_context": False, "use_prev_interaction": True},
        
        # E2: Full (Includes User/Country Context) - Optional variant from original code
        "E2_Full":    {"use_hist_pos_phoneme": False,  "use_lexical": True,  "use_phoneme": True,  "use_context": True,  "use_prev_interaction": True},
    }

    # Choose model to run
    # Use "E1_LACE" to reproduce paper results (AUC ~0.830)
    # Use "E2_Full" to include user/context features
    MODEL_NAME = "E1_LACE" 
    N_RUNS = 1 # Increase to 3 for robustness
    
    all_results = []
    for i in range(N_RUNS):
        res = train_and_evaluate(MODEL_NAME, CONFIGS[MODEL_NAME], n_run=i)
        all_results.append(res)
        
    df_res = pd.DataFrame(all_results)
    df_res.to_csv(os.path.join(RESULTS_DIR, f"summary_{MODEL_NAME}.csv"), index=False)