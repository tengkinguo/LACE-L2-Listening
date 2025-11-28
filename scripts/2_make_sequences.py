# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 08:52:21 2025

@author: T
"""
# scripts/2_make_sequences.py
# Construct sequence data for Transformer model
# Input : data/processed/fe_data.parquet + encoders.pkl
# Output: data/processed/train.npz, val.npz, test.npz


import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "fe_data.parquet")
SEQUENCE_DIR = os.path.join(BASE_DIR, "data", "processed")

MAX_SEQ_LEN = 100
MAX_PHONEME_LEN = 10
SEED = 42

np.random.seed(SEED)

def pad_or_truncate_phonemes(phoneme_ids, max_len=MAX_PHONEME_LEN):
    """Ensures fixed length for phoneme lists."""
    if not isinstance(phoneme_ids, (list, np.ndarray)):
        return [0] * max_len
    return (list(phoneme_ids) + [0] * max_len)[:max_len]

def main():
    if not os.path.exists(FEAT_DATA_PATH):
        raise FileNotFoundError(f"Feature data not found: {FEAT_DATA_PATH}")

    print("Loading feature data...")
    df = pd.read_parquet(FEAT_DATA_PATH)

    # --- User Split ---
    unique_users = df['user'].unique()
    np.random.shuffle(unique_users)
    n_users = len(unique_users)
    n_train = int(0.70 * n_users)
    n_val = int(0.15 * n_users)
    
    train_users = set(unique_users[:n_train])
    val_users = set(unique_users[n_train : n_train + n_val])
    test_users = set(unique_users[n_train + n_val:])
    
    print(f"Users split: Train={len(train_users)}, Val={len(val_users)}, Test={len(test_users)}")

    # --- Sequence Construction ---
    print("Constructing sequences (this may take a while)...")
    
    # Initialize containers
    data = {k: [] for k in [
        'q_seq', 'r_seq', 'pos_seq', 'phoneme_seq', 
        'next_lexical', 'next_phoneme', 'next_context', 'next_prev_interaction',
        'label', 'user_str', 'meta_token_id', 'meta_token_str'
    ]}

    # Group by user (order preserved from previous step)
    for user_str, user_df in tqdm(df.groupby('user', sort=False), total=df['user'].nunique()):
        user_df = user_df.reset_index(drop=True)
        
        # Pre-convert columns to numpy to speed up loop
        token_ids = user_df['token_id'].values
        labels = user_df['label'].values
        pos_tag_ids = user_df['pos_tag_id'].values
        phoneme_lists = [pad_or_truncate_phonemes(p) for p in user_df['phoneme_ids_list']]
        
        # Next item features
        wf_scaled = user_df['word_freq_scaled'].values
        wl_scaled = user_df['word_length_scaled'].values
        ns_scaled = user_df['num_syllables_scaled'].values
        is_chal = user_df['is_challenging_phoneme'].values
        
        # Context features
        u_ids = user_df['user_id'].values
        c_ids = user_df['client_id'].values
        ct_ids = user_df['countries_id'].values
        s_ids = user_df['session_id'].values
        ex_ids = user_df['exercise_id_id'].values
        
        # Prev interaction features
        prev_tok = user_df['prev_token_id'].values
        prev_lab = user_df['prev_label'].values
        
        tokens_str = user_df['token'].values

        for i in range(len(user_df)):
            hist_len = min(i, MAX_SEQ_LEN)
            
            # 1. Historical Sequences (inputs)
            q_s = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
            r_s = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
            p_s = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
            ph_s = np.zeros((MAX_SEQ_LEN, MAX_PHONEME_LEN), dtype=np.int64)
            
            if hist_len > 0:
                q_s[-hist_len:] = token_ids[i-hist_len : i]
                r_s[-hist_len:] = labels[i-hist_len : i]
                p_s[-hist_len:] = pos_tag_ids[i-hist_len : i]
                ph_s[-hist_len:] = phoneme_lists[i-hist_len : i]

            # 2. Current Item Features (target context)
            lex_feat = np.array([wf_scaled[i], wl_scaled[i]], dtype=np.float32)
            pho_feat = np.array([ns_scaled[i], is_chal[i]], dtype=np.float32)
            ctx_feat = np.array([u_ids[i], c_ids[i], ct_ids[i], s_ids[i], ex_ids[i]], dtype=np.int64)
            prev_feat = np.array([prev_tok[i], prev_lab[i]], dtype=np.int64)

            # Append to lists
            data['q_seq'].append(q_s)
            data['r_seq'].append(r_s)
            data['pos_seq'].append(p_s)
            data['phoneme_seq'].append(ph_s)
            
            data['next_lexical'].append(lex_feat)
            data['next_phoneme'].append(pho_feat)
            data['next_context'].append(ctx_feat)
            data['next_prev_interaction'].append(prev_feat)
            
            data['label'].append(int(labels[i]))
            data['user_str'].append(user_str)
            data['meta_token_id'].append(token_ids[i])
            data['meta_token_str'].append(tokens_str[i])

    # --- Saving ---
    print("Saving splits...")
    # Convert lists to numpy arrays for indexing
    for k in data:
        dtype = object if 'str' in k else None
        data[k] = np.array(data[k], dtype=dtype)
        
    all_indices = np.arange(len(data['label']))
    
    split_definitions = {
        "train": train_users,
        "val": val_users,
        "test": test_users
    }

    for name, user_set in split_definitions.items():
        mask = np.isin(data['user_str'], list(user_set))
        indices = all_indices[mask]
        
        save_dict = {k: v[indices] for k, v in data.items() if k != 'user_str'}
        
        # Add metadata specifically for Val/Test analysis
        if name in ['val', 'test']:
            save_dict['meta_user_str'] = data['user_str'][indices]
            # Extract challenging flag for convenient analysis later
            save_dict['meta_is_challenging'] = data['next_phoneme'][indices][:, 1].astype(np.int8)

        out_file = os.path.join(SEQUENCE_DIR, f"{name}.npz")
        np.savez(out_file, **save_dict)
        print(f"Saved {name}.npz: {len(indices)} samples")

if __name__ == "__main__":
    main()