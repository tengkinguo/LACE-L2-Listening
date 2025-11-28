# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 17:18:04 2025

@author: T
"""

# scripts/1_feature_engineering.py
# Feature engineering for LACE model
# Input : data/processed/cleaned_data.parquet
# Output: data/processed/fe_data.parquet, encoders.pkl, scalers.pkl
# -*- coding: utf-8 -*-


import os
import re
import pickle
import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/cmudict.zip')
except LookupError:
    nltk.download('cmudict')
    nltk.download('averaged_perceptron_tagger')

import nltk.corpus

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.parquet")
WORD_FREQ_PATH = os.path.join(BASE_DIR, "assets", "SUBTLEXusfrequencyabove1.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# CMU Dict and Phoneme constants
CMU_DICT = nltk.corpus.cmudict.dict()
CHALLENGING_PHONEMES = {'TH', 'DH', 'SH', 'ZH', 'AW', 'OW', 'ER'}

def get_phoneme_info(word):
    """Extracts phoneme list, syllable count, and challenging phoneme flag."""
    word_lower = str(word).lower()
    if not word_lower or word_lower not in CMU_DICT:
        return [], 0, 0
    
    phonemes_raw = CMU_DICT[word_lower][0]
    phonemes = [re.sub(r'\d', '', p) for p in phonemes_raw] # Remove stress markers
    num_syllables = sum(1 for p in phonemes_raw if any(ch.isdigit() for ch in p))
    is_challenging = int(any(p in CHALLENGING_PHONEMES for p in phonemes))
    
    return phonemes, num_syllables, is_challenging

def load_word_freq_dict(path):
    if not os.path.exists(path):
        print(f"Warning: Frequency file not found at {path}. Frequency features will be 0.")
        return {}
    try:
        df_wf = pd.read_csv(path)
        # Assuming standard SUBTLEX columns; adjust if your CSV differs
        # Commonly 'Word' and 'Lg10WF' or 'SUBTLWF'
        # Trying to find the frequency column dynamically or defaulting to SUBTLWF
        if 'Word' in df_wf.columns:
            word_col = 'Word'
            freq_col = 'SUBTLWF' if 'SUBTLWF' in df_wf.columns else df_wf.columns[1]
            
            df_wf[word_col] = df_wf[word_col].astype(str).str.lower()
            return dict(zip(df_wf[word_col], df_wf[freq_col]))
        return {}
    except Exception as e:
        print(f"Error loading frequency dictionary: {e}")
        return {}

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}. Run preprocessing first.")

    print("Loading preprocessed data...")
    df = pd.read_parquet(INPUT_PATH)

    # --- Sorting ---
    print("Sorting events chronologically...")
    # Create a unified timestamp for sorting: exercise time + instance ID order
    # Note: 'days' and 'time' are per exercise session
    exercise_time_map = df.groupby('exercise_id')[['days', 'time']].first()
    exercise_time_map['exercise_timestamp'] = exercise_time_map['days'] * 100000 + exercise_time_map['time']
    
    df = df.join(exercise_time_map['exercise_timestamp'], on='exercise_id')
    df.sort_values(['user', 'exercise_timestamp', 'instance_id'], inplace=True)
    df.drop(columns=['exercise_timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Linguistic Features ---
    print("Generating linguistic features...")
    word_freq_dict = load_word_freq_dict(WORD_FREQ_PATH)
    
    # POS Tags
    tqdm.pandas(desc="POS Tagging")
    df['pos_tag'] = df['token'].progress_apply(
        lambda w: nltk.pos_tag([str(w)])[0][1] if pd.notna(w) else 'UNK'
    )
    
    # Frequency & Length
    df['word_freq'] = df['token'].str.lower().map(word_freq_dict).fillna(0)
    df['word_length'] = df['token'].str.len().fillna(0)
    
    # Phonemes
    tqdm.pandas(desc="Phoneme Extraction")
    phoneme_data = df['token'].progress_apply(
        lambda w: pd.Series(get_phoneme_info(w), index=['phoneme_list', 'num_syllables', 'is_challenging_phoneme'])
    )
    df = pd.concat([df, phoneme_data], axis=1)

    # --- Encoding Categoricals ---
    print("Encoding categorical features...")
    encoders = {}
    categorical_cols = ['user', 'token', 'pos_tag', 'client', 'countries', 'session', 'exercise_id']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_id'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Phoneme Encoding
    all_phonemes = sorted(list(set(p for sublist in df['phoneme_list'] for p in sublist)))
    phoneme_le = LabelEncoder()
    phoneme_le.fit(['<PAD>', '<UNK>'] + all_phonemes)
    encoders['phoneme'] = phoneme_le
    
    df['phoneme_ids_list'] = df['phoneme_list'].apply(
        lambda lst: phoneme_le.transform(lst).tolist() if isinstance(lst, list) and lst else []
    )

    # --- Contextual & History Features ---
    print("Generating interaction history features...")
    
    # Previous token/label within the same exercise
    grouped = df.groupby(['user_id', 'exercise_id_id'])
    df['prev_token_id'] = grouped['token_id'].shift(1).fillna(0).astype(int) # 0 is PAD/None
    df['prev_label'] = grouped['label'].shift(1).fillna(-1).astype(int) # -1 is None
    
    # User cumulative history (rolling accuracy)
    print("Calculating cumulative accuracy (vectorized)...")
    
    user_grouped = df.groupby('user_id')
    df['user_total_attempts'] = user_grouped.cumcount()
    df['user_correct_cumsum'] = user_grouped['label'].cumsum().shift(1).fillna(0)
    
    # FIX: Use np.where to avoid Division By Zero (infinity)
    # If attempts == 0, we set accuracy to 0.5 (cold start assumption)
    df['user_history_accuracy'] = np.where(
        df['user_total_attempts'] > 0,
        df['user_correct_cumsum'] / df['user_total_attempts'],
        0.5
    )
    
    # Double check to replace any lingering infinities
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # --- Standardization ---
    print("Standardizing continuous features...")
    scalers = {}
    continuous_cols = ['days', 'time', 'word_freq', 'word_length', 
                      'num_syllables', 'user_history_accuracy', 'user_total_attempts']
    
    for col in continuous_cols:
        scaler = StandardScaler()
        # Ensure no NaNs or Infs before scaling
        col_data = df[[col]].fillna(0)
        
        # Safety check for infinity again
        if np.isinf(col_data).values.any():
            print(f"Warning: Infinity found in {col}, replacing with 0.")
            col_data = col_data.replace([np.inf, -np.inf], 0)
            
        df[f'{col}_scaled'] = scaler.fit_transform(col_data)
        scalers[col] = scaler

    # --- Save ---
    output_path = os.path.join(OUTPUT_DIR, 'fe_data.parquet')
    df.to_parquet(output_path, index=False)
    
    encoders_path = os.path.join(OUTPUT_DIR, 'encoders.pkl')
    scalers_path = os.path.join(OUTPUT_DIR, 'scalers.pkl')
    
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)

    print(f"Feature engineering complete. Data saved to {output_path}")

if __name__ == "__main__":
    main()