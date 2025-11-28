# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 17:06:37 2025

@author: T
"""
# scripts/0_data_preprocessing.py
# Clean raw Duolingo SLAM listening logs
# Input : data/raw/slam.listen.train
# Output: data/processed/cleaned_data.parquet
# -*- coding: utf-8 -*-
"""
Script: 0_data_preprocessing.py
Description: 
    Cleans raw listening logs, parses country list strings, handles missing values, 
    and normalizes tokens. Outputs a clean Parquet file.
"""

import os
import re
import ast
import pandas as pd
import numpy as np

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "slam.listen.train")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "cleaned_data.parquet")

COLUMN_NAMES = [
    'user', 'exercise_id', 'instance_id', 'token', 'client', 
    'countries', 'session', 'days', 'time', 'label'
]

def normalize_token(token):
    """
    Normalizes a token by converting to lowercase and removing non-alphabetic characters.
    """
    if not isinstance(token, str):
        return ""
    token = token.lower()
    return re.sub(r"[^a-z]", "", token)

def parse_countries_list(countries_str):
    """
    Parses string representation of lists (e.g., "['CO']") to extract the first country code.
    """
    if not isinstance(countries_str, str):
        return ""
    try:
        lst = ast.literal_eval(countries_str)
        if isinstance(lst, list) and len(lst) > 0:
            code = lst[0]
            return code if isinstance(code, str) else ""
        return ""
    except (ValueError, SyntaxError):
        return ""

def main():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data not found at: {RAW_DATA_PATH}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("Loading raw data...")

    # Load data with specified dtypes to avoid warnings
    df = pd.read_csv(
        RAW_DATA_PATH,
        sep='\t',
        header=None,
        names=COLUMN_NAMES,
        dtype=str,
        na_values=['', 'NA', 'null', 'None']
    )

    # Convert numeric columns
    df['days'] = pd.to_numeric(df['days'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')

    # Parse countries
    print("Parsing country codes...")
    df['countries'] = df['countries'].apply(parse_countries_list)

    # Handle missing values
    print("Handling missing values...")
    df['days'] = df['days'].fillna(df['days'].mean())
    df['time'] = df['time'].fillna(df['time'].mean())
    
    # Fill categorical missings with mode
    for col in ['client', 'countries', 'session']:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # Normalize tokens
    print("Normalizing tokens...")
    df['token'] = df['token'].apply(normalize_token)

    # Filter invalid rows
    initial_len = len(df)
    df = df.dropna(subset=['user', 'exercise_id', 'instance_id', 'token', 'label'])
    df = df[df['token'].str.len() > 0]
    
    print(f"Preprocessing complete. Rows: {initial_len} -> {len(df)}")
    
    # Save
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()