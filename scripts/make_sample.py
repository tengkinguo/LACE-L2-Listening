# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 11:51:08 2025

@author: T
"""
# scripts/make_sample.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
df = pd.read_parquet(ROOT / "data" / "processed" / "cleaned_data.parquet")

sample = df.sample(n=5000, random_state=42)
sample['user'] = sample['user'].apply(lambda x: f"user_{hash(str(x)) & 0xFFFFFF:06x}")

out_path = ROOT / "data" / "sample" / "sample_5000.parquet"
out_path.parent.mkdir(exist_ok=True)
sample.to_parquet(out_path, index=False)
print(f"公开样本已生成（5000 行，user 已脱敏）\n保存至：{out_path}")