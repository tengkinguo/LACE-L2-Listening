# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 10:59:24 2025

@author: T
"""

# -*- coding: utf-8 -*-
"""
Script: generate_paper_figures.py
Description:
    Generates publication-quality figures (Figure 2 & Figure 3) using matplotlib.
    
    Figure 2: Impact of Calibration on ECE (Expected Calibration Error).
    Figure 3: Ablation Study showing the "Single-Word Paradox".
    
    Note: This script uses the reported values from the paper to ensure 
    exact visual reproduction for the repository's README or publication.
"""

import matplotlib.pyplot as plt
import os
import sys

# --- Path Setup ---
# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Global Style Settings ---
# Academic standard styling (Arial font, clean layout)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['savefig.dpi'] = 300

def plot_figure_2_calibration():
    """
    Generates Figure 2: Impact of Calibration on Model Trustworthiness.
    Data: ECE drops from 0.209 (Raw) to 0.006 (Calibrated).
    """
    print("Generating Figure 2...")
    
    labels = ['Before Calibration\n(Raw Output)', 'After Calibration\n(Isotonic)']
    values = [0.209, 0.006] # Exact values from Table 1 in the paper
    colors = ['#e74c3c', '#2ecc71'] # Red (Danger) -> Green (Safe)

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=colors, alpha=0.85, edgecolor='black', width=0.5)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                 f'{height:.3f}', ha='center', va='bottom', 
                 fontweight='bold', fontsize=12)

    plt.ylabel('Expected Calibration Error (ECE)', fontweight='bold')
    plt.title('Impact of Calibration on Model Trustworthiness', pad=15, fontweight='bold')
    plt.ylim(0, 0.25)

    # Annotations for storytelling
    # Arrow for "Unsafe"
    plt.annotate("Unsafe / Overconfident", 
                 xy=(0, 0.1), xytext=(0.6, 0.18),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='gray'),
                 fontsize=10, color='#c0392b', fontweight='bold')

    # Arrow for "Safe"
    plt.annotate("Safe for Deployment", 
                 xy=(1, 0.01), xytext=(1.3, 0.08),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color='gray'),
                 fontsize=10, color='#27ae60', fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, 'Figure2_Calibration.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def plot_figure_3_ablation():
    """
    Generates Figure 3: The "Single-Word Paradox".
    Data: F1 Scores comparing Baseline, Paradox Model, and LACE.
    """
    print("Generating Figure 3...")
    
    models = [
        'Baseline\n(Seq Only)',    # A0
        'With History\n(Paradox)', # A1
        'Lexical +\nPhonological', # C1
        'LACE\n(Proposed)'         # E1
    ]
    
    f1_scores = [0.505, 0.499, 0.515, 0.521] # Exact values from Table 2
    errors = [0.007, 0.010, 0.006, 0.010]    # Std dev for error bars
    
    # Colors: Grey (Base) -> Red (Paradox/Drop) -> Blue (Recovery) -> Green (Best)
    colors = ['#95a5a6', '#c0392b', '#3498db', '#2ecc71']

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Grid behind bars
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    bars = ax.bar(models, f1_scores, yerr=errors, capsize=8, 
                  color=colors, alpha=0.9, edgecolor='black', width=0.6, zorder=3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()*0.6, height + 0.003,
                f'{height:.3f}', ha='left', va='bottom', 
                fontweight='bold', fontsize=12, zorder=4)

    # Zoom in on the relevant Y-axis range to show differences
    ax.set_ylim(0.48, 0.55)
    ax.set_ylabel('F1 Score (Higher is Better)', fontweight='bold')
    ax.set_title('Ablation Study: The "Single-Word Paradox" & Linguistic Features', 
                 pad=20, fontweight='bold')

    # Annotation: The Paradox
    ax.annotate('Paradox:\nPerformance Drops!', 
                xy=(1, 0.499), xytext=(1, 0.535),
                arrowprops=dict(facecolor='#c0392b', shrink=0.05, width=2, headwidth=8),
                ha='center', color='#c0392b', fontweight='bold', fontsize=11, zorder=5)

    # Annotation: LACE Improvement
    ax.annotate('Linguistic Features\nRestore Performance', 
                xy=(2, 0.515), xytext=(2, 0.49),
                arrowprops=dict(facecolor='#3498db', shrink=0.05, width=2, headwidth=8),
                ha='center', color='#2c3e50', fontweight='bold', fontsize=11, zorder=5)

    save_path = os.path.join(OUTPUT_DIR, 'Figure3_SingleWordParadox.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_figure_2_calibration()
    plot_figure_3_ablation()
    print("\nAll figures generated successfully in the 'figures/' folder.")