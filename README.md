
# LACE: Lexico-phonological Approach to Cognitive Load Estimation in L2 Listening

> **Paper Title**: The Single-Word Paradox: Calibrated and Explainable Diagnosis of Cognitive Load in L2 Listening
> 
> **Target Journal**: Computer Assisted Language Learning (Submitted 2026)

This repository contains the official implementation, data samples, and reproduction scripts for the **LACE** framework.

## 📌 Project Information


-   **License**: MIT
    
-   **Last Updated**: November 2025
    

## 🚀 One-Click Full Reproduction

Follow the steps below to replicate the experiments and generate the figures presented in the paper.

### 1. Installation & Setup

First, clone the repository and install the necessary dependencies and NLTK data.

```
# Clone the repository
git clone [https://github.com/tengkinguo/LACE-L2-Listening.git](https://github.com/tengkinguo/LACE-L2-Listening.git)
cd LACE-L2-Listening

# Install Python requirements
pip install -r requirements.txt

# Download required NLTK corpus (CMU Dict)
python -c "import nltk; nltk.download('cmudict')"

```

### 2. Execution Pipeline

Run the scripts in the following order to preprocess data, train the model, and generate analysis.

```
# --- Step 1: Data Pipeline ---
python scripts/0_data_preprocessing.py   # Preprocess raw logs
python scripts/1_feature_engineering.py  # Extract lexico-phonological features
python scripts/2_make_sequences.py       # Construct sequences for training

# --- Step 2: Model Training ---
python scripts/train_lace.py             # Train the LACE model

# --- Step 3: Figure Generation ---
python scripts/generate_shap_figure.py   # Generates Figure 4
python scripts/generate_figures.py       # Generates Figures 2 & 3

```

## 📊 Results & Artifacts

The following table details the correspondence between the generated figures and the scripts used.

**Figure**

**Script**

**Content & Key Metrics**

**Figure 2**

`generate_figures.py`

**Effect of Isotonic Calibration**

  

Demonstrates ECE improvement (0.209 → 0.006).

**Figure 3**

`generate_figures.py`

**The Single-Word Paradox**

  

Ablation study analyzing single-word vs. full-sequence effects.

**Figure 4**

`generate_shap_figure.py`

**Grouped SHAP Feature Importance**

  

Explainability analysis of cognitive load factors.

## 📂 Dataset Overview

-   **Original Source**: Duolingo SLAM listening logs (Settles et al., 2018)
    
-   **Public Sample**: `data/sample/sample_5000.parquet`
    
    -   _Contains 5,000 anonymized interaction records for testing code logic._
        
-   **Word Frequency List**: `assets/SUBTLEXusfrequencyabove1.csv`
    

## 🤝 Future Extensions

We welcome contributions extending LACE to other L1 backgrounds, including but not limited to:

-   🇨🇳 Mandarin
    
-   🇸🇦 Arabic
    
-   🇫🇷 French
    
-   🇰🇷 Korean
