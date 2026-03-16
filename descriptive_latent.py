# -*- coding: utf-8 -*-
"""
=============================================================================
DESCRIPTIVE ANALYSIS OF LATENT MODEL ITEMS (LOCAL VERSION)
Calculates mean, standard deviation, and a correlation matrix for the
indicators used in the Bayesian SEM model and saves the results to CSV files.
=============================================================================
"""

import pandas as pd
from pathlib import Path

# --- 1. SET UP PATHS ---
# Get the directory where the current script is located
BASE_DIR = Path(__file__).resolve().parent

# Set the input and output paths relative to the script location
INPUT_DATA_PATH = BASE_DIR / "basket.csv"
DESCRIPTIVES_OUT = BASE_DIR / "descriptive_statistics.csv"
CORR_MATRIX_OUT = BASE_DIR / "correlation_matrix.csv"

def analyze_and_save_model_items():
    """
    Loads the basket dataset locally, calculates descriptive statistics,
    computes a correlation matrix, and saves both outputs to CSV files.
    """
    # --- 2. Load Data ---
    print(f"📊 Loading data from: {INPUT_DATA_PATH}")
    try:
        if not INPUT_DATA_PATH.exists():
            raise FileNotFoundError(f"basket.csv not found in {BASE_DIR}")
            
        dframe = pd.read_csv(INPUT_DATA_PATH)
        print("✅ Data loaded successfully. Shape:", dframe.shape)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # --- 3. Define Model Items ---
    # These indicators match the factor structure in the Bayesian SEM scripts
    all_indicators = [
        'brand1', 'brand4', 'brand5',
        'hedonic1', 'hedonic2', 'hedonic4',
        'confused1', 'confused2', 'confused4',
        'perf2', 'perf3', 'perf4',
        'careless1', 'careless2', 'careless3',
        'habit1', 'habit2',
        'novelty1', 'novelty2', 'novelty3',
        'value1', 'value2'
    ]

    # Verify that all indicators exist in the dataframe
    missing_cols = [col for col in all_indicators if col not in dframe.columns]
    if missing_cols:
        print(f"\n❌ Error: The following columns are missing from the dataset: {missing_cols}")
        return

    # Create a subset of the dataframe with only the indicators
    indicators_df = dframe[all_indicators]
    print(f"✅ Created subset of data with {len(all_indicators)} model items.")

    # --- 4. Calculate and Save Descriptive Statistics ---
    print("\n" + "="*70)
    print("📈 CALCULATING MEAN AND STANDARD DEVIATION")
    print("="*70)

    descriptives = indicators_df.describe().loc[['mean', 'std']].T
    print(descriptives.to_string())

    try:
        descriptives.to_csv(DESCRIPTIVES_OUT)
        print(f"\n✅ Descriptive statistics saved to: {DESCRIPTIVES_OUT}")
    except Exception as e:
        print(f"\n❌ Failed to save descriptive statistics: {e}")


    # --- 5. Calculate and Save Correlation Matrix ---
    print("\n" + "="*70)
    print("🔗 CALCULATING CORRELATION MATRIX")
    print("="*70)

    corr_matrix = indicators_df.corr()

    # Display a snippet of the correlation matrix
    pd.set_option('display.width', 120) 
    print(corr_matrix.iloc[:10, :5].to_string()) # Show only a small portion in console
    pd.reset_option('display.width')

    try:
        corr_matrix.to_csv(CORR_MATRIX_OUT)
        print(f"\n✅ Correlation matrix saved to: {CORR_MATRIX_OUT}")
    except Exception as e:
        print(f"\n❌ Failed to save correlation matrix: {e}")


if __name__ == '__main__':
    analyze_and_save_model_items()