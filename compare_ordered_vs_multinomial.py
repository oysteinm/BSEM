import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---
# Get the directory where the current script is located
# This ensures BASE_DIR points to the root 'appendix' folder in your repo
BASE_DIR = Path(__file__).resolve().parent

# Define paths relative to the repo root 
ORD_PATH = BASE_DIR / "PS3_measurement_variance_sensitive" / "model_summary.csv"
MULT_PATH = BASE_DIR / "ROBUSTNESS_multinomial_logit" / "model_summary.csv"

# Optional: Output the comparison results back to the root folder
# ROBUSTNESS_OUT = BASE_DIR / "robustness_check_summary.csv"

# Factor names in order used in scripts
FACTOR_NAMES = ['Brand', 'Hedonic', 'Confused', 'Perf', 'Careless', 'Value', 'Habit', 'Novelty']

def load_and_format():
    # 1. Load PS3 (Ordered Logistic)
    ord_df = pd.read_csv(ORD_PATH, index_col=0)
    # Filter for structural factors
    ord_betas = ord_df[ord_df.index.str.contains('beta_factors')].copy()
    ord_betas['Factor'] = FACTOR_NAMES
    ord_betas = ord_betas[['Factor', 'mean', 'hdi_3%', 'hdi_97%']]
    ord_betas.columns = ['Factor', 'Ord_Mean', 'Ord_Lower', 'Ord_Upper']

    # 2. Load Multinomial (Robustness)
    # beta_factors[factor_idx, category_idx] 
    # category_idx 2 = Trolley (relative to Arms)
    mult_df = pd.read_csv(MULT_PATH, index_col=0)
    mult_rows = []
    for i, name in enumerate(FACTOR_NAMES):
        param_name = f"beta_factors[{i}, 2]" # Index 2 is Trolley
        row = mult_df.loc[param_name]
        mult_rows.append({
            'Factor': name,
            'Mult_Mean': row['mean'],
            'Mult_Lower': row['hdi_3%'],
            'Mult_Upper': row['hdi_97%']
        })
    mult_betas = pd.DataFrame(mult_rows)

    # 3. Merge and Compare
    comparison = pd.merge(ord_betas, mult_betas, on='Factor')
    
    # Check if signs match
    comparison['Signs_Match'] = np.sign(comparison['Ord_Mean']) == np.sign(comparison['Mult_Mean'])
    
    return comparison

if __name__ == "__main__":
    try:
        results = load_and_format()
        print("\n=== ROBUSTNESS COMPARISON: ORDERED (PS3) vs. MULTINOMIAL (Trolley) ===")
        print(results.to_string(index=False))
        
        out_file = BASE_DIR / "robustness_check_summary.csv"
        results.to_csv(out_file, index=False)
        print(f"\n✅ Summary saved to: {out_file}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find summary files. Ensure PS3 and Multinomial runs finished. {e}")