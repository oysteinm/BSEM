import os
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import arviz as az

# =============================================================================
# SPEC CONFIG (Robustness Check: Multinomial Logit)
# =============================================================================
from pathlib import Path
import os

SPEC_ID = "ROBUSTNESS_multinomial_logit"

# Get the directory where the current script is located
# This makes the setup portable for any user 
BASE_DIR = Path(__file__).resolve().parent

# Set the input data path relative to the script location 
INPUT_DATA_PATH = BASE_DIR / "basket.csv"

# Create the output directory relative to the script 
OUT_DIR = BASE_DIR / SPEC_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

# For compatibility with older code sections using os.path strings:
OUT_DIR_STR = str(OUT_DIR)

@dataclass
class PreparedData:
    dframe: pd.DataFrame
    basket2_obs: np.ndarray
    handledag_dummies: pd.DataFrame
    age_dummies: pd.DataFrame
    indicators_scaled: np.ndarray
    indicators_dict: dict
    factor_indicators: dict
    scaler: StandardScaler
    all_indicators: list
    n_obs: int
    n_factors: int

def prep_data(dframe: pd.DataFrame) -> PreparedData:
    df = dframe.copy()
    basket_map = {'Arms': 0, 'Basket': 1, 'Cart': 2, 'Trolley': 3}
    df['basket2_idx'] = df['basket2'].map(basket_map)
    basket2_obs = df['basket2_idx'].astype(int).to_numpy()
    df['gender_female'] = (df['gender'] == 'Female').astype(int)
    
    handledag_dummies = pd.get_dummies(df['handledag'], prefix='handledag')
    if 'handledag_Saturday' in handledag_dummies.columns:
        handledag_dummies = handledag_dummies.drop('handledag_Saturday', axis=1)

    age_dummies = pd.get_dummies(df['age'], prefix='age', drop_first=True)

    factor_indicators = {
        'brand': ['brand1', 'brand4', 'brand5'],
        'hedonic': ['hedonic1', 'hedonic2', 'hedonic4'],
        'confused': ['confused1', 'confused2', 'confused4'],
        'perf': ['perf2', 'perf3', 'perf4'],
        'careless': ['careless1', 'careless2', 'careless3'],
        'value': ['value1', 'value2'],
        'habit': ['habit1', 'habit2'],
        'novelty': ['novelty1', 'novelty2', 'novelty3'],
    }

    all_indicators = [item for sublist in factor_indicators.values() for item in sublist]
    X = df[all_indicators].astype(float).to_numpy()
    scaler = StandardScaler()
    indicators_scaled = scaler.fit_transform(X)

    indicators_dict = {}
    idx = 0
    for factor_name, cols in factor_indicators.items():
        k = len(cols)
        indicators_dict[factor_name] = indicators_scaled[:, idx: idx + k]
        idx += k

    return PreparedData(df, basket2_obs, handledag_dummies, age_dummies, 
                        indicators_scaled, indicators_dict, factor_indicators, 
                        scaler, all_indicators, len(df), len(factor_indicators))

# --- MAIN EXECUTION BLOCK ---
# This 'if __name__ == "__main__":' block is MANDATORY for Windows multiprocessing
if __name__ == '__main__':
    print(f"Loading data from: {INPUT_DATA_PATH}")
    raw_data = pd.read_csv(INPUT_DATA_PATH)
    data = prep_data(raw_data)

    with pm.Model() as multinomial_model:
        print("Building Measurement Model...")
        latent_factors_list = []
        
        for factor_name, indicators in data.factor_indicators.items():
            n_ind = len(indicators)
            eta = pm.Normal(f'eta_{factor_name}', mu=0, sigma=1, shape=data.n_obs)
            latent_factors_list.append(eta)
            
            loadings_raw = pm.Normal(f'lambda_{factor_name}', mu=0.7, sigma=0.5, shape=n_ind-1)
            all_loadings = pt.concatenate([pt.ones(1), loadings_raw])
            nu = pm.Normal(f'nu_{factor_name}', mu=0, sigma=2, shape=n_ind)
            psi = pm.HalfCauchy(f'psi_{factor_name}', beta=1, shape=n_ind)
            
            mu_ind = nu + pt.outer(eta, all_loadings)
            pm.Normal(f'obs_{factor_name}', mu=mu_ind, sigma=psi, observed=data.indicators_dict[factor_name])

        print("Building Structural Model (Multinomial Logit)...")
        beta_factors = pm.Normal('beta_factors', mu=0, sigma=1, shape=(data.n_factors, 3))
        beta_gender = pm.Normal('beta_gender', mu=0, sigma=1, shape=3)
        intercepts = pm.Normal('intercepts', mu=0, sigma=2, shape=3)
        
        factor_mat = pt.stack(latent_factors_list, axis=1) 
        gender_vec = pt.constant(data.dframe['gender_female'].values) 
        
        u_other = (intercepts + 
                   pt.dot(factor_mat, beta_factors) + 
                   pt.outer(gender_vec, beta_gender))
        
        u_ref = pt.zeros((data.n_obs, 1))
        utilities = pt.concatenate([u_ref, u_other], axis=1)
        
        p = pm.math.softmax(utilities, axis=1)
        pm.Categorical('outcome', p=p, observed=data.basket2_obs)

        print(f"Starting MCMC Sampling for {SPEC_ID}...")
        # High Target Accept to satisfy reviewer's convergence demands
        trace = pm.sample(draws=4000, tune=2000, chains=4, target_accept=0.99, random_seed=42)
        
        print(f"Saving results to {OUT_DIR}...")
        
        # Use az.to_netcdf instead of pm.to_netcdf
        az.to_netcdf(trace, os.path.join(OUT_DIR, "bayesian_sem_inference_data.nc"))
        
        summary = az.summary(trace, var_names=['beta_factors', 'beta_gender', 'intercepts'])
        summary.to_csv(os.path.join(OUT_DIR, "model_summary.csv"))

    print("✅ Multinomial Robustness Execution Complete.")  
    
    