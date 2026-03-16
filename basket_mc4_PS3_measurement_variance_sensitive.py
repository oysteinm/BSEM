# -*- coding: utf-8 -*-
"""
Local deterministic Bayesian SEM runner (prior sensitivity spec)

This script is designed to run LOCALLY (Windows/Linux/macOS).

Outputs go to:
    <base_out_dir>/<SPEC_ID>/

Expected files per run:
    - bayesian_sem_inference_data.nc
    - processed_data.pkl
    - detailed_results_full.pkl
    - model_summary.csv
    - model_metadata.json

Usage (PowerShell):
    python .\basket_mc4_PS3_measurement_variance_sensitive.py
    python .\basket_mc4_PS3_measurement_variance_sensitive.py --force
    python .\basket_mc4_PS3_measurement_variance_sensitive.py --samples 2000 --tune 2000 --chains 4

IMPORTANT:
- You must edit INPUT_DATA_PATH below to point to your actual input data file.
- The model structure mirrors your basket_mc4.py but is made profile-aware:
    beta_sigma, lambda_mu, lambda_sigma, psi_sigma.

"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# Core Bayesian stack
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from sklearn.preprocessing import StandardScaler


# =============================================================================
# SPEC CONFIG (PS3)
# =============================================================================

SPEC_ID = "PS3_measurement_variance_sensitive"

PRIOR_PROFILE = "measurement_variance_sensitive"
PRIOR_CONFIG = {
    "beta_sigma": 1.0,
    "lambda_mu": 0.7,
    "lambda_sigma": 0.5,
    "psi_sigma": 2.0,
    "notes": "Sensitivity targeted at residual/measurement variance priors (psi_*), where mixing issues were observed."
}

# =============================================================================
# USER CONFIG (EDIT THESE)
# =============================================================================

from pathlib import Path

# Get the directory where the current script is located
BASE_DIR = Path(__file__).resolve().parent

# Set the output directory relative to the script location
DEFAULT_BASE_OUT_DIR = BASE_DIR

# Set the input data path relative to the script location
# Assumes basket.csv is in the same folder as the script
INPUT_DATA_PATH = BASE_DIR / "basket.csv"

# =============================================================================
# Helpers
# =============================================================================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def prior_spec_files(spec_dir: Path) -> Dict[str, Path]:
    spec_dir = Path(spec_dir)
    return {
        "idata_nc": spec_dir / "bayesian_sem_inference_data.nc",
        "processed_pkl": spec_dir / "processed_data.pkl",
        "detailed_results_pkl": spec_dir / "detailed_results_full.pkl",
        "model_summary_csv": spec_dir / "model_summary.csv",
        "metadata_json": spec_dir / "model_metadata.json",
    }


def run_complete(spec_dir: Path) -> bool:
    F = prior_spec_files(spec_dir)
    required = [
        F["idata_nc"],
        F["processed_pkl"],
        F["detailed_results_pkl"],
        F["model_summary_csv"],
        F["metadata_json"],
    ]
    return all(p.exists() for p in required)


# =============================================================================
# Data Loading / Preparation
# =============================================================================

def load_input_data(path: str) -> pd.DataFrame:
    """
    Load the input dataset.

    Supported:
    - .csv
    - .parquet
    - .pkl (pickle of a DataFrame)

    You MUST ensure the resulting df contains the needed columns used in prep_data().
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"INPUT_DATA_PATH does not exist: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".pkl":
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Pickle did not contain a pandas DataFrame.")
        return obj

    raise ValueError(f"Unsupported input format: {suffix}")


@dataclass
class PreparedData:
    dframe: pd.DataFrame
    basket2_obs: np.ndarray
    handledag_dummies: pd.DataFrame
    age_dummies: pd.DataFrame
    indicators_scaled: np.ndarray
    indicators_dict: Dict[str, np.ndarray]
    factor_indicators: Dict[str, List[str]]
    scaler: StandardScaler
    all_indicators: List[str]
    n_obs: int
    n_factors: int


def prep_data(dframe: pd.DataFrame) -> PreparedData:
    """
    Refined data preparation to handle local CSV structure with text-based 
    categorical variables and specific factor indicators.
    """
    df = dframe.copy()

    # --- 1. OUTCOME CODING (Ordered Outcome) ---
    # Map text categories to ordered integers 0-3 for OrderedLogistic
    basket_map = {'Arms': 0, 'Basket': 1, 'Cart': 2, 'Trolley': 3}
    if 'basket2' not in df.columns:
        raise KeyError("Column 'basket2' not found in CSV.")
    
    basket2_obs = df['basket2'].map(basket_map).astype(int).to_numpy()

    # --- 2. CATEGORICAL DUMMIES ---
    # Gender: Convert "Male"/"Female" strings to 0/1 indicator
    df['gender_female'] = (df['gender'] == 'Female').astype(int)

    # Handledag: Create dummies, drop Saturday as the reference category
    handledag_dummies = pd.get_dummies(df['handledag'], prefix='handledag')
    if 'handledag_Saturday' in handledag_dummies.columns:
        handledag_dummies = handledag_dummies.drop('handledag_Saturday', axis=1)

    # Age: Use existing text categories (e.g., '31-35 yr') as dummy variables
    # This automatically handles the grouping found in your CSV
    age_dummies = pd.get_dummies(df['age'], prefix='age', drop_first=True)

    # --- 3. LATENT FACTOR STRUCTURE ---
    # Map factors to the actual column headers found in your basket.csv
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

    # --- 4. SCALING AND PACKAGING ---
    all_indicators = [item for sublist in factor_indicators.values() for item in sublist]
    
    # Scale all indicator variables to mean 0 and variance 1
    X = df[all_indicators].astype(float).to_numpy()
    scaler = StandardScaler()
    indicators_scaled = scaler.fit_transform(X)

    # Build dictionary of scaled matrices for the measurement model
    indicators_dict = {}
    idx = 0
    for factor_name, cols in factor_indicators.items():
        k = len(cols)
        indicators_dict[factor_name] = indicators_scaled[:, idx: idx + k]
        idx += k

    return PreparedData(
        dframe=df,
        basket2_obs=basket2_obs,
        handledag_dummies=handledag_dummies,
        age_dummies=age_dummies,
        indicators_scaled=indicators_scaled,
        indicators_dict=indicators_dict,
        factor_indicators=factor_indicators,
        scaler=scaler,
        all_indicators=all_indicators,
        n_obs=len(df),
        n_factors=len(factor_indicators)
    )

    # -------------------------
    # REQUIRED COLUMNS (adjust to your actual schema)
    # -------------------------
    required_cols = ["basket2", "gender_female", "handledag", "age"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns. You must adapt prep_data() to your schema. "
            f"Missing: {missing}"
        )

    # Outcome: ensure 0..3 coding for OrderedLogistic with 4 categories
    y = df["basket2"].to_numpy()
    # If your data is 1..4, convert to 0..3
    if y.min() == 1 and y.max() == 4:
        y = y - 1
    if not set(np.unique(y)).issubset({0, 1, 2, 3}):
        raise ValueError("basket2 must be coded as 0..3 (or 1..4 which will be converted).")
    basket2_obs = y.astype(int)

    # handledag dummies
    handledag_dummies = pd.get_dummies(df["handledag"], prefix="handledag", drop_first=True)

    # age grouping: 56+ as one group; you can adjust bins
    # Example bins; replace with your original categorization if needed.
    age = df["age"].astype(float)
    bins = [-np.inf, 25, 35, 45, 55, np.inf]
    labels = ["<=25", "26-35", "36-45", "46-55", "56plus"]
    age_group = pd.cut(age, bins=bins, labels=labels, right=True)
    age_dummies = pd.get_dummies(age_group, prefix="age", drop_first=True)

    # -------------------------
    # INDICATOR MODEL SETUP
    # -------------------------
    # You MUST adapt factor_indicators to your real indicator columns.
    # Below is a placeholder example. Replace with your actual indicator list.
    #
    # Each factor must have >=1 indicator. If >1, we estimate loadings for indicators 2..k
    # with the first loading fixed to 1.0 for scale identification.
    factor_indicators = {
        # examples (replace!)
        "quality": ["quality_1", "quality_2", "quality_3"],
        "price": ["price_1", "price_2", "price_3"],
        "convenience": ["convenience_1", "convenience_2", "convenience_3"],
        "value": ["value_1", "value_2", "value_3"],
    }

    all_indicators: List[str] = []
    for f, cols in factor_indicators.items():
        all_indicators.extend(cols)

    missing_ind = [c for c in all_indicators if c not in df.columns]
    if missing_ind:
        raise KeyError(
            "Missing indicator columns. You must edit factor_indicators in prep_data(). "
            f"Missing: {missing_ind[:20]}" + (" ..." if len(missing_ind) > 20 else "")
        )

    # scale indicators
    X = df[all_indicators].astype(float).to_numpy()
    scaler = StandardScaler()
    indicators_scaled = scaler.fit_transform(X)

    # build indicators_dict: factor -> matrix (n_obs x n_indicators)
    indicators_dict: Dict[str, np.ndarray] = {}
    idx = 0
    for factor_name, cols in factor_indicators.items():
        k = len(cols)
        indicators_dict[factor_name] = indicators_scaled[:, idx: idx + k]
        idx += k

    n_obs = df.shape[0]
    n_factors = len(factor_indicators)

    return PreparedData(
        dframe=df,
        basket2_obs=basket2_obs,
        handledag_dummies=handledag_dummies,
        age_dummies=age_dummies,
        indicators_scaled=indicators_scaled,
        indicators_dict=indicators_dict,
        factor_indicators=factor_indicators,
        scaler=scaler,
        all_indicators=all_indicators,
        n_obs=n_obs,
        n_factors=n_factors,
    )


# =============================================================================
# Model
# =============================================================================

def build_and_sample(
    prep: PreparedData,
    prior_config: Dict[str, Any],
    chains: int,
    draws: int,
    tune: int,
    target_accept: float,
    random_seed: int,
) -> az.InferenceData:
    """
    Build the SEM model with profile-aware priors and sample it.
    """
    df = prep.dframe
    basket2_obs = prep.basket2_obs
    handledag_dummies = prep.handledag_dummies
    age_dummies = prep.age_dummies
    indicators_dict = prep.indicators_dict
    factor_indicators = prep.factor_indicators
    n_obs = prep.n_obs
    n_factors = prep.n_factors

    beta_sigma = float(prior_config.get("beta_sigma", 1.0))
    lambda_mu = float(prior_config.get("lambda_mu", 0.7))
    lambda_sigma = float(prior_config.get("lambda_sigma", 0.5))
    psi_sigma = float(prior_config.get("psi_sigma", 1.0))

    with pm.Model() as sem_model_with_value:

        latent_factors: Dict[str, pm.Distribution] = {}

        for factor_name, indicators in factor_indicators.items():
            n_indicators = len(indicators)

            latent_factors[f"eta_{factor_name}"] = pm.Normal(
                f"eta_{factor_name}", mu=0, sigma=1, shape=n_obs
            )

            # Loadings
            if n_indicators > 1:
                if factor_name == "value":
                    # keep special handling but profile-aware
                    loadings = pm.TruncatedNormal(
                        f"lambda_{factor_name}",
                        mu=lambda_mu,
                        sigma=min(lambda_sigma, 1.0),
                        lower=0.1,
                        upper=2.0,
                        shape=n_indicators - 1,
                    )
                else:
                    loadings = pm.Normal(
                        f"lambda_{factor_name}",
                        mu=lambda_mu,
                        sigma=lambda_sigma,
                        shape=n_indicators - 1,
                    )
                all_loadings = pt.concatenate([pt.ones(1), loadings])
            else:
                all_loadings = pt.ones(1)

            intercepts = pm.Normal(f"nu_{factor_name}", mu=0, sigma=2, shape=n_indicators)

            # Measurement error variances:
            # Use HalfNormal so psi_sigma directly controls sensitivity (important for PS3).
            if factor_name == "value":
                error_vars = pm.HalfNormal(
                    f"psi_{factor_name}", sigma=min(psi_sigma, 1.0), shape=n_indicators
                )
            else:
                error_vars = pm.HalfNormal(f"psi_{factor_name}", sigma=psi_sigma, shape=n_indicators)

            X_factor = indicators_dict[factor_name]
            eta = latent_factors[f"eta_{factor_name}"]
            for j in range(n_indicators):
                mu_ind = intercepts[j] + all_loadings[j] * eta
                pm.Normal(
                    f"{factor_name}_ind_{j}",
                    mu=mu_ind,
                    sigma=error_vars[j],
                    observed=X_factor[:, j],
                )

        # Structural priors
        beta_factors = pm.Normal("beta_factors", mu=0, sigma=beta_sigma, shape=n_factors)

        # handledag / age / gender effects
        if handledag_dummies.shape[1] > 0:
            beta_handledag = pm.Normal(
                "beta_handledag", mu=0, sigma=beta_sigma, shape=handledag_dummies.shape[1]
            )
            handledag_effect = pt.dot(handledag_dummies.to_numpy(), beta_handledag)
        else:
            handledag_effect = 0

        if age_dummies.shape[1] > 0:
            beta_age = pm.Normal("beta_age", mu=0, sigma=beta_sigma, shape=age_dummies.shape[1])
            age_effect = pt.dot(age_dummies.to_numpy(), beta_age)
        else:
            age_effect = 0

        beta_gender = pm.Normal("beta_gender", mu=0, sigma=beta_sigma)
        gender_effect = beta_gender * df["gender_female"].to_numpy()

        # Stack latent factors in the same order as factor_indicators keys
        eta_stack = pt.stack([latent_factors[f"eta_{nm}"] for nm in factor_indicators.keys()], axis=1)
        eta_basket = pt.dot(eta_stack, beta_factors) + handledag_effect + age_effect + gender_effect

        cutpoints = pm.Normal(
            "cutpoints",
            mu=[-1, 0, 1],
            sigma=1.5,
            shape=3,
            transform=pm.distributions.transforms.ordered,
        )

        basket2_pred = pm.OrderedLogistic(
            "basket2_pred",
            eta=eta_basket,
            cutpoints=cutpoints,
            observed=basket2_obs,
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    return idata


# =============================================================================
# Save outputs (deterministic)
# =============================================================================

def save_outputs(
    spec_dir: Path,
    prep: PreparedData,
    idata: az.InferenceData,
    chains: int,
    draws: int,
    tune: int,
    target_accept: float,
    prior_profile: str,
    prior_config: Dict[str, Any],
    input_data_path: str,
) -> None:
    F = prior_spec_files(spec_dir)

    # 1) InferenceData
    az.to_netcdf(idata, F["idata_nc"])

    # 2) Processed data
    data_dict = {
        "dframe_original": prep.dframe,
        "basket2_obs": prep.basket2_obs,
        "handledag_dummies": prep.handledag_dummies,
        "age_dummies": prep.age_dummies,
        "gender_female": prep.dframe["gender_female"].to_numpy(),
        "indicators_scaled": prep.indicators_scaled,
        "indicators_dict": prep.indicators_dict,
        "factor_indicators": prep.factor_indicators,
        "scaler": prep.scaler,
        "all_indicators": prep.all_indicators,
        "prior_profile": prior_profile,
        "prior_config": prior_config,
        "spec_id": SPEC_ID,
    }
    with open(F["processed_pkl"], "wb") as f:
        pickle.dump(data_dict, f)

    # 3) Model summary
    var_names_summary = ["beta_factors", "beta_gender", "cutpoints"]
    if prep.handledag_dummies.shape[1] > 0:
        var_names_summary.append("beta_handledag")
    if prep.age_dummies.shape[1] > 0:
        var_names_summary.append("beta_age")
    if "value" in prep.factor_indicators:
        var_names_summary.extend(["lambda_value", "psi_value"])

    summary_df = az.summary(idata, var_names=var_names_summary)
    summary_df.to_csv(F["model_summary_csv"])

    # 4) Detailed results (simple, consistent with earlier appendix approach)
    posterior = idata.posterior
    results_dict: Dict[str, Any] = {}

    # factor effects
    factor_results = {}
    beta_factors_data = posterior["beta_factors"]
    for i, factor_name in enumerate(prep.factor_indicators.keys()):
        beta_factor_i = beta_factors_data.isel(beta_factors_dim_0=i)
        samples = beta_factor_i.values.flatten()
        factor_results[factor_name] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "hdi_2.5": float(np.percentile(samples, 2.5)),
            "hdi_97.5": float(np.percentile(samples, 97.5)),
            "prob_positive": float((samples > 0).mean()),
        }
    results_dict["factor_effects"] = factor_results

    # gender
    gender_samples = posterior["beta_gender"].values.flatten()
    results_dict["gender_effect"] = {
        "mean": float(np.mean(gender_samples)),
        "std": float(np.std(gender_samples)),
        "hdi_2.5": float(np.percentile(gender_samples, 2.5)),
        "hdi_97.5": float(np.percentile(gender_samples, 97.5)),
        "prob_positive": float((gender_samples > 0).mean()),
    }

    # handledag
    if "beta_handledag" in posterior:
        handledag_results = {}
        beta_handledag_data = posterior["beta_handledag"]
        for i, day_col in enumerate(prep.handledag_dummies.columns):
            day_name = str(day_col).replace("handledag_", "")
            beta_day_i = beta_handledag_data.isel(beta_handledag_dim_0=i)
            samples = beta_day_i.values.flatten()
            handledag_results[day_name] = {
                "mean": float(np.mean(samples)),
                "hdi_2.5": float(np.percentile(samples, 2.5)),
                "hdi_97.5": float(np.percentile(samples, 97.5)),
                "prob_positive": float((samples > 0).mean()),
            }
        results_dict["handledag_effects"] = handledag_results

    # age
    if "beta_age" in posterior:
        age_results = {}
        beta_age_data = posterior["beta_age"]
        for i, age_col in enumerate(prep.age_dummies.columns):
            age_name = str(age_col).replace("age_", "")
            beta_age_i = beta_age_data.isel(beta_age_dim_0=i)
            samples = beta_age_i.values.flatten()
            age_results[age_name] = {
                "mean": float(np.mean(samples)),
                "hdi_2.5": float(np.percentile(samples, 2.5)),
                "hdi_97.5": float(np.percentile(samples, 97.5)),
                "prob_positive": float((samples > 0).mean()),
            }
        results_dict["age_effects"] = age_results

    with open(F["detailed_results_pkl"], "wb") as f:
        pickle.dump(results_dict, f)

    # 5) Metadata
    model_metadata = {
        "created_at": datetime.now().isoformat(),
        "spec_id": SPEC_ID,
        "prior_profile": prior_profile,
        "prior_config": prior_config,
        "input_data_path": input_data_path,
        "n_observations": int(prep.n_obs),
        "n_factors": int(prep.n_factors),
        "factor_names": list(prep.factor_indicators.keys()),
        "chains": int(chains),
        "draws": int(draws),
        "tune": int(tune),
        "target_accept": float(target_accept),
        "platform": {
            "python": platform.python_version(),
            "os": platform.platform(),
        },
        "outputs": {k: str(v) for k, v in prior_spec_files(spec_dir).items()},
        "notes": prior_config.get("notes", ""),
    }
    save_json(model_metadata, F["metadata_json"])


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_out_dir", type=str, default=DEFAULT_BASE_OUT_DIR)
    parser.add_argument("--input_data", type=str, default=INPUT_DATA_PATH)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--tune", type=int, default=3000)
    parser.add_argument("--target_accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_out_dir = Path(args.base_out_dir)
    spec_dir = _ensure_dir(base_out_dir / SPEC_ID)

    print("=" * 80)
    print("RUN SPEC:", SPEC_ID)
    print("PRIOR_PROFILE:", PRIOR_PROFILE)
    print("PRIOR_CONFIG:", json.dumps(PRIOR_CONFIG, indent=2))
    print("INPUT_DATA:", args.input_data)
    print("OUT_DIR   :", spec_dir)
    print("=" * 80)

    if run_complete(spec_dir) and not args.force:
        print("✅ Outputs already exist (restart-safe). Use --force to re-run.")
        return

    df = load_input_data(args.input_data)
    prep = prep_data(df)

    idata = build_and_sample(
        prep=prep,
        prior_config=PRIOR_CONFIG,
        chains=args.chains,
        draws=args.samples,
        tune=args.tune,
        target_accept=args.target_accept,
        random_seed=args.seed,
    )

    save_outputs(
        spec_dir=spec_dir,
        prep=prep,
        idata=idata,
        chains=args.chains,
        draws=args.samples,
        tune=args.tune,
        target_accept=args.target_accept,
        prior_profile=PRIOR_PROFILE,
        prior_config=PRIOR_CONFIG,
        input_data_path=args.input_data,
    )

    print("✅ Done. Files written to:", spec_dir)


if __name__ == "__main__":
    main()