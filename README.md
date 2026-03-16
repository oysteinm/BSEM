# BSEM: Bayesian SEM for In-Store Equipment Choice

**Manuscript ID:** JCM-08-2025-8245  
**Title:** *How Shopping Styles Influence In-Store Equipment Choice: Insights from Bayesian SEM Analysis*  
**Journal:** *Journal of Consumer Marketing*

This repository contains the replication materials for the Bayesian structural equation modeling (BSEM) analyses reported in the manuscript. It includes the estimation scripts, robustness checks, marginal-effects scripts, summary output files, and the supplementary appendix used to document model specification and additional results.

## Repository Structure

```text
BSEM/
├── appendix_model_fit_comparison.csv
├── basket.csv
├── basket_mc4_PS0_baseline.py
├── basket_mc4_PS1_tighter_structural.py
├── basket_mc4_PS2_wider_structural.py
├── basket_mc4_PS3_measurement_variance_sensitive.py
├── basket_multinomial_robustness.py
├── compare_ordered_vs_multinomial.py
├── descriptive_latent.py
├── marginal_effects_ps0_ps3_local.py
├── prior_sensitivity_posterior_comparison.csv
├── README.md
├── robustness_check_summary.csv
├── supplementary_appendix_v2.pdf
│
├── marginal_effects_out/
│   ├── marginal_effects_long_all_scenarios.csv
│   ├── marginal_effects_long_PS0_baseline.csv
│   ├── marginal_effects_long_PS1_tighter_structural.csv
│   ├── marginal_effects_long_PS2_wider_structural.csv
│   └── marginal_effects_long_PS3_measurement_variance_sensitive.csv
│
├── PS0_baseline/
│   ├── model_metadata.json
│   └── model_summary.csv
├── PS1_tighter_structural/
│   ├── model_metadata.json
│   └── model_summary.csv
├── PS2_wider_structural/
│   ├── model_metadata.json
│   └── model_summary.csv
├── PS3_measurement_variance_sensitive/
│   ├── model_metadata.json
│   └── model_summary.csv
└── ROBUSTNESS_multinomial_logit/
    └── model_summary.csv
```

## Overview

The repository is organized around four Bayesian SEM specifications and one multinomial robustness model:

* **PS0:** baseline prior specification
* **PS1:** tighter structural priors
* **PS2:** wider structural priors
* **PS3:** measurement-variance-sensitive specification (preferred model)
* **Multinomial robustness model:** non-ordinal alternative used to assess robustness of the ordered-logit results

The main outputs reproduced here are:

* fit comparisons across specifications
* posterior comparisons across prior settings
* ordered-logit versus multinomial robustness comparisons
* marginal probability effects for latent factors and control variables
* supplementary appendix materials

## Requirements

### Python

* Python 3.10+
* PyMC 5.x
* ArviZ
* NumPy
* Pandas
* SciPy
* scikit-learn

Install with:

```bash
pip install pymc arviz numpy pandas scipy scikit-learn
```

### R / Quarto

The supplementary appendix PDF can be regenerated if you have:

* R 4.3+
* `tidyverse`
* `knitr`
* `kableExtra`
* Quarto CLI
* a LaTeX distribution (for example TinyTeX)

## How to Reproduce the Main Results

### 1. Estimate the Bayesian SEM specifications

```bash
python basket_mc4_PS0_baseline.py
python basket_mc4_PS1_tighter_structural.py
python basket_mc4_PS2_wider_structural.py
python basket_mc4_PS3_measurement_variance_sensitive.py
```

### 2. Estimate the multinomial robustness model

```bash
python basket_multinomial_robustness.py
```

### 3. Generate descriptive statistics and marginal effects

```bash
python descriptive_latent.py
python marginal_effects_ps0_ps3_local.py
```

### 4. Build the comparison tables

```bash
python compare_ordered_vs_multinomial.py
```

## Prior Specifications Summary

| Spec    | Structural priors                  | Measurement priors                                          | Purpose                           |
| ------- | ---------------------------------- | ----------------------------------------------------------- | --------------------------------- |
| **PS0** | $\beta \sim \mathcal{N}(0, 1)$     | Standard                                                    | Baseline reference                |
| **PS1** | $\beta \sim \mathcal{N}(0, 0.5^2)$ | Standard                                                    | Tighter structural regularization |
| **PS2** | $\beta \sim \mathcal{N}(0, 2^2)$   | Standard                                                    | Wider structural regularization   |
| **PS3** | $\beta \sim \mathcal{N}(0, 1)$     | Tighter priors for Value Consciousness measurement variance | Preferred specification           |

## Reproducibility Note

Large intermediate files such as ArviZ `InferenceData` objects and temporary serialized outputs are not stored in the repository. Summary outputs and metadata are included, and the estimation scripts can be rerun to regenerate the reported results.

## Data and Usage

The file `basket.csv` is provided for research transparency and replication of the analyses reported in the manuscript. Please cite the manuscript when using these materials, and do not reuse the data for commercial purposes without permission.

## Citation

If you use this repository, please cite the associated manuscript:

> Anon & Anon: *How Shopping Styles Influence In-Store Equipment Choice: Insights from Bayesian SEM Analysis*. Journal of Consumer Marketing, revision under review.

## Contact

For questions about the replication materials, please contact the corresponding author.

