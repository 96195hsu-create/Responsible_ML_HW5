# Individual Homework 05: Applied Security Audit on COMPAS

This repository contains my submission for **Responsible Machine Learning — Individual Homework 05**. The project uses the **ProPublica COMPAS dataset** and extends **adversarial robustness, fairness degradation, poisoning risk, and membership inference leakage**.

The notebook is written as an **audit notebook** rather than a pure coding exercise: each section connects technical outputs to model-risk interpretation, governance implications, and deployment decisions.

## Project Objective

The purpose of this assignment is to evaluate how machine learning systems can fail under adversarial pressure even when traditional performance metrics appear stable. Using logistic regression (`LR`) and gradient boosted trees (`GBT`) trained on the COMPAS recidivism dataset, the notebook answers four questions:

1. How do **PGD evasion attacks** affect group-level fairness metrics across model classes?
2. Can a **label-flip poisoning attack** push fairness outside an acceptable range while leaving AUC nearly unchanged?
3. Does the **generalization gap** predict **membership inference leakage**, and how does LR regularization affect that tradeoff?
4. Which finding creates the **highest deployment risk**, and what governance controls are justified by the evidence?

## Main Deliverable

- Notebook: [L5_Jessie_Hsu_ASSN5.ipynb]

## Analytical Scope

The notebook is organized into the following sections:

- `Setup`: data loading, preprocessing, train/test split, and clean-model fairness baseline
- `Part 1`: PGD evasion audit on both `LR` and `GBT`
- `Part 2`: poisoning loop with fairness monitoring for two target-race variants
- `Part 3`: shadow-model membership inference for both models plus LR `L2` regularization sweep
- `Part 4`: reflection and governance recommendations

## Data and Modeling Setup

The analysis uses the ProPublica COMPAS two-year recidivism dataset. The preprocessing pipeline follows the lecture approach:

- filter `days_b_screening_arrest` to `[-30, 30]`
- remove observations with `is_recid = -1`
- remove charge degree `O`
- model `two_year_recid` as the prediction target
- include demographic and criminal-history covariates used in lecture

Two baseline models are trained:

- `LR`: Logistic Regression
- `GBT`: Gradient Boosting Classifier

Core evaluation metrics:

- `Train AUC`
- `Test AUC`
- `Generalization gap = train AUC - test AUC`
- `FPR by race`
- `AIR = FPR_African-American / FPR_Caucasian`

The clean `LR` baseline AIR reported in the notebook is **1.961**, which already indicates a substantial racial false-positive imbalance before any adversarial manipulation.

## Key Findings

### Part 1: PGD Evasion Audit

The PGD attack is run across `epsilon ∈ {0.25, 0.5, 1.0, 2.0}` for both `LR` and `GBT`.

Main conclusion:

- Neither model crosses the `AIR = 0.80` threshold in the tested epsilon range.
- However, the two models are **not equally vulnerable**.
- `LR` experiences a larger average AIR drop from its clean baseline.
- `GBT` remains comparatively more stable under the same perturbation strengths.

Interpretation:

- model selection in high-stakes settings should not be based only on clean predictive performance
- robustness should be evaluated jointly with fairness sensitivity under adversarial stress
- even when threshold failure is not reached, relative degradation still matters for deployment choice

### Part 2: Poisoning Loop with Fairness Monitoring

The label-flip poisoning attack is extended to compare two target-race variants:

- target `African-American`
- target `Caucasian`

Main conclusion:

- there is a meaningful **stealth-zone problem**
- under the notebook's practical interpretation, fairness can worsen while AUC changes only minimally
- PSI does **not** detect the attack because the attack modifies **labels**, not features

Evidence from the notebook:

- stealth zone for `African-American` targeting: poison rate **0.02 to 0.30**
- stealth zone for `Caucasian` targeting: poison rate **0.02 to 0.08**
- PSI remains effectively zero because feature distributions are unchanged by construction

Interpretation:

- performance monitoring alone is insufficient for training-time integrity risk
- feature-drift metrics such as PSI are structurally incapable of detecting label-only attacks
- fairness monitoring must be treated as a first-class control, not a secondary diagnostic

### Part 3: Membership Inference Depth

The notebook computes **shadow-model membership inference AUC** for both `LR` and `GBT`, then tests whether generalization gap predicts leakage and evaluates the effect of `L2` regularization on LR.

Main conclusion:

- both `LR` and `GBT` have MI AUC values close to **0.50**
- in this experiment, membership inference is only marginally better than random guessing
- `GBT` shows the larger generalization gap and slightly higher MI AUC, which is directionally consistent with lecture, but the evidence is weak rather than decisive

For LR regularization:

- weaker regularization improves test AUC slightly
- MI AUC changes only marginally
- the expected privacy-utility tradeoff is present in form, but not strong in magnitude in this run

Interpretation:

- the relationship between overfitting and privacy leakage is conceptually supported, but empirical strength depends on model behavior
- leakage analysis should be interpreted quantitatively rather than assumed from theory alone
- regularization can be a useful privacy control, but its operational value depends on whether leakage is materially above random baseline

### Part 4: Highest-Risk Finding

The strongest risk identified in the notebook is the **poisoning-driven fairness monitoring gap**.

Notebook evidence:

- in the worst poisoning case (`poison_rate = 0.30`, targeting `African-American`), LR AUC drops by only about **0.0039**
- over the same comparison, AIR increases from **1.961** to **2.860**

Why this matters:

- a conventional monitoring workflow could interpret that AUC movement as negligible
- yet the model's disparate impact becomes materially worse
- this creates a governance failure mode in which harm increases without triggering review

## Governance Implications

The notebook supports two operational recommendations.

### Proactive Mitigation

Add a **fairness gate** before deployment or model promotion.

Recommended rule:

- do not approve a retrained model based only on AUC stability
- require review whenever AIR deteriorates materially relative to the clean baseline
- maintain explicit group-level validation as part of release criteria

Rationale:

- the poisoning results show that acceptable-looking AUC can coexist with worsening group harm

### Reactive Mitigation

Add a **group-metric monitoring and rollback trigger** in production.

Recommended response:

- monitor AIR and group-specific FPR over time
- investigate sharp deviations even when aggregate performance remains stable
- maintain rollback capability to the last trusted model version

Rationale:

- the notebook shows that fairness degradation may emerge without corresponding PSI alarms

## Repository Contents

- [L5_Jessie_Hsu_ASSN5.ipynb]: final notebook submission
- [README.md]: project overview and interpretation guide

## Reproducibility

Suggested Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `statsmodels`

To reproduce the analysis:

1. Open [L5_Jessie_Hsu_ASSN5.ipynb]
2. Run the notebook from top to bottom in a clean session
3. Confirm that the dataset loads from the GitHub source
4. Review the tables, plots, and markdown conclusions in each part

## Limitations

This notebook should be read as a structured course audit, not as a production validation study.

Key limitations include:

- results are based on one dataset and one preprocessing pipeline
- fairness is summarized mainly through `FPR` and `AIR`, not a full fairness metric suite
- PGD on `GBT` uses a finite-difference approximation rather than analytic gradients
- membership inference is tested across two model families only, so the generalization-gap relationship is suggestive rather than statistically general

## Closing Note

The main business lesson from this assignment is that **stable AUC does not imply stable risk**. In a high-stakes setting, the more consequential question is often whether an attack can change **who is harmed**, **whether monitoring would notice**, and **what governance control would have stopped it**. This notebook is designed to answer exactly those questions.
