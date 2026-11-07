# DS471 Machine Learning Project

## Selected dataset

This project uses the **UCI Breast Cancer Wisconsin (Diagnostic)** dataset for
binary classification. It contains diagnostic measurements computed from
digitized breast-mass images and a binary diagnosis target.

The initial EDA script uses the verified `scikit-learn` copy of this UCI
dataset by default so that the workflow is reproducible without committing
data to GitHub. A local CSV can be supplied later when the group downloads the
dataset from UCI.

## Project structure

```text
data/
  raw/          # local source data; ignored by Git
  processed/    # cleaned data; ignored by Git
notebooks/      # analysis notebooks
reports/
  figures/      # generated charts
src/            # reusable project scripts
```

## Environment setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Day 1: initial EDA

Run the default UCI-backed dataset:

```bash
python src/initial_eda.py
```

Run the same analysis on a local CSV:

```bash
python src/initial_eda.py --data-path data/raw/breast_cancer.csv --target diagnosis
```

The script prints the dataset shape, schema, missing-value counts,
descriptive statistics, and an initial observations summary. Cleaning,
encoding, scaling, feature engineering, splitting, model development, and
evaluation will be documented in later project stages.

## Requirements alignment

The project will document all data-cleaning and preprocessing decisions,
justify the train/validation/test split, include at least one neural network
and two traditional machine-learning models, tune hyperparameters, report
appropriate evaluation metrics, and include training/validation curves for the
neural network.
