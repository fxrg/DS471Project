# Breast Cancer Classification (DS471)

A machine learning and deep learning project for binary classification using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Project Structure

```text
data/
  raw/          # Raw data storage
  processed/    # Cleaned and normalized datasets
notebooks/      # Jupyter notebooks for experiments
reports/figures/# Visualizations and performance plots
src/            # Modular Python scripts
```

## Setup & Installation

Clone the repository:

```bash
git clone https://github.com/turki013/DS471Project.git
cd DS471Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start (EDA)

Run the initial exploratory data analysis script:

```bash
python src/initial_eda.py
```

## Preprocessing Pipeline

The authentic WDBC source file is stored locally as
`data/raw/breast_cancer.csv` and is ignored by Git. Run the preprocessing
pipeline to validate the input, encode `M`/`B` as `1`/`0`, create stratified
train/validation/test partitions, and standardize the 30 model features:

```bash
python src/preprocess.py
```

The generated `data/processed/train.csv`, `val.csv`, and `test.csv` files are
also ignored by Git because datasets are local artifacts. The identifier
column is excluded from model features, and the scaler is fitted only on the
training partition to prevent data leakage.
