<div align="center">

# 🎗️ Breast Cancer Classification

### Machine Learning & Deep Learning on the Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

**DS471 · Course Project**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

</div>

---

## 📌 Overview

This project builds and compares machine learning and deep learning models to classify breast tumors as **malignant (M)** or **benign (B)** using the 30 numeric features of the WDBC dataset, computed from digitized images of fine needle aspirates (FNA) of breast masses.

**Goals**

- Perform exploratory data analysis (EDA) to understand feature distributions and class balance.
- Build a leakage-free preprocessing pipeline.
- Train and compare classical ML and deep learning models.
- Evaluate with metrics that matter in a medical setting (recall, precision, F1, ROC-AUC).

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Wisconsin Diagnostic Breast Cancer (WDBC) |
| Samples | 569 |
| Features | 30 numeric features |
| Target | `diagnosis` → `M` (malignant) = 1, `B` (benign) = 0 |
| Excluded | Identifier column (`id`) |

> The raw file is stored locally at `data/raw/breast_cancer.csv` and is **not tracked by Git**.

---

## 🗂️ Project Structure

```text
DS471Project/
├── data/
│   ├── raw/            # Raw data (local only, ignored by Git)
│   └── processed/      # train / val / test splits (local only, ignored by Git)
├── notebooks/          # Jupyter notebooks for experiments
├── reports/
│   └── figures/        # Visualizations and performance plots
├── src/
│   ├── initial_eda.py  # Exploratory data analysis
│   └── preprocess.py   # Preprocessing pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**

```bash
git clone https://github.com/turki013/DS471Project.git
cd DS471Project
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add the dataset**

Place the WDBC file at:

```text
data/raw/breast_cancer.csv
```

---

## 🚀 How to Run

### 1) Exploratory Data Analysis

```bash
python src/initial_eda.py
```

### 2) Preprocessing Pipeline

```bash
python src/preprocess.py
```

This step will:

1. Validate the input file.
2. Encode the target: `M → 1`, `B → 0`.
3. Create **stratified** train / validation / test splits.
4. Standardize the 30 model features.

**Outputs** (local only, ignored by Git):

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

---

## 🛡️ Data Leakage Prevention

- The identifier column is **excluded** from the model features.
- The scaler is **fitted on the training set only**, then applied to validation and test sets.
- Splits are **stratified** to preserve the class ratio across partitions.

---

## 🤖 Models

| Type | Model | Status |
|---|---|---|
| Machine Learning | Logistic Regression| ⏳ |
| Machine Learning |  Random Forest | ⏳ |
| Deep Learning |  Multi-Layer Perceptron (Neural Network) | ⏳ |

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | – | – | – | – | – |
| Random Forest | – | – | – | – | – |
| Multi-Layer Perceptron (MLP) | – | – | – | – | – |

> Plots and figures are available in `reports/figures/`.


## 📚 References

- Wolberg, W. H., Street, W. N., & Mangasarian, O. L. — *Wisconsin Diagnostic Breast Cancer (WDBC)*, UCI Machine Learning Repository.

---

<div align="center">

Made for DS471 

</div>
