# 🔍 Responsible AI: Loan Approval Fairness Analysis

> Internship project analyzing model fairness, bias, and explainability using SHAP/LIME techniques with practical mitigation strategies.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Results](#results)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)

## 🎯 Overview

This project demonstrates end-to-end responsible AI practices on a synthetic loan approval dataset with **injected systemic bias** to simulate real-world discrimination patterns.

**Goal:** Analyze model fairness, detect bias across sensitive groups (gender, race), explain predictions using interpretability techniques, and propose mitigation strategies.

## 📊 Dataset

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Applicant age (18-75) |
| `income` | Numeric | Annual income in USD |
| `credit_score` | Numeric | FICO score (300-850) |
| `employment_years` | Numeric | Years at current job |
| `debt_to_income` | Numeric | DTI ratio (%) |
| `loan_amount` | Numeric | Requested loan amount |
| `gender` | Categorical | Male / Female |
| `race` | Categorical | White, Black, Asian, Hispanic, Other |
| `education` | Categorical | High School, Bachelor, Master, PhD |
| `approved` | Binary | Target: 1 = approved, 0 = rejected |

**Injected Bias:**
- Male applicants: +8% approval boost
- Female applicants: -5% penalty
- Black applicants: -8% vs White +5%

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Distribution analysis
- Approval rate comparison across groups
- Statistical significance tests (chi-square)

### 2. Model Training
- Gradient Boosting Classifier
- 75/25 train-test split
- Feature encoding for categorical variables

### 3. Interpretability (SHAP / LIME)
- **Global:** Feature importance ranking
- **Local:** Individual prediction explanations
- Detection of sensitive attribute usage

### 4. Fairness Metrics
| Metric | Description |
|--------|-------------|
| Demographic Parity | Equal approval rates across groups |
| Equal Opportunity | Equal TPR across groups |
| Equalized Odds | Equal TPR and FPR across groups |

### 5. Mitigation Strategies
1. **Remove Sensitive Features** — Exclude protected attributes
2. **Reweighting** — Balance group influence
3. **Threshold Optimization** — Group-specific decision boundaries

## 📈 Key Findings

### Bias Detection
- **Gender gap:** 14.0 percentage points (Female 52.6% vs Male 66.6%)
- **Race gap:** 15.4 percentage points (Black 50.2% vs Asian 63.8%)
- **Statistical significance:** p < 0.001 for both

### Model Interpretability
- Credit score (28%) and income (22%) are top predictors
- **Race encoding contributes 6%** — model learned discriminatory patterns
- **Gender encoding contributes 4%** — algorithmic bias replication

### Fairness Metrics
| Group | Approval Rate | Precision | Recall | FPR | FNR |
|-------|--------------|-----------|--------|-----|-----|
| Female | 0.518 | 0.589 | 0.712 | 0.352 | 0.288 |
| Male | 0.658 | 0.632 | 0.801 | 0.289 | 0.199 |
| Black | 0.491 | 0.556 | 0.689 | 0.378 | 0.311 |
| White | 0.642 | 0.638 | 0.812 | 0.276 | 0.188 |

## 🛡️ Mitigation Results

| Strategy | Accuracy | Gender Gap | Race Gap | Improvement |
|----------|----------|------------|----------|-------------|
| Baseline | 0.570 | 0.140 | 0.154 | — |
| Remove Features | 0.565 | 0.098 | 0.112 | 30% |
| Reweighting | 0.558 | 0.076 | 0.089 | 46% |
| **Threshold Optimization** | **0.552** | **0.045** | **0.067** | **68%** |

## 🖼️ Visualizations

<p align="center">
  <img src="outputs/01_eda_bias_detection.png" width="800"/>
  <br><em>EDA & Bias Detection</em>
</p>

<p align="center">
  <img src="outputs/03_local_explanations.png" width="700"/>
  <br><em>Model Interpretability</em>
</p>

<p align="center">
  <img src="outputs/05_mitigation_comparison.png" width="700"/>
  <br><em>Mitigation Strategy Comparison</em>
</p>

## 🚀 How to Run

### Prerequisites
```bash
Python 3.9+
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/responsible-ai-loan-fairness.git
cd responsible-ai-loan-fairness
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook notebooks/responsible_ai_analysis.ipynb
```

### Run Individual Scripts
```bash
python src/01_generate_data.py
python src/02_eda_bias_detection.py
python src/03_model_training.py
python src/04_interpretability.py
python src/05_fairness_metrics.py
python src/06_mitigation.py
```

## 📁 Project Structure

```
responsible-ai-loan-fairness/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── notebooks/
│   └── responsible_ai_analysis.ipynb  # Main analysis notebook
├── src/
│   ├── 01_generate_data.py            # Synthetic dataset generation
│   ├── 02_eda_bias_detection.py       # EDA & bias detection
│   ├── 03_model_training.py           # Model training & evaluation
│   ├── 04_interpretability.py         # SHAP/LIME explanations
│   ├── 05_fairness_metrics.py         # Fairness quantification
│   └── 06_mitigation.py               # Bias mitigation strategies
├── outputs/
│   ├── 01_eda_bias_detection.png
│   ├── 02_feature_importance.png
│   ├── 03_local_explanations.png
│   ├── 04_fairness_metrics.png
│   └── 05_mitigation_comparison.png
├── reports/
│   └── bias_mitigation_writeup.md     # Detailed write-up
└── data/
    └── loan_approval_data.csv         # Generated dataset
```

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| NumPy / Pandas | Data manipulation |
| scikit-learn | Machine learning |
| Matplotlib / Seaborn | Visualization |
| SHAP | Model interpretability |
| LIME | Local explanations |
| Jupyter | Interactive analysis |

## 📄 License

MIT License — feel free to use for your own projects and internships.

## 🙋 About

Built as part of an internship project on **Responsible AI & Model Interpretation**.
Focus areas: fairness auditing, explainable AI, bias mitigation.
