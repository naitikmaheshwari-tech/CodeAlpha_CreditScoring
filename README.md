# 💳 CreditIQ — Machine Learning Credit Scoring Model
### CodeAlpha Machine Learning Internship Project

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-indigo.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Executive Overview
**CreditIQ** is an enterprise-grade Machine Learning system and interactive web application designed to automate and explain credit underwriting decisions. Built for the **CodeAlpha Machine Learning Internship**, it demonstrates a complete, leak-free machine learning lifecycle:
1. **Realistic Synthetic Data Generation** mirroring real-world consumer credit bureau distributions.
2. **Robust Preprocessing Pipeline** with median imputation and feature standardization.
3. **Optimized Ensemble Classification** using Random Forest, benchmarked against Gradient Boosting and Logistic Regression.
4. **Calibrated FICO-style Scoring Engine** mapping underwriting probabilities into 300–850 credit scores with tiered risk categories.
5. **Interactive Streamlit Web Dashboard** providing real-time credit scoring, instant applicant presets, automated risk factor explanations, global feature importance charts, and batch CSV applicant evaluation.

---

## 🏗️ Project Architecture & Directory Structure

```text
CodeAlpha_CreditScoring/
├── data/
│   └── credit_scoring_dataset.csv       # Generated synthetic dataset (5,000+ records)
├── models/
│   ├── credit_scoring_model.joblib      # End-to-end trained Pipeline (Imputer + Scaler + RF)
│   ├── preprocessor.joblib              # Serialized scikit-learn ColumnTransformer
│   ├── rf_classifier.joblib             # Serialized Random Forest Classifier
│   ├── feature_importance.csv           # Gini feature importance rankings
│   └── model_metadata.json              # Evaluation metrics, ROC curve data & hyperparameters
├── src/
│   ├── __init__.py                      # Package initializer
│   ├── data_generator.py                # Synthetic credit bureau data generator
│   ├── model_trainer.py                 # Preprocessing, ML pipeline, training & benchmarks
│   └── predictor.py                     # High-level inference engine, FICO score & risk analysis
├── app.py                               # Interactive modern Streamlit web application
├── train.py                             # CLI executable training and evaluation script
├── requirements.txt                     # Production Python dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 🔬 Dataset & Feature Dictionary

The synthetic dataset synthesizes realistic consumer credit profiles based on empirical underwriting principles (FICO methodology):

| Feature Name | Type | Description | Range / Unit |
| :--- | :--- | :--- | :--- |
| `Age` | Integer | Chronological age of applicant | 21 – 72 years |
| `Annual_Income` | Float | Gross annual income (log-normal distribution) | $18,000 – $260,000 |
| `Total_Debt` | Float | Aggregate outstanding debt liabilities | $500 – $220,000 |
| `Num_Open_Accounts` | Integer | Total open credit lines and loans | 1 – 25 accounts |
| `Credit_History_Length` | Float | Age of oldest active credit account | 0.5 – 35.0 years |
| `Delay_in_Payments` | Float | Average payment delay in past 12 months | 0 – 90 days |
| `Credit_Utilization_Ratio` | Float | Revolving credit card balance / Total limit | 0.01 – 0.99 (1% – 99%) |
| `Num_Missed_Payments` | Integer | Number of 30+ day delinquencies in 24 months | 0 – 8 |
| `Savings_Balance` | Float | Liquid cash reserves / checking & savings balance | $100 – $150,000 |
| `Debt_to_Income_Ratio` | Float | Total debt divided by annual income ($DTI$) | 0.02 – 2.50 |
| **`Creditworthy`** *(Target)* | Binary | Underwriting label: `1` (Approved) or `0` (High Risk) | 0 or 1 |

Realistic missing values (~1.5%) are injected across selected features to rigorously test the automated imputation pipeline.

---

## ⚙️ Machine Learning Pipeline & Evaluation

### 1. Preprocessing (`scikit-learn.pipeline.Pipeline`)
- **Missing Value Handling**: `SimpleImputer(strategy='median')` provides robust imputation unaffected by extreme income/debt outliers.
- **Feature Scaling**: `StandardScaler()` standardizes all features to zero mean and unit variance.
- **Leakage Prevention**: All transformations are fitted strictly on training data (`train_test_split` with 80/20 stratified split).

### 2. Model Training & Comparison
The pipeline trains a primary **Random Forest Classifier** (`n_estimators=200`, `max_depth=12`, `class_weight='balanced_subsample'`) and benchmarks against **Gradient Boosting** and **Logistic Regression**.

### 3. Evaluation Metrics
Evaluated on a held-out test set:
- **ROC-AUC Score**: Measures discriminative ability across all decision thresholds.
- **Precision**: Minimizes false approvals (defending against bad credit defaults).
- **Recall**: Maximizes creditworthy capture rate (avoiding wrongful denials).
- **F1-Score**: Harmonic mean balancing precision and recall.
- **Confusion Matrix**: Visualized as an interactive heatmap in the dashboard.

---

## 🚀 Quickstart Guide

### 1. Clone or Open the Repository
```powershell
cd e:\Project\CodeAlpha_CreditScoring
```

### 2. Activate Virtual Environment
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Or Windows Command Prompt
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies (if not already installed)
```powershell
pip install -r requirements.txt
```

### 4. Train the Model (Generates Dataset and Model Artifacts)
```powershell
python train.py
```
*Optional CLI flags:*
- `--samples 8000`: Generate custom number of synthetic records.
- `--seed 42`: Set custom random seed.
- `--artifacts-dir models`: Specify artifact destination.

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Web Application Features

1. **Live Credit Underwriting Tab**:
   - Sliders and inputs for personal, credit, and repayment behavior.
   - **1-Click Presets**: Test "Prime (Approved)", "Subprime (High Risk)", or "Borderline (Fair)" with one click.
   - **Real-Time FICO Meter**: Interactive gauge visualization mapping probability into 300–850 credit scores.
   - **Explainable Strengths & Warnings**: Automatically explains the key positive factors and risk alerts.

2. **Feature Importance & Insights Tab**:
   - Interactive Plotly chart showing Gini impurity reduction across all credit features.
   - Educational breakdown of why payment delays, DTI, and credit utilization dictate underwriting.

3. **Model Evaluation & Benchmarks Tab**:
   - Live KPI cards: Precision, Recall, F1, and ROC-AUC.
   - Interactive Confusion Matrix heatmap and ROC Curve.
   - Side-by-side benchmark comparison table.

4. **Batch CSV Evaluation Tab**:
   - Download a sample CSV template.
   - Upload any batch applicant CSV to score hundreds of records simultaneously with instant downloadable results.

5. **Architecture Documentation Tab**:
   - Comprehensive technical summary, Mermaid flowchart, and underwriting principles.

---

## 👨‍💻 Project Information & Attribution
- **Project Name**: Credit Scoring Model
- **Role**: Senior Machine Learning Engineer & Full-Stack Python Developer
- **Organization**: CodeAlpha Internship Program
- **Version**: 1.0.0 (Production Release)
