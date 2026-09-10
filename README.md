# 🩺 VitalScan AI — Clinical Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.1-emerald.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.7-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An interpretable machine learning clinical screening system that predicts disease risk from **24 routine blood biomarkers**. Built with **Regularized Logistic Regression**, **SMOTE oversampling**, **StandardScaler normalization**, and calibrated decision thresholds to prioritize clinical sensitivity (Recall).

Includes a modern, hospital-grade **Dark Mode web dashboard** designed for live clinical triage demonstrations.

---

## 📌 Project Overview

In clinical pathology and preventive triage, routine blood panels contain rich diagnostic biomarkers. However, subtle multivariate interactions (e.g., slight concurrent elevations in Glucose, HbA1c, Troponin, and CRP) can be overlooked until symptoms escalate.

**VitalScan AI** provides a fast, interpretable screening tool:
1. Evaluates **24 normalized physiological biomarkers** across 4 clinical panels.
2. Applies a tuned **Logistic Regression** classifier with calibrated decision boundaries ($0.45$ cutoff).
3. Produces a binary classification verdict: **HEALTHY (Class 0)** vs. **DISEASE DETECTED (Class 1)**.
4. Explains predictions by computing the exact log-odds contribution ($\beta_i \times z_i$) of each biomarker and comparing abnormal values against healthy reference ranges.

---

## 🔬 Machine Learning Architecture

```
[24 Biomarkers] ──► [Stratified Split (80/20)] ──► [SMOTE (Train Only)] ──► [StandardScaler] 
                                                                                   │
[Prediction Output] ◄── [Calibrated Threshold (0.45)] ◄── [Logistic Regression (L2)] ◄┘
         │
         ├── Predicted Class: Healthy (0) vs. Disease (1)
         ├── Calibrated Probability P(Disease)
         └── Contributing Abnormal Risk Factors (β · z)
```

### Key Technical Highlights
* **Zero Data Leakage**: Stratified train/test splitting performed *before* any synthetic oversampling. The test set remains 100% natural and untouched.
* **Class Imbalance Mitigation**: SMOTE applied exclusively to the training fold to balance minority healthy records.
* **Feature Standardization**: `StandardScaler` ensures $L_2$ regularization penalties act fairly across all biomarkers.
* **Hyperparameter Optimization**: Regularization strength $C$ and solver optimized using 5-Fold Stratified Cross-Validation scoring for **Recall** (Sensitivity).
* **Clinical Explainability**: Unlike black-box algorithms, Logistic Regression provides direct interpretability via learned coefficients ($\beta$) and Odds Ratios ($\exp(\beta)$).

---

## 🧪 The 24 Blood Biomarkers

The input features are organized into 4 clinical panels:

| Panel | Biomarkers Included | Typical Healthy Range |
| :--- | :--- | :---: |
| **1. Metabolic & Glycemic** | Blood Glucose, Insulin, HbA1c, BMI | $0.30 - 0.55$ |
| **2. Complete Blood Count (CBC)** | Hemoglobin, Platelets, WBC, RBC, Hematocrit, MCV, MCH, MCHC | $0.35 - 0.70$ |
| **3. Cardiovascular & Vitals** | Systolic BP, Diastolic BP, Heart Rate, Cardiac Troponin, CRP | $0.20 - 0.55$ |
| **4. Lipids, Hepatic & Renal** | Cholesterol, Triglycerides, LDL, HDL, ALT, AST, Creatinine | $0.25 - 0.55$ |

*All biomarkers in this dataset are normalized between $0.00$ and $1.00$.*

---

## 💻 Web Application Features

* **Compact 2x2 Input Grid**: Displays all 24 inputs cleanly above the fold.
* **Interactive Controls**: Synchronized range sliders and direct numeric boxes with normal reference range tooltips.
* **Quick Clinical Presets**:
  * 🟢 **Healthy Adult**: All biomarkers inside healthy limits $\rightarrow$ `Class 0 (0.2% Risk)`.
  * 🔴 **Type 2 Diabetes**: Spikes Glucose ($0.88$), HbA1c ($0.89$), and Insulin ($0.82$) $\rightarrow$ `Class 1 (92.2% Risk)`.
  * 🟠 **Severe Anemia**: Depletes Hemoglobin ($0.14$), RBC ($0.18$), and Hematocrit ($0.16$) $\rightarrow$ `Class 1`.
  * 🟣 **Acute Cardiac Alert**: Spikes Troponin ($0.92$) and CRP ($0.88$) $\rightarrow$ `Class 1`.
  * 🔵 **Thrombocytopenia**: Depletes Platelets ($0.08$) $\rightarrow$ `Class 1`.
* **Abnormal Factors Comparison Table**: Filters out optimal inputs and highlights only abnormal markers, comparing the patient's value directly against what it *should be* along with its model risk weight.

---

## 🚀 Running Locally

### Prerequisites
* Python 3.10+ (or Anaconda)

### Quick Start (Windows)
Double-click `run_app.bat` to launch the server automatically.

### Manual Setup
```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/vitalscan-ai.git
cd vitalscan-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-train and export model
python train_export_model.py

# 4. Start the web application
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## ☁️ Deploying on Render (Step-by-Step)

Deploy this project on [Render.com](https://render.com) for free in **under 3 minutes**:

### Step 1: Push Project to GitHub
1. Create a new repository on [GitHub](https://github.com/new) (e.g. `vitalscan-ai`).
2. In your terminal, run:
```bash
git init
git add .
git commit -m "Initial release: VitalScan AI Disease Screener"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/vitalscan-ai.git
git push -u origin main
```

### Step 2: Deploy on Render
1. Log in to [Render.com](https://render.com) and click **"New +"** $\rightarrow$ **"Web Service"**.
2. Select **"Build and deploy from a Git repository"** and choose your `vitalscan-ai` repo.
3. Configure the service settings (Render auto-detects most settings from `render.yaml` and `Procfile`):
   * **Name**: `vitalscan-ai`
   * **Environment**: `Python 3`
   * **Region**: Nearest to your users (e.g., Singapore, Frankfurt, Oregon)
   * **Branch**: `main`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `gunicorn app:app`
   * **Instance Type**: `Free`
4. Click **"Deploy Web Service"**.

Render will install dependencies, load the serialized model, and provide a public URL like:
`https://vitalscan-ai.onrender.com`

---

## 📁 Repository Structure

```
├── app.py                  # Flask backend web application & REST API
├── templates/
│   └── index.html          # Dark Mode clinical dashboard UI
├── model.joblib            # Trained Logistic Regression classifier
├── scaler.joblib           # Fitted StandardScaler object
├── features.joblib         # Serialized feature sequence
├── train_export_model.py   # Training script to reproduce model artifacts
├── DP.ipynb                # Jupyter notebook with research, EDA & evaluations
├── DP_backup.ipynb         # Original notebook backup
├── requirements.txt        # Production Python dependencies
├── Procfile                # WSGI process definition for Render/Heroku
├── render.yaml             # Render Blueprint configuration
├── run_app.bat             # 1-click Windows launcher
├── .gitignore              # Ignored files and caches
└── README.md               # Complete project documentation
```

---

## 👥 Team & Academic Context

* **Project Type**: Machine Learning Mini-Project / Clinical Decision Support
* **Focus Algorithm**: Regularized Logistic Regression with Interpretability (Odds Ratios)
* **Evaluation Metrics**: Recall (Sensitivity), Precision, F1-Score, ROC-AUC
