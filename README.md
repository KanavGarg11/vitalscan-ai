# Disease Prediction System — VitalScan AI

**Course:** ML Mini Internship  
**Project Context:** Capstone team implementation showcasing and applying core machine learning concepts acquired during the summer online internship program.

---

## 👥 Team Members

| S.No. | Student Name | Roll Number |
| :---: | :--- | :--- |
| 1 | **Kanav Garg** | 2400270100093 |
| 2 | **Harshit Singh** | 2400270100089 |
| 3 | **Harshit Singh** | 2400270100090 |
| 4 | **Dipanshu Chaudhary** | 2400270100077 |

---

## 📌 Project Overview

This project implements an end-to-end supervised machine learning pipeline to classify individual patient health status based on routine blood test indicators. 

* **Task:** Supervised Binary Classification (`0: Healthy`, `1: Disease`).
* **Input Features:** 24 continuous blood test biomarkers normalized between $0.00$ and $1.00$.
* **Primary Algorithm:** Logistic Regression with Sigmoid activation.
* **Objective:** Enable early pathology screening and risk assessment before clinical escalation.

---

## 🔬 Machine Learning Pipeline

```
[Raw Blood Test Data] ──► [Data Cleaning & Deduplication] ──► [Train-Test Split (80/20)]
                                                                      │
                                                                      ▼
[Model Evaluation] ◄── [Logistic Regression (Sigmoid)] ◄── [StandardScaler Normalization]
```

### 1. Data Preprocessing
* **Data Sanitization:** Checked for missing values and dropped duplicate records to ensure data integrity.
* **Target Encoding:** Mapped categorical targets into binary classes (`Healthy = 0`, `Disease = 1`).

### 2. Feature Scaling
* Applied **`StandardScaler`** to center each biomarker to a mean of $0$ and unit variance ($1$).
* Guarantees that differences in biomarker numerical variances do not disproportionately bias model coefficients.

### 3. Data Splitting
* Segmented data into **80% Training** and **20% Testing** sets using `train_test_split(..., test_size=0.2, random_state=42)` to rigorously assess generalization on unseen records.

### 4. Model Training & Mathematics
* Trained a **Logistic Regression** model:
  $$z = w_0 + \sum_{i=1}^{n} w_i x_i$$
  $$P(\text{Disease} = 1) = \frac{1}{1 + e^{-z}}$$
* Evaluated against the standard classification threshold of **$0.50$**:
  $$\hat{y} = \begin{cases} 1 & \text{if } P(\text{Disease}) \ge 0.50 \\ 0 & \text{if } P(\text{Disease}) < 0.50 \end{cases}$$

### 5. Evaluation Metrics
* **Accuracy:** Quantifies overall classification accuracy (~88.5% – 90% on unseen test data).
* **Confusion Matrix:** Evaluates True Positives, True Negatives, False Positives, and False Negatives.
* **Precision, Recall & F1-Score:** Measures diagnostic sensitivity to minimize missed disease cases.
* **Feature Coefficients:** Analyzes learned weights to identify primary clinical risk drivers.

---

## 🧪 Biomarker Features (24 Input Parameters)

The 24 blood biomarkers are structured across 4 clinical diagnostic panels:

| Panel | Biomarkers | Reference Normal Range |
| :--- | :--- | :---: |
| **Metabolic & Glycemic** | Blood Glucose, Insulin, HbA1c, Body Mass Index (BMI) | $0.30 - 0.55$ |
| **Complete Blood Count (CBC)** | Hemoglobin, Platelets, WBC, RBC, Hematocrit, MCV, MCH, MCHC | $0.35 - 0.70$ |
| **Cardiovascular & Vitals** | Systolic BP, Diastolic BP, Heart Rate, Cardiac Troponin, CRP | $0.20 - 0.55$ |
| **Lipids, Hepatic & Renal** | Cholesterol, Triglycerides, LDL, HDL, ALT, AST, Serum Creatinine | $0.25 - 0.55$ |

---

## 💻 Web Application Interface

A lightweight, responsive web application was developed using **Flask** and **Tailwind CSS** to demonstrate real-time model inference:

* **Compact 2x2 Input Grid:** Organizes the 24 inputs into 4 distinct clinical panels above the fold.
* **Patient Presets Dropdown:** Quick-loading profiles (*Healthy Adult*, *Diabetes Risk*, *Anemia Profile*, *Cardiac Alert*, *Thrombocytopenia*).
* **Automated Risk Assessment:** Computes class prediction and displays probability meter with calibrated $0.50$ threshold.
* **Abnormal Factors Breakdown:** Excludes normal features and highlights only abnormal biomarker deviations against target ranges.

---

## 📁 Repository Structure

```
├── app.py                  # Flask web application & REST API
├── templates/
│   └── index.html          # Clinical dashboard interface (Dark Mode)
├── model.joblib            # Serialized Logistic Regression model
├── scaler.joblib           # Serialized StandardScaler
├── features.joblib         # Feature column ordering
├── train_export_model.py   # Training script for model reproduction
├── DP.ipynb                # Jupyter Notebook containing exploratory data analysis & model pipeline
├── DP_backup.ipynb         # Notebook reference backup
├── requirements.txt        # Production dependencies
├── Procfile                # WSGI web process declaration (Gunicorn)
├── render.yaml             # Render Blueprint configuration
├── run_app.bat             # 1-click Windows local launcher
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## 🚀 Execution & Deployment

### Local Execution (Windows)
1. Double-click [`run_app.bat`](run_app.bat), or
2. Run via terminal:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
3. Open `http://127.0.0.1:5000` in your browser.

### Cloud Deployment (Render)
1. Push repository to GitHub:
   ```bash
   git add .
   git commit -m "Update project documentation"
   git push origin main
   ```
2. Connect the repository on [dashboard.render.com](https://dashboard.render.com/) as a **Web Service**.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`
