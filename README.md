# 🩺 VitalScan AI — Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.1-emerald.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.7-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

A foundational supervised machine learning project for disease screening using **Logistic Regression** on **24 blood biomarkers**. 

Designed for B.Tech Computer Science / Engineering class presentations with a clean, textbook ML pipeline and a live dark-mode web dashboard.

---

## 📌 Problem Statement & Objective

Given routine blood test parameters (such as Glucose, Cholesterol, Hemoglobin, Platelets, WBC, RBC, etc.), predict whether a patient has a disease condition (**Class 1**) or is healthy (**Class 0**).

* **Input**: 24 normalized blood biomarkers ($0.00 - 1.00$).
* **Model**: Logistic Regression classifier.
* **Output**: Predicted class label (`0: Healthy`, `1: Disease`) and predicted probability $P(\text{Disease})$.

---

## 🔬 Standard Machine Learning Workflow

This project strictly follows the core B.Tech Machine Learning curriculum:

```
[Blood Test Dataset (24 Features)]
               │
               ▼
[Data Cleaning: Drop Duplicates & Null Check]
               │
               ▼
[Train-Test Split: 80% Training, 20% Testing]
               │
               ▼
[Feature Scaling: StandardScaler (fit on train, transform test)]
               │
               ▼
[Model Training: LogisticRegression().fit(X_train, y_train)]
               │
               ▼
[Evaluation: Accuracy, Confusion Matrix, Classification Report]
               │
               ▼
[Deployment: Flask Web Application with 2x2 Dark Mode UI]
```

### 1. Data Ingestion & Cleaning
* Checked for missing values (`df.isnull().sum()`).
* Removed duplicate patient rows (`df.drop_duplicates()`).
* Mapped target labels to binary values: `Healthy = 0`, `Disease = 1`.

### 2. Train / Test Split
* Split the dataset into **80% training data** (to fit model weights) and **20% testing data** (to evaluate generalization on unseen data).
* `random_state=42` ensures reproducible results.

### 3. Feature Scaling (`StandardScaler`)
* Scaled features to have a mean of $0$ and standard deviation of $1$.
* Essential for Logistic Regression to ensure features with different variance are weighted fairly.

### 4. Model Training (`LogisticRegression`)
* Fits a linear decision boundary mapped through the standard **Sigmoid activation function**:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* Standard decision threshold of **0.50**:
  * If $P(\text{Disease}) \ge 0.50 \rightarrow \text{Class 1 (Disease)}$
  * If $P(\text{Disease}) < 0.50 \rightarrow \text{Class 0 (Healthy)}$

### 5. Evaluation Metrics
* **Accuracy**: Overall correct predictions on test set (~89-91%).
* **Confusion Matrix**: Visualizes True Positives, True Negatives, False Positives, and False Negatives.
* **Classification Report**: Evaluates Precision, Recall, and F1-Score.
* **Feature Importance**: Plots model coefficients (`model.coef_`) to show which biomarkers most strongly indicate disease risk (e.g. elevated Glucose, Troponin).

---

## 💻 Web Application Features

The project includes an interactive web application built with **Flask** and **Tailwind CSS**:

* **Dark Mode Theme**: Modern clinical UI.
* **2x2 Compact Input Grid**: Groups the 24 biomarkers into 4 panels (Metabolic, CBC, Cardiovascular, Lipids & Organs).
* **Preset Profiles**: One-click dropdown to demo *Healthy Adult*, *Diabetes Risk*, *Anemia Risk*, *Cardiac Alert*, and *Thrombocytopenia*.
* **Compute Analysis**: Scales inputs, runs Logistic Regression, and scrolls down to the prediction verdict.
* **Abnormal Factors Table**: Filters out normal inputs and only displays abnormal markers, comparing entered values against healthy ranges.

---

## 🚀 Running Locally

### 1. Run using Batch File (Windows)
Simply double-click [`run_app.bat`](file:///c:/Users/gargk/Desktop/disease%20prediction%20mini%20inter/run_app.bat).

### 2. Run from Terminal
```bash
# Install dependencies
pip install -r requirements.txt

# Run web app
python app.py
```
Open **`http://localhost:5000`** in any browser.

---

## ☁️ Deploying on Render (Free Hosting)

1. Push this project to your GitHub repository:
   ```bash
   git add .
   git commit -m "Standard B.Tech ML workflow for Render deployment"
   git push origin main
   ```
2. Go to **[dashboard.render.com](https://dashboard.render.com/)** $\rightarrow$ **New +** $\rightarrow$ **Web Service**.
3. Select your GitHub repository.
4. Render automatically configures the service using `render.yaml` and `Procfile`:
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `gunicorn app:app`
5. Click **Deploy Web Service**. Render gives you a live public URL (e.g. `https://vitalscan-ai.onrender.com`).

---

## 🎓 Viva Questions & Answers (Class Presentation Guide)

1. **Q: Why did you choose Logistic Regression?**
   * *A:* "Logistic Regression is the foundational algorithm for binary classification. It maps linear combinations of features to a probability between 0 and 1 using the Sigmoid function, making it easy to interpret."
2. **Q: Why did you scale the data using StandardScaler?**
   * *A:* "Because features can have different numerical scales and variances. StandardScaler centers the data around mean 0 with standard deviation 1 so all features contribute equally during gradient optimization."
3. **Q: Why did you split the data 80-20?**
   * *A:* "To prevent overfitting and test our model on unseen data. 80% is used for training the parameters, and 20% is held out strictly for evaluation."
4. **Q: What is the decision threshold?**
   * *A:* "We used the standard 0.50 threshold: if the Sigmoid probability is 50% or higher, the patient is classified as Class 1 (Disease)."
