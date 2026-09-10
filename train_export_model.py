"""
VitalScan AI - Model Training & Export Script
Trains a clinically consistent Logistic Regression model on normalized blood biomarkers
and serializes the model and scaler for the web application.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import joblib

np.random.seed(42)

FEATURES = [
    'Glucose', 'Cholesterol', 'Hemoglobin', 'Platelets', 'White Blood Cells',
    'Red Blood Cells', 'Hematocrit', 'Mean Corpuscular Volume',
    'Mean Corpuscular Hemoglobin', 'Mean Corpuscular Hemoglobin Concentration',
    'Insulin', 'BMI', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
    'Triglycerides', 'HbA1c', 'LDL Cholesterol', 'HDL Cholesterol',
    'ALT', 'AST', 'Heart Rate', 'Creatinine', 'Troponin', 'C-reactive Protein'
]

def generate_clinical_cohort(n_healthy=600, n_disease=600):
    """Generates synthetic patient profiles consistent with medical diagnostic ranges."""
    healthy_records = []
    for _ in range(n_healthy):
        row = {f: float(np.clip(np.random.normal(0.44, 0.08), 0.20, 0.65)) for f in FEATURES}
        healthy_records.append(row)
    df_healthy = pd.DataFrame(healthy_records)
    df_healthy['Disease'] = 0

    disease_records = []
    for _ in range(n_disease):
        row = {f: float(np.clip(np.random.normal(0.45, 0.10), 0.15, 0.75)) for f in FEATURES}
        subtype = np.random.choice(['diabetes', 'anemia', 'cardiac', 'thromboc', 'hepatic', 'multi'])
        
        if subtype == 'diabetes':
            row['Glucose'] = float(np.random.uniform(0.72, 0.98))
            row['HbA1c'] = float(np.random.uniform(0.70, 0.99))
            row['Insulin'] = float(np.random.uniform(0.65, 0.95))
            row['BMI'] = float(np.random.uniform(0.60, 0.90))
        elif subtype == 'anemia':
            row['Hemoglobin'] = float(np.random.uniform(0.05, 0.28))
            row['Red Blood Cells'] = float(np.random.uniform(0.05, 0.30))
            row['Hematocrit'] = float(np.random.uniform(0.05, 0.28))
            row['Mean Corpuscular Volume'] = float(np.random.uniform(0.10, 0.35))
        elif subtype == 'cardiac':
            row['Troponin'] = float(np.random.uniform(0.72, 0.99))
            row['C-reactive Protein'] = float(np.random.uniform(0.65, 0.98))
            row['Cholesterol'] = float(np.random.uniform(0.68, 0.95))
            row['LDL Cholesterol'] = float(np.random.uniform(0.65, 0.95))
            row['Systolic Blood Pressure'] = float(np.random.uniform(0.65, 0.92))
        elif subtype == 'thromboc':
            row['Platelets'] = float(np.random.uniform(0.02, 0.22))
        elif subtype == 'hepatic':
            row['ALT'] = float(np.random.uniform(0.70, 0.98))
            row['AST'] = float(np.random.uniform(0.68, 0.96))
        elif subtype == 'multi':
            row['Glucose'] = float(np.random.uniform(0.70, 0.95))
            row['Troponin'] = float(np.random.uniform(0.65, 0.95))
            row['C-reactive Protein'] = float(np.random.uniform(0.60, 0.92))
            
        disease_records.append(row)
    df_disease = pd.DataFrame(disease_records)
    df_disease['Disease'] = 1

    df_cohort = pd.concat([df_healthy, df_disease], ignore_index=True)
    return df_cohort

def train_and_export():
    print("Generating clinical training cohort...")
    df = generate_clinical_cohort()
    X = df[FEATURES]
    y = df['Disease']

    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Tuning Logistic Regression with GridSearchCV...")
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring='recall'
    )
    grid.fit(X_scaled, y)
    best_model = grid.best_estimator_

    print(f"Optimal Model Parameters: {grid.best_params_}")
    print(f"Cross-Validation Recall: {grid.best_score_:.4f}")
    print(f"Training Accuracy: {best_model.score(X_scaled, y):.4f}")

    joblib.dump(best_model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(FEATURES, 'features.joblib')
    print("Exported: model.joblib, scaler.joblib, features.joblib successfully!")

if __name__ == '__main__':
    train_and_export()
