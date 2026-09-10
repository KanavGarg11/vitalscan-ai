"""
VitalScan AI - Simple Model Training & Export Script
Standard B.Tech ML Workflow:
1. Load features & target
2. Apply StandardScaler
3. Train Logistic Regression
4. Export model & scaler for web deployment
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

def generate_cohort(n_samples=1000):
    """Generates synthetic cohort matching the dataset blood distribution."""
    records = []
    labels = []
    
    for _ in range(n_samples):
        is_disease = np.random.choice([0, 1], p=[0.4, 0.6])
        row = {f: float(np.clip(np.random.normal(0.45, 0.10), 0.15, 0.70)) for f in FEATURES}
        
        if is_disease:
            subtype = np.random.choice(['diabetes', 'anemia', 'cardiac', 'thromboc'])
            if subtype == 'diabetes':
                row['Glucose'] = float(np.random.uniform(0.70, 0.95))
                row['HbA1c'] = float(np.random.uniform(0.70, 0.95))
                row['Insulin'] = float(np.random.uniform(0.65, 0.90))
            elif subtype == 'anemia':
                row['Hemoglobin'] = float(np.random.uniform(0.05, 0.30))
                row['Red Blood Cells'] = float(np.random.uniform(0.05, 0.32))
                row['Hematocrit'] = float(np.random.uniform(0.05, 0.30))
            elif subtype == 'cardiac':
                row['Troponin'] = float(np.random.uniform(0.70, 0.98))
                row['C-reactive Protein'] = float(np.random.uniform(0.65, 0.95))
                row['Cholesterol'] = float(np.random.uniform(0.65, 0.90))
            elif subtype == 'thromboc':
                row['Platelets'] = float(np.random.uniform(0.02, 0.20))
                
        records.append(row)
        labels.append(is_disease)
        
    df = pd.DataFrame(records)
    df['Disease'] = labels
    return df

def train_and_export():
    print("1. Preparing clinical data...")
    df = generate_cohort()
    X = df[FEATURES]
    y = df['Disease']
    
    # 80/20 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("2. Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("3. Training standard Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"   Train Accuracy: {round(train_acc * 100, 2)}%")
    print(f"   Test Accuracy:  {round(test_acc * 100, 2)}%")
    
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(FEATURES, 'features.joblib')
    print("4. Successfully exported: model.joblib, scaler.joblib, features.joblib")

if __name__ == '__main__':
    train_and_export()
