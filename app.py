"""
VitalScan AI - Simple B.Tech ML Web Application
Standard Workflow:
1. Receives 24 biomarker inputs
2. Scales features using StandardScaler
3. Generates prediction using standard Logistic Regression (threshold = 0.50)
4. Displays abnormal inputs alongside expected healthy ranges
"""
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model, scaler, and features
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'features.joblib')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)

# Feature metadata: grouped into 4 distinct clinical panels for 2x2 grid
FEATURE_METADATA = [
    # Box 1: Metabolic & Glycemic (4 features)
    {
        "id": "Glucose",
        "name": "Blood Glucose",
        "category": "Metabolic & Glycemic",
        "unit": "norm",
        "normal": "0.30 - 0.55",
        "normal_min": 0.30,
        "normal_max": 0.55,
        "default": 0.38,
        "desc": "Fasting blood sugar level"
    },
    {
        "id": "Insulin",
        "name": "Insulin Level",
        "category": "Metabolic & Glycemic",
        "unit": "norm",
        "normal": "0.30 - 0.55",
        "normal_min": 0.30,
        "normal_max": 0.55,
        "default": 0.45,
        "desc": "Pancreatic hormone for glucose regulation"
    },
    {
        "id": "HbA1c",
        "name": "HbA1c (Glycated Hb)",
        "category": "Metabolic & Glycemic",
        "unit": "norm",
        "normal": "0.30 - 0.55",
        "normal_min": 0.30,
        "normal_max": 0.55,
        "default": 0.44,
        "desc": "3-month average blood glucose control"
    },
    {
        "id": "BMI",
        "name": "Body Mass Index",
        "category": "Metabolic & Glycemic",
        "unit": "norm",
        "normal": "0.30 - 0.55",
        "normal_min": 0.30,
        "normal_max": 0.55,
        "default": 0.44,
        "desc": "Body mass fat proportion index"
    },

    # Box 2: Complete Blood Count (CBC) (8 features)
    {
        "id": "Hemoglobin",
        "name": "Hemoglobin (Hb)",
        "category": "Complete Blood Count (CBC)",
        "unit": "norm",
        "normal": "0.45 - 0.70",
        "normal_min": 0.45,
        "normal_max": 0.70,
        "default": 0.56,
        "desc": "Oxygen-carrying protein in red blood cells"
    },
    {
        "id": "Platelets",
        "name": "Platelet Count",
        "category": "Complete Blood Count (CBC)",
        "unit": "norm",
        "normal": "0.40 - 0.70",
        "normal_min": 0.40,
        "normal_max": 0.70,
        "default": 0.51,
        "desc": "Cell fragments responsible for blood clotting"
    },
    {
        "id": "White Blood Cells",
        "name": "White Blood Cells (WBC)",
        "category": "Complete Blood Count (CBC)",
        "unit": "norm",
        "normal": "0.35 - 0.65",
        "normal_min": 0.35,
        "normal_max": 0.65,
        "default": 0.51,
        "desc": "Immune defense leukocytes"
    },
    {
        "id": "Red Blood Cells",
        "name": "Red Blood Cells (RBC)",
        "category": "Complete Blood Count (CBC)",
        "unit": "norm",
        "normal": "0.40 - 0.65",
        "normal_min": 0.40,
        "normal_max": 0.65,
        "default": 0.50,
        "desc": "Oxygen-transporting erythrocytes"
    },
    {
        "id": "Hematocrit",
        "name": "Hematocrit (HCT)",
        "category": "Complete Blood Count (CBC)",
        "unit": "norm",
        "normal": "0.40 - 0.65",
        "normal_min": 0.40,
        "normal_max": 0.65,
        "default": 0.50,
        "desc": "Percentage of blood volume made of RBCs"
    },
    {
        "id": "Mean Corpuscular Volume",
        "name": "MCV",
        "category": "Complete Blood Count (CBC)",
        "unit": "norm",
        "normal": "0.35 - 0.65",
        "normal_min": 0.35,
        "normal_max": 0.65,
        "default": 0.49,
        "desc": "Average volume size of single red cell"
    },
    {
        "id": "Mean Corpuscular Hemoglobin",
        "name": "MCH",
        "category": "Complete Blood Count (CBC)",
        "unit": "norm",
        "normal": "0.35 - 0.65",
        "normal_min": 0.35,
        "normal_max": 0.65,
        "default": 0.48,
        "desc": "Average amount of hemoglobin per RBC"
    },
    {
        "id": "Mean Corpuscular Hemoglobin Concentration",
        "name": "MCHC",
        "category": "Complete Blood Count (CBC)",
        "unit": "norm",
        "normal": "0.40 - 0.70",
        "normal_min": 0.40,
        "normal_max": 0.70,
        "default": 0.55,
        "desc": "Concentration of hemoglobin in packed RBCs"
    },

    # Box 3: Cardiovascular & Vitals (5 features)
    {
        "id": "Systolic Blood Pressure",
        "name": "Systolic Blood Pressure",
        "category": "Cardiovascular & Vitals",
        "unit": "norm",
        "normal": "0.30 - 0.55",
        "normal_min": 0.30,
        "normal_max": 0.55,
        "default": 0.40,
        "desc": "Peak arterial pressure during heart contraction"
    },
    {
        "id": "Diastolic Blood Pressure",
        "name": "Diastolic Blood Pressure",
        "category": "Cardiovascular & Vitals",
        "unit": "norm",
        "normal": "0.30 - 0.55",
        "normal_min": 0.30,
        "normal_max": 0.55,
        "default": 0.43,
        "desc": "Resting arterial pressure between beats"
    },
    {
        "id": "Heart Rate",
        "name": "Heart Rate (Pulse)",
        "category": "Cardiovascular & Vitals",
        "unit": "norm",
        "normal": "0.40 - 0.70",
        "normal_min": 0.40,
        "normal_max": 0.70,
        "default": 0.56,
        "desc": "Resting cardiac beats per minute"
    },
    {
        "id": "Troponin",
        "name": "Cardiac Troponin",
        "category": "Cardiovascular & Vitals",
        "unit": "norm",
        "normal": "0.20 - 0.50",
        "normal_min": 0.20,
        "normal_max": 0.50,
        "default": 0.46,
        "desc": "Myocardial enzyme released during cardiac stress"
    },
    {
        "id": "C-reactive Protein",
        "name": "C-reactive Protein (CRP)",
        "category": "Cardiovascular & Vitals",
        "unit": "norm",
        "normal": "0.20 - 0.50",
        "normal_min": 0.20,
        "normal_max": 0.50,
        "default": 0.44,
        "desc": "Acute-phase systemic inflammatory marker"
    },

    # Box 4: Lipids, Hepatic & Renal (7 features)
    {
        "id": "Cholesterol",
        "name": "Total Cholesterol",
        "category": "Lipids, Hepatic & Renal",
        "unit": "norm",
        "normal": "0.25 - 0.55",
        "normal_min": 0.25,
        "normal_max": 0.55,
        "default": 0.41,
        "desc": "Total blood sterol lipids"
    },
    {
        "id": "Triglycerides",
        "name": "Triglycerides",
        "category": "Lipids, Hepatic & Renal",
        "unit": "norm",
        "normal": "0.25 - 0.55",
        "normal_min": 0.25,
        "normal_max": 0.55,
        "default": 0.39,
        "desc": "Dietary blood fat particles"
    },
    {
        "id": "LDL Cholesterol",
        "name": "LDL (Bad Cholesterol)",
        "category": "Lipids, Hepatic & Renal",
        "unit": "norm",
        "normal": "0.25 - 0.55",
        "normal_min": 0.25,
        "normal_max": 0.55,
        "default": 0.43,
        "desc": "Atherogenic low-density cholesterol"
    },
    {
        "id": "HDL Cholesterol",
        "name": "HDL (Good Cholesterol)",
        "category": "Lipids, Hepatic & Renal",
        "unit": "norm",
        "normal": "0.40 - 0.75",
        "normal_min": 0.40,
        "normal_max": 0.75,
        "default": 0.53,
        "desc": "Cardioprotective high-density cholesterol"
    },
    {
        "id": "ALT",
        "name": "ALT (Alanine Transaminase)",
        "category": "Lipids, Hepatic & Renal",
        "unit": "norm",
        "normal": "0.25 - 0.55",
        "normal_min": 0.25,
        "normal_max": 0.55,
        "default": 0.44,
        "desc": "Liver-specific hepatocellular enzyme"
    },
    {
        "id": "AST",
        "name": "AST (Aspartate Transaminase)",
        "category": "Lipids, Hepatic & Renal",
        "unit": "norm",
        "normal": "0.25 - 0.55",
        "normal_min": 0.25,
        "normal_max": 0.55,
        "default": 0.46,
        "desc": "Liver and cardiac enzyme"
    },
    {
        "id": "Creatinine",
        "name": "Serum Creatinine",
        "category": "Lipids, Hepatic & Renal",
        "unit": "norm",
        "normal": "0.25 - 0.55",
        "normal_min": 0.25,
        "normal_max": 0.55,
        "default": 0.44,
        "desc": "Renal muscle waste filtration byproduct"
    }
]

# Simple Presentation Presets
PRESETS = {
    "healthy": {
        "title": "Healthy Adult Profile",
        "desc": "All 24 biomarkers within healthy reference bounds",
        "values": {f["id"]: f["default"] for f in FEATURE_METADATA}
    },
    "diabetes": {
        "title": "Diabetes Risk Profile",
        "desc": "Elevated Glucose, HbA1c, and Insulin",
        "values": {
            **{f["id"]: f["default"] for f in FEATURE_METADATA},
            "Glucose": 0.88,
            "HbA1c": 0.89,
            "Insulin": 0.82,
            "BMI": 0.74,
            "Triglycerides": 0.65
        }
    },
    "anemia": {
        "title": "Anemia Risk Profile",
        "desc": "Depleted Hemoglobin, RBC count, and Hematocrit",
        "values": {
            **{f["id"]: f["default"] for f in FEATURE_METADATA},
            "Hemoglobin": 0.14,
            "Red Blood Cells": 0.18,
            "Hematocrit": 0.16,
            "Mean Corpuscular Volume": 0.22,
            "Heart Rate": 0.70
        }
    },
    "cardiac": {
        "title": "Cardiac Risk Alert",
        "desc": "Elevated Troponin, CRP, and Blood Pressure",
        "values": {
            **{f["id"]: f["default"] for f in FEATURE_METADATA},
            "Troponin": 0.92,
            "C-reactive Protein": 0.88,
            "Cholesterol": 0.82,
            "LDL Cholesterol": 0.79,
            "Systolic Blood Pressure": 0.78,
            "Heart Rate": 0.85
        }
    },
    "thromboc": {
        "title": "Thrombocytopenia Alert",
        "desc": "Critically low Platelet count",
        "values": {
            **{f["id"]: f["default"] for f in FEATURE_METADATA},
            "Platelets": 0.08,
            "White Blood Cells": 0.42
        }
    }
}

@app.route('/')
def index():
    return render_template('index.html', metadata=FEATURE_METADATA, presets=PRESETS)

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    return jsonify({
        "features": FEATURE_METADATA,
        "presets": PRESETS
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        
        # 1. Build input vector
        input_vector = []
        meta_dict = {f["id"]: f for f in FEATURE_METADATA}
        
        for feat in features:
            val = float(data.get(feat, meta_dict.get(feat, {}).get("default", 0.45)))
            val = max(0.0, min(1.0, val))
            input_vector.append(val)
        
        # 2. Scale features
        X_df = pd.DataFrame([input_vector], columns=features)
        X_scaled = scaler.transform(X_df)
        
        # 3. Standard Logistic Regression Prediction & Probability
        prediction = int(model.predict(X_scaled)[0])
        prob_disease = float(model.predict_proba(X_scaled)[0][1])
        prob_healthy = float(1.0 - prob_disease)
        
        # Standard 0.50 decision threshold
        if prediction == 1:
            status = "DISEASE DETECTED (CLASS 1)"
            status_badge = "Class 1"
        else:
            status = "HEALTHY (CLASS 0)"
            status_badge = "Class 0"
        
        # 4. Identify only abnormal factors (filter out optimal ones)
        abnormal_factors = []
        
        for feat, raw_val in zip(features, input_vector):
            f_meta = meta_dict.get(feat, {})
            norm_min = f_meta.get("normal_min", 0.30)
            norm_max = f_meta.get("normal_max", 0.55)
            norm_label = f_meta.get("normal", "0.30 - 0.55")
            default_val = f_meta.get("default", 0.45)
            
            if raw_val > norm_max + 0.05:
                diff_pct = round(((raw_val - norm_max) / norm_max) * 100)
                abnormal_factors.append({
                    "name": f_meta.get("name", feat),
                    "category": f_meta.get("category", "General"),
                    "raw_value": round(raw_val, 2),
                    "healthy_range": norm_label,
                    "healthy_typical": default_val,
                    "status_badge": "high",
                    "status_text": f"+{diff_pct}% Above Normal"
                })
            elif raw_val < norm_min - 0.05:
                diff_pct = round(((norm_min - raw_val) / norm_min) * 100)
                abnormal_factors.append({
                    "name": f_meta.get("name", feat),
                    "category": f_meta.get("category", "General"),
                    "raw_value": round(raw_val, 2),
                    "healthy_range": norm_label,
                    "healthy_typical": default_val,
                    "status_badge": "low",
                    "status_text": f"-{diff_pct}% Below Normal"
                })

        return jsonify({
            "status": "success",
            "prediction": status,
            "prediction_class": prediction,
            "status_badge": status_badge,
            "probability_disease": round(prob_disease * 100, 1),
            "probability_healthy": round(prob_healthy * 100, 1),
            "abnormal_factors": abnormal_factors,
            "has_abnormal": len(abnormal_factors) > 0
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/healthz')
def healthz():
    """Health check endpoint for Render."""
    return jsonify({"status": "healthy", "service": "vitalscan-ai"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    print(f"Starting VitalScan AI on http://0.0.0.0:{port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
