from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ── Load Model & Scaler on Startup ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'models/xgb_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models/scaler.pkl'))

with open(os.path.join(BASE_DIR, 'models/feature_names.json')) as f:
    feature_names = json.load(f)

print("✅ Model loaded successfully!")
print("✅ Scaler loaded successfully!")

# ── Helper: Feature Engineering ─────────────────────────────
def engineer_features(data):
    # Derived features (same as training)
    data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']
    data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
    data['chol_glucose_risk'] = data['cholesterol'] + data['gluc']
    data['hypertension'] = int(
        data['ap_hi'] >= 140 or data['ap_lo'] >= 90
    )

    # Age group
    age = data['age']
    if age < 40:
        data['age_group'] = 0
    elif age < 50:
        data['age_group'] = 1
    elif age < 60:
        data['age_group'] = 2
    else:
        data['age_group'] = 3

    # BMI category
    bmi = data['bmi']
    if bmi < 18.5:
        data['bmi_category'] = 0
    elif bmi < 25:
        data['bmi_category'] = 1
    elif bmi < 30:
        data['bmi_category'] = 2
    else:
        data['bmi_category'] = 3

    return data

# ── Helper: Get Risk Level ───────────────────────────────────
def get_risk_level(probability):
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Moderate"
    elif probability < 0.80:
        return "High"
    else:
        return "Very High"

# ── Home Route ───────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({
        "message": "CardioPredict AI API is running!",
        "version": "1.0",
        "endpoints": ["/predict", "/health"]
    })

# ── Health Check ─────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

# ── Predict Route ────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Validate required fields
        required = ['age', 'gender', 'height', 'weight',
                    'ap_hi', 'ap_lo', 'cholesterol',
                    'gluc', 'smoke', 'alco', 'active']

        for field in required:
            if field not in data:
                return jsonify({
                    "error": f"Missing field: {field}"
                }), 400

        # Apply feature engineering
        data = engineer_features(data)

        # Build feature array in correct order
        features = [data[f] for f in feature_names]
        features_array = np.array(features).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features_array)

        # Make prediction
        probability = model.predict_proba(features_scaled)[0][1]
        risk_score = round(probability * 100, 1)
        risk_level = get_risk_level(probability)

        # Return result
        return jsonify({
    "risk_score": round(float(probability * 100), 1),
    "risk_level": risk_level,
    "probability": round(float(probability), 4),
    "message": "Prediction successful"
})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)