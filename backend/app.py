from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import json
import numpy as np
from datetime import timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))
import os
from models import db, User, Prediction

app = Flask(__name__, template_folder='../frontend')
CORS(app)

# ── Configuration ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'database', 'cardiopredict.db')

app.config['SECRET_KEY'] = 'cardiopredict-secret-key-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ── Initialize Extensions ────────────────────────────────────
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)

# ── Load Model & Scaler ──────────────────────────────────────
model = joblib.load(os.path.join(BASE_DIR, 'models/xgb_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models/scaler.pkl'))

with open(os.path.join(BASE_DIR, 'models/feature_names.json')) as f:
    feature_names = json.load(f)

print("✅ Model loaded!")
print("✅ Database configured!")

# ── User Loader ──────────────────────────────────────────────
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Helper: Feature Engineering ─────────────────────────────
def engineer_features(data):
    data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']
    data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
    data['chol_glucose_risk'] = data['cholesterol'] + data['gluc']
    data['hypertension'] = int(data['ap_hi'] >= 140 or data['ap_lo'] >= 90)

    age = data['age']
    if age < 40: data['age_group'] = 0
    elif age < 50: data['age_group'] = 1
    elif age < 60: data['age_group'] = 2
    else: data['age_group'] = 3

    bmi = data['bmi']
    if bmi < 18.5: data['bmi_category'] = 0
    elif bmi < 25: data['bmi_category'] = 1
    elif bmi < 30: data['bmi_category'] = 2
    else: data['bmi_category'] = 3

    return data

# ── Helper: Risk Level ───────────────────────────────────────
def get_risk_level(probability):
    if probability < 0.30: return "Low"
    elif probability < 0.60: return "Moderate"
    elif probability < 0.80: return "High"
    else: return "Very High"

# ══════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        full_name = data.get('full_name')
        email = data.get('email')
        password = data.get('password')

        if not full_name or not email or not password:
            return jsonify({"error": "All fields are required"}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already registered"}), 400

        hashed_password = generate_password_hash(password)
        new_user = User(
            full_name=full_name,
            email=email,
            password=hashed_password
        )
        db.session.add(new_user)
        db.session.commit()

        return jsonify({
            "message": "Registration successful!",
            "user": {"name": full_name, "email": email}
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            return jsonify({"error": "Invalid email or password"}), 401

        login_user(user)
        return jsonify({
            "message": "Login successful!",
            "user": {
                "name": user.full_name,
                "email": user.email,
                "id": user.id
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

@app.route('/api/me', methods=['GET'])
def get_current_user():
    if current_user.is_authenticated:
        return jsonify({
            "logged_in": True,
            "user": {
                "name": current_user.full_name,
                "email": current_user.email,
                "id": current_user.id
            }
        })
    return jsonify({"logged_in": False})

# ══════════════════════════════════════════════════════════════
# PREDICTION ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required = ['age', 'gender', 'height', 'weight',
                    'ap_hi', 'ap_lo', 'cholesterol',
                    'gluc', 'smoke', 'alco', 'active']

        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        data = engineer_features(data)
        features = [data[f] for f in feature_names]
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        probability = model.predict_proba(features_scaled)[0][1]
        risk_score = round(float(probability * 100), 1)
        risk_level = get_risk_level(probability)

        # Save to database if user_id provided
        user_id = data.get('user_id')
        if user_id:
            prediction = Prediction(
                user_id=user_id,
                age=data['age'],
                gender=data['gender'],
                height=data['height'],
                weight=data['weight'],
                ap_hi=data['ap_hi'],
                ap_lo=data['ap_lo'],
                cholesterol=data['cholesterol'],
                gluc=data['gluc'],
                smoke=data['smoke'],
                alco=data['alco'],
                active=data['active'],
                risk_score=risk_score,
                risk_level=risk_level,
                probability=round(float(probability), 4)
            )
            db.session.add(prediction)
            db.session.commit()

        return jsonify({
            "risk_score": risk_score,
            "risk_level": risk_level,
            "probability": round(float(probability), 4),
            "message": "Prediction successful",
            "saved": user_id is not None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID required"}), 400

        predictions = Prediction.query.filter_by(
            user_id=int(user_id)
        ).order_by(Prediction.created_at.desc()).limit(10).all()

        history = []
        for p in predictions:
            history.append({
                "id": p.id,
                "risk_score": p.risk_score,
                "risk_level": p.risk_level,
                "age": p.age,
                "date": p.created_at.replace(tzinfo=timezone.utc).astimezone(IST).strftime("%d %b %Y, %I:%M %p")
            })

        return jsonify({"history": history})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# ══════════════════════════════════════════════════════════════
# TRIAGE ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/api/triage', methods=['POST'])
def triage():
    try:
        data = request.get_json()

        # Get symptoms
        chest_pain = int(data.get('chest_pain', 0))
        chest_pain_severity = int(data.get('chest_pain_severity', 0))
        radiating_pain = int(data.get('radiating_pain', 0))
        shortness_of_breath = int(data.get('shortness_of_breath', 0))
        breath_severity = data.get('breath_severity', 'none')
        palpitations = int(data.get('palpitations', 0))
        dizziness = int(data.get('dizziness', 0))
        fainting = int(data.get('fainting', 0))
        sweating = int(data.get('sweating', 0))
        nausea = int(data.get('nausea', 0))
        fatigue = int(data.get('fatigue', 0))

        # Get vitals
        heart_rate = int(data.get('heart_rate', 0))
        systolic_bp = int(data.get('systolic_bp', 0))

        # Get context
        symptom_duration = data.get('symptom_duration', 'hours')
        cardiac_history = int(data.get('cardiac_history', 0))

        # ── Triage Logic ─────────────────────────────────────

        # EMERGENCY conditions
        emergency = False
        emergency_reasons = []

        if chest_pain and chest_pain_severity >= 7:
            emergency = True
            emergency_reasons.append("Severe chest pain")

        if chest_pain and radiating_pain:
            emergency = True
            emergency_reasons.append("Chest pain radiating to arm/jaw")

        if chest_pain and shortness_of_breath and breath_severity == 'severe':
            emergency = True
            emergency_reasons.append("Chest pain with severe shortness of breath")

        if fainting:
            emergency = True
            emergency_reasons.append("Fainting episode")

        if heart_rate > 150 or heart_rate < 40:
            if heart_rate > 0:
                emergency = True
                emergency_reasons.append(f"Dangerous heart rate: {heart_rate} BPM")

        if systolic_bp > 180:
            if systolic_bp > 0:
                emergency = True
                emergency_reasons.append(f"Severely high BP: {systolic_bp} mmHg")

        if emergency:
            return jsonify({
                "triage_level": "Emergency",
                "color": "red",
                "action": "Call 911 immediately or go to the nearest Emergency Room",
                "wait_time": "Immediate — do not wait",
                "reasons": emergency_reasons,
                "advice": [
                    "Call emergency services (911) immediately",
                    "Do not drive yourself to the hospital",
                    "Chew an aspirin if not allergic",
                    "Sit down and stay calm",
                    "Have someone stay with you"
                ]
            })

        # URGENT conditions
        urgent = False
        urgent_reasons = []

        if chest_pain and chest_pain_severity >= 4:
            urgent = True
            urgent_reasons.append("Moderate chest pain")

        if shortness_of_breath and breath_severity in ['moderate', 'severe']:
            urgent = True
            urgent_reasons.append("Significant shortness of breath")

        if palpitations and cardiac_history:
            urgent = True
            urgent_reasons.append("Palpitations with cardiac history")

        if dizziness and chest_pain:
            urgent = True
            urgent_reasons.append("Dizziness with chest pain")

        if heart_rate > 120:
            if heart_rate > 0:
                urgent = True
                urgent_reasons.append(f"Elevated heart rate: {heart_rate} BPM")

        if systolic_bp > 160:
            if systolic_bp > 0:
                urgent = True
                urgent_reasons.append(f"High blood pressure: {systolic_bp} mmHg")

        if symptom_duration == 'now' and (chest_pain or shortness_of_breath):
            urgent = True
            urgent_reasons.append("Sudden onset of cardiac symptoms")

        if urgent:
            return jsonify({
                "triage_level": "Urgent",
                "color": "orange",
                "action": "Visit Emergency Room or Urgent Care within 2 hours",
                "wait_time": "Within 1-2 hours",
                "reasons": urgent_reasons,
                "advice": [
                    "Go to Emergency Room or Urgent Care now",
                    "Do not wait until tomorrow",
                    "Bring list of current medications",
                    "Have someone drive you",
                    "Monitor symptoms — call 911 if they worsen"
                ]
            })

        # NON-URGENT
        non_urgent_reasons = []
        if chest_pain: non_urgent_reasons.append("Mild chest discomfort")
        if palpitations: non_urgent_reasons.append("Occasional palpitations")
        if fatigue: non_urgent_reasons.append("Fatigue reported")
        if nausea: non_urgent_reasons.append("Nausea reported")
        if dizziness: non_urgent_reasons.append("Mild dizziness")
        if not non_urgent_reasons:
            non_urgent_reasons.append("No significant cardiac symptoms detected")

        return jsonify({
            "triage_level": "Non-Urgent",
            "color": "green",
            "action": "Schedule an appointment with your doctor within 1-2 weeks",
            "wait_time": "Within 1-2 weeks",
            "reasons": non_urgent_reasons,
            "advice": [
                "Schedule a checkup with your primary care doctor",
                "Monitor your symptoms — note if they worsen",
                "Maintain a record of when symptoms occur",
                "If symptoms suddenly worsen, seek immediate care",
                "Consider lifestyle modifications"
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ══════════════════════════════════════════════════════════════
# GENERAL ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return jsonify({
        "message": "CardioPredict AI API is running!",
        "version": "2.0",
        "endpoints": ["/predict", "/api/register",
                      "/api/login", "/api/logout",
                      "/api/me", "/api/history"]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

# ── Create Database Tables ───────────────────────────────────
with app.app_context():
    db.create_all()
    print("✅ Database tables created!")

if __name__ == '__main__':
    app.run(debug=True)