from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import json
import numpy as np
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
login_manager.login_view = 'login'

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

# ── Register ─────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        full_name = data.get('full_name')
        email = data.get('email')
        password = data.get('password')

        if not full_name or not email or not password:
            return jsonify({"error": "All fields are required"}), 400

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already registered"}), 400

        # Create new user
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

# ── Login ─────────────────────────────────────────────────────
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
            "user": {"name": user.full_name, "email": user.email, "id": user.id}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Logout ───────────────────────────────────────────────────
@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

# ── Get Current User ─────────────────────────────────────────
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

        # Save to database if user is logged in
        if current_user.is_authenticated:
            prediction = Prediction(
                user_id=current_user.id,
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
            "saved": current_user.is_authenticated
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Get Prediction History ────────────────────────────────────
@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    predictions = Prediction.query.filter_by(
        user_id=current_user.id
    ).order_by(Prediction.created_at.desc()).limit(10).all()

    history = []
    for p in predictions:
        history.append({
            "id": p.id,
            "risk_score": p.risk_score,
            "risk_level": p.risk_level,
            "age": p.age,
            "date": p.created_at.strftime("%d %b %Y, %I:%M %p")
        })

    return jsonify({"history": history})

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