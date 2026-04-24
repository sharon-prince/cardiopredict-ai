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

        chest_pain          = int(data.get('chest_pain', 0))
        chest_pain_severity = int(data.get('chest_pain_severity', 0))
        radiating_pain      = int(data.get('radiating_pain', 0))
        shortness_of_breath = int(data.get('shortness_of_breath', 0))
        breath_severity     = data.get('breath_severity', 'none')
        palpitations        = int(data.get('palpitations', 0))
        dizziness           = int(data.get('dizziness', 0))
        fainting            = int(data.get('fainting', 0))
        sweating            = int(data.get('sweating', 0))
        nausea              = int(data.get('nausea', 0))
        fatigue             = int(data.get('fatigue', 0))
        heart_rate          = int(data.get('heart_rate', 0))
        systolic_bp         = int(data.get('systolic_bp', 0))
        symptom_duration    = data.get('symptom_duration', 'hours')
        cardiac_history     = int(data.get('cardiac_history', 0))

        # ── Determine Urgency Level ──────────────────────────
        emergency = False
        urgent    = False
        symptoms_detected = []
        possible_conditions = []

        # Emergency triggers
        if chest_pain and chest_pain_severity >= 7:
            emergency = True
            symptoms_detected.append("Severe chest pain")
        if chest_pain and radiating_pain:
            emergency = True
            symptoms_detected.append("Chest pain radiating to arm or jaw")
            possible_conditions.append("Heart Attack (Myocardial Infarction)")
        if fainting:
            emergency = True
            symptoms_detected.append("Fainting or loss of consciousness")
            possible_conditions.append("Cardiac Arrhythmia or Severe Drop in Blood Pressure")
        if heart_rate > 150 or (0 < heart_rate < 40):
            emergency = True
            symptoms_detected.append(f"Dangerous heart rate: {heart_rate} BPM")
            possible_conditions.append("Serious Heart Rhythm Problem (Arrhythmia)")
        if systolic_bp > 180:
            emergency = True
            symptoms_detected.append(f"Severely high BP: {systolic_bp} mmHg")
            possible_conditions.append("Hypertensive Crisis")
        if chest_pain and shortness_of_breath and breath_severity == 'severe':
            emergency = True
            symptoms_detected.append("Chest pain with severe breathlessness")
            possible_conditions.append("Heart Attack or Pulmonary Embolism")

        # Urgent triggers
        if not emergency:
            if chest_pain and chest_pain_severity >= 4:
                urgent = True
                symptoms_detected.append("Moderate chest pain")
                possible_conditions.append("Unstable Angina or Cardiac Stress")
            if shortness_of_breath and breath_severity in ['moderate','severe']:
                urgent = True
                symptoms_detected.append("Significant shortness of breath")
                possible_conditions.append("Heart Failure or Respiratory Issue")
            if palpitations and cardiac_history:
                urgent = True
                symptoms_detected.append("Palpitations with known cardiac history")
                possible_conditions.append("Cardiac Arrhythmia")
            if dizziness and chest_pain:
                urgent = True
                symptoms_detected.append("Dizziness combined with chest pain")
                possible_conditions.append("Reduced Blood Flow to Brain")
            if heart_rate > 120:
                urgent = True
                symptoms_detected.append(f"Elevated heart rate: {heart_rate} BPM")
            if systolic_bp > 160:
                urgent = True
                symptoms_detected.append(f"High blood pressure: {systolic_bp} mmHg")
            if symptom_duration == 'now' and (chest_pain or shortness_of_breath):
                urgent = True
                symptoms_detected.append("Sudden onset of cardiac symptoms")

        # Non-urgent symptoms
        if not emergency and not urgent:
            if chest_pain: symptoms_detected.append("Mild chest discomfort")
            if palpitations: symptoms_detected.append("Occasional palpitations")
            if fatigue: symptoms_detected.append("Fatigue")
            if nausea: symptoms_detected.append("Nausea")
            if dizziness: symptoms_detected.append("Mild dizziness")
            if sweating: symptoms_detected.append("Unusual sweating")
            possible_conditions.append("Non-cardiac cause — stress, dehydration or minor issue")

        if not symptoms_detected:
            symptoms_detected = ["No significant symptoms detected"]

        # Remove duplicates
        possible_conditions = list(dict.fromkeys(possible_conditions))

        # ── Build Response Based on Level ───────────────────
        if emergency:
            return jsonify({
                "level": "Emergency",
                "color": "red",
                "headline": "Seek Immediate Medical Attention",
                "description": "Your symptoms suggest a possible cardiac emergency. Every minute matters.",
                "symptoms_detected": symptoms_detected,
                "possible_conditions": possible_conditions,
                "immediate_steps": [
                    "Call 112 (National Emergency) or 108 (Ambulance) immediately",
                    "Do NOT drive yourself — wait for the ambulance",
                    "Sit or lie down in a comfortable position",
                    "Loosen any tight clothing around your chest",
                    "Chew one aspirin (325mg) if available and not allergic",
                    "Have someone stay with you until help arrives",
                    "Unlock your front door so paramedics can enter"
                ],
                "contacts": [
                    {"name": "National Emergency", "number": "112", "type": "emergency"},
                    {"name": "Ambulance (108)", "number": "108", "type": "emergency"},
                    {"name": "Cardiac Helpline", "number": "1800-112-114", "type": "helpline"},
                ]
            })

        elif urgent:
            return jsonify({
                "level": "Urgent",
                "color": "orange",
                "headline": "Visit a Hospital or Clinic Within 2 Hours",
                "description": "Your symptoms need medical evaluation soon. Do not wait until tomorrow.",
                "symptoms_detected": symptoms_detected,
                "possible_conditions": possible_conditions,
                "immediate_steps": [
                    "Go to the nearest hospital Emergency or Urgent Care department",
                    "Do not drive alone — have someone accompany you",
                    "Carry a list of your current medications",
                    "Note when symptoms started and how they have changed",
                    "Call ahead if possible so the hospital can prepare",
                    "If symptoms worsen on the way, call 112 immediately"
                ],
                "contacts": [
                    {"name": "National Emergency", "number": "112", "type": "emergency"},
                    {"name": "Ambulance (108)", "number": "108", "type": "emergency"},
                    {"name": "Find Nearest Hospital", "number": "maps", "type": "maps"},
                ]
            })

        else:
            return jsonify({
                "level": "Non-Urgent",
                "color": "green",
                "headline": "Schedule a Doctor's Appointment",
                "description": "Your symptoms do not suggest an immediate emergency, but should be discussed with your doctor.",
                "symptoms_detected": symptoms_detected,
                "possible_conditions": possible_conditions,
                "immediate_steps": [
                    "Schedule an appointment with your doctor within 1–2 weeks",
                    "Keep a symptom diary — note when symptoms occur",
                    "Monitor for any worsening and seek urgent care if needed",
                    "Avoid strenuous activity until you have spoken to a doctor",
                    "Stay hydrated and get adequate rest"
                ],
                "contacts": [
                    {"name": "National Emergency", "number": "112", "type": "emergency"},
                    {"name": "Find Nearest Clinic", "number": "maps", "type": "maps"},
                ]
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# ══════════════════════════════════════════════════════════════
# RECOMMENDATIONS ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/api/recommendations', methods=['POST'])
def recommendations():
    try:
        data = request.get_json()

        risk_level = data.get('risk_level', 'Low')
        risk_score = float(data.get('risk_score', 0))
        cholesterol = int(data.get('cholesterol', 1))
        gluc = int(data.get('gluc', 1))
        smoke = int(data.get('smoke', 0))
        alco = int(data.get('alco', 0))
        active = int(data.get('active', 1))
        age = int(data.get('age', 40))
        bmi = float(data.get('bmi', 22))
        ap_hi = int(data.get('ap_hi', 120))

        diet = []
        exercise = []
        lifestyle = []
        monitoring = []
        priority_actions = []

        # ── Diet Recommendations ─────────────────────────────
        if cholesterol >= 2:
            diet.append("Reduce saturated fats — avoid red meat, full-fat dairy and fried foods")
            diet.append("Increase soluble fibre — eat oats, beans, lentils and barley daily")
            diet.append("Add omega-3 rich foods — salmon, mackerel or walnuts 2-3 times per week")
            diet.append("Include nuts and seeds — a small handful of almonds daily can lower LDL")
            priority_actions.append("Lower your cholesterol through diet — reduce saturated fats and increase fibre")

        if gluc >= 2:
            diet.append("Limit refined carbohydrates — avoid white bread, white rice and sugary drinks")
            diet.append("Choose low glycaemic index foods — sweet potato, brown rice, whole grain bread")
            diet.append("Eat smaller, more frequent meals to keep blood sugar stable")
            priority_actions.append("Manage your blood sugar — cut refined carbs and sugary foods")

        if ap_hi >= 140:
            diet.append("Reduce sodium intake — target less than 2,300 mg per day")
            diet.append("Follow the DASH diet — rich in fruits, vegetables, whole grains and low-fat dairy")
            diet.append("Increase potassium-rich foods — bananas, sweet potatoes and spinach help lower BP")
            priority_actions.append("Lower your blood pressure through the DASH diet and sodium reduction")

        if bmi >= 25:
            diet.append("Aim for a calorie deficit of 300-500 calories per day for gradual weight loss")
            diet.append("Replace processed snacks with whole foods — fruits, vegetables and nuts")
            diet.append("Practice portion control — use smaller plates and eat slowly")

        if not diet:
            diet.append("Maintain your healthy diet — continue eating plenty of fruits and vegetables")
            diet.append("Stay hydrated — drink at least 8 glasses of water daily")
            diet.append("Limit processed foods and added sugars as a preventive measure")

        # ── Exercise Recommendations ─────────────────────────
        if active == 0:
            if age >= 60:
                exercise.append("Start with gentle walks — 10 minutes, 3 times per day is a great beginning")
                exercise.append("Try chair yoga or tai chi — low impact, improves balance and flexibility")
                exercise.append("Gradually increase to 30 minutes of light activity most days of the week")
            else:
                exercise.append("Begin with 15-minute walks daily — consistency matters more than intensity")
                exercise.append("Progress to 30 minutes of moderate activity (brisk walking, cycling) 5 days/week")
                exercise.append("Consider swimming — excellent full-body, low-impact cardiovascular workout")
            priority_actions.append("Start exercising regularly — even 15 minutes of walking daily makes a difference")
        else:
            exercise.append("Maintain your current activity level — you are meeting basic recommendations")
            exercise.append("Consider adding strength training 2 days per week to complement cardio")
            exercise.append("Try interval training — alternating moderate and brisk pace improves heart health")

        if risk_level in ["High", "Very High"]:
            exercise.append("Consult your doctor before starting any new exercise programme at your risk level")
            exercise.append("Avoid heavy weightlifting and high-intensity interval training until medically cleared")

        # ── Lifestyle Recommendations ────────────────────────
        if smoke == 1:
            lifestyle.append("Quit smoking — cardiovascular risk reduces by 50% within just 1 year of quitting")
            lifestyle.append("Seek support — nicotine replacement therapy doubles your chances of success")
            lifestyle.append("Avoid secondhand smoke environments as much as possible")
            priority_actions.append("Quit smoking — this single change has the biggest impact on your heart health")

        if alco == 1:
            lifestyle.append("Reduce alcohol intake — aim for no more than 1 drink per day for women, 2 for men")
            lifestyle.append("Have at least 2-3 alcohol-free days per week")
            lifestyle.append("Replace alcoholic drinks with sparkling water, herbal tea or fresh juices")

        lifestyle.append("Aim for 7-9 hours of quality sleep every night — poor sleep raises cardiovascular risk")
        lifestyle.append("Practise stress management — try 10 minutes of deep breathing or meditation daily")
        lifestyle.append("Maintain social connections — loneliness and isolation increase heart disease risk")

        if bmi >= 30:
            lifestyle.append("Set a realistic weight loss goal — losing just 5-10% of body weight significantly reduces risk")
            priority_actions.append("Work towards a healthier weight — even modest weight loss improves heart health")

        # ── Monitoring Recommendations ───────────────────────
        if ap_hi >= 130:
            monitoring.append("Check your blood pressure at home regularly — aim for readings below 130/80")
            monitoring.append("Keep a BP diary — note readings, time of day and any symptoms")

        if cholesterol >= 2:
            monitoring.append("Get a full lipid panel blood test every 6 months")
            monitoring.append("Ask your doctor about your LDL target — for high risk patients it is typically below 70 mg/dL")

        if risk_level == "Very High":
            monitoring.append("Schedule a cardiology consultation within 2 weeks")
            monitoring.append("Do not skip any prescribed medications")
        elif risk_level == "High":
            monitoring.append("See your doctor within 4 weeks to discuss your cardiovascular risk")
        elif risk_level == "Moderate":
            monitoring.append("Schedule a general health check-up within 2-3 months")
        else:
            monitoring.append("Continue with your annual health check-up routine")

        monitoring.append("Know your numbers — keep track of BP, cholesterol and blood sugar over time")
        monitoring.append("Seek immediate care if you experience chest pain, breathlessness or dizziness")

        # ── Top 3 Priority Actions ───────────────────────────
        if not priority_actions:
            priority_actions = [
                "Maintain your current healthy habits — you are doing well",
                "Schedule a routine annual health check-up",
                "Continue monitoring your blood pressure and cholesterol regularly"
            ]

        return jsonify({
            "risk_level": risk_level,
            "risk_score": risk_score,
            "priority_actions": priority_actions[:3],
            "diet": diet,
            "exercise": exercise,
            "lifestyle": lifestyle,
            "monitoring": monitoring
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