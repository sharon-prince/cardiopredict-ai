# ❤️ CardioPredict AI

**An Intelligent Cardiovascular Disease Risk Prediction, Triage Classification
and Personalised Care Recommendation System**

> BCA Final Year Project — Academic Year 2025–2026
> 
> ⚠️ For educational purposes only. Not a certified medical device.

---

## 📋 Project Overview

CardioPredict AI is a full-stack web application that integrates three
core modules into a single cardiovascular health management platform:

| Module | Description |
|--------|-------------|
| 📊 Risk Assessment | ML-based CVD risk prediction using XGBoost trained on 68,517 patients |
| 🩺 Symptom Triage | Priority-based symptom checker with emergency contact guidance |
| 💊 Health Advice | Personalised diet, exercise and lifestyle recommendations |

---

## 🏆 Model Performance

| Model | Validation AUC | Test AUC |
|-------|---------------|----------|
| Logistic Regression | 0.7976 | — |
| Random Forest | 0.8046 | — |
| **XGBoost (Selected)** | **0.8061** | **0.7977** |

---

## 🛠️ Tech Stack

- **Backend:** Python 3.12, Flask 3.1, Flask-SQLAlchemy, Flask-Login
- **ML:** XGBoost, scikit-learn, SHAP, pandas, numpy
- **Database:** SQLite
- **Frontend:** HTML5, Bootstrap 5, Vanilla JavaScript
- **Auth:** werkzeug PBKDF2-SHA256 password hashing
- **Version Control:** Git + GitHub

---

## 📁 Project Structure
cardiopredict-ai/
├── data/                    # Dataset files and generated charts
│   ├── cardio_train.csv     # Original Kaggle dataset (70,000 records)
│   ├── cardio_cleaned.csv   # Cleaned dataset (68,517 records)
│   └── *.png                # Generated visualization charts
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_visualizations.ipynb
├── backend/
│   ├── app.py               # Flask application with all API routes
│   ├── models.py            # SQLAlchemy database models
│   └── models/              # Saved ML model files
│       ├── xgb_model.pkl    # Trained XGBoost model
│       ├── scaler.pkl       # Fitted StandardScaler
│       └── feature_names.json
├── frontend/
│   ├── home.html            # Landing page
│   ├── index.html           # Risk Assessment (Module 1)
│   ├── triage.html          # Symptom Checker (Module 2)
│   ├── recommendations.html # Health Advice + Goals (Module 3)
│   └── login.html           # Login and Registration
├── database/                # SQLite database (excluded from git)
├── requirements.txt         # Python dependencies
└── README.md

---

## 🚀 Setup and Installation

### Prerequisites
- Python 3.10 or higher
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/sharon-prince/cardiopredict-ai.git
cd cardiopredict-ai
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the Flask Server
```bash
cd backend
python app.py
```

You should see:
✅ Model loaded!
✅ Database configured!
✅ Database tables created!

Running on http://127.0.0.1:5000 

### Step 4 — Open the Application
Open a new terminal window and run:
```bash
start frontend/home.html    # Windows
open frontend/home.html     # macOS
```

Or manually open `frontend/home.html` in your browser.

---

## 📊 Dataset

- **Source:** Kaggle — Cardiovascular Disease Dataset by Svetlana Ulianova
- **Link:** https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
- **Size:** 70,000 patient records (68,517 after cleaning)
- **Features:** 11 clinical and lifestyle features + 6 engineered features
- **Target:** Binary — cardiovascular disease present (1) or absent (0)

---

## 🔑 Key Features

- ✅ XGBoost classifier with 80.6% AUC score
- ✅ SHAP explainability showing which factors drive each prediction
- ✅ Priority-first triage logic — Emergency always checked first
- ✅ Personalised recommendations based on individual risk factors
- ✅ User authentication with secure password hashing
- ✅ Prediction history for logged-in users
- ✅ Assessment streak and goal tracking
- ✅ Form pre-fill for returning users
- ✅ Responsive design with mobile menu

---

## 📱 Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | home.html | Landing page with module overview |
| Risk Assessment | index.html | 11-field health metrics form |
| Symptom Checker | triage.html | Tap-based symptom assessment |
| Health Advice | recommendations.html | Personalised recommendations + goals |
| Login / Register | login.html | User authentication |

---

## ⚠️ Disclaimer

CardioPredict AI is an academic prototype developed as a BCA Final Year
Project. It is not a certified medical device and has not been clinically
validated. All outputs are for educational and informational purposes only.
Always consult a qualified healthcare professional for medical decisions.

---

## 👥 Team

- Sharon Prince Thomas
- Chinkki R

**Institution:**ST FRANCIS COLLEGE

**Guide:** Dr Geetha.S

**Academic Year:** 2025–2026