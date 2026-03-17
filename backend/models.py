from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

# ── User Table ───────────────────────────────────────────────
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.email}>'

# ── Prediction Table ─────────────────────────────────────────
class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Input features
    age = db.Column(db.Integer)
    gender = db.Column(db.Integer)
    height = db.Column(db.Integer)
    weight = db.Column(db.Float)
    ap_hi = db.Column(db.Integer)
    ap_lo = db.Column(db.Integer)
    cholesterol = db.Column(db.Integer)
    gluc = db.Column(db.Integer)
    smoke = db.Column(db.Integer)
    alco = db.Column(db.Integer)
    active = db.Column(db.Integer)

    # Results
    risk_score = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    probability = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.risk_level} for User {self.user_id}>'