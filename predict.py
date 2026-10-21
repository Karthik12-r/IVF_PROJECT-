# predict.py
import os
import joblib
import json
import numpy as np
from datetime import datetime, timezone
from functools import lru_cache

from main import app  # Use the existing app instance
from models import db, PatientData, WellnessLog, Prediction, User

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "ivf_success_model.pkl")
META_FILE = os.path.join(MODEL_DIR, "ivf_model_metadata.json")

@lru_cache(maxsize=1)
def load_model_and_meta():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(META_FILE):
        raise FileNotFoundError("Model or metadata files not found. Run train_model.py first.")
    model = joblib.load(MODEL_FILE)
    with open(META_FILE, "r", encoding="utf-8") as metadata_file:
        meta = json.load(metadata_file)
    return model, meta


def _predict_with_probability(model, feature_vector):
    if not hasattr(model, "predict_proba"):
        return int(model.predict(feature_vector)[0]), None

    probabilities = model.predict_proba(feature_vector)[0]
    classes = getattr(model, "classes_", None)
    prediction = classes[int(np.argmax(probabilities))] if classes is not None else model.predict(feature_vector)[0]
    return int(prediction), probabilities

def build_feature_vector(user_id, meta):
    """
    Read PatientData and latest WellnessLog for user_id and return np.array([features])
    Fallbacks are used for missing values.
    """
    with app.app_context():
        pat = PatientData.query.filter_by(user_id=user_id).first()
        if pat is None:
            raise ValueError(f"No PatientData found for user_id={user_id}. Please complete your medical profile.")
        # latest wellness log
        wl = WellnessLog.query.filter_by(user_id=user_id).order_by(WellnessLog.date.desc()).first()

        # Map model features to DB columns with sensible fallbacks
        # Note: BMI calculation fallback if not directly stored or if height/weight are missing
        bmi_val = getattr(pat, "bmi", None)
        if bmi_val is None and getattr(pat, "height", None) and getattr(pat, "weight", None):
            height_m = pat.height / 100
            if height_m > 0:
                bmi_val = pat.weight / (height_m ** 2)
        
        mapping = {
            "age": getattr(pat, "age", None),
            "bmi": bmi_val,
            "amh": getattr(pat, "amh_level", None),
            "fsh": getattr(pat, "fsh_level", None),
            "previous_ivf": getattr(pat, "previous_ivf_cycles", None),
            "stress": wl.stress_level if wl and wl.stress_level is not None else 3,
            "sleep_hours": wl.sleep_hours if wl and wl.sleep_hours is not None else 7.0,
            "exercise_min": wl.exercise_minutes if wl and wl.exercise_minutes is not None else 30
        }
        defaults = {"age": 30, "bmi": 22.5, "amh": 2.5, "fsh": 6.0, "previous_ivf": 0}
        mapping = {feature: defaults.get(feature) if value is None else value
                   for feature, value in mapping.items()}

        feat_order = meta["feature_order"]
        # Ensure all values are float for the model
        fv = np.array([[float(mapping.get(f, 0)) for f in feat_order]]) # Use .get with default 0 for safety
        return fv

def predict_and_store(user_id):
    model, meta = load_model_and_meta()
    fv = build_feature_vector(user_id, meta)
    mapping = {feature: float(value) for feature, value in zip(meta["feature_order"], fv[0])}
    pred, proba = _predict_with_probability(model, fv)

    # store in Prediction table
    with app.app_context():
        p = Prediction(
            user_id=user_id,
            prediction_date=datetime.now(timezone.utc), # Use timezone-aware datetime
            success_probability=float(proba[1]) if proba is not None else float(pred),
            protocol_recommendation=None, # This model doesn't predict protocol, so leave as None or add placeholder
            llm_analysis=json.dumps({"model_features": mapping}), # Store features used for transparency
            model_metadata=json.dumps(meta)
        )
        db.session.add(p)
        db.session.commit()
        return {"prediction": pred, "probability": float(proba[1]) if proba is not None else None, "prediction_id": p.id}

def predict_from_features(features_dict, meta):
    """
    Predict IVF success from a dictionary of features.
    features_dict should have keys matching the model's feature_order, with defaults if missing.
    Returns dict with prediction, probability, etc.
    """
    model, _ = load_model_and_meta()  # meta is already loaded, but we need model

    # Map features_dict to the expected keys, with defaults
    defaults = {
        "age": 30, "bmi": 22.5, "amh": 2.5, "fsh": 6.0,
        "previous_ivf": 0, "stress": 3, "sleep_hours": 7.0, "exercise_min": 30
    }
    mapping = {feature: defaults[feature] if features_dict.get(feature) is None
               else features_dict[feature] for feature in defaults}

    feat_order = meta["feature_order"]
    fv = np.array([[float(mapping.get(f, 0)) for f in feat_order]])
    pred, proba = _predict_with_probability(model, fv)

    probability = float(proba[1]) if proba is not None else float(pred)
    prediction_text = "Success Likely" if probability >= 0.5 else "Success Unlikely"

    return {
        "prediction": pred,
        "success_probability": round(probability * 100, 1),  # as percentage
        "prediction_text": prediction_text,
        "features_used": mapping
    }

if __name__ == "__main__":
    # quick manual test - change user_id as needed
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <user_id>")
        sys.exit(1)
    uid = int(sys.argv[1])
    out = predict_and_store(uid)
    print("Stored prediction:", out)
