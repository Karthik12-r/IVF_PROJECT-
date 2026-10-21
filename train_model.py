# train_model.py
import os
import json
import joblib
import numpy as np
import pandas as pd

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import your app context and models
from main import app  # Use the existing app instance
from models import db, PatientData, WellnessLog, Prediction, User

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "ivf_success_model.pkl")
META_FILE = os.path.join(MODEL_DIR, "ivf_model_metadata.json")

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_training_rows(min_rows=50):
    """
    Try to fetch historical rows by joining PatientData with the most recent WellnessLog per user.
    Returns a pandas DataFrame. If not enough rows found, returns None.
    """
    with app.app_context():
        # Query PatientData and join with latest wellness log per user
        rows = []
        patients = PatientData.query.all()
        for p in patients:
            # find latest wellness log for this user (if any)
            wl = WellnessLog.query.filter_by(user_id=p.user_id).order_by(WellnessLog.date.desc()).first()
            
            # We'll only use rows if the DB has an explicit 'ivf_success' column somewhere.
            # Many production DBs don't have labels; so return None to use synthetic dataset.
            # For this prototype, we assume 'ivf_success' is not directly in PatientData/WellnessLog
            # and thus will always fall back to synthetic data unless explicitly added to models.py
            rows.append({
                "user_id": p.user_id, # Keep user_id for potential future use
                "age": p.age,
                "bmi": p.bmi,
                "amh": p.amh_level,
                "fsh": p.fsh_level,
                "previous_ivf": p.previous_ivf_cycles,
                "stress": wl.stress_level if wl else np.nan,
                "sleep_hours": wl.sleep_hours if wl else np.nan,
                "exercise_min": wl.exercise_minutes if wl else np.nan,
                "ivf_success": None # Placeholder, will be None for synthetic fallback
            })
        df = pd.DataFrame(rows)
        # if no explicit label column exists, return None to indicate synthetic fallback
        # For this project, we're explicitly falling back to synthetic data as there's no 'ivf_success' column in models.py
        return None # Always return None to use synthetic data for now

def generate_synthetic_dataset(n=200, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "age": np.random.randint(25, 45, n),
        "bmi": np.round(np.random.uniform(18, 35, n), 2),
        "amh": np.round(np.random.uniform(0.5, 6.0, n), 2),
        "fsh": np.round(np.random.uniform(3.0, 12.0, n), 2),
        "previous_ivf": np.random.randint(0, 4, n),
        "stress": np.random.randint(1, 6, n),
        "sleep_hours": np.round(np.random.uniform(4, 9, n), 1),
        "exercise_min": np.random.randint(0, 60, n)
    })
    # Create a heuristic label probability and sample success
    prob = (
        (1 - (df["age"] - 25) / 30) * 0.28 +
        (df["amh"] / 6.0) * 0.32 +
        (1 - (df["fsh"] - 3) / 9) * 0.2 +
        ((6 - df["stress"]) / 5) * 0.2
    )
    prob = np.clip(prob, 0.05, 0.95)
    df["ivf_success"] = np.random.binomial(1, prob)
    return df

def train_and_save(df):
    ensure_dirs()
    feature_cols = ["age", "bmi", "amh", "fsh", "previous_ivf", "stress", "sleep_hours", "exercise_min"]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["ivf_success"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Test accuracy:", acc)
    print(classification_report(y_test, preds))

    # Save model and metadata
    joblib.dump(clf, MODEL_FILE)
    meta = {"feature_order": feature_cols, "model": "RandomForestClassifier", "accuracy": float(acc)}
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved model ->", MODEL_FILE)
    print("Saved metadata ->", META_FILE)

if __name__ == "__main__":
    print("Attempting to fetch labeled data from your DB...")
    df_db = fetch_training_rows()
    if df_db is None:
        print("Not enough labeled rows in DB or no explicit 'ivf_success' column. Generating synthetic dataset instead.")
        df = generate_synthetic_dataset(300)
    else:
        df = df_db

    print("Training model on", len(df), "rows...")
    train_and_save(df)
    print("Training finished.")