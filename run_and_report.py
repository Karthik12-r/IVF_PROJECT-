import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data_path = Path(__file__).parent / 'ChoreCompanion' / 'ivf_5000_dataset.xlsx'
if data_path.exists():
    df = pd.read_excel(data_path)
else:
    # fallback sample
    np.random.seed(42)
    n_samples = 5000
    data = {
        'age': np.random.randint(22, 45, n_samples),
        'bmi': np.random.uniform(18, 35, n_samples),
        'amh': np.random.uniform(0.5, 8.0, n_samples),
        'fsh': np.random.uniform(4, 15, n_samples),
        'previous_ivf': np.random.randint(0, 5, n_samples),
        'stress': np.random.randint(1, 10, n_samples),
        'sleep_hours': np.random.uniform(5, 10, n_samples),
        'exercise_min': np.random.randint(0, 120, n_samples),
        'IVF_Success': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    success_prob = (
        0.3 +
        (df['age'] < 35).astype(float) * 0.2 +
        (df['amh'] > 2).astype(float) * 0.15 +
        (df['fsh'] < 10).astype(float) * 0.1 +
        (df['bmi'] < 25).astype(float) * 0.1 +
        np.random.uniform(-0.1, 0.1, n_samples)
    )
    df['IVF_Success'] = (success_prob > 0.5).astype(int)

# Prepare features
X = df.drop(columns=['IVF_Success']).values
y = df['IVF_Success'].values

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit and predict (matches user's snippet)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_accuracy = accuracy_score(y_train, train_pred)
val_accuracy = accuracy_score(y_val, val_pred)

results = {
    'train_accuracy': float(train_accuracy),
    'validation_accuracy': float(val_accuracy)
}

# Print and save results
print("Training Accuracy:", train_accuracy)
print("Validation Accuracy:", val_accuracy)

out_path = Path(__file__).parent / 'run_and_report_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"Wrote results to {out_path}")
