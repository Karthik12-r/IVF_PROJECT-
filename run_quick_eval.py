import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Try to load dataset from ChoreCompanion folder
data_path = Path(__file__).parent / 'ChoreCompanion' / 'ivf_5000_dataset.xlsx'

if data_path.exists():
    try:
        df = pd.read_excel(data_path)
        print(f"Loaded dataset from {data_path} with shape {df.shape}")
    except Exception as e:
        print(f"Error loading excel: {e}")
        df = None
else:
    df = None

if df is None:
    # create sample data similar to model_training._create_sample_data
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
    print(f"Created sample dataset with shape {df.shape}")

# Prepare X/y
target = 'IVF_Success'
X = df.drop(columns=[target]).values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

results = {
    'train_accuracy': float(train_acc),
    'validation_accuracy': float(test_acc),
    'classification_report_validation': classification_report(y_test, y_test_pred, output_dict=True)
}

import sys
print('\n===BEGIN_EVAL_RESULTS===')
print(json.dumps(results, indent=2))
print('===END_EVAL_RESULTS===\n')
sys.stdout.flush()

# Also write results to a file so the runner can read them reliably
out_path = Path(__file__).parent / 'run_quick_eval_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print(f"Wrote results to {out_path}")
sys.stdout.flush()
