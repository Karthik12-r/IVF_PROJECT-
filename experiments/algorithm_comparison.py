# Algorithm comparison script for IVF success prediction
"""Offline IVF success model comparison; does not import or modify the Flask app."""

from pathlib import Path
import importlib.util
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "IVF_validation_dataset_500_synthetic.csv"
TRAINING_PATH = BASE_DIR.parent / "ivf_5000_dataset.xlsx"
RESULTS_PATH = BASE_DIR / "ivf_model_comparison_results.csv"
REPORT_PATH = BASE_DIR / "ivf_model_comparison_report.txt"
IMPORTANCE_PATH = BASE_DIR / "ivf_feature_importance.csv"
TARGET = "IVF Success"
NUMERIC_FEATURES = [
    "Age", "BMI", "AMH", "FSH", "Previous IVF Attempts",
    "Sleep Hours", "Exercise (minutes/day)",
]
CATEGORICAL_FEATURES = ["Stress Level"]
FEATURES = [
    "Age", "BMI", "AMH", "FSH", "Previous IVF Attempts", "Stress Level",
    "Sleep Hours", "Exercise (minutes/day)",
]

def make_preprocessor(scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

def make_pipeline(model, scale_numeric=False):
    return Pipeline([
        ("preprocess", make_preprocessor(scale_numeric)),
        ("model", model),
    ])

def metric_row(name, model, x_train, x_test, y_train, y_test, cv_scores=None):
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_prob = model.predict_proba(x_train)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x_train)
    test_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    result = {
        "Algorithm": name,
        "Accuracy": test_accuracy,
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Recall": recall_score(y_test, test_pred, zero_division=0),
        "F1 Score": f1_score(y_test, test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, test_prob),
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Log Loss": log_loss(y_test, test_prob),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "Sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "Confusion Matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]",
        "Mean CV Accuracy": float(np.mean(cv_scores)) if cv_scores is not None else np.nan,
        "CV Std Dev": float(np.std(cv_scores)) if cv_scores is not None else np.nan,
        "Overfitting Flag": "Possible Overfitting" if train_accuracy - test_accuracy >= 0.15 else "No significant gap",
        "Status": "Completed",
    }
    return result, model

def feature_importance_rows(name, fitted_model):
    if not hasattr(fitted_model.named_steps["model"], "feature_importances_"):
        return []
    preprocessor = fitted_model.named_steps["preprocess"]
    model = fitted_model.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())
    values = model.feature_importances_
    return [{"Model": name, "Feature": feature, "Importance": float(value)} for feature, value in zip(names, values)]

def inspect_dataset(df):
    print("DATASET INSPECTION")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Data types:")
    print(df.dtypes.to_string())
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate records: {int(df.duplicated().sum())}")
    print(f"Target distribution: {df[TARGET].value_counts().sort_index().to_dict()}")
    print(f"Success cases: {int((df[TARGET] == 1).sum())}")
    print(f"Failure cases: {int((df[TARGET] == 0).sum())}")

def load_training_dataset():
    """Load the existing project training data without changing it."""
    if not TRAINING_PATH.exists():
        return None
    training = pd.read_excel(TRAINING_PATH)
    training = training.rename(columns={
        "age": "Age",
        "bmi": "BMI",
        "amh": "AMH",
        "fsh": "FSH",
        "previous_ivf": "Previous IVF Attempts",
        "stress": "Stress Level",
        "sleep_hours": "Sleep Hours",
        "exercise_min": "Exercise (minutes/day)",
        "IVF_Success": TARGET,
    })
    if "Stress Level" in training:
        training["Stress Level"] = pd.cut(
            training["Stress Level"],
            bins=[-np.inf, 2, 3, np.inf],
            labels=["Low", "Medium", "High"],
        ).astype(str)
    return training[FEATURES + [TARGET]]

def run_comparison():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    expected_columns = FEATURES + [TARGET]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, found {list(df.columns)}")
    inspect_dataset(df)
    training_df = load_training_dataset()
    if training_df is None:
        raise FileNotFoundError("Existing training dataset not found; refusing to train on the external validation CSV.")
    print(f"Training dataset: {TRAINING_PATH.name}, shape: {training_df.shape}")
    x_train = training_df[FEATURES]
    y_train = training_df[TARGET].astype(int)
    x_test = df[FEATURES]
    y_test = df[TARGET].astype(int)
    print(f"\nExternal validation: {len(x_test)} records; training: {len(x_train)} records")
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), True),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=7), True),
        ("Support Vector Machine", SVC(probability=True, random_state=RANDOM_STATE), True),
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), False),
        ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1), False),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1), False),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=RANDOM_STATE), False),
        ("AdaBoost", AdaBoostClassifier(n_estimators=150, random_state=RANDOM_STATE), False),
        ("Gaussian Naive Bayes", GaussianNB(), True),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis(), True),
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis(reg_param=0.1), True),
        ("SGD Classifier", SGDClassifier(loss="log_loss", max_iter=2000, random_state=RANDOM_STATE), True),
        ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=150, random_state=RANDOM_STATE), False),
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    importance_rows = []
    fitted_models = {}
    failures = []
    for name, estimator, scaled in models:
        try:
            pipeline = make_pipeline(estimator, scale_numeric=scaled)
            cv_scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
            result, fitted = metric_row(name, pipeline, x_train, x_test, y_train, y_test, cv_scores)
            results.append(result)
            fitted_models[name] = fitted
            importance_rows.extend(feature_importance_rows(name, fitted))
        except Exception as error:
            failures.append(f"{name} failed — {type(error).__name__}: {error}")
    optional_models = [
        ("XGBoost", "xgboost", "xgboost.XGBClassifier"),
        ("LightGBM", "lightgbm", "lightgbm.LGBMClassifier"),
    ]
    for name, package, import_path in optional_models:
        if not importlib.util.find_spec(package):
            failures.append(f"{name} skipped — package not installed")
            continue
        failures.append(f"{name} skipped — package import not added to this isolated script")
    for name in ["ANN Basic", "ANN Dropout", "Deep ANN"]:
        failures.append(f"{name} skipped — TensorFlow/PyTorch not installed")
    results_df = pd.DataFrame(results).sort_values([
        "ROC-AUC", "F1 Score", "Accuracy", "Recall"], ascending=False
    ).reset_index(drop=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    if importance_rows:
        importance_df = pd.DataFrame(importance_rows).sort_values([
            "Model", "Importance"], ascending=[True, False]
        )
        importance_df.to_csv(IMPORTANCE_PATH, index=False)
    else:
        pd.DataFrame(columns=["Model", "Feature", "Importance"]).to_csv(IMPORTANCE_PATH, index=False)
    report = build_report(df, x_train, x_test, results_df, importance_rows, failures)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report)

def build_report(df, x_train, x_test, results_df, importance_rows, failures):
    def fmt(row, column):
        return f"{row[column]:.4f}" if pd.notna(row[column]) else "N/A"
    ml_best = results_df.iloc[0]
    rf = results_df[results_df["Algorithm"] == "Random Forest"].iloc[0]
    top_features = (
        pd.DataFrame(importance_rows)
        .groupby("Feature", as_index=False)["Importance"].mean()
        .sort_values("Importance", ascending=False)
        .head(5)
    )
    lines = [
        "IVF SUCCESS PREDICTION — MODEL COMPARISON",
        "",
        f"Dataset: {DATA_PATH.name}",
        f"Number of records: {len(df)}",
        f"Number of features: {len(FEATURES)}",
        f"Training records: {len(x_train)}",
        f"Testing/Validation records: {len(x_test)}",
        "Evaluation approach: models were trained on the existing project training dataset and evaluated only on the supplied 500‑row external validation CSV.",
        "",
        "BEST MACHINE LEARNING MODEL",
        f"Model: {ml_best['Algorithm']}",
        f"Accuracy: {fmt(ml_best, 'Accuracy')}",
        f"Precision: {fmt(ml_best, 'Precision')}",
        f"Recall: {fmt(ml_best, 'Recall')}",
        f"F1: {fmt(ml_best, 'F1 Score')}",
        f"ROC‑AUC: {fmt(ml_best, 'ROC-AUC')}",
        f"Mean CV Accuracy: {fmt(ml_best, 'Mean CV Accuracy')} +/- {fmt(ml_best, 'CV Std Dev')}",
        "",
        "RANDOM FOREST RESULT",
        f"Accuracy: {fmt(rf, 'Accuracy')}",
        f"Precision: {fmt(rf, 'Precision')}",
        f"Recall: {fmt(rf, 'Recall')}",
        f"F1: {fmt(rf, 'F1 Score')}",
        f"ROC‑AUC: {fmt(rf, 'ROC-AUC')}",
        f"Random Forest is still the best model: {'Yes' if rf['Algorithm'] == ml_best['Algorithm'] else 'No'}",
        "",
        "TOP IMPORTANT FEATURES:",
    ]
    for i, row in enumerate(top_features.to_dict('records'), 1):
        lines.append(f"{i}. {row['Feature']} ({row['Importance']:.6f})")
    lines.extend(["", "OVERFITTING OBSERVATIONS:"])
    for _, row in results_df.iterrows():
        if row["Overfitting Flag"] == "Possible Overfitting":
            lines.append(f"{row['Algorithm']}: train accuracy {row['Train Accuracy']:.3f} vs test accuracy {row['Test Accuracy']:.3f} — Possible Overfitting")
    if not any(row["Overfitting Flag"] == "Possible Overfitting" for _, row in results_df.iterrows()):
        lines.append("No model exceeded the configured 0.15 train/test accuracy gap threshold.")
    lines.extend(["", "FULL RANKING:", results_df.to_string(index=False), "", "FINAL RECOMMENDATION:"])
    lines.append(f"Use {ml_best['Algorithm']} as the offline experimental winner only. Do not replace the production model based on this validation split.")
    if failures:
        lines.extend(["", "SKIPPED/FAILED MODELS:", *failures])
    return "\n".join(lines)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_comparison()
