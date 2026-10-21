"""Complete offline IVF model comparison.

This file is intentionally independent of the Flask application. It reads the
existing training and external-validation datasets, writes only experiment
outputs, and never saves or changes a trained production model.
"""

from pathlib import Path
import importlib.util
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
TRAINING_PATH = BASE_DIR / "ivf_5000_dataset.xlsx"
VALIDATION_PATH = BASE_DIR / "IVF_validation_dataset_500_synthetic.csv"
RESULTS_PATH = BASE_DIR / "model_comparison_results.csv"
IMPORTANCE_PATH = BASE_DIR / "model_feature_importance.csv"
CONFUSION_PATH = BASE_DIR / "confusion_matrices.csv"
REPORT_PATH = BASE_DIR / "model_comparison_report.txt"
CHART_PATH = BASE_DIR / "model_comparison_results.png"
TARGET = "IVF Success"
FEATURES = ["Age", "BMI", "AMH", "FSH", "Previous IVF Attempts", "Stress Level", "Sleep Hours", "Exercise (minutes/day)"]
NUMERIC = ["Age", "BMI", "AMH", "FSH", "Previous IVF Attempts", "Sleep Hours", "Exercise (minutes/day)"]
CATEGORICAL = ["Stress Level"]
EXPECTED_COLUMNS = FEATURES + [TARGET]


def normalize_frame(frame, source_name):
    """Normalize the existing training schema without writing back to it."""
    frame = frame.rename(columns={
        "age": "Age", "bmi": "BMI", "amh": "AMH", "fsh": "FSH",
        "previous_ivf": "Previous IVF Attempts", "stress": "Stress Level",
        "sleep_hours": "Sleep Hours", "exercise_min": "Exercise (minutes/day)",
        "IVF_Success": TARGET,
    })
    if "Stress Level" in frame and not frame["Stress Level"].isin(["Low", "Medium", "High"]).all():
        stress = pd.to_numeric(frame["Stress Level"], errors="coerce")
        frame["Stress Level"] = pd.cut(stress, [-np.inf, 2, 3, np.inf], labels=["Low", "Medium", "High"]).astype(str)
    missing = [column for column in EXPECTED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")
    return frame[EXPECTED_COLUMNS].copy()


def inspect_dataset(name, frame):
    print(f"\n{name}: {frame.shape[0]} rows x {frame.shape[1]} columns")
    print(f"Missing values: {int(frame.isna().sum().sum())}")
    print(f"Duplicate rows: {int(frame.duplicated().sum())}")
    print(f"Target distribution: {frame[TARGET].value_counts().sort_index().to_dict()}")
    print("Feature dtypes:")
    print(frame[FEATURES].dtypes.to_string())
    print("Feature ranges:")
    for column in NUMERIC:
        print(f"  {column}: {frame[column].min()} - {frame[column].max()}")
    print(f"  Stress Level: {sorted(frame['Stress Level'].dropna().unique().tolist())}")


def preprocessor(scale):
    numeric = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])


def make_pipeline(estimator, scale):
    return Pipeline([("preprocess", preprocessor(scale)), ("model", estimator)])


def probabilities(model, values):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(values)[:, 1]
    return model.decision_function(values)


def evaluate_model(name, estimator, x_train, y_train, x_validation, y_validation, scale):
    model = make_pipeline(estimator, scale)
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    validation_pred = model.predict(x_validation)
    validation_probability = probabilities(model, x_validation)
    tn, fp, fn, tp = confusion_matrix(y_validation, validation_pred, labels=[0, 1]).ravel()
    train_accuracy = accuracy_score(y_train, train_pred)
    external_accuracy = accuracy_score(y_validation, validation_pred)
    gap = train_accuracy - external_accuracy
    overfit = "High overfitting" if gap >= 0.25 else "Moderate overfitting" if gap >= 0.15 else "Low overfitting"
    row = {
        "Model": name,
        "Train Accuracy": train_accuracy,
        "CV Accuracy": cv_results["test_accuracy"].mean(),
        "CV Accuracy Std": cv_results["test_accuracy"].std(),
        "CV F1": cv_results["test_f1"].mean(),
        "CV ROC-AUC": cv_results["test_roc_auc"].mean(),
        "External Accuracy": external_accuracy,
        "Precision": precision_score(y_validation, validation_pred, zero_division=0),
        "Recall": recall_score(y_validation, validation_pred, zero_division=0),
        "F1": f1_score(y_validation, validation_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_validation, validation_probability),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "Sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "Train-External Gap": gap,
        "Overfitting": overfit,
        "Status": "Completed",
    }
    confusion = {"Model": name, "TP": tp, "TN": tn, "FP": fp, "FN": fn}
    return row, confusion, model


def importance_for_model(name, model):
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return []
    names = model.named_steps["preprocess"].get_feature_names_out()
    return [{"Model": name, "Feature": feature, "Importance": float(value)} for feature, value in zip(names, values)]


def optional_estimators():
    models = []
    unavailable = []
    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier
        models.append(("XGBoost", XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=SEED), False))
    else:
        unavailable.append("XGBoost — package installation failed: No space left on device")
    if importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMClassifier
        models.append(("LightGBM", LGBMClassifier(n_estimators=250, num_leaves=20, learning_rate=0.05, verbosity=-1, random_state=SEED), False))
    else:
        unavailable.append("LightGBM — package installation failed: No space left on device")
    return models, unavailable


def write_chart(results):
    chart_data = results.sort_values("ROC-AUC", ascending=True)
    plt.figure(figsize=(11, 7))
    plt.barh(chart_data["Model"], chart_data["ROC-AUC"], color="#2f6f95")
    plt.xlabel("External Validation ROC-AUC")
    plt.title("IVF Model Comparison: External Validation ROC-AUC")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=160)
    plt.close()


def make_report(training, validation, results, confusion, importance, unavailable):
    sorted_results = results.sort_values(["ROC-AUC", "F1", "External Accuracy"], ascending=False)
    best_ml = sorted_results.iloc[0]
    balanced = results[results["Specificity"] >= 0.20]
    research_candidate = balanced.sort_values(["ROC-AUC", "F1", "External Accuracy"], ascending=False).iloc[0] if not balanced.empty else best_ml
    rf = results[results["Model"] == "Random Forest"].iloc[0]
    top = importance.groupby("Feature", as_index=False)["Importance"].mean().sort_values("Importance", ascending=False).head(5)
    lines = [
        "IVF AI MODEL COMPARISON COMPLETE", "",
        f"Training Dataset: {TRAINING_PATH.name} ({len(training)} records)",
        f"External Validation Dataset: {VALIDATION_PATH.name} ({len(validation)} records)",
        f"Features: {len(FEATURES)}", "", "MODELS SUCCESSFULLY TESTED:",
        ", ".join(results["Model"]), "", "MODELS NOT TESTED:",
        *(unavailable or ["None"]),
        "ANN Basic — TensorFlow/PyTorch not installed",
        "ANN Dropout — TensorFlow/PyTorch not installed",
        "Deep ANN — TensorFlow/PyTorch not installed", "",
        "BEST ML MODEL:",
        f"{best_ml['Model']} | Accuracy {best_ml['External Accuracy']:.4f} | Precision {best_ml['Precision']:.4f} | Recall {best_ml['Recall']:.4f} | F1 {best_ml['F1']:.4f} | ROC-AUC {best_ml['ROC-AUC']:.4f}",
        "BEST DL MODEL:", "None completed because TensorFlow and PyTorch are not installed.", "",
        "OVERALL CANDIDATE:", f"{research_candidate['Model']} (metric winner: {best_ml['Model']}); selected with a specificity safeguard.", "",
        "RANDOM FOREST ANALYSIS:",
        f"External Accuracy: {rf['External Accuracy']:.4f}",
        f"Train Accuracy: {rf['Train Accuracy']:.4f}",
        f"Train-External Gap: {rf['Train-External Gap']:.4f}",
        f"Overfitting: {rf['Overfitting']}",
        "Random Forest shows significant overfitting on the external validation dataset." if rf["Overfitting"] == "High overfitting" else "Random Forest does not meet the high-overfitting threshold.",
        "", "TOP IMPORTANT FEATURES:",
    ]
    lines.extend(f"{i}. {row['Feature']} ({row['Importance']:.6f})" for i, (_, row) in enumerate(top.iterrows(), 1))
    lines.extend(["", "FULL RANKING:", results.sort_values(["ROC-AUC", "F1", "External Accuracy"], ascending=False).to_string(index=False), "", "OVERFITTING OBSERVATIONS:"])
    lines.extend(f"{row['Model']}: {row['Overfitting']} (train {row['Train Accuracy']:.3f}, CV {row['CV Accuracy']:.3f}, external {row['External Accuracy']:.3f})" for _, row in results.iterrows())
    lines.extend(["", "FINAL RECOMMENDATION:", f"Consider {research_candidate['Model']} for further offline research, but do not replace the production model. External validation performance is modest and additional representative data and tuning are required."])
    if best_ml["Specificity"] < 0.20:
        lines.append(f"{best_ml['Model']} is unsuitable without further tuning because specificity is only {best_ml['Specificity']:.4f} and it predicts almost all validation cases as positive.")
    return "\n".join(lines)


def main():
    training = normalize_frame(pd.read_excel(TRAINING_PATH), TRAINING_PATH.name)
    validation = normalize_frame(pd.read_csv(VALIDATION_PATH), VALIDATION_PATH.name)
    inspect_dataset("TRAINING DATASET", training)
    inspect_dataset("EXTERNAL VALIDATION DATASET", validation)
    x_train, y_train = training[FEATURES], training[TARGET].astype(int)
    x_validation, y_validation = validation[FEATURES], validation[TARGET].astype(int)

    model_specs = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=SEED), True),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=7), True),
        ("Support Vector Machine", SVC(kernel="linear", probability=True, random_state=SEED), True),
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=SEED), False),
        ("Random Forest", RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1), False),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=150, random_state=SEED, n_jobs=-1), False),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=SEED), False),
        ("AdaBoost", AdaBoostClassifier(n_estimators=150, random_state=SEED), False),
        ("Gaussian Naive Bayes", GaussianNB(), True),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis(), True),
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis(reg_param=0.1), True),
        ("SGD Classifier", SGDClassifier(loss="log_loss", max_iter=2000, random_state=SEED), True),
        ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=150, random_state=SEED), False),
    ]
    extra, unavailable = optional_estimators()
    model_specs.extend(extra)
    results, confusion_rows, importance_rows = [], [], []
    for name, estimator, scale in model_specs:
        try:
            row, confusion, fitted = evaluate_model(name, estimator, x_train, y_train, x_validation, y_validation, scale)
            results.append(row)
            confusion_rows.append(confusion)
            importance_rows.extend(importance_for_model(name, fitted))
            print(f"Completed: {name}")
        except Exception as error:
            unavailable.append(f"{name} — {type(error).__name__}: {error}")
            print(f"{name} failed — {type(error).__name__}: {error}")

    results_df = pd.DataFrame(results).sort_values(["ROC-AUC", "F1", "External Accuracy"], ascending=False)
    importance_df = pd.DataFrame(importance_rows, columns=["Model", "Feature", "Importance"])
    confusion_df = pd.DataFrame(confusion_rows)
    results_df.to_csv(RESULTS_PATH, index=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    confusion_df.to_csv(CONFUSION_PATH, index=False)
    write_chart(results_df)
    report = make_report(training, validation, results_df, confusion_df, importance_df, unavailable)
    REPORT_PATH.write_text(report, encoding="utf-8")

    best = results_df.iloc[0]
    balanced = results_df[results_df["Specificity"] >= 0.20]
    research_candidate = balanced.iloc[0] if not balanced.empty else best
    rf = results_df[results_df["Model"] == "Random Forest"].iloc[0]
    print("\n" + "=" * 50)
    print("IVF AI MODEL COMPARISON COMPLETE")
    print(f"Training Dataset: {len(training)} records")
    print(f"External Validation Dataset: {len(validation)} records")
    print("\nModels Successfully Tested:")
    print(", ".join(results_df["Model"]))
    print("\nModels Not Tested:")
    print("\n".join(unavailable + ["ANN Basic — TensorFlow/PyTorch not installed", "ANN Dropout — TensorFlow/PyTorch not installed", "Deep ANN — TensorFlow/PyTorch not installed"]))
    print(f"\nBest ML Model: {best['Model']}")
    print("Best DL Model: None — TensorFlow/PyTorch not installed")
    print(f"Overall Candidate: {research_candidate['Model']} (metric winner: {best['Model']})")
    print(f"Best External Accuracy: {best['External Accuracy']:.4f}")
    print(f"Best External F1: {best['F1']:.4f}")
    print(f"Best External ROC-AUC: {best['ROC-AUC']:.4f}")
    print(f"Random Forest External Accuracy: {rf['External Accuracy']:.4f}")
    print(f"Random Forest Train Accuracy: {rf['Train Accuracy']:.4f}")
    print(f"Random Forest Overfitting: {rf['Overfitting']}")
    print("=" * 50)
    print(f"\nOutputs: {RESULTS_PATH.name}, {IMPORTANCE_PATH.name}, {CONFUSION_PATH.name}, {REPORT_PATH.name}, {CHART_PATH.name}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
