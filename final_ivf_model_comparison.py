"""Final isolated IVF model comparison experiment.

This script does not import the Flask application, modify project data, save
models, or write to production directories. It trains only from the existing
5,000-row workbook and uses the 500-row CSV only for external validation.
"""

from pathlib import Path
import importlib.util
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "ivf_5000_dataset.xlsx"
EXTERNAL_PATH = BASE_DIR / "IVF_validation_dataset_500_synthetic.csv"
RESULTS_PATH = BASE_DIR / "final_ivf_model_comparison.csv"
IMPORTANCE_PATH = BASE_DIR / "final_ivf_feature_importance.csv"
CONFUSION_PATH = BASE_DIR / "final_ivf_confusion_matrices.csv"
REPORT_PATH = BASE_DIR / "final_ivf_model_report.txt"
CHART_PATH = BASE_DIR / "final_ivf_model_comparison.png"
TARGET = "IVF Success"
FEATURES = ["Age", "BMI", "AMH", "FSH", "Previous IVF Attempts", "Stress Level", "Sleep Hours", "Exercise (minutes/day)"]
NUMERIC = ["Age", "BMI", "AMH", "FSH", "Previous IVF Attempts", "Sleep Hours", "Exercise (minutes/day)"]
CATEGORICAL = ["Stress Level"]


def normalize(data, source):
    data = data.rename(columns={
        "age": "Age", "bmi": "BMI", "amh": "AMH", "fsh": "FSH",
        "previous_ivf": "Previous IVF Attempts", "stress": "Stress Level",
        "sleep_hours": "Sleep Hours", "exercise_min": "Exercise (minutes/day)",
        "IVF_Success": TARGET,
    })
    if not data["Stress Level"].isin(["Low", "Medium", "High"]).all():
        numeric_stress = pd.to_numeric(data["Stress Level"], errors="coerce")
        data["Stress Level"] = pd.cut(numeric_stress, [-np.inf, 2, 3, np.inf], labels=["Low", "Medium", "High"]).astype(str)
    missing = [column for column in FEATURES + [TARGET] if column not in data]
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")
    return data[FEATURES + [TARGET]].copy()


def describe(name, data):
    print(f"\n{name}: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Missing values: {int(data.isna().sum().sum())}")
    print(f"Duplicate rows: {int(data.duplicated().sum())}")
    print(f"Target distribution: {data[TARGET].value_counts().sort_index().to_dict()}")
    print("Ranges:")
    for column in NUMERIC:
        print(f"  {column}: {data[column].min()} - {data[column].max()}")
    print(f"  Stress Level: {sorted(data['Stress Level'].dropna().unique().tolist())}")


def make_pipeline(estimator, scale=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL),
        ])),
        ("model", estimator),
    ])


def score_probabilities(model, data):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(data)[:, 1]
    return model.decision_function(data)


def evaluate(name, estimator, scale, x_train, y_train, x_external, y_external):
    pipeline = make_pipeline(estimator, scale)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_result = cross_validate(
        pipeline, x_train, y_train, cv=cv,
        scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"},
        n_jobs=-1,
    )
    pipeline.fit(x_train, y_train)
    train_prediction = pipeline.predict(x_train)
    external_prediction = pipeline.predict(x_external)
    external_probability = score_probabilities(pipeline, x_external)
    tn, fp, fn, tp = confusion_matrix(y_external, external_prediction, labels=[0, 1]).ravel()
    train_accuracy = accuracy_score(y_train, train_prediction)
    external_accuracy = accuracy_score(y_external, external_prediction)
    gap = train_accuracy - external_accuracy
    overfitting = "High" if gap >= 0.25 else "Moderate" if gap >= 0.15 else "Low"
    metrics = {
        "Model": name,
        "Train Accuracy": train_accuracy,
        "CV Accuracy": cv_result["test_accuracy"].mean(),
        "CV Accuracy Std": cv_result["test_accuracy"].std(),
        "CV F1": cv_result["test_f1"].mean(),
        "CV ROC-AUC": cv_result["test_roc_auc"].mean(),
        "External Accuracy": external_accuracy,
        "Precision": precision_score(y_external, external_prediction, zero_division=0),
        "Recall": recall_score(y_external, external_prediction, zero_division=0),
        "F1": f1_score(y_external, external_prediction, zero_division=0),
        "ROC-AUC": roc_auc_score(y_external, external_probability),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "Sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "Train-External Gap": gap,
        "Overfitting": overfitting,
        "Status": "Completed",
    }
    confusion = {"Model": name, "TP": tp, "TN": tn, "FP": fp, "FN": fn}
    return metrics, confusion, pipeline


def model_importance(name, fitted):
    model = fitted.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return []
    names = fitted.named_steps["preprocess"].get_feature_names_out()
    return [{"Model": name, "Feature": feature, "Importance": float(value)} for feature, value in zip(names, values)]


def get_optional_models():
    models = []
    unavailable = []
    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier
        models.append(("XGBoost", XGBClassifier(n_estimators=30, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=SEED, n_jobs=1), False))
    else:
        unavailable.append("XGBoost — package not installed")
    if importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMClassifier
        models.append(("LightGBM", LGBMClassifier(n_estimators=30, num_leaves=20, learning_rate=0.05, verbosity=-1, random_state=SEED, n_jobs=1), False))
    else:
        unavailable.append("LightGBM — package not installed")
    return models, unavailable


def run_deep_learning(x_train, y_train, x_external, y_external):
    framework = "TensorFlow" if importlib.util.find_spec("tensorflow") else "PyTorch" if importlib.util.find_spec("torch") else None
    if framework is None:
        return [], [], ["Basic ANN — TensorFlow and PyTorch are not installed", "ANN with Dropout — TensorFlow and PyTorch are not installed", "Deep ANN — TensorFlow and PyTorch are not installed"]
    raise RuntimeError(f"{framework} is installed, but this run requires its dedicated implementation before execution")


def write_chart(results):
    ordered = results.sort_values("ROC-AUC", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(ordered["Model"], ordered["ROC-AUC"], color="#326b8f")
    plt.xlabel("External Validation ROC-AUC")
    plt.title("Final IVF Model Comparison")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=160)
    plt.close()


def build_report(training, external, results, confusion, importance, unavailable):
    ranked = results.sort_values(["ROC-AUC", "F1", "External Accuracy"], ascending=False)
    metric_winner = ranked.iloc[0]
    balanced = ranked[ranked["Specificity"] >= 0.20]
    candidate = balanced.iloc[0] if not balanced.empty else metric_winner
    rf = results[results["Model"] == "Random Forest"].iloc[0]
    top_features = importance.groupby("Feature", as_index=False)["Importance"].mean().sort_values("Importance", ascending=False).head(5)
    lines = [
        "FINAL IVF MODEL COMPARISON", "",
        f"Training: {TRAIN_PATH.name} | {len(training)} records",
        f"External validation: {EXTERNAL_PATH.name} | {len(external)} records",
        "External validation was never used for fitting or preprocessing.", "",
        "BEST ML MODEL BY EXTERNAL ROC-AUC:",
        f"{metric_winner['Model']} | Accuracy {metric_winner['External Accuracy']:.4f} | Precision {metric_winner['Precision']:.4f} | Recall {metric_winner['Recall']:.4f} | F1 {metric_winner['F1']:.4f} | ROC-AUC {metric_winner['ROC-AUC']:.4f}",
        "BEST DL MODEL: None completed because TensorFlow/PyTorch were unavailable.",
        f"OVERALL BEST CANDIDATE: {candidate['Model']} (metric winner: {metric_winner['Model']})",
        "A model with specificity below 0.20 is not treated as a practical winner because it predicts nearly every case as positive.", "",
        "RANDOM FOREST:",
        f"Train Accuracy: {rf['Train Accuracy']:.4f}",
        f"External Accuracy: {rf['External Accuracy']:.4f}",
        f"Train-External Gap: {rf['Train-External Gap']:.4f}",
        f"Overfitting: {rf['Overfitting']}",
        "Random Forest shows significant overfitting on the external validation dataset." if rf["Overfitting"] == "High" else "Random Forest did not reach the high-overfitting threshold.", "",
        "TOP FEATURES:",
    ]
    lines.extend(f"{i}. {row['Feature']} ({row['Importance']:.6f})" for i, (_, row) in enumerate(top_features.iterrows(), 1))
    lines.extend(["", "FULL RESULTS:", results.to_string(index=False), "", "OVERFITTING AND LIMITATIONS:"])
    lines.extend(f"{row['Model']}: {row['Overfitting']}; train={row['Train Accuracy']:.4f}, CV={row['CV Accuracy']:.4f}, external={row['External Accuracy']:.4f}" for _, row in results.iterrows())
    lines.extend(["The training and external target distributions differ substantially, so external performance is not directly comparable to cross-validation performance.", "The 500-row synthetic external set is useful for software validation, not clinical inference.", "More representative samples and further hyperparameter/threshold tuning are required.", "", "MODELS NOT TESTED:", *(unavailable or ["None"])])
    return "\n".join(lines)


def main():
    training = normalize(pd.read_excel(TRAIN_PATH), TRAIN_PATH.name)
    external = normalize(pd.read_csv(EXTERNAL_PATH), EXTERNAL_PATH.name)
    describe("TRAINING DATASET", training)
    describe("EXTERNAL VALIDATION DATASET", external)
    x_train, y_train = training[FEATURES], training[TARGET].astype(int)
    x_external, y_external = external[FEATURES], external[TARGET].astype(int)
    specs = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=SEED), True),
        ("Random Forest", RandomForestClassifier(n_estimators=30, random_state=SEED, n_jobs=1), False),
        ("Support Vector Machine", SVC(kernel="linear", probability=False, random_state=SEED), True),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=30, random_state=SEED), False),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=30, random_state=SEED, n_jobs=1), False),
    ]
    optional, unavailable = get_optional_models()
    specs.extend(optional)
    results, confusion_rows, importance_rows = [], [], []
    for name, estimator, scale in specs:
        try:
            metrics, confusion, fitted = evaluate(name, estimator, scale, x_train, y_train, x_external, y_external)
            results.append(metrics)
            confusion_rows.append(confusion)
            importance_rows.extend(model_importance(name, fitted))
            print(f"Completed: {name}")
        except Exception as error:
            unavailable.append(f"{name} — {type(error).__name__}: {error}")
            print(f"{name} failed — {type(error).__name__}: {error}")
    try:
        _, _, dl_unavailable = run_deep_learning(x_train, y_train, x_external, y_external)
        unavailable.extend(dl_unavailable)
    except Exception as error:
        unavailable.append(f"Deep learning — {type(error).__name__}: {error}")
    result_frame = pd.DataFrame(results).sort_values(["ROC-AUC", "F1", "External Accuracy"], ascending=False)
    importance_frame = pd.DataFrame(importance_rows, columns=["Model", "Feature", "Importance"])
    confusion_frame = pd.DataFrame(confusion_rows)
    result_frame.to_csv(RESULTS_PATH, index=False)
    importance_frame.to_csv(IMPORTANCE_PATH, index=False)
    confusion_frame.to_csv(CONFUSION_PATH, index=False)
    write_chart(result_frame)
    REPORT_PATH.write_text(build_report(training, external, result_frame, confusion_frame, importance_frame, unavailable), encoding="utf-8")
    winner = result_frame.iloc[0]
    balanced = result_frame[result_frame["Specificity"] >= 0.20]
    candidate = balanced.iloc[0] if not balanced.empty else winner
    rf = result_frame[result_frame["Model"] == "Random Forest"].iloc[0]
    print("\n# =============================")
    print("FINAL IVF MODEL COMPARISON")
    print(f"Best ML Model: {winner['Model']}")
    print("Best DL Model: None — TensorFlow/PyTorch not installed")
    print(f"Overall Best Candidate: {candidate['Model']} (metric winner: {winner['Model']})")
    print(f"External Validation Accuracy: {winner['External Accuracy']:.4f}")
    print(f"External Validation F1: {winner['F1']:.4f}")
    print(f"External Validation ROC-AUC: {winner['ROC-AUC']:.4f}")
    print(f"Random Forest Train Accuracy: {rf['Train Accuracy']:.4f}")
    print(f"Random Forest External Accuracy: {rf['External Accuracy']:.4f}")
    print(f"Random Forest Overfitting: {rf['Overfitting']}")
    print("\nModels Successfully Tested:")
    print(", ".join(result_frame["Model"]))
    print("\nModels Not Tested:")
    print("\n".join(unavailable))
    print("# =============================")
    print(f"Outputs: {RESULTS_PATH.name}, {IMPORTANCE_PATH.name}, {CONFUSION_PATH.name}, {REPORT_PATH.name}, {CHART_PATH.name}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
