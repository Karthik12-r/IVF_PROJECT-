"""Reproducible, offline algorithm comparison for the IVF research prototype.

Uses the project's existing 5,000-row synthetic training dataset. It does not
import the Flask app, retrain or replace its saved model, or use patient data.
Run from the project root: python results/algorithm_comparison/run_comparison.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "ivf_5000_dataset.csv"
TARGET = "IVF_Success"
FEATURES = ["age", "bmi", "amh", "fsh", "previous_ivf", "stress", "sleep_hours", "exercise_min"]
SEED = 42


def build_models():
    specs = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=SEED), True),
        ("Decision Tree", DecisionTreeClassifier(random_state=SEED), False),
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                                   min_samples_leaf=2, random_state=SEED, n_jobs=-1), False),
        ("SVM", SVC(probability=True, random_state=SEED), True),
        ("KNN", KNeighborsClassifier(n_neighbors=5), True),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=SEED), False),
    ]
    result = {}
    for name, estimator, _scale in specs:
        # Use the same imputation and scaling steps for every algorithm.
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        from sklearn.pipeline import Pipeline
        result[name] = Pipeline(steps + [("model", estimator)])
    return result


def markdown_table(frame):
    """Render a small dependency-free Markdown table."""
    headers = list(frame.columns)
    values = [[str(v) if isinstance(v, str) else (f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v))
               for v in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(str(h)), *(len(row[i]) for row in values)) for i, h in enumerate(headers)]
    line = lambda row: "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
    return "\n".join([line([str(h) for h in headers]), line(["-" * w for w in widths]), *map(line, values)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        raise FileNotFoundError(f"Expected existing project dataset: {DATA}")
    raw = pd.read_csv(DATA)
    expected = FEATURES + [TARGET]
    if set(raw.columns) != set(expected):
        raise ValueError(f"Dataset schema mismatch. Expected {expected}; found {list(raw.columns)}")
    data = raw[expected].copy()
    if data[TARGET].isna().any() or data[TARGET].nunique() != 2:
        raise ValueError("Target must be complete and binary for this classification experiment.")
    if not set(pd.unique(data[TARGET])).issubset({0, 1}):
        raise ValueError("IVF_Success must contain binary 0/1 labels.")

    profile = {
        "dataset": str(DATA.relative_to(ROOT)), "rows": len(data), "columns": len(data.columns),
        "feature_columns": FEATURES, "target": TARGET,
        "target_distribution": {str(k): int(v) for k, v in data[TARGET].value_counts().sort_index().items()},
        "missing_values_by_column": {k: int(v) for k, v in data.isna().sum().items()},
        "duplicate_rows": int(data.duplicated().sum()),
        "dtypes": {k: str(v) for k, v in data.dtypes.items()},
        "feature_types": {"numeric": FEATURES, "categorical": []},
        "leakage_review": (
            "IVF_Success is excluded from features. The eight input columns match the existing model metadata; "
            "no explicit target-derived column is present. Inspection of generate_ivf_5000.py shows labels are "
            "synthetically sampled from probabilities assigned by hand-written feature thresholds. Thus the "
            "features are intentionally related to the target, and measured performance reflects recovery of "
            "those synthetic rules, not observed IVF outcomes or clinical validity."
        ),
    }
    (OUT / "dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    X, y = data[FEATURES], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, matrices = [], {}
    for name, pipeline in build_models().items():
        cv_result = cross_validate(
            clone(pipeline), X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1"}, n_jobs=-1,
            return_train_score=True,
        )
        fitted = clone(pipeline).fit(X_train, y_train)
        prediction = fitted.predict(X_test)
        probability = fitted.predict_proba(X_test)[:, 1]
        train_prediction = fitted.predict(X_train)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        train_acc = accuracy_score(y_train, train_prediction)
        test_acc = accuracy_score(y_test, prediction)
        rows.append({
            "Algorithm": name,
            "Accuracy": test_acc,
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1-score": f1_score(y_test, prediction, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probability),
            "Train Accuracy": train_acc,
            "Train F1-score": f1_score(y_train, train_prediction, zero_division=0),
            "Train-Test Accuracy Gap": train_acc - test_acc,
            "CV Accuracy Mean": float(cv_result["test_accuracy"].mean()),
            "CV Accuracy Std": float(cv_result["test_accuracy"].std(ddof=1)),
            "CV F1 Mean": float(cv_result["test_f1"].mean()),
            "CV F1 Std": float(cv_result["test_f1"].std(ddof=1)),
            "CV Train Accuracy Mean": float(cv_result["train_accuracy"].mean()),
            "Confusion Matrix TN,FP,FN,TP": f"{tn},{fp},{fn},{tp}",
        })
        matrices[name] = (fitted, confusion_matrix(y_test, prediction, labels=[0, 1]))

    results = pd.DataFrame(rows).sort_values("Algorithm").reset_index(drop=True)
    results.to_csv(OUT / "comparison_results.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    for metric, filename, title in [
        ("Accuracy", "accuracy_comparison.png", "Held-out Accuracy by Algorithm"),
        ("Precision", "precision_comparison.png", "Held-out Precision by Algorithm"),
        ("Recall", "recall_comparison.png", "Held-out Recall by Algorithm"),
        ("F1-score", "f1_comparison.png", "Held-out F1-score by Algorithm"),
        ("ROC-AUC", "roc_auc_comparison.png", "Held-out ROC-AUC by Algorithm"),
    ]:
        ax = results.set_index("Algorithm")[metric].sort_values().plot.barh(figsize=(9, 5), color="#4385be")
        ax.set(xlabel=metric, ylabel="Algorithm", title=title, xlim=(0, 1))
        plt.tight_layout(); plt.savefig(OUT / filename, dpi=160); plt.close()
    ax = results.set_index("Algorithm")[["CV Accuracy Mean", "CV F1 Mean"]].plot.bar(
        figsize=(10, 5), ylim=(0, 1), color=["#4385be", "#e09f3e"]
    )
    ax.set(ylabel="Score", title="5-fold Cross-validation Performance", xlabel="Algorithm")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout(); plt.savefig(OUT / "cv_comparison.png", dpi=160); plt.close()
    for name, (_, matrix) in matrices.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(matrix, display_labels=["Failure (0)", "Success (1)"]).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d"
        )
        ax.set_title(f"{name}: held-out confusion matrix")
        plt.tight_layout(); plt.savefig(OUT / f"confusion_matrix_{name.lower().replace(' ', '_').replace('-', '_')}.png", dpi=160); plt.close()

    existing = None
    model_path = ROOT / "models" / "ivf_success_model.pkl"
    existing_note = "Existing application model: RandomForestClassifier according to models/ivf_model_metadata.json; main.py loads models/ivf_success_model.pkl."
    try:
        import joblib
        saved = joblib.load(model_path)
        if hasattr(saved, "n_features_in_") and saved.n_features_in_ == len(FEATURES):
            pred = saved.predict(X_test)
            prob = saved.predict_proba(X_test)[:, 1] if hasattr(saved, "predict_proba") else None
            existing = {
                "Algorithm": "Existing saved application model",
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1-score": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, prob) if prob is not None else np.nan,
            }
            existing_note += " Its test metrics are descriptive only because its original training records are not documented; overlap with this dataset may make them optimistic."
        else:
            existing_note += " The saved artifact's input width does not match the eight-column comparison dataset, so it was not scored here."
    except Exception as exc:
        existing_note += f" The saved artifact could not be evaluated: {type(exc).__name__}: {exc}."
    if existing is not None:
        pd.DataFrame([existing]).to_csv(OUT / "existing_model_reference.csv", index=False)

    best = {metric: results.loc[results[metric].idxmax(), ["Algorithm", metric]].to_dict()
            for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]}
    summary = [
        "# Algorithm Comparison / Ablation Study", "",
        "> Research prototype only. The experiment uses synthetic project data; results are not clinical evidence.", "",
        "## Objective", "Compare common classifiers under one reproducible dataset, split, feature set, and training-only preprocessing setup.", "",
        "## Experimental setup", f"- Dataset: `{DATA.relative_to(ROOT)}` ({len(data)} rows, {len(data.columns)} columns).",
        f"- Features: {', '.join(FEATURES)}; target: `{TARGET}` (binary classification).",
        f"- Target counts: {profile['target_distribution']}; missing values: {sum(profile['missing_values_by_column'].values())}; duplicate rows: {profile['duplicate_rows']}.",
        "- Preprocessing: median imputation followed by standard scaling in every model pipeline. Each fold fits preprocessing on its training partition only.",
        f"- Hold-out: stratified 80/20 split, random seed {SEED}; five-fold shuffled stratified CV runs on training data only.",
        "- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, CV mean/std, and train/test accuracy gap.",
        "- Leakage review: target excluded and no target-derived input column identified. However, `generate_ivf_5000.py` creates labels from a hand-written score over the input features and samples them using score-based probabilities. The target is therefore synthetic and structurally related to these features; this is a major validity limitation, though not direct target-column leakage.", "",
        "## Algorithms compared", "Logistic Regression, Decision Tree, Random Forest, SVM, KNN, and Gradient Boosting.", "",
        "## Results", markdown_table(results), "",
        "## Findings", *[f"- Highest {m}: {v['Algorithm']} ({v[m]:.4f})." for m, v in best.items()],
        "- Possible overfit gap (descriptive flag at accuracy gap ≥ 0.10): " +
        (", ".join(f"{r['Algorithm']} ({r['Train-Test Accuracy Gap']:.3f})" for _, r in results[results["Train-Test Accuracy Gap"] >= .10].iterrows()) or "none by this threshold"),
        "- False negatives are shown in each confusion matrix (lower-left cell). Their consequences require clinical context and are not assessed by this experiment.",
        "- CV standard deviations indicate variation across the five folds; they do not replace external validation.", "",
        "## Existing application model", existing_note,
    ]
    if existing is not None:
        summary.extend(["", "Reference metrics (interpret with the training-overlap caveat above):", markdown_table(pd.DataFrame([existing]))])
    summary += ["", "## Conclusion", "These measurements compare experimental predictive performance on the available synthetic dataset. They do not establish clinical validity or support clinical use; external validation on appropriately governed real-world data would be required."]
    (OUT / "report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Dataset {DATA.name}: {len(data)} rows x {len(data.columns)} columns; target={TARGET}; counts={profile['target_distribution']}")
    print(f"Train/test sizes: {len(X_train)}/{len(X_test)}; seed={SEED}; five-fold stratified CV")
    print(results.to_string(index=False, float_format=lambda n: f"{n:.4f}"))
    print(existing_note)
    print(f"Artifacts: {OUT}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
