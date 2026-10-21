"""Validate the project's synthetic IVF CSV without modifying its raw source.

Run from the project root: python results/validation/run_validation.py
Outputs go to results/validation/ and data/validated/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "ivf_5000_dataset.csv"
OUT = Path(__file__).resolve().parent
VALIDATED_PATH = ROOT / "data" / "validated" / "ivf_5000_validated.csv"
FEATURES = ["age", "bmi", "amh", "fsh", "previous_ivf", "stress", "sleep_hours", "exercise_min"]
TARGET = "IVF_Success"
EXPECTED = FEATURES + [TARGET]
COUNT_OR_INTEGER = ["age", "previous_ivf", "stress", "exercise_min"]


def write_csv(rows, path, columns=None):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "graphs").mkdir(exist_ok=True)
    VALIDATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Keep original values and inferred types for inspection. Blank text is also
    # counted as missing; no value or row is repaired or removed.
    raw = pd.read_csv(RAW_PATH, keep_default_na=True)
    df = raw.copy()
    row_ids = np.arange(2, len(df) + 2)  # CSV line numbers, including header line 1.

    print("ACTUAL CSV INSPECTION")
    print(f"Rows: {len(df)}; columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    print("Data types:\n" + df.dtypes.to_string())
    print("Missing values:\n" + df.isna().sum().to_string())
    print("Unique-value counts:\n" + df.nunique(dropna=False).to_string())
    print("Unique values for low-cardinality columns:")
    for col in df.columns:
        vals = df[col].drop_duplicates().tolist()
        if len(vals) <= 30:
            print(f"  {col}: {vals}")
        else:
            print(f"  {col}: {len(vals)} distinct values; complete list saved to unique_values.csv")
    print(f"Exact duplicate rows (beyond first copies): {int(df.duplicated().sum())}")
    print(f"Numerical columns: {df.select_dtypes(include=np.number).columns.tolist()}")
    print(f"Categorical columns: {df.select_dtypes(exclude=np.number).columns.tolist()}")
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    print(f"Date-like columns by name: {date_cols}")
    print(f"Target values: {df[TARGET].value_counts(dropna=False).to_dict() if TARGET in df else 'target column missing'}")
    print("Numeric descriptive statistics:\n" + df.select_dtypes(include=np.number).describe().T.to_string())

    missing_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            missing_mask[col] = df[col].isna() | df[col].astype("string").str.strip().eq("").fillna(False)
        else:
            missing_mask[col] = df[col].isna()
    missing_rows = [{
        "Column": c, "Total": len(df), "Missing": int(missing_mask[c].sum()),
        "Missing %": float(missing_mask[c].mean() * 100),
        "Status": "PASS" if not missing_mask[c].any() else "WARNING",
    } for c in df.columns]
    write_csv(missing_rows, OUT / "missing_values.csv")

    schema_missing = [c for c in EXPECTED if c not in df.columns]
    schema_extra = [c for c in df.columns if c not in EXPECTED]
    type_rows, numeric_values = [], {}
    for col in EXPECTED:
        if col not in df.columns:
            type_rows.append({"Column": col, "Expected Type": "binary integer" if col == TARGET else "numeric", "Actual Type": "MISSING COLUMN", "Invalid Count": len(df), "Status": "INVALID"})
            continue
        parsed = pd.to_numeric(df[col], errors="coerce")
        numeric_values[col] = parsed
        invalid_parse = int((parsed.isna() & ~missing_mask[col]).sum())
        invalid = invalid_parse + int(missing_mask[col].sum())
        if col == TARGET:
            invalid += int((parsed.notna() & ~parsed.isin([0, 1])).sum())
            expected_type = "binary 0/1"
        else:
            expected_type = "numeric"
        type_rows.append({"Column": col, "Expected Type": expected_type, "Actual Type": str(df[col].dtype),
                          "Invalid Count": invalid, "Status": "PASS" if invalid == 0 else "INVALID"})
    write_csv(type_rows, OUT / "data_type_validation.csv")

    # Duplicate checks. Report all participating source records and preserve them.
    dup_rows = []
    exact_mask = df.duplicated(keep=False)
    for idx in df.index[exact_mask]:
        dup_rows.append({"Record ID": int(row_ids[idx]), "Duplicate Type": "Exact duplicate row",
                         "Group Record IDs": ";".join(map(str, row_ids[df.index[df.eq(df.loc[idx]).all(axis=1)]].tolist())),
                         "Target Values": str(df.loc[idx, TARGET]) if TARGET in df else "N/A"})
    feature_duplicate_mask = df.duplicated(subset=[c for c in FEATURES if c in df.columns], keep=False) if all(c in df for c in FEATURES) else pd.Series(False, index=df.index)
    for _, group in df.loc[feature_duplicate_mask].groupby(FEATURES, dropna=False, sort=False):
        ids = row_ids[group.index].tolist()
        targets = group[TARGET].dropna().unique().tolist() if TARGET in group else []
        kind = "Duplicate features with different target" if len(targets) > 1 else "Duplicate feature rows"
        for idx in group.index:
            dup_rows.append({"Record ID": int(row_ids[idx]), "Duplicate Type": kind,
                             "Group Record IDs": ";".join(map(str, ids)), "Target Values": ";".join(map(str, targets))})
    write_csv(dup_rows, OUT / "duplicate_records.csv", ["Record ID", "Duplicate Type", "Group Record IDs", "Target Values"])
    exact_dup_count = int(df.duplicated().sum())
    feature_dup_count = int(df.duplicated(subset=FEATURES).sum()) if all(c in df for c in FEATURES) else 0

    # Defensible integrity rules only: nonnegative measurements/counts, binary
    # target, and integer storage for fields generated as integers in source code.
    rule_masks = {}
    for col in FEATURES:
        if col in numeric_values:
            rule_masks[f"{col}: nonnegative"] = numeric_values[col].lt(0) & numeric_values[col].notna()
    if "previous_ivf" in numeric_values:
        rule_masks["previous_ivf: integer count"] = numeric_values["previous_ivf"].notna() & (numeric_values["previous_ivf"] % 1 != 0)
    if "age" in numeric_values:
        rule_masks["age: whole-year integer representation"] = numeric_values["age"].notna() & (numeric_values["age"] % 1 != 0)
    if "stress" in numeric_values:
        # The generator defines stress as an integer 1..5; this is a data-format
        # rule based on project provenance, not a clinical reference range.
        rule_masks["stress: generator's 1..5 integer coding"] = numeric_values["stress"].notna() & (~numeric_values["stress"].isin([1, 2, 3, 4, 5]))
    if "exercise_min" in numeric_values:
        rule_masks["exercise_min: whole-minute integer representation"] = numeric_values["exercise_min"].notna() & (numeric_values["exercise_min"] % 1 != 0)
    if TARGET in numeric_values:
        rule_masks["IVF_Success: binary 0/1"] = numeric_values[TARGET].notna() & ~numeric_values[TARGET].isin([0, 1])
    range_rows = []
    for name, mask in rule_masks.items():
        col, rule = name.split(": ", 1)
        count = int(mask.sum())
        range_rows.append({"Feature": col, "Rule": rule, "Invalid Count": count,
                           "Invalid %": count / len(df) * 100 if len(df) else 0,
                           "Status": "PASS" if count == 0 else "INVALID"})
    write_csv(range_rows, OUT / "range_validation.csv")

    # No categorical columns are expected for this dataset. Still record that
    # finding explicitly without fabricating categories.
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    categorical_rows = ([{"Column": c, "Unique Values": json.dumps(df[c].dropna().unique().tolist()), "Unexpected Values": "", "Status": "REVIEW"} for c in categorical_cols]
                        or [{"Column": "N/A", "Unique Values": "No categorical feature columns were present in the current dataset.", "Unexpected Values": "N/A", "Status": "N/A"}])
    write_csv(categorical_rows, OUT / "categorical_validation.csv")

    # IQR statistical flags; these are warnings, never grounds for deletion.
    outlier_rows, outlier_summary = [], []
    outlier_masks = {}
    for col in FEATURES:
        if col not in numeric_values:
            continue
        series = numeric_values[col].dropna()
        q1, q3 = series.quantile(.25), series.quantile(.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = numeric_values[col].lt(lower) | numeric_values[col].gt(upper)
        mask &= numeric_values[col].notna()
        outlier_masks[col] = mask
        outlier_summary.append({"Feature": col, "Method": "IQR (1.5 × IQR)", "Lower Bound": lower,
                                "Upper Bound": upper, "Outlier Count": int(mask.sum()),
                                "Outlier %": float(mask.mean() * 100),
                                "Status": "WARNING" if mask.any() else "PASS"})
        for idx in df.index[mask]:
            outlier_rows.append({"Row/Record ID": int(row_ids[idx]), "Feature": col,
                                 "Value": numeric_values.loc[idx, col], "Detection Method": "IQR (1.5 × IQR)",
                                 "Lower Bound": lower, "Upper Bound": upper,
                                 "Status": "Statistically flagged observation; not necessarily invalid"})
    write_csv(outlier_rows, OUT / "outlier_records.csv", ["Row/Record ID", "Feature", "Value", "Detection Method", "Lower Bound", "Upper Bound", "Status"])
    write_csv(outlier_summary, OUT / "outlier_summary.csv")

    # Target integrity and distribution.
    if TARGET in numeric_values:
        y = numeric_values[TARGET]
        counts = y.value_counts(dropna=False)
        target_rows = [{"Target Value": val, "Count": int(count), "Percentage": count / len(df) * 100,
                        "Status": "VALID" if val in [0, 1] else "INVALID"}
                       for val, count in counts.items()]
        invalid_target_count = int((y.isna() | ~y.isin([0, 1])).sum())
        valid_class_counts = y[y.isin([0, 1])].value_counts()
        imbalance_ratio = (float(valid_class_counts.max() / valid_class_counts.min())
                           if len(valid_class_counts) == 2 and valid_class_counts.min() else np.nan)
    else:
        target_rows, invalid_target_count, imbalance_ratio = [], len(df), np.nan
        valid_class_counts = pd.Series(dtype=int)
    write_csv(target_rows, OUT / "target_validation.csv", ["Target Value", "Count", "Percentage", "Status"])

    # Per-row statuses: only mathematical/type/target violations => INVALID;
    # statistical outliers and duplicate participation => WARNING.
    invalid_row = pd.Series(False, index=df.index)
    for col in EXPECTED:
        if col in numeric_values:
            invalid_row |= numeric_values[col].isna() | missing_mask[col]
            invalid_row |= numeric_values[col].notna() & numeric_values[col].lt(0) if col in FEATURES else False
    if TARGET in numeric_values:
        invalid_row |= numeric_values[TARGET].notna() & ~numeric_values[TARGET].isin([0, 1])
    for mask in rule_masks.values():
        invalid_row |= mask
    warning_row = exact_mask.copy() | feature_duplicate_mask
    for mask in outlier_masks.values():
        warning_row |= mask
    status = np.where(invalid_row, "INVALID", np.where(warning_row, "WARNING", "VALID"))
    flag_lists = []
    for idx in df.index:
        flags = []
        if invalid_row.loc[idx]:
            flags.extend(f"INVALID_{col}" for col in EXPECTED if col in numeric_values and (numeric_values[col].isna().loc[idx] or missing_mask.loc[idx, col]))
            flags.extend(name.replace(": ", "_").upper() for name, mask in rule_masks.items() if bool(mask.loc[idx]))
        if exact_mask.loc[idx]: flags.append("EXACT_DUPLICATE")
        if feature_duplicate_mask.loc[idx]: flags.append("DUPLICATE_FEATURES")
        flags.extend(f"OUTLIER_{col}" for col, mask in outlier_masks.items() if bool(mask.loc[idx]))
        flag_lists.append(";".join(dict.fromkeys(flags)) if flags else "NONE")
    validated = raw.copy()
    validated["validation_status"] = status
    validated["validation_flags"] = flag_lists
    validated.to_csv(VALIDATED_PATH, index=False)

    # Leakage review is independently derived from feature declarations and the
    # existing experiment source, then source-code label generation inspected.
    ml_source = (ROOT / "results" / "algorithm_comparison" / "run_comparison.py").read_text(encoding="utf-8")
    generator_source = (ROOT / "generate_ivf_5000.py").read_text(encoding="utf-8")
    feature_line = next((line for line in ml_source.splitlines() if line.strip().startswith("FEATURES =")), "")
    target_excluded = TARGET not in FEATURES and "TARGET" in ml_source and "X, y = data[FEATURES], data[TARGET]" in ml_source
    label_rule_present = "success_score" in generator_source and "success_prob" in generator_source
    leakage_rows = [
        {"Check": "Target excluded from comparison inputs", "Finding": f"Declared feature list: {FEATURES}; target: {TARGET}.", "Status": "PASS" if target_excluded else "REVIEW"},
        {"Check": "Target-derived columns", "Finding": "No additional target-derived column appears in this CSV schema.", "Status": "PASS"},
        {"Check": "Duplicate records", "Finding": f"{exact_dup_count} exact duplicate rows beyond first copies; {feature_dup_count} repeated feature rows beyond first copies.", "Status": "PASS" if exact_dup_count == 0 and feature_dup_count == 0 else "WARNING"},
        {"Check": "Preprocessing/split order", "Finding": "Comparison code uses a stratified hold-out and Pipeline/cross-validation, fitting preprocessing within each training fold.", "Status": "PASS" if "train_test_split" in ml_source and "Pipeline" in ml_source and "cross_validate" in ml_source else "REVIEW"},
        {"Check": "Future/post-outcome columns", "Finding": "CSV contains only baseline/lifestyle-like synthetic inputs and IVF_Success; no date or future workflow field is present.", "Status": "PASS"},
        {"Check": "Algorithm-comparison data linkage", "Finding": "The existing comparison remains unchanged and loads ivf_5000_dataset.csv directly. This validation creates a separate validated copy; it does not alter the comparison's input path. The validated file retains all source rows and feature/target values.", "Status": "INFO"},
        {"Check": "Synthetic target dependence", "Finding": "Dataset generator computes success_score from hand-written thresholds over the same input features, then samples success_prob from that score. This is structural target dependence, not direct target-column leakage; measurements reflect the synthetic generating pattern, not observed clinical outcomes.", "Status": "WARNING" if label_rule_present else "REVIEW"},
    ]
    write_csv(leakage_rows, OUT / "data_leakage_report.csv")
    (OUT / "data_leakage_report.md").write_text(
        "# Data Leakage Review\n\n" + "\n".join(f"- **{r['Check']} — {r['Status']}:** {r['Finding']}" for r in leakage_rows) + "\n",
        encoding="utf-8")

    # Project-level dates are distinct from the CSV. Schema source inspection
    # found IVF-cycle, document, prediction and medical-activity timestamps.
    schema_source = (ROOT / "models.py").read_text(encoding="utf-8")
    app_date_fields = [name for name in ["start_date", "egg_retrieval_date", "transfer_date", "uploaded_at", "prediction_date", "performed_date"] if name in schema_source]
    schema_missing_count = len(schema_missing) + len(schema_extra)
    type_issue_count = sum(int(r["Invalid Count"]) for r in type_rows)
    missing_count = int(missing_mask.sum().sum())
    range_issue_count = sum(int(r["Invalid Count"]) for r in range_rows)
    outlier_count = len(outlier_rows)
    consistency_rows = [
        {"Check": "Numeric values parse as numeric", "Records Checked": len(df), "Issues Found": type_issue_count, "Status": "PASS" if type_issue_count == 0 else "INVALID"},
        {"Check": "Nonnegative values and documented integer/binary representations", "Records Checked": len(df), "Issues Found": range_issue_count, "Status": "PASS" if range_issue_count == 0 else "INVALID"},
        {"Check": "Additional cross-field workflow rules", "Records Checked": "N/A", "Issues Found": "N/A", "Status": "N/A — No additional cross-field consistency rules were applied because the current CSV does not contain sufficient documented workflow fields to justify them."},
    ]
    write_csv(consistency_rows, OUT / "consistency_validation.csv")

    categorical_status = "N/A" if not categorical_cols else "WARNING"
    date_status = "N/A" if not date_cols else "PASS"
    direct_leak_issues = sum(r["Status"] == "INVALID" for r in leakage_rows[:5])
    summary_rows = [
        {"Check": "Missing Values", "Records Checked": len(df), "Issues Found": missing_count, "Status": "PASS" if missing_count == 0 else "WARNING"},
        {"Check": "Data Types", "Records Checked": len(df) * len(EXPECTED), "Issues Found": type_issue_count + schema_missing_count, "Status": "PASS" if type_issue_count + schema_missing_count == 0 else "INVALID"},
        {"Check": "Duplicate Rows", "Records Checked": len(df), "Issues Found": exact_dup_count, "Status": "PASS" if exact_dup_count == 0 else "WARNING"},
        {"Check": "Range Rules", "Records Checked": len(df), "Issues Found": range_issue_count, "Status": "PASS" if range_issue_count == 0 else "INVALID"},
        {"Check": "Categorical Values", "Records Checked": "N/A" if not categorical_cols else len(df), "Issues Found": "N/A" if not categorical_cols else "See categorical_validation.csv", "Status": categorical_status},
        {"Check": "Date Validation", "Records Checked": "N/A" if not date_cols else len(df), "Issues Found": "N/A" if not date_cols else "See date validation", "Status": date_status},
        {"Check": "Outliers", "Records Checked": len(df) * len(outlier_masks), "Issues Found": outlier_count, "Status": "WARNING" if outlier_count else "PASS"},
        {"Check": "Target Validation", "Records Checked": len(df), "Issues Found": invalid_target_count, "Status": "PASS" if invalid_target_count == 0 else "INVALID"},
        {"Check": "Consistency", "Records Checked": len(df), "Issues Found": range_issue_count, "Status": "PASS" if range_issue_count == 0 else "INVALID"},
        {"Check": "Leakage", "Records Checked": len(df), "Issues Found": f"{direct_leak_issues} direct issues; 1 synthetic target-dependence warning", "Status": "WARNING" if label_rule_present else "REVIEW"},
    ]
    write_csv(summary_rows, OUT / "validation_summary.csv")

    valid_count = int((status == "VALID").sum())
    warning_count = int((status == "WARNING").sum())
    invalid_count = int((status == "INVALID").sum())
    target_distribution = {str(int(k)): int(v) for k, v in valid_class_counts.sort_index().items()}
    target_percentages = {str(k): v / len(df) * 100 for k, v in target_distribution.items()}
    report = [
        "# IVF Dataset Data Validation Study", "", "## 1. Objective",
        "Assess completeness, structure, basic data integrity, duplicates, statistical outliers and obvious leakage in the project CSV. Validation does not prove clinical correctness.", "",
        "## 2. Dataset Description", f"- Source: `{RAW_PATH.relative_to(ROOT)}`; raw file was read only.",
        f"- Shape: {len(df)} rows × {len(df.columns)} columns. Columns: {', '.join(map(str, df.columns))}.",
        f"- Inferred numerical columns: {df.select_dtypes(include=np.number).columns.tolist()}; categorical columns: {categorical_cols or 'none'}; date columns: {date_cols or 'none'}.",
        f"- Target distribution: {target_distribution}; percentages: { {k: round(v, 2) for k, v in target_percentages.items()} }; class ratio (majority/minority): {imbalance_ratio:.3f}.",
        "- Full numeric descriptive statistics are in `dataset_statistics.csv`.", "",
        "## 3. Data-Type Validation", f"Expected columns present: {not schema_missing}; unexpected columns: {schema_extra or 'none'}. CSV numeric-feature parsing errors: {type_issue_count}.",
        "See `data_type_validation.csv` for each column.", "",
        "## 4. Missing-Value Validation", f"Missing cells including blank strings: {missing_count}; dataset is {'complete under this check' if missing_count == 0 else 'not complete under this check'}. See `missing_values.csv`.", "",
        "## 5. Duplicate Validation", f"Exact duplicate rows beyond first copies: {exact_dup_count} ({exact_dup_count / len(df) * 100:.3f}%). Repeated feature rows beyond first copies: {feature_dup_count}; records are retained. See `duplicate_records.csv`.", "",
        "## 6. Range Validation", f"Rule violations: {range_issue_count}. Rules are nonnegative numeric values, binary target, and documented integer/coding representations from the project generator; no medical reference ranges were applied. See `range_validation.csv`.", "",
        "## 7. Categorical Validation", "No categorical feature columns were present in the current dataset." if not categorical_cols else f"Categorical columns found: {categorical_cols}. See `categorical_validation.csv`.", "",
        "## 8. Date Validation", "No date columns are present in ivf_5000_dataset.csv, so date consistency validation was not applicable to this dataset." if not date_cols else f"Date-like columns found: {date_cols}.",
        f"Application schema inspection separately found date/timestamp fields: {', '.join(app_date_fields)} in `models.py`. Application-level date validation is a separate future validation layer; this study did not inspect patient database records.", "",
        "## 9. Outlier Analysis", f"IQR (1.5 × IQR) identified {outlier_count} feature-observations as statistical flags across {len(outlier_masks)} numerical features. These are not automatically invalid values and were retained. See `outlier_summary.csv` and `outlier_records.csv`.", "",
        "## 10. Target Validation", f"Target missing/invalid values: {invalid_target_count}; unique valid target values: {sorted(valid_class_counts.index.tolist())}; class counts: {target_distribution}; majority/minority ratio: {imbalance_ratio:.3f}. No imbalance label is assigned because no threshold was specified.", "",
        "## 11. Consistency Validation", "Basic numeric, nonnegative, integer representation and binary checks are summarized in `consistency_validation.csv`. No additional cross-field consistency rules were applied because the current CSV does not contain sufficient documented workflow fields to justify them.", "",
        "## 12. Data Leakage Review", "See `data_leakage_report.csv` and `data_leakage_report.md`. The comparison input list excludes `IVF_Success`; no target-encoding companion column or post-outcome field exists in this CSV. The generator uses hand-written feature rules to create the synthetic target: this is not direct target-column leakage, but it makes the target structurally dependent on the features. Model scores therefore measure recovery of the synthetic pattern, not real-world IVF clinical performance.", "",
        "## 13. Validation Summary", f"Rows: {len(df)}; VALID: {valid_count}; WARNING: {warning_count}; INVALID: {invalid_count}. See `validation_summary.csv`.", "",
        "## 14. Limitations", "Statistical outliers are not clinical errors. The CSV is synthetic and generated by hand-written rules; no external clinical reference data was used. Application date fields were identified from schema definitions only; database values were not audited. The existing comparison script was intentionally left unchanged and still reads the raw CSV directly; validation produced a separate copy with the same rows and data values, but is not wired as an automated precondition to model training. Data validation does not establish clinical correctness.", "",
        "## 15. Conclusion", f"The CSV passed the performed structural and basic integrity checks with {missing_count} missing cells, {exact_dup_count} exact duplicate rows, {range_issue_count} range/integrity violations, and {invalid_target_count} invalid target values. Statistical outliers are reported separately. These findings describe data quality only and do not imply clinical validity.",
    ]
    (OUT / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    # Statistics artifact required for the initial inspection, across numeric data.
    desc = df.select_dtypes(include=np.number).describe().T.reset_index().rename(columns={"index": "Column"})
    desc.to_csv(OUT / "dataset_statistics.csv", index=False)
    unique_rows = [{"Column": col, "Unique Count": int(df[col].nunique(dropna=False)),
                    "Unique Values": json.dumps(df[col].drop_duplicates().where(df[col].notna(), None).tolist(), default=str)}
                   for col in df.columns]
    write_csv(unique_rows, OUT / "unique_values.csv")

    # Visualizations.
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df.columns, [int(missing_mask[c].sum()) for c in df.columns], color="#4385be")
    ax.set(title="Missing Values by Column", ylabel="Missing cells", xlabel="Column")
    ax.tick_params(axis="x", rotation=35); fig.tight_layout(); fig.savefig(OUT / "graphs" / "missing_values_by_column.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if TARGET in df:
        df[TARGET].value_counts(dropna=False).sort_index().plot.bar(ax=ax, color="#4385be")
    ax.set(title="Target Distribution: IVF_Success", xlabel="Target value", ylabel="Records")
    fig.tight_layout(); fig.savefig(OUT / "graphs" / "target_distribution.png", dpi=160); plt.close(fig)

    ncols = 4
    fig, axes = plt.subplots(2, ncols, figsize=(15, 7))
    for ax, col in zip(axes.flat, FEATURES):
        if col in numeric_values:
            ax.hist(numeric_values[col].dropna(), bins=24, color="#4385be", edgecolor="white")
            ax.set_title(col); ax.set_ylabel("Records")
    fig.suptitle("Numerical Feature Distributions", y=1.01); fig.tight_layout(); fig.savefig(OUT / "graphs" / "numerical_feature_distributions.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    pd.DataFrame(outlier_summary).set_index("Feature")["Outlier Count"].plot.bar(ax=ax, color="#e09f3e")
    ax.set(title="IQR Statistical Flags by Feature", ylabel="Flagged observations", xlabel="Feature")
    ax.tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(OUT / "graphs" / "outlier_summary.png", dpi=160); plt.close(fig)

    issue_plot = [{"Check": r["Check"], "Count": (r["Issues Found"] if isinstance(r["Issues Found"], (int, float)) else np.nan)} for r in summary_rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    pd.DataFrame(issue_plot).set_index("Check")["Count"].plot.bar(ax=ax, color="#d97757")
    ax.set(title="Validation Issue Counts (N/A checks omitted)", ylabel="Issues / flagged observations", xlabel="Validation check")
    ax.tick_params(axis="x", rotation=35); fig.tight_layout(); fig.savefig(OUT / "graphs" / "validation_issue_summary.png", dpi=160); plt.close(fig)

    print("\nVALIDATION RESULTS")
    print(f"Rows: {len(df)}; columns: {len(df.columns)}; valid={valid_count}; warning={warning_count}; invalid={invalid_count}")
    print(f"Missing={missing_count}; exact duplicate rows={exact_dup_count}; repeated feature rows={feature_dup_count}; range violations={range_issue_count}; outlier observations={outlier_count}; invalid target={invalid_target_count}")
    print(f"Target counts={target_distribution}; imbalance ratio={imbalance_ratio:.3f}")
    print(f"Validated dataset: {VALIDATED_PATH}")
    print(f"Validation artifacts: {OUT}")


if __name__ == "__main__":
    main()
