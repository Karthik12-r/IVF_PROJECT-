"""Generate and validate a reproducible synthetic IVF validation dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
ROW_COUNT = 500
OUTPUT_FILE = Path(__file__).with_name("IVF_validation_dataset_500_synthetic.csv")
EXPECTED_COLUMNS = [
    "Age",
    "BMI",
    "AMH",
    "FSH",
    "Previous IVF Attempts",
    "Stress Level",
    "Sleep Hours",
    "Exercise (minutes/day)",
    "IVF Success",
]
STRESS_LEVELS = {"Low", "Medium", "High"}


def generate_dataset(row_count=ROW_COUNT, seed=RANDOM_SEED):
    """Generate synthetic records with noisy, non-deterministic success probabilities."""
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 46, row_count)
    bmi = np.round(rng.uniform(18.0, 35.0, row_count), 1)
    amh = np.round(rng.uniform(0.3, 8.0, row_count), 2)
    fsh = np.round(rng.uniform(2.0, 15.0, row_count), 2)
    previous_attempts = rng.integers(0, 6, row_count)
    stress = rng.choice(["Low", "Medium", "High"], row_count, p=[0.35, 0.45, 0.20])
    sleep_hours = np.round(rng.uniform(4.0, 9.0, row_count), 1)
    exercise = rng.integers(0, 121, row_count)

    bmi_score = -0.04 * np.abs(bmi - 22.5)
    stress_score = np.select(
        [stress == "Low", stress == "Medium", stress == "High"],
        [0.35, 0.0, -0.35],
    )
    sleep_score = 0.16 * np.clip(sleep_hours - 6.5, -2.5, 2.5)
    exercise_score = 0.18 * np.minimum(exercise, 60) / 20
    logit = (
        -0.20
        - 0.075 * (age - 32)
        + 0.16 * (amh - 2.5)
        - 0.12 * (fsh - 7.0)
        - 0.18 * previous_attempts
        + bmi_score
        + stress_score
        + sleep_score
        + exercise_score
        + rng.normal(0.0, 0.45, row_count)
    )
    success_probability = 1.0 / (1.0 + np.exp(-logit))
    success = (rng.random(row_count) < success_probability).astype(int)

    return pd.DataFrame(
        {
            "Age": age,
            "BMI": bmi,
            "AMH": amh,
            "FSH": fsh,
            "Previous IVF Attempts": previous_attempts,
            "Stress Level": stress,
            "Sleep Hours": sleep_hours,
            "Exercise (minutes/day)": exercise,
            "IVF Success": success,
        }
    )


def validate_dataset(df):
    """Raise an error if the generated dataset violates any requested constraint."""
    if len(df) != ROW_COUNT:
        raise ValueError(f"Expected {ROW_COUNT} rows, found {len(df)}")
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if df.isna().sum().sum() != 0:
        raise ValueError("Dataset contains missing values")
    if df.duplicated().sum() != 0:
        raise ValueError("Dataset contains duplicate rows")

    ranges = {
        "Age": (21, 45),
        "BMI": (18.0, 35.0),
        "AMH": (0.3, 8.0),
        "FSH": (2.0, 15.0),
        "Previous IVF Attempts": (0, 5),
        "Sleep Hours": (4.0, 9.0),
        "Exercise (minutes/day)": (0, 120),
    }
    for column, (minimum, maximum) in ranges.items():
        if not df[column].between(minimum, maximum).all():
            raise ValueError(f"{column} contains values outside {minimum}-{maximum}")
    if not set(df["Stress Level"]).issubset(STRESS_LEVELS):
        raise ValueError("Stress Level contains an invalid category")
    if not set(df["IVF Success"]).issubset({0, 1}):
        raise ValueError("IVF Success must contain only 0 or 1")


def print_summary(df):
    """Print the requested validation and numerical range summary."""
    success_count = int(df["IVF Success"].sum())
    failure_count = ROW_COUNT - success_count
    print("\nValidation summary:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Success count: {success_count}")
    print(f"Failure count: {failure_count}")
    print(f"Success percentage: {df['IVF Success'].mean() * 100:.1f}%")
    print(f"Missing-value count: {int(df.isna().sum().sum())}")
    print(f"Duplicate-row count: {int(df.duplicated().sum())}")
    print("Numerical ranges:")
    for column in EXPECTED_COLUMNS:
        if pd.api.types.is_numeric_dtype(df[column]):
            print(f"  {column}: {df[column].min()} - {df[column].max()}")


def main():
    dataset = generate_dataset()
    validate_dataset(dataset)
    dataset.to_csv(OUTPUT_FILE, index=False)

    print("Dataset generated successfully!")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"File: {OUTPUT_FILE.name}")
    print("\nFirst 10 rows:")
    print(dataset.head(10).to_string(index=False))
    print_summary(dataset)


if __name__ == "__main__":
    main()
