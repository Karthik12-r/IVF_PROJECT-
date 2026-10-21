"""Pure helpers for safe reuse of the existing medical-document OCR flow."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import date, datetime


OCR_TO_PATIENT_FIELD = {"age": "age", "amh": "amh_level", "fsh": "fsh_level"}
OCR_TO_DASHBOARD_FIELD = {"age": "Age", "amh": "AMH", "fsh": "FSH"}
PREDICTION_FIELD_MAP = {
    "Age": "age", "BMI": "bmi", "AMH": "amh", "FSH": "fsh",
    "Previous_IVF_Attempts": "previous_ivf", "Stress_Level": "stress",
    "Sleep_Hours": "sleep_hours", "Exercise_Min_per_Day": "exercise_min",
}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _parse_date(value: str):
    value = value.strip().replace(".", "-")
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%d %B %Y",
                "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def extract_report_date(text: str):
    """Return a date only when it follows an explicit report/test date label."""
    date_token = r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})"
    label = re.compile(r"\b(?:report(?:ed)?\s+date|date\s+of\s+report|test\s+date|sample\s+date|specimen\s+date|collection\s+date|collected\s+on)\b\s*[:=\-]?\s*" + date_token, re.I)
    for line in (text or "").splitlines():
        match = label.search(line)
        if match:
            parsed = _parse_date(match.group(1))
            if parsed:
                return parsed
    return None


def explicit_measurement_evidence(text: str, field: str):
    """Find one unambiguous field-labelled value/unit; avoid cross-line guesses."""
    aliases = {"amh": r"(?:\bAMH\b(?:\s+level)?|anti[- ]?mullerian(?:\s+ hormone)?)",
               "fsh": r"(?:\bFSH\b(?:\s+level)?|follicle[- ]stimulating(?:\s+ hormone)?)"}
    label = aliases[field]
    pattern = re.compile(r"" + label + r"\s*[:=\-]?\s*([+-]?\d+(?:\.\d+)?)\s*([A-Za-zµμ/]+)?", re.I)
    matches = []
    for line in (text or "").splitlines():
        match = pattern.search(line)
        if match:
            matches.append((float(match.group(1)), (match.group(2) or "").strip().lower()))
    distinct = {(value, unit) for value, unit in matches}
    if len(distinct) > 1:
        return None, "ambiguous"
    if matches:
        return matches[0]
    return None, ""


def validate_ocr_ai_features(extracted: dict, text: str, report_date, date_confirmed: bool = False):
    """Validate only the OCR fields mapped into the predictor; never infer confidence."""
    output = {}
    for field in OCR_TO_PATIENT_FIELD:
        value = extracted.get(field)
        if value is None:
            output[field] = {"value": None, "status": "NOT_FOUND", "reason": "Field was not extracted."}
            continue
        try:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError
        except (TypeError, ValueError):
            output[field] = {"value": None, "status": "REQUIRES_REVIEW", "reason": "Value is not a finite number."}
            continue
        if numeric < 0 or (field == "age" and (numeric <= 0 or not numeric.is_integer())):
            output[field] = {"value": numeric, "status": "REQUIRES_REVIEW", "reason": "Value violates a basic nonnegative/integer data-integrity rule."}
            continue
        if field in ("amh", "fsh"):
            evidence_value, unit = explicit_measurement_evidence(text, field)
            if unit == "ambiguous":
                output[field] = {"value": numeric, "status": "REQUIRES_REVIEW", "reason": "Multiple different field-labelled readings were found in the report."}
                continue
            allowed_units = {"amh": {"ng/ml", "ng/mL"}, "fsh": {"miu/ml", "miu/mL", "iu/l"}}
            if evidence_value is None or not math.isclose(evidence_value, numeric, rel_tol=0, abs_tol=1e-9):
                output[field] = {"value": numeric, "status": "REQUIRES_REVIEW", "reason": "Could not confirm the field-labelled value from its source line."}
                continue
            if unit.lower() not in {u.lower() for u in allowed_units[field]}:
                output[field] = {"value": numeric, "status": "REQUIRES_REVIEW", "reason": "Expected a recognized unit (AMH ng/mL; FSH mIU/mL or IU/L)."}
                continue
        if not report_date and not date_confirmed:
            output[field] = {"value": numeric, "status": "REQUIRES_REVIEW", "reason": "Report date was not identified or confirmed; freshness cannot be established."}
            continue
        output[field] = {"value": int(numeric) if field == "age" else numeric,
                         "status": "VALID", "reason": "Passed numeric, unit (where applicable), and report-freshness checks."}
    return output


def validate_prediction_payload(data: dict):
    """Return complete numeric model inputs in the established order or raise ValueError."""
    if not isinstance(data, dict):
        raise ValueError("Prediction input must be a JSON object.")
    values = {}
    missing = []
    for form_key, model_key in PREDICTION_FIELD_MAP.items():
        value = data.get(form_key)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(form_key)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{form_key} must be numeric.")
        if not math.isfinite(number):
            raise ValueError(f"{form_key} must be a finite number.")
        values[form_key] = number
    if missing:
        raise ValueError("Prediction requires additional validated information: " + ", ".join(missing))
    if values["Age"] <= 0 or not values["Age"].is_integer():
        raise ValueError("Age must be a positive whole number.")
    if values["Previous_IVF_Attempts"] < 0 or not values["Previous_IVF_Attempts"].is_integer():
        raise ValueError("Previous IVF attempts must be a nonnegative whole number.")
    if values["Stress_Level"] not in (1, 2, 3, 4, 5):
        raise ValueError("Stress level must use the existing 1–5 scale.")
    if not values["Exercise_Min_per_Day"].is_integer():
        raise ValueError("Exercise minutes must be a whole number.")
    for key in ("BMI", "AMH", "FSH", "Sleep_Hours", "Exercise_Min_per_Day"):
        if values[key] < 0:
            raise ValueError(f"{key} must be nonnegative.")
    return values
