"""Integration coverage for the existing upload/OCR/dashboard workflow."""

import io
import json
import tempfile
from datetime import date
from pathlib import Path

import pytest


def report_text(report_date="2026-10-01", age=31, amh_line="AMH: 2.8 ng/mL", fsh_line="FSH: 6.2 mIU/mL"):
    return f"Report Date: {report_date}\nPatient Age: {age} years\n{amh_line}\n{fsh_line}\n"


@pytest.fixture
def patient_client(app, client, db_session):
    from models import User, PatientData, WellnessLog

    user = User(username="docflow_patient", email="docflow@example.test", first_name="Test",
                last_name="Patient", user_type="patient")
    user.set_password("test-password")
    db_session.add(user)
    db_session.flush()
    patient = PatientData(user_id=user.id, age=35, previous_ivf_cycles=1)
    patient.bmi = 24.5
    patient.amh_level = 2.4
    patient.fsh_level = 7.0
    db_session.add(patient)
    db_session.add(WellnessLog(user_id=user.id, date=date(2026, 10, 1), stress_level=3,
                               sleep_hours=7.0, exercise_minutes=30))
    db_session.commit()
    with client.session_transaction() as session:
        session["user_id"] = user.id
        session["user_type"] = "patient"
    return client, user, patient


def upload(client, content, filename="report.txt", report_date_value=None):
    data = {"file": (io.BytesIO(content.encode("utf-8")), filename)}
    if report_date_value:
        data["report_date"] = report_date_value
    return client.post("/upload_document", data=data, content_type="multipart/form-data",
                       headers={"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"})


def test_new_report_is_ocr_processed_validated_saved_and_mapped(patient_client, app, db_session, tmp_path):
    from models import MedicalDocument, MedicalFeatureHistory

    client, _, patient = patient_client
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    response = upload(client, report_text())
    result = response.get_json()
    assert response.status_code == 200
    assert result["status"] == "VALIDATED"
    assert result["auto_saved_fields"] == ["age", "amh", "fsh"]
    assert patient.age == 31 and patient.amh_level == 2.8 and patient.fsh_level == 6.2
    document = MedicalDocument.query.one()
    assert document.file_hash and len(document.file_hash) == 64
    assert document.is_latest is True
    history = MedicalFeatureHistory.query.order_by(MedicalFeatureHistory.feature_name).all()
    assert len(history) == 3
    assert all(item.validation_status == "VALID" and item.confidence is None for item in history)
    assert result["feature_sources"]["AMH"]["document"] == "report.txt"


def test_exact_duplicate_skips_extraction_and_does_not_create_a_second_document(patient_client, app, db_session, tmp_path, monkeypatch):
    import main
    from models import MedicalDocument

    client, _, _ = patient_client
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    content = report_text()
    first = upload(client, content)
    assert first.get_json()["status"] == "VALIDATED"
    monkeypatch.setattr(main, "_read_medical_document_text", lambda *args: pytest.fail("duplicate triggered OCR"))
    second = upload(client, content, filename="same-bytes-different-name.txt")
    assert second.get_json()["status"] == "DUPLICATE"
    assert second.get_json()["ocr_skipped"] is True
    assert MedicalDocument.query.count() == 1


def test_document_download_is_restricted_to_owner(patient_client, app, db_session):
    from models import MedicalDocument, User

    client, _, _ = patient_client
    with tempfile.TemporaryDirectory(prefix="document-owner-test-", dir=app.instance_path) as upload_dir:
        app.config["UPLOAD_FOLDER"] = upload_dir
        result = upload(client, report_text()).get_json()
        document = db_session.get(MedicalDocument, result["document_id"])
        other_user = User(username="other_document_user", email="other-document@example.test",
                          first_name="Other", last_name="User", user_type="patient")
        other_user.set_password("test-password")
        db_session.add(other_user)
        db_session.commit()
        (Path(upload_dir) / document.filename).write_bytes(b"private report")

        with client.session_transaction() as session:
            session["user_id"] = other_user.id
            session["user_type"] = "patient"
        response = client.get(f"/download_document/{document.id}")
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/my_documents")


def test_older_report_is_kept_without_feature_extraction_or_overwrite(patient_client, app, db_session, tmp_path, monkeypatch):
    import main
    from models import MedicalDocument

    client, _, patient = patient_client
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    newer = upload(client, report_text("2026-10-01", 31, "AMH: 2.8 ng/mL", "FSH: 6.2 mIU/mL"))
    assert newer.get_json()["status"] == "VALIDATED"
    monkeypatch.setattr(main, "extract_medical_params", lambda *_: pytest.fail("old report entered feature extraction"))
    older = upload(client, report_text("2026-08-01", 30, "AMH: 1.9 ng/mL", "FSH: 8.2 mIU/mL"), "old-report.txt")
    assert older.get_json()["status"] == "OLD_REPORT"
    assert patient.age == 31 and patient.amh_level == 2.8 and patient.fsh_level == 6.2
    assert MedicalDocument.query.count() == 2
    assert MedicalDocument.query.filter_by(processing_status="OLD_REPORT").count() == 1


def test_partial_report_updates_only_present_valid_values(patient_client, app, db_session, tmp_path):
    client, _, patient = patient_client
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    response = upload(client, "Report Date: 2026-10-01\nAMH: 3.1 ng/mL\n", "amh-only.txt")
    assert response.get_json()["status"] == "VALIDATED"
    assert response.get_json()["auto_saved_fields"] == ["amh"]
    assert patient.amh_level == 3.1
    assert patient.age == 35 and patient.fsh_level == 7.0


def test_unconfirmed_units_require_review_and_do_not_replace_existing_value(patient_client, app, db_session, tmp_path):
    from models import MedicalFeatureHistory

    client, _, patient = patient_client
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    response = upload(client, "Report Date: 2026-10-01\nAMH: 4.2\n")
    result = response.get_json()
    assert result["status"] == "VALIDATION_REQUIRED"
    assert result["validation"]["amh"]["status"] == "REQUIRES_REVIEW"
    assert patient.amh_level == 2.4
    assert MedicalFeatureHistory.query.filter_by(feature_name="amh", validation_status="REQUIRES_REVIEW").count() == 1


def test_conflicting_measurements_require_manual_review(patient_client, app, db_session, tmp_path):
    client, _, patient = patient_client
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    response = upload(client, "Report Date: 2026-10-01\nAMH: 2.8 ng/mL\nAMH: 3.4 ng/mL\n")
    result = response.get_json()
    assert result["status"] == "VALIDATION_REQUIRED"
    assert result["validation"]["amh"]["status"] == "REQUIRES_REVIEW"
    assert patient.amh_level == 2.4


def test_user_edit_is_persisted_and_audited_when_existing_prediction_endpoint_runs(patient_client, app, db_session, tmp_path, monkeypatch):
    import prediction_service
    from models import MedicalFeatureHistory, Prediction

    client, _, patient = patient_client
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    uploaded = upload(client, report_text())
    document_id = uploaded.get_json()["document_id"]

    def known_existing_service(subject):
        assert subject.age == 31 and subject.amh_level == 3.0 and subject.fsh_level == 6.2
        return {"success_rate": 68.0, "confidence": 80.0, "prediction_label": "Research estimate", "model_type": "test-double", "factors": [], "interpretation": ""}

    monkeypatch.setattr(prediction_service, "calculate_ivf_success_prediction", known_existing_service)
    response = client.post("/predict_ivf", json={
        "Age": 31, "BMI": 24.5, "AMH": 3.0, "FSH": 6.2,
        "Previous_IVF_Attempts": 1, "Stress_Level": 3, "Sleep_Hours": 7,
        "Exercise_Min_per_Day": 30, "source_document_id": document_id,
        "edited_fields": ["AMH"],
    })
    assert response.status_code == 200
    assert patient.amh_level == 3.0
    edit = MedicalFeatureHistory.query.filter_by(feature_name="amh", user_modified=True).one()
    assert edit.old_value == pytest.approx(2.8) and edit.new_value == pytest.approx(3.0)
    assert edit.document_id == document_id
    prediction = Prediction.query.one()
    assert json.loads(prediction.model_metadata)["source_document_id"] == document_id


def test_newer_valid_report_updates_current_fields_and_keeps_history(patient_client, app, db_session, tmp_path):
    from models import MedicalDocument, MedicalFeatureHistory

    client, _, patient = patient_client
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    first = upload(client, report_text("2026-08-01", 30, "AMH: 2.2 ng/mL", "FSH: 7.5 mIU/mL"), "aug.txt")
    second = upload(client, report_text("2026-10-01", 31, "AMH: 2.8 ng/mL", "FSH: 6.2 mIU/mL"), "oct.txt")
    assert first.get_json()["status"] == second.get_json()["status"] == "VALIDATED"
    assert patient.age == 31 and patient.amh_level == 2.8
    assert MedicalFeatureHistory.query.filter_by(feature_name="amh").count() == 2
    assert MedicalDocument.query.filter_by(is_latest=True).count() == 1
    assert MedicalDocument.query.filter_by(original_filename="aug.txt").one().is_latest is False


def test_dashboard_refresh_loads_saved_values_without_running_ocr(patient_client, app, db_session, tmp_path, monkeypatch):
    import main
    from models import WellnessLog

    client, user, _ = patient_client
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    WellnessLog.query.filter_by(user_id=user.id).delete()
    db_session.commit()
    monkeypatch.setattr(main, "_read_medical_document_text", lambda *args: pytest.fail("dashboard refresh ran OCR"))
    response = client.get("/ivf_predictor")
    assert response.status_code == 200
    assert b"Use a Medical Report" in response.data
    assert b"source_document_id" in response.data
    assert b"Prediction requires additional validated information" in response.data


def test_prediction_endpoint_rejects_missing_or_invalid_features(patient_client, app, db_session):
    client, _, _ = patient_client
    response = client.post("/predict_ivf", json={"Age": 31})
    assert response.status_code == 400
    assert "Prediction requires additional validated information" in response.get_json()["error"]
    invalid = client.post("/predict_ivf", json={
        "Age": 31, "BMI": 24.5, "AMH": 2.8, "FSH": 6.2,
        "Previous_IVF_Attempts": 1, "Stress_Level": 8, "Sleep_Hours": 7,
        "Exercise_Min_per_Day": 30,
    })
    assert invalid.status_code == 400


def test_prediction_endpoint_runs_existing_prediction_service(patient_client, app, db_session):
    client, _, _ = patient_client
    response = client.post("/predict_ivf", json={
        "Age": 31, "BMI": 24.5, "AMH": 2.8, "FSH": 6.2,
        "Previous_IVF_Attempts": 1, "Stress_Level": 3, "Sleep_Hours": 7,
        "Exercise_Min_per_Day": 30,
    })
    assert response.status_code == 200, response.get_json()
    result = response.get_json()
    assert 0 <= result["success_probability"] <= 100
    assert result["prediction_id"]
