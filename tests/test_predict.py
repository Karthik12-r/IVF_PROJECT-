import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from predict import load_model_and_meta, build_feature_vector, predict_from_features


class TestPredictionService:
    """Unit tests for IVF prediction service functions"""

    @pytest.fixture
    def sample_meta(self):
        """Sample metadata for testing"""
        return {
            "feature_order": ["age", "bmi", "amh", "fsh", "previous_ivf", "stress", "sleep_hours", "exercise_min"],
            "model": "RandomForestClassifier",
            "accuracy": 0.75
        }

    def test_load_model_and_meta(self):
        """Test model and metadata loading"""
        model, meta = load_model_and_meta()
        assert model is not None
        assert meta is not None
        assert "feature_order" in meta
        assert len(meta["feature_order"]) == 8

    def test_predict_from_features_success(self, sample_meta):
        """Test successful prediction with valid features"""
        features_dict = {
            "age": 30,
            "bmi": 22.5,
            "amh": 2.5,
            "fsh": 6.0,
            "previous_ivf": 0,
            "stress": 3,
            "sleep_hours": 7.0,
            "exercise_min": 30
        }

        result = predict_from_features(features_dict, sample_meta)

        assert "prediction" in result
        assert "success_probability" in result
        assert "prediction_text" in result
        assert "features_used" in result
        assert isinstance(result["success_probability"], (int, float))
        assert result["prediction_text"] in ["Success Likely", "Success Unlikely"]

    def test_predict_from_features_missing_data(self, sample_meta):
        """Test prediction with missing feature data (should use defaults)"""
        features_dict = {
            "age": 30,
            "bmi": None,  # Missing BMI
            "amh": 2.5,
            "fsh": 6.0,
            "previous_ivf": 0,
            "stress": 3,
            "sleep_hours": 7.0,
            "exercise_min": 30
        }

        result = predict_from_features(features_dict, sample_meta)

        assert "prediction" in result
        assert result["features_used"]["bmi"] == 22.5  # Default value

    @patch('predict.app')
    def test_build_feature_vector_with_data(self, mock_app, sample_meta):
        """Test feature vector building with mock patient data"""
        # Mock the app context and database query
        mock_app.app_context.return_value.__enter__ = MagicMock()
        mock_app.app_context.return_value.__exit__ = MagicMock()

        # Mock patient data
        mock_patient = MagicMock()
        mock_patient.age = 32
        mock_patient.bmi = 24.5
        mock_patient.amh_level = 2.1
        mock_patient.fsh_level = 6.2
        mock_patient.previous_ivf_cycles = 1

        # Mock wellness log
        mock_wellness = MagicMock()
        mock_wellness.stress_level = 4
        mock_wellness.sleep_hours = 6.5
        mock_wellness.exercise_minutes = 45

        with patch('predict.PatientData') as mock_patient_model, \
             patch('predict.WellnessLog') as mock_wellness_model:

            mock_patient_model.query.filter_by.return_value.first.return_value = mock_patient
            mock_wellness_model.query.filter_by.return_value.order_by.return_value.first.return_value = mock_wellness

            features = build_feature_vector(1, sample_meta)

            assert isinstance(features, np.ndarray)
            assert features.shape == (1, 8)
            assert features[0][0] == 32  # age
            assert features[0][1] == 24.5  # bmi
            assert features[0][2] == 2.1  # amh
            assert features[0][3] == 6.2  # fsh
            assert features[0][4] == 1  # previous_ivf
            assert features[0][5] == 4  # stress
            assert features[0][6] == 6.5  # sleep_hours
            assert features[0][7] == 45  # exercise_min

    def test_build_feature_vector_no_patient_data(self, sample_meta, app, db_session):
        """Raise a clear error when the database has no row for this user."""
        with pytest.raises(ValueError, match="No PatientData found"):
            build_feature_vector(999999999, sample_meta)

    def test_predict_from_features_edge_cases(self, sample_meta):
        """Test prediction with edge case values"""
        # Very young age
        features_dict = {
            "age": 20,
            "bmi": 18.5,
            "amh": 5.0,
            "fsh": 4.0,
            "previous_ivf": 0,
            "stress": 1,
            "sleep_hours": 8.0,
            "exercise_min": 60
        }

        result = predict_from_features(features_dict, sample_meta)
        assert "prediction" in result

        # Advanced age
        features_dict["age"] = 45
        result = predict_from_features(features_dict, sample_meta)
        assert "prediction" in result
