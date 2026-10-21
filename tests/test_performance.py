import pytest
import time
import psutil
import os
from unittest.mock import patch
from predict import predict_from_features


class TestPerformance:
    """Performance tests for IVF prediction system"""

    @pytest.fixture
    def sample_meta(self):
        """Sample metadata for performance testing"""
        return {
            "feature_order": ["age", "bmi", "amh", "fsh", "previous_ivf", "stress", "sleep_hours", "exercise_min"],
            "model": "RandomForestClassifier",
            "accuracy": 0.75
        }

    def test_prediction_response_time(self, sample_meta):
        """Test prediction response time under normal load"""
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

        predict_from_features(features_dict, sample_meta)
        start_time = time.perf_counter()
        results = [predict_from_features(features_dict, sample_meta) for _ in range(10)]
        average_time = (time.perf_counter() - start_time) / len(results)

        assert results
        assert "prediction" in results[0]

        # Assert reasonable response time (< 100ms)
        assert average_time < 0.1

    def test_concurrent_predictions(self, sample_meta):
        """Test multiple concurrent predictions"""
        import threading
        import queue

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

        results = queue.Queue()
        errors = queue.Queue()

        def predict_worker(worker_id):
            try:
                result = predict_from_features(features_dict, sample_meta)
                results.put((worker_id, result))
            except Exception as e:
                errors.put((worker_id, str(e)))

        # Start 10 concurrent predictions
        threads = []
        for i in range(10):
            t = threading.Thread(target=predict_worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check results
        assert results.qsize() == 10  # All predictions should succeed
        assert errors.qsize() == 0  # No errors should occur

    def test_memory_usage_during_prediction(self, sample_meta):
        """Test memory usage during prediction"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

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

        # Run multiple predictions
        for _ in range(100):
            predict_from_features(features_dict, sample_meta)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB)
        assert memory_increase < 50

    def test_model_loading_performance(self):
        """Test model loading performance"""
        from predict import load_model_and_meta

        start_time = time.time()
        model, meta = load_model_and_meta()
        load_time = time.time() - start_time

        assert model is not None
        assert meta is not None
        # Model loading should be fast (< 1 second)
        assert load_time < 1.0

    def test_batch_prediction_performance(self, sample_meta):
        """Test batch prediction performance"""
        # Create batch of 50 prediction requests
        batch_features = []
        for i in range(50):
            features_dict = {
                "age": 25 + i % 20,  # Vary age between 25-44
                "bmi": 20 + (i % 10),  # Vary BMI
                "amh": 1.0 + (i % 5),  # Vary AMH
                "fsh": 4.0 + (i % 8),  # Vary FSH
                "previous_ivf": i % 3,  # Vary previous cycles
                "stress": 1 + (i % 5),  # Vary stress
                "sleep_hours": 5 + (i % 5),  # Vary sleep
                "exercise_min": 10 + (i % 50)  # Vary exercise
            }
            batch_features.append(features_dict)

        start_time = time.time()
        results = []
        for features in batch_features:
            result = predict_from_features(features, sample_meta)
            results.append(result)

        batch_time = time.time() - start_time

        assert len(results) == 50
        # Batch processing should be reasonably fast (< 5 seconds for 50 predictions)
        assert batch_time < 5.0

        # Average time per prediction
        avg_time = batch_time / 50
        assert avg_time < 0.1  # < 100ms per prediction

    def test_database_query_performance(self, db_session):
        """Test database query performance for common operations"""
        from models import User, PatientData, WellnessLog

        # Create test data
        users = []
        for i in range(10):
            user = User(
                username=f'perf_user_{i}',
                email=f'perf{i}@example.com',
                first_name=f'Perf{i}',
                last_name='User',
                user_type='patient'
            )
            user.set_password('password123')
            users.append(user)

        db_session.add_all(users)
        db_session.commit()

        # Test user query performance
        start_time = time.time()
        queried_users = User.query.all()
        query_time = time.time() - start_time

        assert len(queried_users) == 10
        # Database queries should be fast (< 100ms)
        assert query_time < 0.1

    def test_large_dataset_handling(self, sample_meta):
        """Test handling of large feature datasets"""
        # Test with edge case values
        edge_cases = [
            {"age": 20, "bmi": 15, "amh": 0.1, "fsh": 2, "previous_ivf": 0, "stress": 1, "sleep_hours": 3, "exercise_min": 0},
            {"age": 50, "bmi": 40, "amh": 10, "fsh": 20, "previous_ivf": 5, "stress": 5, "sleep_hours": 12, "exercise_min": 120},
            {"age": 35, "bmi": 25, "amh": 2.5, "fsh": 8, "previous_ivf": 2, "stress": 3, "sleep_hours": 7, "exercise_min": 45}
        ]

        for features in edge_cases:
            result = predict_from_features(features, sample_meta)
            assert "prediction" in result
            assert "success_probability" in result

    def test_api_response_time_simulation(self, client):
        """Simulate API response time for web requests"""
        # Test static page load time
        start_time = time.time()
        response = client.get('/')
        load_time = time.time() - start_time

        assert response.status_code == 200
        # Page should load quickly (< 500ms)
        assert load_time < 0.5

    def test_scalability_with_feature_complexity(self, sample_meta):
        """Test prediction performance with varying feature complexity"""
        # Test with minimal features (defaults used)
        minimal_features = {"age": 30}

        start_time = time.time()
        result = predict_from_features(minimal_features, sample_meta)
        minimal_time = time.time() - start_time

        # Test with all features provided
        full_features = {
            "age": 30, "bmi": 22.5, "amh": 2.5, "fsh": 6.0,
            "previous_ivf": 0, "stress": 3, "sleep_hours": 7.0, "exercise_min": 30
        }

        start_time = time.time()
        result = predict_from_features(full_features, sample_meta)
        full_time = time.time() - start_time

        # Full feature prediction should not be significantly slower
        assert full_time < minimal_time * 2
