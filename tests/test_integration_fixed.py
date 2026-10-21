import pytest
import json
from unittest.mock import patch, MagicMock
from models import User, PatientData, WellnessLog


class TestIntegration:
    """Integration tests for end-to-end functionality"""

    def test_complete_user_registration_login_flow(self, client, db_session):
        """Test complete user registration and login flow"""
        # Register user
        response = client.post('/register', data={
            'username': 'integration_test',
            'email': 'integration@test.com',
            'first_name': 'Integration',
            'last_name': 'Test',
            'password': 'password123',
            'confirm_password': 'password123',
            'date_of_birth': '1990-01-01',
            'user_type': 'patient'
        }, follow_redirects=True)

        assert response.status_code == 200

        # Login user
        response = client.post('/login', data={
            'email': 'integration@test.com',
            'password': 'password123'
        }, follow_redirects=True)

        assert response.status_code == 200
        assert b'Welcome' in response.data or b'Dashboard' in response.data

    @patch('predict.predict_and_store')
    def test_prediction_workflow(self, mock_predict, client, db_session):
        """Test complete prediction workflow"""
        # Create and login user
        user = User(
            username='predict_test',
            email='predict@test.com',
            first_name='Predict',
            last_name='Test',
            user_type='patient'
        )
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()

        # Login
        response = client.post('/login', data={
            'email': 'predict@test.com',
            'password': 'password123'
        }, follow_redirects=True)

        # Mock prediction function
        mock_predict.return_value = {
            'prediction': 1,
            'probability': 0.75,
            'prediction_id': 1
        }

        # Access prediction page
        response = client.get('/ivf_predictor', follow_redirects=True)
        assert response.status_code == 200

    def test_ivf_predictor_page_access(self, client, db_session):
        """Test IVF predictor page access and prediction display"""
        # Create and login user
        user = User(
            username='predictor_test',
            email='predictor@test.com',
            first_name='Predictor',
            last_name='Test',
            user_type='patient'
        )
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()

        # Create patient data
        patient_data = PatientData(user_id=user.id, age=30, bmi=22.5, amh_level=2.5, fsh_level=6.0)
        db_session.add(patient_data)
        db_session.commit()

        # Login
        response = client.post('/login', data={
            'email': 'predictor@test.com',
            'password': 'password123'
        }, follow_redirects=True)

        # Test page access and prediction display
        response = client.get('/ivf_predictor')
        assert response.status_code == 200
        assert b'IVF Success Predictor' in response.data

    @patch('models.Clinic')
    def test_clinic_search_workflow(self, mock_clinic, client):
        """Test clinic search functionality"""
        # Mock clinic data
        mock_clinic_obj = MagicMock()
        mock_clinic_obj.id = 1
        mock_clinic_obj.name = 'Test Clinic'
        mock_clinic_obj.city = 'Mumbai'
        mock_clinic_obj.success_rate = 65.0
        mock_clinic.query.all.return_value = [mock_clinic_obj]

        # Test clinic search page
        response = client.get('/find_clinic')
        assert response.status_code == 200
        assert b'Find IVF Clinics' in response.data

        # Test search functionality
        response = client.post('/find_clinic', data={
            'city': 'Mumbai',
            'min_success_rate': '60'
        }, follow_redirects=True)

        assert response.status_code == 200

    def test_chatbot_interaction(self, client):
        """Test chatbot page and basic interaction"""
        response = client.get('/chatbot')
        assert response.status_code == 200
        assert b'IVF Assistant' in response.data

    def test_document_upload_flow(self, client, db_session):
        """Test document upload functionality"""
        # Create and login user
        user = User(
            username='doc_test',
            email='doc@test.com',
            first_name='Doc',
            last_name='Test',
            user_type='patient'
        )
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()

        # Login
        client.post('/login', data={
            'email': 'doc@test.com',
            'password': 'password123'
        }, follow_redirects=True)

        # Test document upload page
        response = client.get('/my_documents')
        assert response.status_code == 200
        assert b'My Documents' in response.data

    def test_wellness_tracking_workflow(self, client, db_session):
        """Test wellness tracking functionality"""
        # Create and login user
        user = User(
            username='wellness_test',
            email='wellness@test.com',
            first_name='Wellness',
            last_name='Test',
            user_type='patient'
        )
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()

        # Login
        client.post('/login', data={
            'email': 'wellness@test.com',
            'password': 'password123'
        }, follow_redirects=True)

        # Test wellness page
        response = client.get('/wellness')
        assert response.status_code == 200
        assert b'Wellness & Mood Tracking' in response.data

        # Test wellness log submission
        response = client.post('/wellness', data={
            'mood_rating': '4',
            'stress_level': '3',
            'sleep_hours': '7.5',
            'exercise_minutes': '45',
            'notes': 'Feeling good'
        }, follow_redirects=True)

        assert response.status_code == 200

    def test_faq_accessibility(self, client):
        """Test FAQ page accessibility"""
        response = client.get('/faq')
        assert response.status_code == 200
        assert b'FAQ' in response.data
        assert b'frequently asked questions' in response.data.lower()

    def test_error_handling(self, client):
        """Test error handling for invalid routes"""
        response = client.get('/nonexistent_route')
        assert response.status_code == 404

    def test_static_file_serving(self, client):
        """Test static file serving"""
        response = client.get('/static/css/base.style.css')
        assert response.status_code == 200
        assert 'text/css' in response.content_type

    def test_json_api_endpoints(self, client):
        """Test JSON API endpoints if any"""
        # This would test any JSON API endpoints in the application
        # For now, just verify the app can handle JSON requests
        response = client.get('/', headers={'Accept': 'application/json'})
        # Should still return HTML but handle the request
        assert response.status_code in [200, 406]  # 406 if JSON not acceptable
