import pytest
import json
from unittest.mock import patch, MagicMock


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
            'user_type': 'patient',
            'agree_terms': 'on'
        }, follow_redirects=True)

        assert response.status_code == 200

        # Login user
        response = client.post('/login', data={
            'email': 'integration@test.com',
            'password': 'password123'
        }, follow_redirects=True)

        assert response.status_code == 200
        assert b'Welcome' in response.data or b'Dashboard' in response.data

    def test_prediction_workflow(self, client, db_session):
        """Test complete prediction workflow"""
        # Create and login user
        from models import User
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

        # Access prediction page
        response = client.get('/predict', follow_redirects=True)
        assert response.status_code == 200

    def test_ivf_predictor_requires_login(self, client):
        response = client.get('/ivf_predictor')
        assert response.status_code == 302
        assert '/login' in response.headers['Location']

    def test_clinic_search_workflow(self, client):
        """Test clinic search functionality"""
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

    def test_chat_api_fallback_and_input_validation(self, client, db_session, monkeypatch):
        import openai_service
        from models import ChatMessage, User

        user = User(username='chat_api_test', email='chat-api@test.com', first_name='Chat',
                    last_name='Test', user_type='patient')
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()
        monkeypatch.setattr(openai_service, 'GOOGLE_API_KEY', '')
        with client.session_transaction() as flask_session:
            flask_session['user_id'] = user.id
            flask_session['user_type'] = user.user_type

        response = client.post('/api/chat', json={'message': 'What is the IVF process?'})
        assert response.status_code == 200
        assert 'fertility specialist' in response.get_json()['response']
        assert ChatMessage.query.filter_by(user_id=user.id).count() == 1
        assert client.post('/api/chat', json={'message': '  '}).status_code == 400
        assert client.post('/api/chat', json={'message': 'x' * 4001}).status_code == 400

    def test_document_upload_flow(self, client, db_session):
        """Test document upload functionality"""
        # Create and login user
        from models import User
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
        from models import User
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
        assert b'Wellness &amp; Mood Tracking' in response.data or b'Wellness & Mood Tracking' in response.data

        # Test wellness log submission
        response = client.post('/wellness', data={
            'mood_rating': '4',
            'stress_level': '3',
            'sleep_hours': '7.5',
            'exercise_minutes': '45',
            'notes': 'Feeling good'
        }, follow_redirects=True)

        assert response.status_code == 200
        from models import WellnessLog
        saved_log = WellnessLog.query.filter_by(user_id=user.id).one()
        assert saved_log.mood_rating == 4
        assert saved_log.stress_level == 3

    def test_treatment_cycle_and_medication_reminder_persist_for_owner(self, client, db_session):
        from models import IVFCycle, MedicationReminder, User

        patient = User(username='treatment_owner', email='treatment-owner@test.com',
                       first_name='Treatment', last_name='Owner', user_type='patient')
        patient.set_password('password123')
        other = User(username='treatment_other', email='treatment-other@test.com',
                     first_name='Other', last_name='Patient', user_type='patient')
        other.set_password('password123')
        db_session.add_all([patient, other])
        db_session.commit()

        with client.session_transaction() as flask_session:
            flask_session['user_id'] = patient.id
            flask_session['user_type'] = 'patient'
        response = client.post('/my_treatment', data={
            'protocol': 'Antagonist', 'start_date': '2026-10-17', 'patient_notes': 'Integration check'
        }, follow_redirects=True)
        assert response.status_code == 200
        cycle = IVFCycle.query.filter_by(patient_id=patient.id).one()
        assert cycle.protocol == 'Antagonist' and cycle.status == 'stimulation'

        response = client.post('/my_reminders', data={
            'medication_name': 'Test medication', 'dosage': '1 dose', 'frequency': 'Daily',
            'time_of_day': '08:30', 'start_date': '2026-10-17'
        }, follow_redirects=True)
        assert response.status_code == 200
        reminder = MedicationReminder.query.filter_by(user_id=patient.id).one()
        assert reminder.medication_name == 'Test medication' and reminder.is_active is True

        with client.session_transaction() as flask_session:
            flask_session['user_id'] = other.id
            flask_session['user_type'] = 'patient'
        response = client.post(f'/toggle_reminder/{reminder.id}', follow_redirects=True)
        assert response.status_code == 200
        db_session.refresh(reminder)
        assert reminder.is_active is True

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
