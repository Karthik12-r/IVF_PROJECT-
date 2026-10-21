import pytest
from unittest.mock import patch, MagicMock
from flask import session


class TestRoutes:
    """Integration tests for Flask routes"""

    def test_home_page(self, client):
        """Test home page loads successfully"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'IVF Journey Tracker' in response.data

    def test_login_required_redirect(self, client):
        """Test that protected routes redirect to login"""
        response = client.get('/patient_dashboard')
        assert response.status_code == 302  # Redirect
        assert '/login' in response.headers['Location']

    def test_register_page(self, client):
        """Test registration page loads"""
        response = client.get('/register')
        assert response.status_code == 200
        assert b'Register' in response.data

    def test_register_user_success(self, client, db_session):
        """Test successful user registration"""
        response = client.post('/register', data={
            'username': 'testuser',
            'email': 'test@example.com',
            'first_name': 'Test',
            'last_name': 'User',
            'password': 'password123',
            'confirm_password': 'password123',
            'user_type': 'patient',
            'agree_terms': 'on'
        }, follow_redirects=True)

        assert response.status_code == 200
        user = db_session.query(__import__('models').User).filter_by(email='test@example.com').first()
        assert user is not None
        assert user.check_password('password123')

    def test_login_page(self, client):
        """Test login page loads"""
        response = client.get('/login')
        assert response.status_code == 200
        assert b'Login' in response.data

    def test_login_success(self, client, db_session):
        """Test successful login"""
        from models import User
        user = User(username='login_test', email='login@test.com', first_name='Login', last_name='Test', user_type='patient')
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()

        with client:
            response = client.post('/login', data={
                'email': 'login@test.com',
                'password': 'password123'
            }, follow_redirects=True)
            assert session.get('user_id') == user.id

        assert response.status_code == 200

    def test_logout(self, client):
        """Test logout functionality"""
        with client:
            # First login
            with client.session_transaction() as sess:
                sess['user_id'] = 1

            response = client.get('/logout', follow_redirects=True)
            assert response.status_code == 200
            assert 'user_id' not in session

    def test_prediction_route(self, client, db_session):
        """Test IVF prediction route"""
        from models import User
        user = User(username='predict_route_test', email='predict-route@test.com', first_name='Predict', last_name='Test', user_type='patient')
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()

        with client:
            with client.session_transaction() as sess:
                sess['user_id'] = user.id

            response = client.get('/predict')
            assert response.status_code == 302
            assert response.headers['Location'].endswith('/ivf_predictor')

    def test_ivf_predictor_page(self, client):
        """The personalized predictor is available only after authentication."""
        response = client.get('/ivf_predictor')
        assert response.status_code == 302
        assert '/login' in response.headers['Location']

    def test_find_clinic_page(self, client):
        """Test find clinic page"""
        response = client.get('/find_clinic')
        assert response.status_code == 200
        assert b'Find IVF Clinics' in response.data

    def test_chatbot_page(self, client):
        """Test chatbot page loads"""
        response = client.get('/chatbot')
        assert response.status_code == 200
        assert b'IVF Assistant' in response.data

    def test_faq_page(self, client):
        """Test FAQ page loads"""
        response = client.get('/faq')
        assert response.status_code == 200
        assert b'FAQ' in response.data

    def test_wellness_page(self, client):
        """Test wellness page loads"""
        response = client.get('/wellness')
        assert response.status_code == 200
        assert b'Wellness &amp; Mood Tracking' in response.data or b'Wellness & Mood Tracking' in response.data

    def test_partner_registration_page(self, client):
        """Test partner registration page loads"""
        response = client.get('/register/partner')
        assert response.status_code == 200
        assert b'Partner' in response.data

    def test_partner_dashboard_requires_login(self, client):
        """Test partner dashboard rejects unauthenticated access"""
        response = client.get('/partner/dashboard')
        assert response.status_code == 302
        assert '/login' in response.headers['Location']
