import pytest
import os
import sys
from flask import Flask
from database import db

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models to ensure they are registered with SQLAlchemy
from models import User, PatientData, WellnessLog, Clinic, IVFCycle, MedicalDocument, Prediction, ChatMessage, MedicationReminder, CycleNote, MedicalActivity

@pytest.fixture(scope="session")
def app():
    """Create and configure a test app instance."""
    from main import app as flask_app

    # Configure for testing
    flask_app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'SECRET_KEY': 'test-secret-key',
        'WTF_CSRF_ENABLED': False,
    })

    # Create tables within app context
    with flask_app.app_context():
        db.create_all()

    return flask_app

@pytest.fixture(scope="session")
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture(scope="function")
def db_session(app):
    """Create a fresh database for each test."""
    with app.app_context():
        db.create_all()
        yield db.session
        db.session.remove()
        db.drop_all()

@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        'username': 'testuser',
        'email': 'test@example.com',
        'first_name': 'Test',
        'last_name': 'User',
        'password': 'password123',
        'user_type': 'patient'
    }

@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing."""
    return {
        'age': 32,
        'bmi': 22.5,
        'amh_level': 2.1,
        'fsh_level': 6.2,
        'previous_ivf_cycles': 1,
        'diagnosis': 'Unexplained infertility'
    }
