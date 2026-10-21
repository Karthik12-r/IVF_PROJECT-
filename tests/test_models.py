import pytest
from datetime import datetime, timezone
from models import User, PatientData, Prediction, WellnessLog, db


class TestModels:
    """Unit tests for database models"""

    def test_user_creation(self, db_session):
        """Test User model creation"""
        user = User(
            username='testuser',
            email='test@example.com',
            first_name='Test',
            last_name='User',
            user_type='patient'
        )
        user.set_password('password123')

        db_session.add(user)
        db_session.commit()

        assert user.id is not None
        assert user.username == 'testuser'
        assert user.check_password('password123')
        assert not user.check_password('wrongpassword')

    def test_patient_data_creation(self, db_session):
        """Test PatientData model creation"""
        user = User(
            username='patient',
            email='patient@example.com',
            first_name='Patient',
            last_name='Test',
            user_type='patient'
        )
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()

        patient_data = PatientData(
            user_id=user.id,
            age=32,
            bmi=24.5,
            amh_level=2.1,
            fsh_level=6.2,
            previous_ivf_cycles=1,
            diagnosis='Unexplained infertility'
        )

        db_session.add(patient_data)
        db_session.commit()

        assert patient_data.id is not None
        assert patient_data.age == 32
        assert patient_data.user_id == user.id

    def test_wellness_log_creation(self, db_session):
        """Test WellnessLog model creation"""
        user = User(
            username='wellness_user',
            email='wellness@example.com',
            first_name='Wellness',
            last_name='User',
            user_type='patient'
        )
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()

        wellness_log = WellnessLog(
            user_id=user.id,
            date=datetime.now(timezone.utc).date(),
            mood_rating=4,
            stress_level=3,
            sleep_hours=7.5,
            exercise_minutes=45,
            notes='Feeling good today'
        )

        db_session.add(wellness_log)
        db_session.commit()

        assert wellness_log.id is not None
        assert wellness_log.mood_rating == 4
        assert wellness_log.user_id == user.id

    def test_prediction_creation(self, db_session):
        """Test Prediction model creation"""
        user = User(
            username='predict_user',
            email='predict@example.com',
            first_name='Predict',
            last_name='User',
            user_type='patient'
        )
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()

        prediction = Prediction(
            user_id=user.id,
            success_probability=0.75,
            protocol_recommendation='Standard IVF',
            llm_analysis='{"analysis": "Good prognosis"}',
            model_metadata='{"version": "1.0"}'
        )

        db_session.add(prediction)
        db_session.commit()

        assert prediction.id is not None
        assert prediction.success_probability == 0.75
        assert prediction.user_id == user.id

    def test_user_relationships(self, db_session):
        """Test User model relationships"""
        user = User(
            username='relations_user',
            email='relations@example.com',
            first_name='Relations',
            last_name='User',
            user_type='patient'
        )
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()

        # Create related records
        patient_data = PatientData(user_id=user.id, age=30)
        wellness_log = WellnessLog(user_id=user.id, date=datetime.now(timezone.utc).date(), mood_rating=5)
        prediction = Prediction(user_id=user.id, success_probability=0.8)

        db_session.add_all([patient_data, wellness_log, prediction])
        db_session.commit()

        # Test relationships
        assert user.patient_data is not None
        assert len(user.wellness_logs) == 1
        assert len(user.predictions) == 1

    def test_user_type_validation(self, db_session):
        """Test user type validation"""
        user = User(
            username='type_user',
            email='type@example.com',
            first_name='Type',
            last_name='User',
            user_type='invalid_type'  # Invalid type
        )
        user.set_password('password123')

        db_session.add(user)
        # This should work as we don't have strict validation in model
        db_session.commit()
        assert user.id is not None

    def test_unique_constraints(self, db_session):
        """Test unique constraints on username and email"""
        user1 = User(
            username='unique_user',
            email='unique@example.com',
            first_name='Unique',
            last_name='User',
            user_type='patient'
        )
        user1.set_password('password123')
        db_session.add(user1)
        db_session.commit()

        # Try to create duplicate username
        user2 = User(
            username='unique_user',  # Duplicate
            email='different@example.com',
            first_name='Different',
            last_name='User',
            user_type='patient'
        )
        user2.set_password('password123')

        with pytest.raises(Exception):  # Should raise IntegrityError
            db_session.add(user2)
            db_session.commit()

    def test_data_types(self, db_session):
        """Test correct data types are stored"""
        user = User(
            username='datatype_user',
            email='datatype@example.com',
            first_name='DataType',
            last_name='User',
            user_type='patient'
        )
        user.set_password('password123')
        db_session.add(user)
        db_session.commit()

        patient_data = PatientData(
            user_id=user.id,
            age=35,
            bmi=23.5,
            amh_level=3.2,
            fsh_level=5.8,
            previous_ivf_cycles=2
        )

        db_session.add(patient_data)
        db_session.commit()

        # Verify data types
        assert isinstance(patient_data.age, int)
        assert isinstance(patient_data.bmi, float)
        assert isinstance(patient_data.amh_level, float)
        assert patient_data.previous_ivf_cycles == 2
