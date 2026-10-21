from datetime import datetime
from main import app
from models import User, db, PartnerConnection, PartnerSharingPermission

with app.app_context():
    patient = User.query.filter_by(email='patientdemo@test.com').first()
    if patient is None:
        patient = User(
            username='patientdemo',
            email='patientdemo@test.com',
            first_name='Demo',
            last_name='Patient',
            user_type='patient'
        )
        patient.set_password('demo123')
        db.session.add(patient)
        db.session.commit()

    partner = User.query.filter_by(email='partnerdemo@test.com').first()
    if partner is None:
        partner = User(
            username='partnerdemo',
            email='partnerdemo@test.com',
            first_name='Demo',
            last_name='Partner',
            user_type='partner'
        )
        partner.set_password('demo123')
        db.session.add(partner)
        db.session.commit()

    connection = PartnerConnection.query.filter_by(
        patient_id=patient.id,
        partner_id=partner.id,
        status='active'
    ).first()

    if connection is None:
        connection = PartnerConnection(
            patient_id=patient.id,
            partner_id=partner.id,
            status='active',
            accepted_at=datetime.utcnow()
        )
        db.session.add(connection)
        db.session.commit()

        permissions = PartnerSharingPermission(
            connection_id=connection.id,
            treatment_shared=True,
            appointments_shared=True,
            medications_shared=True,
            wellness_shared=True,
            mood_shared=True,
            nutrition_shared=True,
            documents_shared=False,
            doctor_notes_shared=False,
            messages_enabled=True,
            tasks_enabled=True,
        )
        db.session.add(permissions)
        db.session.commit()
        print('NEW_CONNECTION_CREATED')
    else:
        print('ACTIVE_CONNECTION_ALREADY_PRESENT')

    client = app.test_client()

    with client.session_transaction() as sess:
        sess['user_id'] = partner.id
        sess['user_type'] = 'partner'

    partner_dashboard = client.get('/partner/dashboard')
    patient_dashboard = client.get('/patient/dashboard')
    print('PARTNER_DASHBOARD_STATUS', partner_dashboard.status_code)
    print('PATIENT_DASHBOARD_STATUS', patient_dashboard.status_code)
    print('PATIENT_EMAIL', patient.email)
    print('PARTNER_EMAIL', partner.email)
    print('CONNECTION_STATUS', connection.status)
    print('PARTNER_CONNECTED', True)
