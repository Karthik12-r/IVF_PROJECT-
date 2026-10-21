from main import app
from models import User, db

with app.app_context():
    user = User.query.filter_by(user_type='partner').first()
    if user is None:
        user = User(
            username='partnerdemo',
            email='partnerdemo@test.com',
            first_name='Partner',
            last_name='User',
            user_type='partner'
        )
        user.set_password('test123')
        db.session.add(user)
        db.session.commit()

    client = app.test_client()
    with client.session_transaction() as sess:
        sess['user_id'] = user.id
        sess['user_type'] = 'partner'

    resp = client.get('/partner/profile')
    print('status', resp.status_code)
    print(resp.data[:400].decode('utf-8', errors='ignore'))
