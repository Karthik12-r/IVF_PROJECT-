from main import app
from models import Clinic
from database import db

with app.app_context():
    clinics = Clinic.query.filter(Clinic.name.in_(['Nova IVF Fertility', 'Oasis Fertility', 'Iswarya Fertility Centre', 'Gaudium IVF Centre', 'Indira IVF'])).all()
    print('Updating coordinates...')
    for c in clinics:
        if c.city == 'Bangalore':
            lat, lng = 12.9716, 77.5946
        elif c.city == 'Hubli':
            lat, lng = 15.3647, 75.1240
        else:
            lat, lng = 12.9716, 77.5946  # default to Bangalore
        c.latitude = lat
        c.longitude = lng
        print(f'Updated {c.name} in {c.city}: lat={lat}, lng={lng}')
    db.session.commit()
    print('Coordinates updated successfully.')
