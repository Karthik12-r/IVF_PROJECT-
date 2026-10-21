from models import *
from database import db
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ivf_tracker.db'
db.init_app(app)

with app.app_context():
    # Drop and recreate the medical_document table
    db.drop_all()
    db.create_all()
    print('Database recreated successfully')
