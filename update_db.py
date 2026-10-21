from database import db
from models import *
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ivf_tracker.db'
db.init_app(app)

with app.app_context():
    db.create_all()
    print('Database tables created/updated')
