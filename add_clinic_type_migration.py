#!/usr/bin/env python3
"""
Migration script to add clinic_type and zip_code columns to the clinic table.
Run this script once after updating the models.py file.
"""

import os
import sys
from database import db
from models import Clinic

def run_migration():
    """Add clinic_type and zip_code columns to existing clinic table."""

    # Set up database connection
    database_url = os.environ.get("DATABASE_URL")
    if not database_url or database_url.strip() == "":
        database_url = "sqlite:///ivf_tracker.db"

    # Configure SQLAlchemy
    from flask import Flask
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    with app.app_context():
        try:
            # Check if columns already exist
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            columns = [col['name'] for col in inspector.get_columns('clinic')]

            if 'clinic_type' not in columns:
                # Add clinic_type column using text() for raw SQL
                from sqlalchemy import text
                db.session.execute(text("ALTER TABLE clinic ADD COLUMN clinic_type VARCHAR(50) DEFAULT 'IVF'"))
                print("✓ Added clinic_type column to clinic table")

            if 'zip_code' not in columns:
                # Add zip_code column using text() for raw SQL
                from sqlalchemy import text
                db.session.execute(text("ALTER TABLE clinic ADD COLUMN zip_code VARCHAR(10)"))
                print("✓ Added zip_code column to clinic table")

            db.session.commit()
            print("Migration completed successfully!")

        except Exception as e:
            db.session.rollback()
            print(f"Migration failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    run_migration()
