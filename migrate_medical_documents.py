"""Standalone command for the safe medical_document schema upgrade."""

import os

from sqlalchemy import create_engine

from database import db
from schema_migration import upgrade_database_schema
import models


def migrate(db_path=None):
    """Add missing columns in place; never drop or recreate the table."""
    if db_path is None:
        db_path = os.path.join("instance", "ivf_tracker.db")

    if not os.path.exists(db_path):
        print(f"Database not found at {os.path.abspath(db_path)}.")
        return

    engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
    added_columns = upgrade_database_schema(engine, db.metadata)
    engine.dispose()

    if added_columns:
        print(f"Added columns to medical_document: {', '.join(added_columns)}")
    else:
        print("medical_document is already up to date.")


if __name__ == "__main__":
    migrate()