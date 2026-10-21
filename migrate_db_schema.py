"""
Migration script to update database schema for new features.
Handles SQLite limitations by recreating tables with new columns.
"""
import os
import sys
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'ivf_tracker.db')

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. It will be created automatically when the app starts.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if medical_document table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='medical_document'")
    if not cursor.fetchone():
        print("medical_document table does not exist yet. New tables will be created by SQLAlchemy.")
        conn.close()
        return

    # Check if new columns already exist
    cursor.execute("PRAGMA table_info(medical_document)")
    columns = [col[1] for col in cursor.fetchall()]
    
    new_columns = ['title', 'category', 'version', 'uploaded_by', 'tags']
    missing_columns = [c for c in new_columns if c not in columns]

    if not missing_columns:
        print("All new columns already exist in medical_document.")
    else:
        print(f"Adding missing columns to medical_document: {missing_columns}")
        
        # SQLite doesn't support multiple ALTER TABLE ADD COLUMN in one statement easily
        # We'll recreate the table
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='medical_document'")
        create_sql = cursor.fetchone()[0]
        
        # Backup existing data
        cursor.execute("SELECT * FROM medical_document")
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        
        print(f"Backing up {len(rows)} documents...")
        
        # Drop old table and create new one with extra columns
        cursor.execute("ALTER TABLE medical_document RENAME TO medical_document_old")
        
        # Create new table with all columns
        new_create_sql = create_sql.rstrip().rstrip(')')
        if 'uploaded_at' not in new_create_sql:
            new_create_sql += ",\n    uploaded_at DATETIME"
        new_create_sql += ",\n    title VARCHAR(255),\n    category VARCHAR(50) DEFAULT 'other',\n    version INTEGER DEFAULT 1,\n    uploaded_by VARCHAR(10) DEFAULT 'patient',\n    tags VARCHAR(255)\n)"
        
        cursor.execute(new_create_sql)
        
        # Restore data
        placeholders = ', '.join(['?' for _ in column_names])
        insert_sql = f"INSERT INTO medical_document ({', '.join(column_names)}) VALUES ({placeholders})"
        for row in rows:
            cursor.execute(insert_sql, row)
        
        # Drop old table
        cursor.execute("DROP TABLE medical_document_old")
        
        print(f"Migrated {len(rows)} documents successfully.")

    # Create new tables if they don't exist
    new_tables = {
        'notification': """
            CREATE TABLE IF NOT EXISTS notification (
                id INTEGER NOT NULL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                type VARCHAR(20) DEFAULT 'info',
                category VARCHAR(30) DEFAULT 'system',
                is_read BOOLEAN DEFAULT 0,
                link VARCHAR(255),
                created_at DATETIME,
                FOREIGN KEY(user_id) REFERENCES user (id)
            )
        """,
        'document_version': """
            CREATE TABLE IF NOT EXISTS document_version (
                id INTEGER NOT NULL PRIMARY KEY,
                document_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                filename VARCHAR(255) NOT NULL,
                uploaded_at DATETIME,
                change_note TEXT,
                FOREIGN KEY(document_id) REFERENCES medical_document (id)
            )
        """,
        'document_annotation': """
            CREATE TABLE IF NOT EXISTS document_annotation (
                id INTEGER NOT NULL PRIMARY KEY,
                document_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                annotation_text TEXT NOT NULL,
                created_at DATETIME,
                FOREIGN KEY(document_id) REFERENCES medical_document (id),
                FOREIGN KEY(doctor_id) REFERENCES user (id)
            )
        """,
        'document_share': """
            CREATE TABLE IF NOT EXISTS document_share (
                id INTEGER NOT NULL PRIMARY KEY,
                document_id INTEGER NOT NULL,
                shared_with_user_id INTEGER NOT NULL,
                shared_by_user_id INTEGER NOT NULL,
                shared_at DATETIME,
                can_view BOOLEAN DEFAULT 1,
                FOREIGN KEY(document_id) REFERENCES medical_document (id),
                FOREIGN KEY(shared_with_user_id) REFERENCES user (id),
                FOREIGN KEY(shared_by_user_id) REFERENCES user (id)
            )
        """
    }

    for table_name, create_sql in new_tables.items():
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            print(f"Creating new table: {table_name}")
            cursor.execute(create_sql)
        else:
            print(f"Table {table_name} already exists.")

    conn.commit()
    conn.close()
    print("Migration completed successfully!")

if __name__ == '__main__':
    migrate()
