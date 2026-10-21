import sqlite3
import os

def migrate_database():
    db_path = 'ivf_tracker.db'

    if not os.path.exists(db_path):
        print("Database file not found. No migration needed.")
        return

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if extracted_text column exists
        cursor.execute("PRAGMA table_info(medical_document)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        if 'extracted_text' in column_names:
            print("extracted_text column already exists. No migration needed.")
            conn.close()
            return

        print("Migrating medical_document table to add extracted_text column...")

        # Create new table with extracted_text column
        cursor.execute('''
            CREATE TABLE medical_document_new (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                filename VARCHAR(255) NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(100),
                file_size INTEGER,
                description TEXT,
                extracted_text TEXT,
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user (id)
            )
        ''')

        # Copy data from old table to new table
        cursor.execute('''
            INSERT INTO medical_document_new (
                id, user_id, filename, original_filename,
                file_type, file_size, description, uploaded_at
            )
            SELECT id, user_id, filename, original_filename,
                   file_type, file_size, description, uploaded_at
            FROM medical_document
        ''')

        # Drop old table
        cursor.execute('DROP TABLE medical_document')

        # Rename new table to old name
        cursor.execute('ALTER TABLE medical_document_new RENAME TO medical_document')

        # Commit changes
        conn.commit()

        print("Migration completed successfully!")

    except Exception as e:
        print(f"Migration failed: {e}")
        conn.rollback()

    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()
