"""
Fix script to add missing columns to medical_document table.
Targets the active database (instance/ivf_tracker.db).
"""
import sqlite3
import os

# The app is using instance/ivf_tracker.db (confirmed by schema check)
DB_PATH = 'instance/ivf_tracker.db'

if not os.path.exists(DB_PATH):
    print(f"Database not found at {DB_PATH}")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check current columns
cursor.execute("PRAGMA table_info(medical_document)")
columns = [col[1] for col in cursor.fetchall()]
print(f"Current columns: {columns}")

# Columns to add
new_columns = {
    'title': 'VARCHAR(255)',
    'category': "VARCHAR(50) DEFAULT 'other'",
    'version': 'INTEGER DEFAULT 1',
    'uploaded_by': "VARCHAR(10) DEFAULT 'patient'",
    'tags': 'VARCHAR(255)'
}

added = []
for col_name, col_type in new_columns.items():
    if col_name not in columns:
        try:
            cursor.execute(f"ALTER TABLE medical_document ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
            print(f"Added column: {col_name} ({col_type})")
        except Exception as e:
            print(f"Error adding {col_name}: {e}")
    else:
        print(f"Column {col_name} already exists")

conn.commit()

# Verify
cursor.execute("PRAGMA table_info(medical_document)")
updated_columns = [col[1] for col in cursor.fetchall()]
print(f"\nUpdated columns: {updated_columns}")

conn.close()

if added:
    print(f"\nSuccessfully added {len(added)} column(s): {', '.join(added)}")
else:
    print("\nNo new columns needed to be added.")

