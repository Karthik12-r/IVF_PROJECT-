import sqlite3
import os

# Find the database file
db_paths = ['ivf_tracker.db', 'instance/ivf_tracker.db']
db_path = None
for p in db_paths:
    if os.path.exists(p):
        db_path = p
        break

if not db_path:
    print("Database not found!")
    exit(1)

print(f"Using database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(medical_document)")
cols = cursor.fetchall()
print("\nCurrent columns in medical_document:")
for c in cols:
    print(f"  {c[1]} ({c[2]})")

conn.close()

