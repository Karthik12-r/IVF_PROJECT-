import sqlite3
import os

for db_name in ['ivf_tracker.db', 'instance/ivf_tracker.db']:
    if os.path.exists(db_name):
        print(f"\n=== Database: {db_name} ===")
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute("PRAGMA table_info(medical_document)")
        cols = c.fetchall()
        if cols:
            for col in cols:
                print(f"  {col[1]} ({col[2]})")
        else:
            print("  Table 'medical_document' does not exist")
        conn.close()
    else:
        print(f"\nDatabase {db_name} not found")

