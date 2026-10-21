import sqlite3

conn = sqlite3.connect('ivf_tracker.db')
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(medical_document)")
columns = cursor.fetchall()

print("Medical Document Table Schema:")
for col in columns:
    print(f"  {col[1]}: {col[2]}")

conn.close()
