import sqlite3

conn = sqlite3.connect('ivf_tracker.db')
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(medical_document)")
columns = cursor.fetchall()

print("Medical Document Table Schema:")
for col in columns:
    print(f"  {col[1]}: {col[2]}")

# Check if extracted_text column exists
column_names = [col[1] for col in columns]
if 'extracted_text' in column_names:
    print("\nextracted_text column EXISTS")
else:
    print("\nextracted_text column DOES NOT EXIST")

conn.close()
