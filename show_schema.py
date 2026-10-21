import sqlite3

conn = sqlite3.connect('ivf_tracker.db')
cursor = conn.cursor()
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
schemas = cursor.fetchall()
with open('schema_output.txt', 'w') as f:
    for schema in schemas:
        f.write(schema[0] + '\n---\n')
conn.close()
print("Schema written to schema_output.txt")
