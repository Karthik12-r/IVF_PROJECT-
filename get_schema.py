import sqlite3

conn = sqlite3.connect('ivf_tracker.db')
cursor = conn.cursor()
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
schemas = cursor.fetchall()
for schema in schemas:
    print(schema[0])
    print('---')
conn.close()
