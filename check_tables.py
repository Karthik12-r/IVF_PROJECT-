import sqlite3

conn = sqlite3.connect('ivf_tracker.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tables:')
for table in tables:
    print(f'  {table[0]}')

# Also check if there are any records
cursor.execute("SELECT COUNT(*) FROM user")
user_count = cursor.fetchone()[0]
print(f'\nUser count: {user_count}')

conn.close()
