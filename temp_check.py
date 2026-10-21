import sqlite3

conn = sqlite3.connect('ivf_tracker.db')
cursor = conn.cursor()

cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print('Tables:', [t[0] for t in tables])

if 'clinic' in [t[0] for t in tables]:
    cursor.execute('SELECT COUNT(*) FROM clinic')
    count = cursor.fetchone()[0]
    print('Clinics in DB:', count)
else:
    print('Clinic table does not exist')

conn.close()
