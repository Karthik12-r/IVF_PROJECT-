import os
import sys

# Check DATABASE_URL
print(f"DATABASE_URL env: {os.environ.get('DATABASE_URL', 'not set')}")

# Check which db files exist and their sizes
for db_name in ['ivf_tracker.db', 'instance/ivf_tracker.db']:
    if os.path.exists(db_name):
        size = os.path.getsize(db_name)
