from vectra.observability import SQLiteLogger
import os
import time

DB_PATH = 'test_traces.db'
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

config = type('Config', (), {})()
config.enabled = True
config.sqlite_path = DB_PATH
config.project_id = 'test-project'
config.track_traces = True
config.track_metrics = False
config.track_logs = False
config.session_tracking = False

logger = SQLiteLogger(config)

logger.log_trace({
    'trace_id': 'trace-1',
    'span_id': 'span-1',
    'name': 'test-span',
    'start_time': int(time.time() * 1000),
    'end_time': int(time.time() * 1000) + 100,
    'provider': 'openai',
    'model_name': 'gpt-4'
})

import sqlite3
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT * FROM traces WHERE trace_id = 'trace-1'")
row = cursor.fetchone()

# Get column names
col_names = [description[0] for description in cursor.description]
row_dict = dict(zip(col_names, row))

print('Trace found:', row_dict)

if row_dict.get('provider') == 'openai' and row_dict.get('model_name') == 'gpt-4':
    print('Provider and Model Name verification passed!')
else:
    print('Verification failed:', row_dict)

conn.close()
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
