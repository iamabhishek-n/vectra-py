import sqlite3
import json
import uuid
import time
import os
import threading

class SQLiteLogger:
    def __init__(self, config):
        self.enabled = config.enabled
        if not self.enabled:
            return

        self.project_id = config.project_id
        self.track_metrics = config.track_metrics
        self.track_traces = config.track_traces
        self.track_logs = config.track_logs
        self.session_tracking = config.session_tracking
        self.db_path = config.sqlite_path
        
        self._buffer_lock = threading.Lock()
        self._trace_buffer = []
        self._metric_buffer = []
        self._log_buffer = []
        self._flush_timer = None
        self._is_closing = False
        
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()
            self._start_flush_timer()
        except Exception as e:
            print(f"Failed to initialize SQLite logger: {e}")
            self.enabled = False

    def _start_flush_timer(self):
        if self.enabled and not self._is_closing:
            self._flush_timer = threading.Timer(5.0, self._periodic_flush)
            self._flush_timer.daemon = True
            self._flush_timer.start()

    def _periodic_flush(self):
        self.flush()
        self._start_flush_timer()

    def _init_schema(self):
        cursor = self._conn.cursor()
        cursor.executescript("""
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    trace_id TEXT,
                    span_id TEXT,
                    parent_span_id TEXT,
                    name TEXT,
                    start_time INTEGER,
                    end_time INTEGER,
                    duration INTEGER,
                    status TEXT,
                    attributes TEXT,
                    input TEXT,
                    output TEXT,
                    error TEXT,
                    provider TEXT,
                    model_name TEXT
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    name TEXT,
                    value REAL,
                    timestamp INTEGER,
                    tags TEXT
                );
                CREATE TABLE IF NOT EXISTS logs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    level TEXT,
                    message TEXT,
                    timestamp INTEGER,
                    context TEXT
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    session_id TEXT,
                    user_id TEXT,
                    start_time INTEGER,
                    last_activity_time INTEGER,
                    metadata TEXT
                );
            """)
            
        # Migration: Add new columns if they don't exist
        try:
            cursor.execute("ALTER TABLE traces ADD COLUMN provider TEXT")
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute("ALTER TABLE traces ADD COLUMN model_name TEXT")
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def log_trace(self, trace):
        if not self.enabled or not self.track_traces:
            return
        with self._buffer_lock:
            self._trace_buffer.append(trace)
            if len(self._trace_buffer) >= 50:
                self.flush()

    def log_metric(self, metric):
        if not self.enabled or not self.track_metrics:
            return
        with self._buffer_lock:
            self._metric_buffer.append(metric)
            if len(self._metric_buffer) >= 50:
                self.flush()

    def flush(self):
        if not self.enabled:
            return
        with self._buffer_lock:
            if not self._trace_buffer and not self._metric_buffer and not self._log_buffer:
                return
            
            try:
                cursor = self._conn.cursor()
                if self._trace_buffer:
                    for trace in self._trace_buffer:
                        cursor.execute("""
                            INSERT INTO traces (id, project_id, trace_id, span_id, parent_span_id, name, start_time, end_time, duration, status, attributes, input, output, error, provider, model_name)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(uuid.uuid4()),
                            self.project_id,
                            trace.get('trace_id'),
                            trace.get('span_id'),
                            trace.get('parent_span_id'),
                            trace.get('name'),
                            trace.get('start_time'),
                            trace.get('end_time'),
                            (trace.get('end_time', 0) - trace.get('start_time', 0)),
                            trace.get('status', 'ok'),
                            json.dumps(trace.get('attributes', {})),
                            json.dumps(trace.get('input', {})),
                            json.dumps(trace.get('output', {})),
                            json.dumps(trace.get('error')) if trace.get('error') else None,
                            trace.get('provider'),
                            trace.get('model_name')
                        ))
                    self._trace_buffer = []

                if self._metric_buffer:
                    for metric in self._metric_buffer:
                        cursor.execute("""
                            INSERT INTO metrics (id, project_id, name, value, timestamp, tags)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            str(uuid.uuid4()),
                            self.project_id,
                            metric.get('name'),
                            metric.get('value'),
                            int(time.time() * 1000),
                            json.dumps(metric.get('tags', {}))
                        ))
                    self._metric_buffer = []
                
                if self._log_buffer:
                    for log in self._log_buffer:
                         cursor.execute("""
                            INSERT INTO logs (id, project_id, level, message, timestamp, context)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            str(uuid.uuid4()),
                            self.project_id,
                            log.get('level'),
                            log.get('message'),
                            log.get('timestamp', int(time.time() * 1000)),
                            json.dumps(log.get('context', {}))
                        ))
                    self._log_buffer = []

                self._conn.commit()
            except Exception as e:
                print(f"Failed to flush SQLite logs: {e}")

    def log(self, level, message, context=None):
        if not self.enabled or not self.track_logs:
            return
        with self._buffer_lock:
            self._log_buffer.append({'level': level, 'message': message, 'context': context or {}})
            if len(self._log_buffer) >= 50:
                self.flush()

    def update_session(self, session_id, user_id=None, metadata=None):
        if not self.enabled or not self.session_tracking:
            return
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT id FROM sessions WHERE session_id = ?", (session_id,))
            existing = cursor.fetchone()
            
            now = int(time.time() * 1000)
            if existing:
                cursor.execute("""
                    UPDATE sessions 
                    SET last_activity_time = ?, metadata = ?
                    WHERE session_id = ?
                """, (now, json.dumps(metadata or {}), session_id))
            else:
                cursor.execute("""
                    INSERT INTO sessions (id, project_id, session_id, user_id, start_time, last_activity_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    self.project_id,
                    session_id,
                    user_id,
                    now,
                    now,
                    json.dumps(metadata or {})
                ))
            self._conn.commit()
        except Exception as e:
            print(f"Failed to update session: {e}")
            
    def close(self):
        if hasattr(self, '_conn'):
            self._is_closing = True
            if self._flush_timer:
                self._flush_timer.cancel()
            self.flush()
            self._conn.close()

    def __del__(self):
        self.close()
