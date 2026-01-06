import json
import os
import threading
import webbrowser
import sqlite3
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .config import VectraConfig, ProviderType, ChunkingStrategy, RetrievalStrategy
from .telemetry import telemetry

def _get_db_connection(config_path):
    try:
        if not os.path.exists(config_path):
            return None
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        obs = cfg.get('observability', {})
        if not obs.get('enabled'):
            return None
            
        db_path = obs.get('sqlite_path', 'vectra-observability.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"Failed to open observability DB: {e}")
        return None

def _default_config():
    return {
        "embedding": {
            "provider": ProviderType.OPENAI.value,
            "api_key": "",
            "model_name": "text-embedding-3-small",
            "dimensions": None
        },
        "llm": {
            "provider": ProviderType.GEMINI.value,
            "api_key": "",
            "model_name": "gemini-1.5-pro-latest",
            "temperature": 0.0,
            "max_tokens": 1024,
            "base_url": None,
            "default_headers": None
        },
        "database": {
            "type": "prisma",
            "table_name": "Document",
            "column_map": {"content": "content", "vector": "vector", "metadata": "metadata"}
        },
        "chunking": {
            "strategy": ChunkingStrategy.RECURSIVE.value,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separators": ["\n\n", "\n", " ", ""]
        },
        "retrieval": {
            "strategy": RetrievalStrategy.NAIVE.value,
            "hybrid_alpha": 0.5
        },
        "reranking": {
            "enabled": False,
            "provider": "llm",
            "top_n": 5,
            "window_size": 20
        },
        "metadata": None,
        "query_planning": None,
        "grounding": None,
        "generation": None,
        "prompts": None,
        "tracing": None,
        "callbacks": []
    }

class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, path, content_type, folder='ui'):
        try:
            # Locate the ui directory relative to this file
            base_dir = os.path.dirname(__file__)
            file_path = os.path.join(base_dir, folder, path)
            
            with open(file_path, 'rb') as f:
                data = f.read()
                
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self.send_error(404)

    def do_GET(self):
        p = urlparse(self.path)
        
        # --- Dashboard Routes ---
        if p.path == "/dashboard":
            self.send_response(301)
            self.send_header('Location', '/dashboard/')
            self.end_headers()
            return
            
        if p.path == "/dashboard/":
            self._serve_static("index.html", "text/html; charset=utf-8", folder="dashboard")
            return
            
        if p.path.startswith("/dashboard/"):
            asset_name = p.path.replace("/dashboard/", "")
            content_type = "text/plain"
            if asset_name.endswith(".css"): content_type = "text/css"
            elif asset_name.endswith(".js"): content_type = "application/javascript"
            elif asset_name.endswith(".html"): content_type = "text/html"
            elif asset_name.endswith(".png"): content_type = "image/png"
            self._serve_static(asset_name, content_type, folder="dashboard")
            return

        # --- Observability API ---
        if p.path.startswith("/api/observability/"):
            print(f"DEBUG: Handling observability request: {p.path}")
            conn = _get_db_connection(self.server.config_path)
            if not conn:
                self._send_json(400, {"error": "Observability not enabled or DB not found"})
                return

            try:
                qs = parse_qs(p.query)
                project_id = qs.get('projectId', [None])[0]
                
                # Filter Logic
                where_clauses = []
                params = []
                if project_id and project_id != 'all':
                    where_clauses.append("project_id = ?")
                    params.append(project_id)
                
                where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                and_sql = "AND " + " AND ".join(where_clauses) if where_clauses else ""

                if p.path.endswith("/stats"):
                    cur = conn.cursor()
                    
                    # Total Requests
                    cur.execute(f"SELECT COUNT(*) as count FROM traces WHERE name = 'query_rag' {and_sql.replace('AND', 'WHERE', 1) if not where_sql else and_sql}", params)
                    total_req = cur.fetchone()['count']
                    
                    # Avg Latency
                    cur.execute(f"SELECT AVG(value) as val FROM metrics WHERE name = 'query_latency' {and_sql}", params)
                    avg_lat = cur.fetchone()['val'] or 0
                    
                    # Token Counts
                    cur.execute(f"SELECT SUM(value) as val FROM metrics WHERE name = 'prompt_chars' {and_sql}", params)
                    tokens_p = cur.fetchone()['val'] or 0
                    
                    # Error Rate
                    cur.execute(f"SELECT COUNT(*) as count FROM traces WHERE name = 'query_rag' AND status = 'error' {and_sql}", params)
                    error_count = cur.fetchone()['count']
                    error_rate = (error_count / total_req * 100) if total_req > 0 else 0
                    
                    cur.execute(f"SELECT SUM(value) as val FROM metrics WHERE name = 'completion_chars' {and_sql}", params)
                    tokens_c = cur.fetchone()['val'] or 0

                    # History
                    cur.execute(f"""
                        SELECT m.timestamp, m.value as latency, 
                        (SELECT value FROM metrics m2 WHERE m2.timestamp = m.timestamp AND m2.name = 'prompt_chars') + 
                        (SELECT value FROM metrics m3 WHERE m3.timestamp = m.timestamp AND m3.name = 'completion_chars') as tokens
                        FROM metrics m 
                        WHERE m.name = 'query_latency' {and_sql}
                        ORDER BY m.timestamp DESC LIMIT 50
                    """, params)
                    history = [dict(row) for row in cur.fetchall()]
                    history.reverse()

                    self._send_json(200, {
                        "totalRequests": total_req,
                        "avgLatency": avg_lat,
                        "totalPromptChars": tokens_p,
                        "totalCompletionChars": tokens_c,
                        "errorRate": error_rate,
                        "history": history
                    })
                
                elif p.path.endswith("/projects"):
                    cur = conn.cursor()
                    cur.execute("SELECT DISTINCT project_id FROM traces")
                    rows = cur.fetchall()
                    projects = [row['project_id'] for row in rows if row['project_id']]
                    self._send_json(200, projects)

                elif p.path.endswith("/traces"):
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM traces WHERE name = 'query_rag' ORDER BY start_time DESC LIMIT 50")
                    rows = [dict(row) for row in cur.fetchall()]
                    self._send_json(200, rows)

                elif "/traces/" in p.path:
                    trace_id = p.path.split("/")[-1]
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
                    rows = [dict(row) for row in cur.fetchall()]
                    self._send_json(200, rows)

                elif p.path.endswith("/sessions"):
                    cur = conn.cursor()
                    # Fix session filter
                    s_where = where_sql.replace('project_id', 'project_id') 
                    cur.execute(f"SELECT * FROM sessions {s_where} ORDER BY last_active DESC LIMIT 50", params)
                    rows = [dict(row) for row in cur.fetchall()]
                    
                    # Parse metadata and keep snake_case to match JS dashboard
                    results = []
                    for r in rows:
                        d = dict(r)
                        if d.get('metadata'):
                            try:
                                d['metadata'] = json.loads(d['metadata'])
                            except:
                                d['metadata'] = {}
                        else:
                            d['metadata'] = {}
                        results.append(d)
                        
                    self._send_json(200, results)
                
                else:
                    self._send_json(404, {"error": "Unknown endpoint"})

            except Exception as e:
                self._send_json(500, {"error": str(e)})
            finally:
                conn.close()
            return

        # --- Legacy Config UI ---
        if p.path == "/" or p.path == "/index.html":
            self._serve_static("index.html", "text/html; charset=utf-8")
            return
            
        if p.path == "/style.css":
            self._serve_static("style.css", "text/css")
            return
            
        if p.path == "/script.js":
            self._serve_static("script.js", "application/javascript")
            return
            
        if p.path == "/config":
            cfg_path = self.server.config_path
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    val = VectraConfig.model_validate(raw)
                    self._send_json(200, val.model_dump())
                    return
                except Exception as e:
                    self._send_json(400, {"error": str(e)})
                    return
            self._send_json(200, _default_config())
            return
        self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path)
        if p.path == "/config":
            ln = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(ln) if ln > 0 else b"{}"
            try:
                raw = json.loads(body.decode("utf-8"))
                val = VectraConfig.model_validate(raw)
                out = val.model_dump(exclude_none=True)
                os.makedirs(os.path.dirname(self.server.config_path) or ".", exist_ok=True)
                with open(self.server.config_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                self._send_json(200, {"message": "Saved"})
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return
        self.send_error(404)

def start(config_path, mode='webconfig', host="127.0.0.1", port=8765, open_browser=True):
    server = None
    while server is None:
        try:
            server = ThreadingHTTPServer((host, port), _Handler)
        except OSError:
            print(f"Port {port} is in use, trying {port + 1}...")
            port += 1
            if port > 65535:
                raise Exception("No available ports found")

    server.config_path = os.path.abspath(config_path)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f"http://{host}:{port}/"
    if mode == 'dashboard':
        url = f"http://{host}:{port}/dashboard"
    
    # Try to init telemetry if not already (it's safe to call multiple times)
    telemetry.init()
    telemetry.track('feature_used', {'feature': mode})

    print(f"Vectra WebConfig running at {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return server

