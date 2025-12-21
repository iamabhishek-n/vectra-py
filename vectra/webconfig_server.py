import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from .config import RAGConfig, ProviderType, ChunkingStrategy, RetrievalStrategy

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

    def _serve_static(self, path, content_type):
        try:
            # Locate the ui directory relative to this file
            base_dir = os.path.dirname(__file__)
            file_path = os.path.join(base_dir, 'ui', path)
            
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
                    val = RAGConfig.model_validate(raw)
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
                val = RAGConfig.model_validate(raw)
                out = val.model_dump(exclude_none=True)
                os.makedirs(os.path.dirname(self.server.config_path) or ".", exist_ok=True)
                with open(self.server.config_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                self._send_json(200, {"message": "Saved"})
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return
        self.send_error(404)

def start(config_path, host="127.0.0.1", port=8765, open_browser=True):
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
    print(f"Vectra WebConfig running at {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return server

