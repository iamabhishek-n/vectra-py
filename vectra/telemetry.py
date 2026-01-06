import os
import json
import uuid
import time
import threading
import platform
import atexit
from pathlib import Path
from datetime import datetime

try:
    from importlib.metadata import version
    SDK_VERSION = version("vectra-rag-py")
except Exception:
    SDK_VERSION = "0.9.8"

TELEMETRY_DIR = Path.home() / ".vectra"
TELEMETRY_FILE = TELEMETRY_DIR / "telemetry.json"

BATCH_SIZE = 10
FLUSH_INTERVAL_SEC = 60

API_ENDPOINT = os.getenv(
    "VECTRA_TELEMETRY_ENDPOINT",
    "https://thwcefdrkimerqztvfjj.supabase.co/functions/v1/vectra-collect"
)


class TelemetryManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.distinct_id = None
        self.queue = []
        self.enabled = True
        self.lock = threading.Lock()
        self.worker = None
        self.stop_event = threading.Event()

        self.global_properties = {
            "sdk": "vectra-python",
            "sdk_version": SDK_VERSION,
            "language": "python",
            "runtime": f"python-{platform.python_version()}",
            "os": platform.system().lower(),
            "ci": os.getenv("CI", "").lower() == "true",
            "telemetry_version": 1,
        }

        self._initialized = True

    def init(self, config=None):
        telemetry_cfg = {}

        if config:
            if hasattr(config, "telemetry"):
                telemetry_cfg = (
                    config.telemetry
                    if isinstance(config.telemetry, dict)
                    else config.telemetry.dict()
                )
            elif isinstance(config, dict):
                telemetry_cfg = config.get("telemetry", {})

        if telemetry_cfg.get("enabled") is False:
            self.enabled = False
            return

        if os.getenv("VECTRA_TELEMETRY_DISABLED") == "1" or os.getenv("DO_NOT_TRACK") == "1":
            self.enabled = False
            return

        self._load_identity()
        self._start_worker()

    def _load_identity(self):
        try:
            TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)

            if TELEMETRY_FILE.exists():
                with open(TELEMETRY_FILE, "r") as f:
                    data = json.load(f)
                    if "distinct_id" in data:
                        self.distinct_id = data["distinct_id"]
                        return

            self.distinct_id = f"anon_{uuid.uuid4()}"
            with open(TELEMETRY_FILE, "w") as f:
                json.dump({"distinct_id": self.distinct_id}, f, indent=2)

        except Exception:
            self.enabled = False

    def track(self, event: str, properties: dict | None = None):
        if not self.enabled or not self.distinct_id:
            return

        payload = {
            "event": event,
            "distinct_id": self.distinct_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "properties": {
                **self.global_properties,
                **(properties or {}),
            },
        }

        if os.getenv("VECTRA_TELEMETRY_DEBUG"):
            print(f"[Telemetry] Queued event: {event}")

        with self.lock:
            self.queue.append(payload)

    def _start_worker(self):
        if self.worker:
            return

        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self.worker.start()

    def _worker_loop(self):
        while not self.stop_event.is_set():
            time.sleep(FLUSH_INTERVAL_SEC)
            self._flush_background()

    def _flush_background(self):
        if os.getenv("VECTRA_TELEMETRY_DEBUG"):
            print("[Telemetry] flushing background...")
        with self.lock:
            if not self.queue:
                if os.getenv("VECTRA_TELEMETRY_DEBUG"):
                    print("[Telemetry] Queue empty, nothing to flush")
                return
            batch = self.queue[:]
            self.queue.clear()

       

        try:
            import requests

            if os.getenv("VECTRA_TELEMETRY_DEBUG"):
                print(f"[Telemetry] Sending {len(batch)} events to {API_ENDPOINT}")

            response = requests.post(
                API_ENDPOINT,
                headers={
                   'Content-Type': 'application/json'
                },
                json=batch,
                timeout=6,
            )

            # print response
            if os.getenv("VECTRA_TELEMETRY_DEBUG"):
                print(f"[Telemetry] Response: {response.status_code} {response.text}")

            if os.getenv("VECTRA_TELEMETRY_DEBUG"):
                print(f"[Telemetry] Flushed {len(batch)} events")

        except Exception as e:
            if os.getenv("VECTRA_TELEMETRY_DEBUG"):
                print(f"[Telemetry] Flush error: {e}")

    def shutdown(self):
        self.stop_event.set()
        self._flush_background()

telemetry = TelemetryManager()
atexit.register(telemetry.shutdown)
