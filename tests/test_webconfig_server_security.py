import json
import os
import tempfile
import urllib.request
import urllib.error

from vectra.webconfig_server import start


def _request(url, headers=None, method="GET", data=None):
    req = urllib.request.Request(url, headers=headers or {}, method=method, data=data)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


class TestWebconfigServerSecurity:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="vectra-webconfig-test-")
        self.config_path = os.path.join(self.tmp_dir, "vectra-config.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump({"embedding": {"provider": "openai", "api_key": "sk-secret-123", "model_name": "x"},
                       "llm": {"provider": "openai", "api_key": "sk-secret-123", "model_name": "x"},
                       "database": {"type": "postgres"}}, f)
        self.server = start(self.config_path, "webconfig", host="127.0.0.1", port=0, open_browser=False)
        self.port = self.server.server_address[1]
        self.token = self.server.auth_token
        self.base = f"http://127.0.0.1:{self.port}"

    def teardown_method(self):
        self.server.shutdown()
        self.server.server_close()

    def test_binds_to_loopback(self):
        assert self.server.server_address[0] == "127.0.0.1"

    def test_blocks_path_traversal_on_dashboard_asset_route(self):
        status, body = _request(f"{self.base}/dashboard/../../../../../../etc/passwd")
        assert status != 200
        assert "root:" not in body

    def test_blocks_path_traversal_with_url_encoded_dots(self):
        status, body = _request(f"{self.base}/dashboard/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/etc/passwd")
        assert status != 200
        assert "root:" not in body

    def test_rejects_get_config_with_no_token(self):
        status, _ = _request(f"{self.base}/config")
        assert status == 401

    def test_rejects_get_config_with_wrong_token(self):
        status, _ = _request(f"{self.base}/config", headers={"X-Vectra-Token": "wrong"})
        assert status == 401

    def test_allows_get_config_with_correct_token(self):
        status, body = _request(f"{self.base}/config", headers={"X-Vectra-Token": self.token})
        assert status == 200
        assert "sk-secret-123" in body

    def test_rejects_post_config_with_no_token_and_does_not_write(self):
        before = open(self.config_path, "r", encoding="utf-8").read()
        payload = json.dumps({"embedding": {"provider": "openai", "api_key": "PWNED", "model_name": "x"},
                               "llm": {"provider": "openai", "api_key": "PWNED", "model_name": "x"},
                               "database": {"type": "postgres"}}).encode("utf-8")
        status, _ = _request(f"{self.base}/config", method="POST",
                              headers={"Content-Type": "application/json"}, data=payload)
        assert status == 401
        assert open(self.config_path, "r", encoding="utf-8").read() == before

    def test_allows_post_config_with_correct_token_and_writes(self):
        payload = json.dumps({"embedding": {"provider": "openai", "api_key": "sk-new-key", "model_name": "x"},
                               "llm": {"provider": "openai", "api_key": "sk-new-key", "model_name": "x"},
                               "database": {"type": "postgres"}}).encode("utf-8")
        status, _ = _request(f"{self.base}/config", method="POST",
                              headers={"Content-Type": "application/json", "X-Vectra-Token": self.token},
                              data=payload)
        assert status == 200
        with open(self.config_path, "r", encoding="utf-8") as f:
            written = json.load(f)
        assert written["embedding"]["api_key"] == "sk-new-key"

    def test_rejects_observability_api_with_no_token(self):
        status, _ = _request(f"{self.base}/api/observability/stats")
        assert status == 401

    def test_injects_auth_token_into_served_html(self):
        status, body = _request(f"{self.base}/dashboard/")
        assert status == 200
        assert self.token in body
