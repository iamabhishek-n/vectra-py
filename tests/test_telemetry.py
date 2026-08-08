import os
import importlib


def _fresh_telemetry_module():
    import vectra.telemetry as telemetry_module
    importlib.reload(telemetry_module)
    return telemetry_module


class TestTelemetryDefaultOff:
    def test_disabled_by_construction_before_init(self):
        mod = _fresh_telemetry_module()
        assert mod.telemetry.enabled is False

    def test_stays_disabled_when_init_called_with_no_config(self):
        mod = _fresh_telemetry_module()
        mod.telemetry.init(None)
        assert mod.telemetry.enabled is False

    def test_stays_disabled_when_telemetry_enabled_is_omitted(self):
        mod = _fresh_telemetry_module()
        mod.telemetry.init({"telemetry": {}})
        assert mod.telemetry.enabled is False

    def test_enables_only_when_telemetry_enabled_is_explicitly_true(self, tmp_path, monkeypatch):
        mod = _fresh_telemetry_module()
        monkeypatch.setattr(mod, "TELEMETRY_DIR", tmp_path)
        monkeypatch.setattr(mod, "TELEMETRY_FILE", tmp_path / "telemetry.json")
        mod.telemetry.init({"telemetry": {"enabled": True}})
        assert mod.telemetry.enabled is True

    def test_stays_disabled_when_env_var_disables_even_if_config_enables(self, monkeypatch):
        mod = _fresh_telemetry_module()
        monkeypatch.setenv("VECTRA_TELEMETRY_DISABLED", "1")
        mod.telemetry.init({"telemetry": {"enabled": True}})
        assert mod.telemetry.enabled is False
