"""Test emoji fallback logic for GUI status icons."""
import importlib
import sys

import pytest  # type: ignore

pytestmark = pytest.mark.gui


def test_icon_fallback_ascii(monkeypatch):
    monkeypatch.setenv("DISABLE_EMOJI", "1")
    # Reload module to apply env var
    sys.modules.pop("machine_learning_model.gui.icon_utils", None)
    mod = importlib.import_module("machine_learning_model.gui.icon_utils")
    completed = mod.icon_for_status("COMPLETED")
    not_started = mod.icon_for_status("NOT_STARTED")
    assert completed.startswith("[") and completed.endswith("]")
    assert not_started.startswith("[") and not_started.endswith("]")
