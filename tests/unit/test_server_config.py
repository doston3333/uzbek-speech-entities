from __future__ import annotations

import pytest

from uzbek_speech_entities.api.server import server_address


def test_server_address_uses_config_and_explicit_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APP_HOST", raising=False)
    monkeypatch.delenv("APP_PORT", raising=False)
    assert server_address() == ("127.0.0.1", 8000)

    monkeypatch.setenv("APP_HOST", "127.0.0.2")
    monkeypatch.setenv("APP_PORT", "8766")
    assert server_address() == ("127.0.0.2", 8766)


@pytest.mark.parametrize("value", ["not-a-port", "0", "65536"])
def test_server_address_rejects_invalid_port_override(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("APP_PORT", value)

    with pytest.raises(ValueError, match="port|APP_PORT"):
        server_address()
