from __future__ import annotations

import pytest

import uzbek_speech_entities.config as config_module
from uzbek_speech_entities.config import frontend_directory, packaged_resource_path


def _asset(name: str) -> str:
    return (frontend_directory() / name).read_text(encoding="utf-8")


def test_frontend_falls_back_to_packaged_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_source_checkout_root", lambda: None)

    assert frontend_directory() == packaged_resource_path("web")


def test_packaged_frontend_assets_match_the_checkout_copies() -> None:
    for name in ("index.html", "app.js", "styles.css"):
        assert packaged_resource_path("web", name).read_text(encoding="utf-8") == _asset(name)


def test_frontend_markup_exposes_required_local_controls() -> None:
    markup = _asset("index.html")
    for token in (
        'lang="uz"',
        'id="record-button"',
        'id="stop-button"',
        'id="audio-file"',
        'accept=".wav,.mp3,.m4a,.ogg,.webm,.flac,audio/*"',
        'id="audio-preview"',
        'id="analyze-button"',
        'id="clear-button"',
        'id="health-retry"',
        'id="raw-transcript"',
        'id="normalized-transcript"',
        'id="highlighted-transcript"',
        'id="time-total"',
        'href="#main-content"',
    ):
        assert token in markup
    assert "https://" not in markup
    assert "http://" not in markup


def test_frontend_script_keeps_untrusted_text_out_of_html_sinks() -> None:
    script = _asset("app.js")
    for unsafe_sink in ("innerHTML", "insertAdjacentHTML", "document.write"):
        assert unsafe_sink not in script
    for required_token in (
        "MediaRecorder",
        "getUserMedia",
        "URL.createObjectURL",
        "URL.revokeObjectURL",
        'fetch("/api/health"',
        'fetch("/api/analyze-audio"',
        'formData.append("file"',
        "AbortController",
        "replaceChildren",
        "createTextNode",
        "createElement",
        "safeEntities",
        "clearAll",
        '".ogg"',
    ):
        assert required_token in script


def test_frontend_recorder_stops_streams_and_invalidates_failed_recordings() -> None:
    script = _asset("app.js")
    stream_assignment = script.index("state.stream = stream;")
    recorder_construction = script.index("new MediaRecorder(stream")
    error_handler_start = script.index('recorder.addEventListener("error"')
    stop_handler_start = script.index('recorder.addEventListener("stop"')
    error_handler = script[error_handler_start:stop_handler_start]

    assert stream_assignment < recorder_construction
    assert "nextOperation();" in error_handler
    assert "resetTimer();" in error_handler
    assert "stopTracks();" in error_handler
    assert "resetAudio();" in error_handler


def test_frontend_hidden_state_overrides_component_display_rules() -> None:
    styles = _asset("styles.css")

    assert "[hidden] { display:none !important; }" in styles
