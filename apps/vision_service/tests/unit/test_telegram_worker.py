from __future__ import annotations

import importlib
import sys
import types


def _load_worker_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy-chat")
    if "apps.telegram_worker.main" in sys.modules:
        del sys.modules["apps.telegram_worker.main"]
    if "redis" not in sys.modules:
        sys.modules["redis"] = types.SimpleNamespace(Redis=object)
    return importlib.import_module("apps.telegram_worker.main")


def test_build_caption_from_v1_payload(monkeypatch):
    worker = _load_worker_module(monkeypatch)

    payload = {
        "camera_id": "cam_001",
        "event_id": "evt-abc",
        "timestamp": 1710000000,
        "confidence": 0.876,
    }
    caption = worker._build_caption(payload)

    assert "cam_001" in caption
    assert "evt-abc" in caption
    assert "Độ tin cậy" in caption


def test_extract_snapshot_path_prefers_v1_field(monkeypatch):
    worker = _load_worker_module(monkeypatch)

    payload = {"snapshot_path": "/workspace/storage/snapshots/a.jpg"}
    assert worker._extract_snapshot_path(payload) == "/workspace/storage/snapshots/a.jpg"


def test_send_photo_fallbacks_to_text_when_file_missing(monkeypatch):
    worker = _load_worker_module(monkeypatch)
    captured: list[str] = []

    monkeypatch.setattr(worker.os.path, "exists", lambda _path: False)

    def _fake_send_text(text: str, bot_token: str, chat_id: str):
        captured.append(f"{text}|{bot_token}|{chat_id}")

    monkeypatch.setattr(worker, "send_text_to_telegram", _fake_send_text)

    worker.send_photo_to_telegram(
        "/workspace/storage/snapshots/missing.jpg",
        "caption",
        "token",
        "chat",
    )

    assert captured
    assert captured[0].startswith("caption")
