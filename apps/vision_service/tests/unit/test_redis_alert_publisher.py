from __future__ import annotations

import json

from apps.vision_service.src.services.redis_alert_publisher import RedisAlertPublisher
from apps.vision_service.src.settings import RootSettings


def _build_settings() -> RootSettings:
    return RootSettings.model_validate(
        {
            "app": {"name": "app", "env": "dev", "log_level": "INFO", "gpu_id": 0},
            "storage": {"logs_dir": "/tmp/logs"},
            "events": {"output_file": "/tmp/logs/events.jsonl"},
            "streams": {"scan_camera_dir": "/tmp/cameras"},
            "pipeline": {
                "streammux_width": 960,
                "streammux_height": 544,
                "batched_push_timeout_usec": 40000,
                "max_sources": 16,
                "sink": "rtmp",
                "frame_log_interval_sec": 5.0,
            },
            "infer": {
                "enabled": True,
                "config_file": "/tmp/model.txt",
                "unique_id": 1,
                "summary_interval_sec": 5.0,
                "emit_frame_events": False,
            },
            "visualization": {
                "enabled": True,
                "display_text": True,
                "display_bbox": True,
                "display_clock": False,
                "osd_process_mode": 0,
            },
            "rtsp": {
                "enabled": False,
                "host": "127.0.0.1",
                "udp_port": 5400,
                "rtsp_port": 8554,
                "mount_point": "/vision",
                "codec": "h264",
                "bitrate": 2500000,
                "iframe_interval": 15,
                "payload_type": 96,
                "rtp_mtu": 1400,
                "udp_buffer_size": 2097152,
                "sps_pps_interval": -1,
                "rtsp_repay_enabled": True,
                "rtsp_repay_jitter_latency_ms": 0,
                "rtsp_repay_jitter_drop_on_latency": True,
                "rtsp_repay_leaky_queue_enabled": True,
                "rtsp_transport": "tcp",
                "udpsink_sync": True,
                "udpsink_async": False,
                "udpsink_qos": False,
                "debug_h264_output_file": "",
            },
            "rtmp": {
                "enabled": True,
                "location": "rtmp://mediamtx:1935/vision1 live=1",
                "sink_sync": True,
                "sink_async": False,
                "streamable_mux": True,
            },
            "tracker": {
                "enabled": True,
                "gpu_id": 0,
                "tracker_width": 640,
                "tracker_height": 640,
                "ll_lib_file": "/tmp/lib.so",
                "ll_config_file": "/tmp/tracker.yml",
                "display_tracking_id": False,
            },
            "metrics": {"enabled": True, "port": 9100},
            "telegram": {
                "enabled": True,
                "snapshot_source": "probe",
                "redis_host": "redis",
                "redis_port": 6379,
                "redis_topic": "helmet_violations",
                "cooldown_sec": 0.0,
                "min_consecutive_no_helmet_frames": 3,
                "snapshot_dir": "/tmp/snapshots",
                "snapshot_rtmp_url": "rtmp://mediamtx:1935/vision1",
                "snapshot_hls_url": "http://mediamtx:8888/vision1/index.m3u8",
            },
            "cameras": [],
        }
    )


class _FakeRedisClient:
    def __init__(self, fail_publish: bool = False):
        self.fail_publish = fail_publish
        self.published: list[tuple[str, str]] = []

    def publish(self, topic: str, payload: str) -> None:
        if self.fail_publish:
            raise RuntimeError("redis down")
        self.published.append((topic, payload))


def test_process_event_publishes_snapshot_path_when_source_is_rtmp(monkeypatch):
    settings = _build_settings()
    settings.telegram.snapshot_source = "rtmp"
    publisher = RedisAlertPublisher(settings, start_thread=False)
    fake_client = _FakeRedisClient()

    monkeypatch.setattr(
        publisher,
        "_grab_snapshot_from_cache",
        lambda camera_id, event_id: f"/workspace/storage/snapshots/violation_{camera_id}_{event_id}.jpg",
    )
    monkeypatch.setattr(publisher, "_save_snapshot_from_probe_frame", lambda **_kwargs: None)

    event = publisher._normalize_event(
        {
            "event_id": "evt-1",
            "camera_id": "cam_01",
            "confidence": 0.91,
            "frame_num": 42,
            "bbox": [1, 2, 3, 4],
        }
    )
    out_client = publisher._process_event(fake_client, event)

    assert out_client is fake_client
    assert len(fake_client.published) == 1
    topic, raw_payload = fake_client.published[0]
    payload = json.loads(raw_payload)

    assert topic == "helmet_violations"
    assert payload["event_id"] == "evt-1"
    assert payload["snapshot_path"].endswith("violation_cam_01_evt-1.jpg")


def test_process_event_fallback_without_snapshot(monkeypatch):
    publisher = RedisAlertPublisher(_build_settings(), start_thread=False)
    fake_client = _FakeRedisClient()

    def _stream_should_not_be_called(*_args, **_kwargs):
        raise AssertionError("stream snapshot must not be used in probe mode")

    monkeypatch.setattr(publisher, "_grab_snapshot_from_cache", _stream_should_not_be_called)
    monkeypatch.setattr(publisher, "_save_snapshot_from_probe_frame", lambda **_kwargs: None)

    event = publisher._normalize_event({"event_id": "evt-2", "camera_id": "cam_02"})
    publisher._process_event(fake_client, event)

    topic, raw_payload = fake_client.published[0]
    payload = json.loads(raw_payload)

    assert topic == "helmet_violations"
    assert payload["snapshot_path"] == ""


def test_publish_payload_reconnects_when_first_client_fails(monkeypatch):
    publisher = RedisAlertPublisher(_build_settings(), start_thread=False)
    first_client = _FakeRedisClient(fail_publish=True)
    second_client = _FakeRedisClient()

    monkeypatch.setattr(publisher, "_connect_redis", lambda: second_client)

    event = publisher._normalize_event({"event_id": "evt-3", "camera_id": "cam_03"})
    out_client = publisher._publish_payload(first_client, event)

    assert out_client is second_client
    assert len(second_client.published) == 1


def test_process_event_prefers_probe_snapshot_when_available(monkeypatch):
    settings = _build_settings()
    settings.telegram.snapshot_source = "probe"
    publisher = RedisAlertPublisher(settings, start_thread=False)
    fake_client = _FakeRedisClient()

    monkeypatch.setattr(
        publisher,
        "_save_snapshot_from_probe_frame",
        lambda **_kwargs: "/workspace/storage/snapshots/from_probe.jpg",
    )

    event = publisher._normalize_event({"event_id": "evt-4", "camera_id": "cam_04"})
    publisher._process_event(fake_client, event, snapshot_frame=object())

    _topic, raw_payload = fake_client.published[0]
    payload = json.loads(raw_payload)
    assert payload["snapshot_path"] == "/workspace/storage/snapshots/from_probe.jpg"
