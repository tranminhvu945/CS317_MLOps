from __future__ import annotations

from pathlib import Path

import yaml

from apps.vision_service.src.settings import load_settings


def test_load_settings_parses_telegram_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    camera_dir = config_dir / "camera"
    camera_dir.mkdir(parents=True)

    app_yaml = {
        "app": {"name": "test-app", "env": "test", "log_level": "INFO", "gpu_id": 0},
        "storage": {"logs_dir": str(tmp_path / "logs")},
        "events": {"output_file": str(tmp_path / "logs" / "events.jsonl")},
        "streams": {"scan_camera_dir": "camera"},
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
            "redis_host": "redis",
            "redis_port": 6379,
            "redis_topic": "helmet_violations",
            "cooldown_sec": 7.5,
            "snapshot_dir": "/workspace/storage/snapshots",
            "snapshot_rtmp_url": "rtmp://mediamtx:1935/vision1",
            "snapshot_hls_url": "http://mediamtx:8888/vision1/index.m3u8",
        },
    }
    (config_dir / "app.yaml").write_text(yaml.safe_dump(app_yaml), encoding="utf-8")

    camera_yaml = {
        "camera_id": "cam_001",
        "name": "Cam 001",
        "enabled": True,
        "stream": {"type": "file", "uri": "file:///tmp/test.mp4", "loop": True},
        "detection": {"min_confidence": 0.5, "roi": {"enabled": False, "polygon": []}},
    }
    (camera_dir / "cam_001.yaml").write_text(yaml.safe_dump(camera_yaml), encoding="utf-8")

    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    settings = load_settings()

    assert settings.telegram.enabled is True
    assert settings.telegram.redis_host == "redis"
    assert settings.telegram.redis_port == 6379
    assert settings.telegram.redis_topic == "helmet_violations"
    assert settings.telegram.cooldown_sec == 7.5
    assert settings.telegram.snapshot_rtmp_url == "rtmp://mediamtx:1935/vision1"
    assert Path(settings.storage.logs_dir).exists()
