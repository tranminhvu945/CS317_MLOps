from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field

from apps.vision_service.src.domain.camera_schema import CameraConfig
from apps.vision_service.src.utils.file_utils import ensure_dir
from apps.vision_service.src.logger import get_logger

logger = get_logger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


class AppConfig(BaseModel):
    name: str = "helmet-violation-service"
    env: str = "development"
    log_level: str = "INFO"
    gpu_id: int = Field(default=1, ge=0)


class StorageConfig(BaseModel):
    logs_dir: str


class EventsConfig(BaseModel):
    output_file: str


class StreamsConfig(BaseModel):
    scan_camera_dir: str


class PipelineConfig(BaseModel):
    streammux_width: int = Field(default=960, ge=1)
    streammux_height: int = Field(default=544, ge=1)
    batched_push_timeout_usec: int = Field(default=40000, ge=1)
    max_sources: int = Field(default=16, ge=1)
    sink: Literal["fake", "display", "rtsp", "rtmp"] = "rtsp"
    frame_log_interval_sec: float = Field(default=5.0, ge=0.5)


class InferConfig(BaseModel):
    enabled: bool = True
    config_file: str
    unique_id: int = Field(default=1, ge=1)
    summary_interval_sec: float = Field(default=5.0, ge=0.5)
    emit_frame_events: bool = False


class VisualizationConfig(BaseModel):
    enabled: bool = True
    display_text: bool = True
    display_bbox: bool = True
    display_clock: bool = False
    osd_process_mode: int = Field(default=0, ge=0, le=1)


class RtspConfig(BaseModel):
    enabled: bool = True
    host: str = "127.0.0.1"
    udp_port: int = Field(default=5400, ge=1024, le=65535)
    rtsp_port: int = Field(default=8554, ge=1024, le=65535)
    mount_point: str = "/vision"
    codec: str = "h264"
    bitrate: int = Field(default=4000000, ge=100000)
    iframe_interval: int = Field(default=30, ge=1)
    payload_type: int = Field(default=96, ge=96, le=127)
    rtp_mtu: int = Field(default=1400, ge=576, le=9000)
    udp_buffer_size: int = Field(default=2097152, ge=65536)
    sps_pps_interval: int = Field(default=-1, ge=-1, le=3600)
    rtsp_repay_enabled: bool = True
    rtsp_repay_jitter_latency_ms: int = Field(default=0, ge=0, le=2000)
    rtsp_repay_jitter_drop_on_latency: bool = True
    rtsp_repay_leaky_queue_enabled: bool = True
    rtsp_transport: Literal["all", "tcp", "udp"] = "all"
    udpsink_sync: bool = True
    udpsink_async: bool = False
    udpsink_qos: bool = False
    debug_h264_output_file: str = ""


class RtmpConfig(BaseModel):
    enabled: bool = True
    location: str = "rtmp://127.0.0.1:1935/vision1"
    sink_sync: bool = False
    sink_async: bool = False
    streamable_mux: bool = True


class TrackerConfig(BaseModel):
    enabled: bool = True
    gpu_id: int = Field(default=0, ge=0)
    tracker_width: int = Field(default=960, ge=1)
    tracker_height: int = Field(default=544, ge=1)
    ll_lib_file: str
    ll_config_file: str
    display_tracking_id: bool = False


class MetricsConfig(BaseModel):
    enabled: bool = True
    port: int = Field(default=9100, ge=1024, le=65535)


class TelegramConfig(BaseModel):
    enabled: bool = False
    snapshot_source: Literal["probe", "rtmp"] = "probe"
    redis_host: str = "redis"
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_topic: str = "helmet_violations"
    cooldown_sec: float = Field(default=5.0, ge=0.0)
    min_consecutive_no_helmet_frames: int = Field(default=3, ge=1)
    snapshot_dir: str = "/workspace/storage/snapshots"
    snapshot_rtmp_url: str = "rtmp://mediamtx:1935/vision1"
    snapshot_hls_url: str = "http://mediamtx:8888/vision1/index.m3u8?cookieCheck=1"


class TilerConfig(BaseModel):
    enabled: bool = True
    rows: int = Field(default=0, ge=0, description="0 = tự động tính theo sqrt(n_cameras)")
    cols: int = Field(default=0, ge=0, description="0 = tự động tính theo sqrt(n_cameras)")
    width: int = Field(default=1920, ge=1, description="Tổng chiều rộng output sau khi ghép tile")
    height: int = Field(default=1080, ge=1, description="Tổng chiều cao output sau khi ghép tile")


class RootSettings(BaseModel):
    app: AppConfig
    storage: StorageConfig
    events: EventsConfig
    streams: StreamsConfig
    pipeline: PipelineConfig
    infer: InferConfig
    tracker: TrackerConfig
    tiler: TilerConfig = Field(default_factory=TilerConfig)
    visualization: VisualizationConfig
    rtsp: RtspConfig
    rtmp: RtmpConfig = Field(default_factory=RtmpConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    cameras: List[CameraConfig] = Field(default_factory=list)


def _read_yaml_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in file: {path}")
    return data


def _load_camera_configs(camera_dir: Path) -> List[CameraConfig]:
    if not camera_dir.exists():
        raise FileNotFoundError(f"Camera config directory not found: {camera_dir}")
    cameras: List[CameraConfig] = []
    for file_path in sorted(camera_dir.glob("*.yaml")):
        raw_data = _read_yaml_file(file_path)
        camera = CameraConfig.model_validate(raw_data)
        if camera.enabled:
            cameras.append(camera)
    return cameras


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_settings() -> RootSettings:
    config_dir = Path(
        os.getenv("CONFIG_DIR", "/workspace/apps/vision_service/configs")
    ).resolve()
    # Repo root (CS317_MLOps) từ configs/
    project_root = config_dir.parents[2] if len(config_dir.parents) >= 3 else config_dir.parent
    app_config_path = config_dir / "app.yaml"

    raw_app = _read_yaml_file(app_config_path)

    settings = RootSettings.model_validate(
        {
            **raw_app,
            "app": {
                **raw_app.get("app", {}),
                "name": os.getenv(
                    "APP_NAME",
                    raw_app.get("app", {}).get("name", "uit-medseg-vision"),
                ),
                "env": os.getenv(
                    "APP_ENV",
                    raw_app.get("app", {}).get("env", "development"),
                ),
                "log_level": os.getenv(
                    "LOG_LEVEL",
                    raw_app.get("app", {}).get("log_level", "INFO"),
                ),
                "gpu_id": int(
                    os.getenv("GPU_ID", raw_app.get("app", {}).get("gpu_id", 0))
                ),
            },
            "rtmp": {
                **raw_app.get("rtmp", {}),
                "location": os.getenv(
                    "RTMP_LOCATION",
                    raw_app.get("rtmp", {}).get(
                        "location",
                        "rtmp://127.0.0.1:1935/vision1",
                    ),
                ),
            },
            "telegram": {
                **raw_app.get("telegram", {}),
                "enabled": _env_bool(
                    "TELEGRAM_ENABLED",
                    bool(raw_app.get("telegram", {}).get("enabled", False)),
                ),
                "snapshot_source": os.getenv(
                    "TELEGRAM_SNAPSHOT_SOURCE",
                    raw_app.get("telegram", {}).get("snapshot_source", "probe"),
                ),
                "redis_host": os.getenv(
                    "TELEGRAM_REDIS_HOST",
                    raw_app.get("telegram", {}).get("redis_host", "redis"),
                ),
                "redis_port": _env_int(
                    "TELEGRAM_REDIS_PORT",
                    int(raw_app.get("telegram", {}).get("redis_port", 6379)),
                ),
                "redis_topic": os.getenv(
                    "TELEGRAM_REDIS_TOPIC",
                    raw_app.get("telegram", {}).get("redis_topic", "helmet_violations"),
                ),
                "cooldown_sec": _env_float(
                    "TELEGRAM_COOLDOWN_SEC",
                    float(raw_app.get("telegram", {}).get("cooldown_sec", 5.0)),
                ),
                "min_consecutive_no_helmet_frames": _env_int(
                    "TELEGRAM_MIN_CONSEC_NO_HELMET_FRAMES",
                    int(
                        raw_app.get("telegram", {}).get(
                            "min_consecutive_no_helmet_frames", 3
                        )
                    ),
                ),
                "snapshot_dir": os.getenv(
                    "TELEGRAM_SNAPSHOT_DIR",
                    raw_app.get("telegram", {}).get(
                        "snapshot_dir", "/workspace/storage/snapshots"
                    ),
                ),
                "snapshot_rtmp_url": os.getenv(
                    "TELEGRAM_SNAPSHOT_RTMP_URL",
                    raw_app.get("telegram", {}).get(
                        "snapshot_rtmp_url", "rtmp://mediamtx:1935/vision1"
                    ),
                ),
                "snapshot_hls_url": os.getenv(
                    "TELEGRAM_SNAPSHOT_HLS_URL",
                    raw_app.get("telegram", {}).get(
                        "snapshot_hls_url", "http://mediamtx:8888/vision1/index.m3u8?cookieCheck=1"
                    ),
                ),
            },
        }
    )

    resolved_logs_dir = _resolve_path(settings.storage.logs_dir, project_root)
    resolved_events_output = _resolve_path(settings.events.output_file, project_root)
    resolved_snapshot_dir = _resolve_path(settings.telegram.snapshot_dir, project_root)

    settings = settings.model_copy(
        update={
            "storage": settings.storage.model_copy(
                update={"logs_dir": str(resolved_logs_dir)}
            ),
            "events": settings.events.model_copy(
                update={"output_file": str(resolved_events_output)}
            ),
            "telegram": settings.telegram.model_copy(
                update={"snapshot_dir": str(resolved_snapshot_dir)}
            ),
        }
    )

    camera_dir = _resolve_path(settings.streams.scan_camera_dir, config_dir)
    cameras = _load_camera_configs(camera_dir)

    # Filter reachable cameras if enabled
    if _env_bool("CHECK_CAMERA_REACHABILITY", True):
        import urllib.request
        import urllib.error
        import socket
        import time
        from urllib.parse import urlparse
        from concurrent.futures import ThreadPoolExecutor

        def is_uri_reachable(uri: str) -> bool:
            parsed = urlparse(uri)
            if not parsed.scheme or parsed.scheme == "file":
                return Path(parsed.path).exists()

            max_attempts = 5
            for attempt in range(max_attempts):
                reached = False
                if parsed.scheme in {"http", "https"}:
                    try:
                        req = urllib.request.Request(uri, method="HEAD")
                        with urllib.request.urlopen(req, timeout=1.0) as resp:
                            if resp.status == 200:
                                reached = True
                    except Exception:
                        try:
                            with urllib.request.urlopen(uri, timeout=1.0) as resp:
                                if resp.status == 200:
                                    reached = True
                        except Exception:
                            pass
                elif parsed.scheme == "rtsp":
                    port = parsed.port if parsed.port is not None else 554
                    host = parsed.hostname
                    if host:
                        try:
                            with socket.create_connection((host, port), timeout=1.0):
                                reached = True
                        except Exception:
                            pass
                elif parsed.scheme == "rtmp":
                    port = parsed.port if parsed.port is not None else 1935
                    host = parsed.hostname
                    if host:
                        try:
                            with socket.create_connection((host, port), timeout=1.0):
                                reached = True
                        except Exception:
                            pass
                else:
                    return True

                if reached:
                    return True
                if attempt < max_attempts - 1:
                    time.sleep(1.0)
            return False

        logger.info("Verifying camera stream reachability (with retries for slow-starting mock streams)...")
        reachable_cameras = []
        with ThreadPoolExecutor(max_workers=max(1, len(cameras))) as executor:
            futures = {executor.submit(is_uri_reachable, cam.stream.uri): cam for cam in cameras}
            for future in futures:
                cam = futures[future]
                try:
                    is_ok = future.result()
                except Exception as e:
                    logger.warning("Error checking reachability for %s: %s", cam.camera_id, e)
                    is_ok = False
                
                if is_ok:
                    reachable_cameras.append(cam)
                else:
                    logger.warning(
                        "Camera %s is unreachable (uri=%s) - excluding from pipeline to prevent empty tiler tiles.",
                        cam.camera_id,
                        cam.stream.uri,
                    )

        if reachable_cameras:
            cameras = reachable_cameras
        else:
            logger.warning("All cameras are unreachable! Keeping the original camera list as fallback.")

    settings = settings.model_copy(update={"cameras": cameras})

    ensure_dir(settings.storage.logs_dir)
    ensure_dir(Path(settings.events.output_file).parent)
    return settings
