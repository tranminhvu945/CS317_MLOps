from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field

from apps.vision_service.src.domain.camera_schema import CameraConfig
from apps.vision_service.src.utils.file_utils import ensure_dir


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
    location: str = "rtmp://127.0.0.1/live/vision live=1"
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
        }
    )

    camera_dir = _resolve_path(settings.streams.scan_camera_dir, config_dir)
    cameras = _load_camera_configs(camera_dir)
    settings = settings.model_copy(update={"cameras": cameras})

    ensure_dir(settings.storage.logs_dir)
    return settings
