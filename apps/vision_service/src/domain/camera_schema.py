from __future__ import annotations

from typing import List, Literal, Tuple

from pydantic import BaseModel, Field, HttpUrl, field_validator


Point = Tuple[int, int]


class RoiConfig(BaseModel):
    enabled: bool = False
    polygon: List[Point] = Field(default_factory=list)

class StreamConfig(BaseModel):
    type: Literal["hls", "file", "rtsp"] = "hls"
    uri: str
    reconnect_interval_sec: int = Field(default=10, ge=1)
    timeout_sec: int = Field(default=15, ge=1)
    decoder_drop_frame_interval: int = Field(default=0, ge=0, le=30)
    loop: bool = Field(default=True, description="Lặp lại file sau khi phát xong (chỉ áp dụng cho type=file)")

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        if (
            value.startswith("http://")
            or value.startswith("https://")
            or value.startswith("file://")
            or value.startswith("rtsp://")
        ):
            return value
        raise ValueError("uri must start with http://, https://, file://, or rtsp://")

class DetectionConfig(BaseModel):
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    roi: RoiConfig = Field(default_factory=RoiConfig)


class CameraConfig(BaseModel):
    camera_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    enabled: bool = True
    stream: StreamConfig
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
