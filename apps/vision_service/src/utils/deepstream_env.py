from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


REQUIRED_FACTORIES = [
    "nvstreammux",
    "nvinfer",
]


def validate_gstreamer_factories(sink: str | None = None) -> None:
    Gst.init(None)

    required = list(REQUIRED_FACTORIES)
    if sink in {"rtsp", "rtmp"}:
        required.extend(["nvv4l2h264enc", "h264parse"])
    if sink == "rtsp":
        required.extend(["rtph264pay", "udpsink"])
    if sink == "rtmp":
        required.append("flvmux")

    missing = []
    for name in required:
        if Gst.ElementFactory.find(name) is None:
            missing.append(name)

    if sink == "rtmp":
        has_rtmp_path = (
            Gst.ElementFactory.find("rtmpsink") is not None
            or Gst.ElementFactory.find("rtmp2sink") is not None
        )
        if not has_rtmp_path:
            missing.append("rtmpsink or rtmp2sink")

    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing required GStreamer/DeepStream factories: {missing_str}. "
            f"Run inside the DeepStream container or fix your GStreamer plugin environment first."
        )
