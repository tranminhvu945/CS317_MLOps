from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


REQUIRED_FACTORIES = [
    "nvstreammux",
    "nvinfer",
]


def validate_gstreamer_factories() -> None:
    Gst.init(None)

    missing = []
    for name in REQUIRED_FACTORIES:
        if Gst.ElementFactory.find(name) is None:
            missing.append(name)

    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing required GStreamer/DeepStream factories: {missing_str}. "
            f"Run inside the DeepStream container or fix your GStreamer plugin environment first."
        )