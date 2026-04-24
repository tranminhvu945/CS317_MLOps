from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from apps.vision_service.src.logger import get_logger


logger = get_logger(__name__)


def make_element(factory_name: str, element_name: str) -> Gst.Element:
    element = Gst.ElementFactory.make(factory_name, element_name)
    if element is None:
        raise RuntimeError(
            f"Failed to create GStreamer element '{factory_name}' with name '{element_name}'."
        )
    logger.debug("Created element: %s (%s)", element_name, factory_name)
    return element


def request_streammux_sink_pad(streammux: Gst.Element, index: int) -> Gst.Pad:
    pad_name = f"sink_{index}"
    sink_pad = streammux.request_pad_simple(pad_name)
    if sink_pad is None:
        sink_pad = streammux.get_request_pad(pad_name)

    if sink_pad is None:
        raise RuntimeError(f"Failed to request nvstreammux sink pad: {pad_name}")

    return sink_pad