from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from apps.vision_service.src.logger import get_logger


logger = get_logger(__name__)


def _make_element(factory: str, name: str) -> Gst.Element:
    element = Gst.ElementFactory.make(factory, name)
    if element is None:
        raise RuntimeError(f"Failed to create element '{factory}' with name '{name}'")
    return element


def _set_property_if_exists(element: Gst.Element, property_name: str, value: object) -> None:
    if element.find_property(property_name) is not None:
        element.set_property(property_name, value)


def _configure_live_queue(queue: Gst.Element) -> None:
    # Prefer smoother playback over lowest-possible latency.
    queue.set_property("max-size-buffers", 8)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    _set_property_if_exists(queue, "flush-on-eos", True)
    _set_property_if_exists(queue, "silent", True)


def _make_optional_element(factory: str, name: str) -> Gst.Element | None:
    if Gst.ElementFactory.find(factory) is None:
        return None
    return _make_element(factory, name)


def _configure_clocksync(clocksync: Gst.Element) -> None:
    # Re-pace HLS bursts according to buffer timestamps before frames enter the
    # downstream inference/encode stages.
    _set_property_if_exists(clocksync, "sync", True)
    _set_property_if_exists(clocksync, "sync-to-first", True)
    _set_property_if_exists(clocksync, "qos", False)
    _set_property_if_exists(clocksync, "silent", True)


def _configure_decode_child(child: Gst.Element, drop_frame_interval: int) -> None:
    name = child.get_name()

    if "decodebin" in name:
        child.connect(
            "child-added",
            lambda _parent, nested_child, _name: _configure_decode_child(
                nested_child,
                drop_frame_interval,
            ),
        )
        return

    if "decoder" not in name:
        return

    _set_property_if_exists(child, "low-latency-mode", True)
    _set_property_if_exists(child, "enable-max-performance", True)
    # NOTE:
    # `disable-dpb=true` can stall decode on streams with B-frames.
    # Keep DPB enabled for robust HLS/RTSP ingestion.

    if drop_frame_interval > 1:
        _set_property_if_exists(child, "drop-frame-interval", drop_frame_interval)
        logger.info(
            "Configured decoder drop-frame-interval=%d on %s",
            drop_frame_interval,
            name,
        )


def _normalize_hls_uri(uri: str) -> str:
    """
    MediaMTX HLS often redirects `/index.m3u8` to `?cookieCheck=1`.
    Some GStreamer HTTP stacks can fail on this redirect/cookie flow and return 404.
    To keep ingestion stable, inject cookieCheck=1 directly for plain HTTP HLS URLs.
    """
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"}:
        return uri
    if not parsed.path.endswith("/index.m3u8"):
        return uri

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if "cookieCheck" in query:
        return uri

    query["cookieCheck"] = "1"
    normalized = parsed._replace(query=urlencode(query))
    return urlunparse(normalized)


def create_hls_source_bin(
    index: int,
    uri: str,
    drop_frame_interval: int = 0,
) -> Gst.Bin:
    """
    Source bin: uridecodebin -> queue -> clocksync? -> ghost_pad.

    Hỗ trợ HLS, RTSP, file input.
    setBboxContext được gọi từ main thread trong builder.py.
    """
    bin_name = f"source-bin-{index:02d}"
    source_bin = Gst.Bin.new(bin_name)
    if source_bin is None:
        raise RuntimeError(f"Failed to create source bin: {bin_name}")

    normalized_uri = _normalize_hls_uri(uri)
    if normalized_uri != uri:
        logger.info(
            "Normalized HLS URI for compatibility | original=%s | normalized=%s",
            uri,
            normalized_uri,
        )

    decodebin = _make_element("uridecodebin", f"uri-decode-bin-{index:02d}")
    decodebin.set_property("uri", normalized_uri)
    _set_property_if_exists(decodebin, "use-buffering", False)
    _set_property_if_exists(decodebin, "buffer-duration", 0)

    queue = _make_element("queue", f"source-queue-{index:02d}")
    _configure_live_queue(queue)
    clocksync = _make_optional_element("clocksync", f"source-clocksync-{index:02d}")
    if clocksync is not None:
        _configure_clocksync(clocksync)

    source_bin.add(decodebin)
    source_bin.add(queue)
    if clocksync is not None:
        source_bin.add(clocksync)
        if not queue.link(clocksync):
            raise RuntimeError(f"Failed to link queue -> clocksync for {bin_name}")
        logger.info("Enabled clocksync pacing in source bin: %s", bin_name)
    else:
        logger.warning(
            "clocksync plugin not available; HLS source will keep original pacing | bin=%s",
            bin_name,
        )

    def on_pad_added(_decodebin: Gst.Element, pad: Gst.Pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return

        structure = caps.get_structure(0)
        media_type = structure.get_name()

        # Chỉ nhận video pad
        if not media_type.startswith("video/"):
            return

        sink_pad = queue.get_static_pad("sink")
        if sink_pad is None:
            raise RuntimeError(f"Failed to get queue sink pad for {bin_name}")

        if sink_pad.is_linked():
            return

        result = pad.link(sink_pad)
        if result != Gst.PadLinkReturn.OK:
            raise RuntimeError(
                f"Failed to link decodebin video pad -> queue for {bin_name}, result={result}"
            )

        logger.info("Linked decodebin video pad to source bin: %s", bin_name)

    decodebin.connect("pad-added", on_pad_added)
    decodebin.connect(
        "child-added",
        lambda _parent, child, _name: _configure_decode_child(
            child,
            drop_frame_interval,
        ),
    )

    src_element = clocksync if clocksync is not None else queue
    src_pad = src_element.get_static_pad("src")
    if src_pad is None:
        raise RuntimeError(f"Failed to get source output pad for {bin_name}")

    ghost_pad = Gst.GhostPad.new("src", src_pad)
    if ghost_pad is None:
        raise RuntimeError(f"Failed to create ghost pad for {bin_name}")

    if not source_bin.add_pad(ghost_pad):
        raise RuntimeError(f"Failed to add ghost pad to source bin: {bin_name}")

    logger.info("Created source bin: %s | uri=%s", bin_name, normalized_uri)
    return source_bin
