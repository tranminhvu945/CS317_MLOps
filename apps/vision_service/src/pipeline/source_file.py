from __future__ import annotations

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


def create_file_source_bin(
    index: int,
    uri: str,
    loop: bool = True,
) -> Gst.Bin:
    """
    Create a file source bin: uridecodebin → queue → ghost src pad.

    Args:
        index:  Chỉ số source (0-based), dùng để đặt tên unique cho elements.
        uri:    URI file video (dạng file:///path/to/file.mp4).
        loop:   Được xử lý ở cấp pipeline bus (BusHandler), không ở đây.
                Tham số này được giữ lại để BusHandler biết source nào cần loop.
    """
    bin_name = f"source-bin-{index:02d}"
    source_bin = Gst.Bin.new(bin_name)
    if source_bin is None:
        raise RuntimeError(f"Failed to create source bin: {bin_name}")

    decodebin = _make_element("uridecodebin", f"uri-decode-bin-{index:02d}")
    queue = _make_element("queue", f"source-queue-{index:02d}")

    decodebin.set_property("uri", uri)
    # Tắt buffering để giảm độ trễ với file local
    if decodebin.find_property("use-buffering") is not None:
        decodebin.set_property("use-buffering", False)

    # Cấu hình queue cho live-like playback
    queue.set_property("max-size-buffers", 8)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    if queue.find_property("flush-on-eos") is not None:
        queue.set_property("flush-on-eos", True)
    if queue.find_property("silent") is not None:
        queue.set_property("silent", True)

    source_bin.add(decodebin)
    source_bin.add(queue)

    def on_pad_added(_decodebin: Gst.Element, pad: Gst.Pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            logger.warning("pad-added: caps empty for %s", bin_name)
            return

        structure = caps.get_structure(0)
        media_type = structure.get_name()
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

        logger.info("Linked decodebin video pad to file source bin: %s", bin_name)

    decodebin.connect("pad-added", on_pad_added)

    src_pad = queue.get_static_pad("src")
    if src_pad is None:
        raise RuntimeError(f"Failed to get queue src pad for {bin_name}")

    ghost_pad = Gst.GhostPad.new("src", src_pad)
    if ghost_pad is None:
        raise RuntimeError(f"Failed to create ghost pad for {bin_name}")

    if not source_bin.add_pad(ghost_pad):
        raise RuntimeError(f"Failed to add ghost pad to source bin: {bin_name}")

    # Lưu metadata cho BusHandler biết cách xử lý EOS
    source_bin._loop = loop  # type: ignore[attr-defined]
    source_bin._decodebin = decodebin  # type: ignore[attr-defined]

    logger.info(
        "Created file source bin: %s | uri=%s | loop=%s",
        bin_name,
        uri,
        loop,
    )
    return source_bin