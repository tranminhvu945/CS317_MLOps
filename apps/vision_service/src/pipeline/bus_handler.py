from __future__ import annotations

from typing import List, Optional

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from apps.vision_service.src.logger import get_logger


logger = get_logger(__name__)


class BusHandler:
    def __init__(self, main_loop: GLib.MainLoop) -> None:
        self.main_loop = main_loop
        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop_decodebins: List[Gst.Element] = []

    def register_loop_sources(
        self,
        pipeline: Gst.Pipeline,
        source_bins: List[Gst.Bin],
    ) -> None:
        """
        Đăng ký các file source bins cần loop.

        Khi pipeline nhận EOS, BusHandler sẽ seek các source này về đầu
        thay vì dừng pipeline.
        """
        self._pipeline = pipeline
        for sbin in source_bins:
            if getattr(sbin, "_loop", False):
                decodebin = getattr(sbin, "_decodebin", None)
                if decodebin is not None:
                    self._loop_decodebins.append(decodebin)

        logger.info(
            "Registered %d file source(s) for loop-on-EOS",
            len(self._loop_decodebins),
        )

    def attach(self, pipeline: Gst.Pipeline) -> None:
        bus = pipeline.get_bus()
        if bus is None:
            raise RuntimeError("Failed to get pipeline bus.")

        bus.add_signal_watch()
        bus.connect("message", self.on_message)

    def on_message(self, _bus: Gst.Bus, message: Gst.Message) -> None:
        message_type = message.type

        if message_type == Gst.MessageType.EOS:
            if self._pipeline is not None and self._loop_decodebins:
                # Có file sources cần loop: seek về đầu thay vì quit
                logger.info(
                    "EOS received — seeking %d loop source(s) back to start",
                    len(self._loop_decodebins),
                )
                success = self._pipeline.seek_simple(
                    Gst.Format.TIME,
                    Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                    0,
                )
                if not success:
                    logger.warning("Pipeline seek-to-start failed; stopping.")
                    self.main_loop.quit()
            else:
                logger.warning("Received EOS. Stopping main loop.")
                self.main_loop.quit()
            return

        if message_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("GStreamer ERROR: %s | debug=%s", err, debug)
            self.main_loop.quit()
            return

        if message_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning("GStreamer WARNING: %s | debug=%s", err, debug)
            return

        if message_type == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending_state = message.parse_state_changed()
            if isinstance(message.src, Gst.Pipeline):
                logger.info(
                    "Pipeline state changed: %s -> %s (pending=%s)",
                    Gst.Element.state_get_name(old_state),
                    Gst.Element.state_get_name(new_state),
                    Gst.Element.state_get_name(pending_state),
                )
            return