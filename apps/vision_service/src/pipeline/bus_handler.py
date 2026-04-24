from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from apps.vision_service.src.logger import get_logger


logger = get_logger(__name__)


class BusHandler:
    def __init__(self, main_loop: GLib.MainLoop) -> None:
        self.main_loop = main_loop

    def attach(self, pipeline: Gst.Pipeline) -> None:
        bus = pipeline.get_bus()
        if bus is None:
            raise RuntimeError("Failed to get pipeline bus.")

        bus.add_signal_watch()
        bus.connect("message", self.on_message)

    def on_message(self, _bus: Gst.Bus, message: Gst.Message) -> None:
        message_type = message.type

        if message_type == Gst.MessageType.EOS:
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