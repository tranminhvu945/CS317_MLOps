from __future__ import annotations

import signal

import gi

gi.require_version("GLib", "2.0")
from gi.repository import GLib  # noqa: E402

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.pipeline.builder import PipelineBuilder
from apps.vision_service.src.settings import RootSettings


logger = get_logger(__name__)


class VisionApp:
    def __init__(self, settings: RootSettings) -> None:
        self.settings = settings
        self.main_loop = GLib.MainLoop()
        self.pipeline_builder = PipelineBuilder(settings, self.main_loop)

    def run(self) -> None:
        self._register_signal_handlers()

        try:
            self.pipeline_builder.build()
            self.pipeline_builder.start()
            self.main_loop.run()
        finally:
            self.pipeline_builder.stop()

    def stop(self) -> None:
        try:
            self.pipeline_builder.stop()
        finally:
            try:
                self.main_loop.quit()
            except Exception:
                pass

    def _register_signal_handlers(self) -> None:
        def _handle_signal(signum, _frame) -> None:
            logger.info("Received shutdown signal: %s", signum)
            self.stop()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)