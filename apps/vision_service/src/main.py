import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)

import signal
import sys

from apps.vision_service.src.app import VisionApp
from apps.vision_service.src.logger import get_logger, setup_logger
from apps.vision_service.src.settings import load_settings
from apps.vision_service.src.utils.deepstream_env import validate_gstreamer_factories

logger = get_logger(__name__)


def main() -> int:
    setup_logger("INFO")
    settings = load_settings()
    setup_logger(settings.app.log_level)

    if settings.metrics.enabled:
        try:
            from apps.vision_service.src.services.metrics_exporter import (
                get_metrics_exporter,
            )
            exporter = get_metrics_exporter()
            exporter.start_server(settings.metrics.port)
        except Exception as exc:
            logger.warning("Failed to start Prometheus server early: %s", exc)

    logger.info(
        "Validating GStreamer/DeepStream environment early... | sink=%s",
        settings.pipeline.sink,
    )
    validate_gstreamer_factories(settings.pipeline.sink)

    app = VisionApp(settings)

    def _handle_shutdown(signum: int, _frame: object) -> None:
        logger.info("Received shutdown signal: %s", signum)
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
