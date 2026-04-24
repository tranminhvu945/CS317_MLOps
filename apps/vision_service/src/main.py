from __future__ import annotations

import signal
import sys

from apps.vision_service.src.app import VisionApp
from apps.vision_service.src.logger import get_logger, setup_logger
from apps.vision_service.src.settings import load_settings
from apps.vision_service.src.utils.deepstream_env import validate_gstreamer_factories


logger = get_logger(__name__)


def main() -> int:
    logger.info("Validating GStreamer/DeepStream environment early...")
    validate_gstreamer_factories()

    settings = load_settings()
    setup_logger(settings.app.log_level)

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