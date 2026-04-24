from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.utils.file_utils import ensure_dir


logger = get_logger(__name__)


class JsonlEventPublisher:
    def __init__(self, output_file: str) -> None:
        self.output_path = Path(output_file)
        ensure_dir(self.output_path.parent)
        self._lock = Lock()

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }

        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.debug("Published event: %s", event_type)