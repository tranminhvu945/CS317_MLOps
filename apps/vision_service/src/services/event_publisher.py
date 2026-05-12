from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.utils.file_utils import ensure_dir

logger = get_logger(__name__)


class JsonlEventPublisher:
    """
    Publish pipeline events to a JSONL file and update Prometheus metrics.

    Two responsibilities:
      1. Append every event as a JSON-Lines record to *output_file*.
      2. Forward the event payload to the Prometheus metrics exporter so that
         Grafana dashboards are updated in real time (lazy import to avoid
         circular deps / import-time side-effects).
    """

    def __init__(self, output_file: str) -> None:
        self.output_path = Path(output_file)
        ensure_dir(self.output_path.parent)
        self._lock = Lock()

        # Rolling counters for consume_window_stats()
        self._window_events: int = 0
        self._window_events_by_type: Dict[str, int] = defaultdict(int)
        self._total_events: int = 0
        self._window_lock = Lock()

        # Lazily resolved — avoids prometheus_client import at module level
        # when running tests that don't have it installed.
        self._metrics_exporter: Optional[Any] = None
        self._metrics_enabled: bool = True  # can be disabled via disable_metrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def disable_metrics(self) -> None:
        """Opt-out of Prometheus updates (e.g. in unit tests)."""
        self._metrics_enabled = False

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append event to JSONL file and push to Prometheus metrics."""
        record = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }

        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

        with self._window_lock:
            self._window_events += 1
            self._window_events_by_type[event_type] += 1
            self._total_events += 1

        logger.debug("Published event: %s", event_type)

        if self._metrics_enabled:
            self._forward_to_prometheus(event_type, payload)

    def consume_window_stats(self, elapsed_sec: float) -> Dict[str, Any]:
        """
        Return and reset the rolling event counters for the current window.
        Called by RuntimeMetricsProbe to embed event stats in pipeline metrics.
        """
        with self._window_lock:
            events_in_window = self._window_events
            events_by_type = dict(self._window_events_by_type)
            total_events = self._total_events

            self._window_events = 0
            self._window_events_by_type = defaultdict(int)

        elapsed = max(elapsed_sec, 1e-6)
        event_rate = events_in_window / elapsed

        return {
            "event_rate": round(event_rate, 3),
            "events_in_window": events_in_window,
            "events_by_type": events_by_type,
            "total_events": total_events,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_metrics_exporter(self) -> Optional[Any]:
        """Lazy-load the metrics exporter to avoid circular imports."""
        if not self._metrics_enabled:
            return None
        if self._metrics_exporter is None:
            try:
                from apps.vision_service.src.services.metrics_exporter import (  # noqa: PLC0415
                    get_metrics_exporter,
                )
                self._metrics_exporter = get_metrics_exporter()
            except ImportError:
                logger.warning(
                    "prometheus_client not installed — Prometheus metrics disabled"
                )
                self._metrics_enabled = False
                return None
        return self._metrics_exporter

    def _forward_to_prometheus(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Route events to the appropriate Prometheus metric updater."""
        exporter = self._get_metrics_exporter()
        if exporter is None:
            return

        try:
            if event_type == "pipeline_debug_metrics":
                exporter.update_pipeline_debug_metrics(payload)
            elif event_type == "detection_window_summary":
                exporter.update_detection_window_summary(payload)
            elif event_type == "helmet_violation":
                exporter.update_helmet_violation(payload)
        except Exception as exc:  # noqa: BLE001
            # Never let metric update errors crash the pipeline
            logger.error(f"Prometheus update failed: {exc}", exc_info=True)

            logger.debug("Prometheus update error [%s]: %s", event_type, exc)