"""
metrics_exporter.py — Prometheus metrics exporter cho DeepStream vision-service.

Expose một HTTP endpoint /metrics (chuẩn Prometheus text format) trên port cấu hình.
Được cập nhật bởi event_publisher khi nhận events từ pipeline.

Metric registry:
  pipeline_input_fps            Gauge  camera_id
  pipeline_infer_fps            Gauge  camera_id
  pipeline_output_fps           Gauge  camera_id
  pipeline_latency_ms           Gauge  camera_id
  pipeline_dropped_frames_total Counter camera_id
  pipeline_queue_level          Gauge  queue_name
  system_cpu_utilization_pct    Gauge
  system_gpu_utilization_pct    Gauge
  system_ram_utilization_pct    Gauge
  system_vram_utilization_pct   Gauge
  helmet_violation_total        Counter camera_id
  detection_objects_total       Counter label, camera_id
  probe_callback_avg_ms         Gauge   (probe overhead)
  probe_callback_p95_ms         Gauge
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    start_http_server,
)

from apps.vision_service.src.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Singleton registry wrapper
# ---------------------------------------------------------------------------

class PrometheusMetricsExporter:
    """Thread-safe singleton that manages all Prometheus metrics for the pipeline."""

    _instance: Optional["PrometheusMetricsExporter"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PrometheusMetricsExporter":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._server_started = False

        # ── Pipeline throughput ──────────────────────────────────────────────
        self.pipeline_input_fps = Gauge(
            "pipeline_input_fps",
            "Input frame rate per camera source (frames/sec)",
            ["camera_id"],
        )
        self.pipeline_infer_fps = Gauge(
            "pipeline_infer_fps",
            "Inference throughput per camera source (frames/sec)",
            ["camera_id"],
        )
        self.pipeline_output_fps = Gauge(
            "pipeline_output_fps",
            "Output buffer rate of the pipeline sink (frames/sec)",
            ["camera_id"],
        )

        # ── Latency ─────────────────────────────────────────────────────────
        self.pipeline_latency_ms = Gauge(
            "pipeline_latency_ms",
            "Average input-to-infer latency per camera (milliseconds)",
            ["camera_id"],
        )
        self.pipeline_rtmp_delay_ms = Gauge(
            "pipeline_rtmp_delay_ms",
            "Average infer-to-output (RTMP/sink) delay per camera (milliseconds)",
            ["camera_id"],
        )

        # ── Frame drops ──────────────────────────────────────────────────────
        self.pipeline_dropped_frames_total = Counter(
            "pipeline_dropped_frames_total",
            "Cumulative count of frames dropped before inference per camera",
            ["camera_id"],
        )

        # ── Queue health ─────────────────────────────────────────────────────
        self.pipeline_queue_level = Gauge(
            "pipeline_queue_level",
            "Current buffer occupancy of pipeline queues (buffers)",
            ["queue_name"],
        )

        # ── System resources ─────────────────────────────────────────────────
        self.system_cpu_utilization_pct = Gauge(
            "system_cpu_utilization_pct",
            "Host CPU utilization percentage",
        )
        self.system_gpu_utilization_pct = Gauge(
            "system_gpu_utilization_pct",
            "NVIDIA GPU compute utilization percentage",
        )
        self.system_ram_utilization_pct = Gauge(
            "system_ram_utilization_pct",
            "Host RAM utilization percentage",
        )
        self.system_vram_utilization_pct = Gauge(
            "system_vram_utilization_pct",
            "NVIDIA GPU VRAM utilization percentage",
        )

        # ── Detection events ─────────────────────────────────────────────────
        self.helmet_violation_total = Counter(
            "helmet_violation_total",
            "Total helmet violation detections per camera",
            ["camera_id"],
        )
        self.detection_objects_total = Counter(
            "detection_objects_total",
            "Total detected objects per label per camera",
            ["label", "camera_id"],
        )

        # ── Probe overhead ───────────────────────────────────────────────────
        self.probe_callback_avg_ms = Gauge(
            "probe_callback_avg_ms",
            "Average GStreamer probe callback duration (milliseconds)",
        )
        self.probe_callback_p95_ms = Gauge(
            "probe_callback_p95_ms",
            "P95 GStreamer probe callback duration (milliseconds)",
        )

        # ── Detection window rate ────────────────────────────────────────────
        self.detection_buffer_rate = Gauge(
            "detection_buffer_rate",
            "Detection probe buffer processing rate (buffers/sec)",
        )
        self.pipeline_event_rate = Gauge(
            "pipeline_event_rate",
            "Published event rate per second across the window",
            ["camera_id"],
        )

        logger.info("PrometheusMetricsExporter initialized (metrics registered)")

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    def start_server(self, port: int) -> None:
        """Start Prometheus HTTP server in a daemon thread (call once)."""
        if self._server_started:
            logger.warning("Prometheus HTTP server already started — skipping")
            return

        start_http_server(port)
        self._server_started = True
        logger.info("Prometheus metrics HTTP server started on port %d", port)

    # -----------------------------------------------------------------------
    # Update helpers — called by JsonlEventPublisher
    # -----------------------------------------------------------------------

    def update_pipeline_debug_metrics(self, payload: Dict[str, Any]) -> None:
        """Handle a 'pipeline_debug_metrics' event payload."""
        camera_id: str = str(payload.get("camera_id", "unknown"))

        _set_gauge(self.pipeline_input_fps.labels(camera_id=camera_id), payload.get("input_fps"))
        _set_gauge(self.pipeline_infer_fps.labels(camera_id=camera_id), payload.get("infer_fps"))
        _set_gauge(self.pipeline_output_fps.labels(camera_id=camera_id), payload.get("output_fps"))
        _set_gauge(self.pipeline_latency_ms.labels(camera_id=camera_id), payload.get("latency_ms"))
        _set_gauge(self.pipeline_rtmp_delay_ms.labels(camera_id=camera_id), payload.get("rtmp_delay_ms"))
        _set_gauge(self.pipeline_event_rate.labels(camera_id=camera_id), payload.get("event_rate"))

        dropped = payload.get("dropped_frames")
        if dropped is not None and dropped > 0:
            self.pipeline_dropped_frames_total.labels(camera_id=camera_id).inc(dropped)

        # Queue levels (detail dict: {queue_name: level})
        queue_detail: Dict[str, int] = payload.get("queue_level_detail") or {}
        for queue_name, level in queue_detail.items():
            if level >= 0:
                self.pipeline_queue_level.labels(queue_name=queue_name).set(level)

        # System
        _set_gauge(self.system_cpu_utilization_pct, payload.get("cpu_utilization_pct"))
        _set_gauge(self.system_gpu_utilization_pct, payload.get("gpu_utilization_pct"))
        _set_gauge(self.system_ram_utilization_pct, payload.get("ram_utilization_pct"))
        _set_gauge(self.system_vram_utilization_pct, payload.get("vram_utilization_pct"))

    def update_detection_window_summary(self, payload: Dict[str, Any]) -> None:
        """Handle a 'detection_window_summary' event payload."""
        # Object counts by label (counts_by_label: {"helmet #5": 3, "no_helmet #7": 1, ...})
        counts_by_label: Dict[str, int] = payload.get("counts_by_label") or {}
        # Normalize label to base category (strip track ID suffix)
        label_totals: Dict[str, int] = {}
        for raw_label, count in counts_by_label.items():
            base_label = raw_label.split(" #")[0].strip()
            label_totals[base_label] = label_totals.get(base_label, 0) + count

        # detection_objects_total is per-label, camera unknown at this level → use "all"
        for label, count in label_totals.items():
            if count > 0:
                self.detection_objects_total.labels(label=label, camera_id="all").inc(count)

        # Probe callback stats
        cb_stats = payload.get("probe_callback_ms") or {}
        _set_gauge(self.probe_callback_avg_ms, cb_stats.get("avg"))
        _set_gauge(self.probe_callback_p95_ms, cb_stats.get("p95"))
        _set_gauge(self.detection_buffer_rate, payload.get("buffer_rate"))

    def update_helmet_violation(self, payload: Dict[str, Any]) -> None:
        """Handle a 'helmet_violation' event payload."""
        camera_id: str = str(payload.get("camera_id", "unknown"))
        self.helmet_violation_total.labels(camera_id=camera_id).inc()
        # Also count no_helmet per camera
        self.detection_objects_total.labels(label="no_helmet", camera_id=camera_id).inc()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _set_gauge(gauge: Any, value: Any) -> None:
    """Safely set a Gauge, ignoring None values."""
    if value is not None:
        try:
            gauge.set(float(value))
        except (TypeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

def get_metrics_exporter() -> PrometheusMetricsExporter:
    """Return the singleton PrometheusMetricsExporter."""
    return PrometheusMetricsExporter()
