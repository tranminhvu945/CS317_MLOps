from __future__ import annotations

import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, List, Optional

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import pyds

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.services.event_publisher import JsonlEventPublisher
from apps.vision_service.src.settings import RootSettings


logger = get_logger(__name__)

CLOCK_TIME_NONE = int(getattr(Gst, "CLOCK_TIME_NONE", 18446744073709551615))
MAX_PENDING_PTS_PER_SOURCE = 4096


@dataclass
class _SourceWindow:
    input_frames: int = 0
    infer_frames: int = 0
    latency_samples_ms: List[float] = field(default_factory=list)
    rtmp_delay_samples_ms: List[float] = field(default_factory=list)


class RuntimeMetricsProbe:
    def __init__(
        self,
        settings: RootSettings,
        publisher: JsonlEventPublisher,
        queue_elements: Dict[str, Gst.Element],
        log_interval_sec: float,
    ) -> None:
        self.settings = settings
        self.publisher = publisher
        self.queue_elements = queue_elements
        self.log_interval_sec = max(float(log_interval_sec), 0.5)

        self._lock = Lock()
        self._window_started_ns = time.monotonic_ns()

        self._source_windows: Dict[int, _SourceWindow] = defaultdict(_SourceWindow)
        self._source_camera_map: Dict[int, str] = {
            idx: camera.camera_id for idx, camera in enumerate(self.settings.cameras)
        }

        self._pending_input_pts_ns: Dict[int, Dict[int, int]] = defaultdict(dict)
        self._pending_input_pts_order: Dict[int, Deque[int]] = defaultdict(deque)

        self._window_output_buffers = 0
        self._last_infer_seen_ns_by_source: Dict[int, int] = {}

        self._cpu_last_total_idle: Optional[tuple[int, int]] = self._read_cpu_total_idle()

    def attach_input_stage(
        self,
        source_bin: Gst.Element,
        *,
        source_id: int,
        camera_id: str,
        pad_name: str = "src",
    ) -> None:
        pad = source_bin.get_static_pad(pad_name)
        if pad is None:
            raise RuntimeError(
                f"Failed to get pad '{pad_name}' from source element '{source_bin.get_name()}'."
            )

        context = {"source_id": int(source_id), "camera_id": camera_id}
        pad.add_probe(Gst.PadProbeType.BUFFER, self._on_input_buffer_probe, context)
        logger.info(
            "RuntimeMetricsProbe input attached | source_id=%d | camera_id=%s | element=%s | pad=%s",
            source_id,
            camera_id,
            source_bin.get_name(),
            pad_name,
        )

    def attach_infer_stage(self, element: Gst.Element, *, pad_name: str = "src") -> None:
        pad = element.get_static_pad(pad_name)
        if pad is None:
            raise RuntimeError(
                f"Failed to get pad '{pad_name}' from infer element '{element.get_name()}'."
            )

        pad.add_probe(Gst.PadProbeType.BUFFER, self._on_infer_buffer_probe, None)
        logger.info(
            "RuntimeMetricsProbe infer attached | element=%s | pad=%s",
            element.get_name(),
            pad_name,
        )

    def attach_output_stage(self, element: Gst.Element, *, pad_name: str = "src") -> None:
        pad = element.get_static_pad(pad_name)
        if pad is None:
            raise RuntimeError(
                f"Failed to get pad '{pad_name}' from output element '{element.get_name()}'."
            )

        pad.add_probe(Gst.PadProbeType.BUFFER, self._on_output_buffer_probe, None)
        logger.info(
            "RuntimeMetricsProbe output attached | element=%s | pad=%s",
            element.get_name(),
            pad_name,
        )

    def flush(self) -> None:
        with self._lock:
            self._emit_locked(force=True)

    def _on_input_buffer_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        user_data: object,
    ) -> Gst.PadProbeReturn:
        context = user_data if isinstance(user_data, dict) else {}
        source_id = int(context.get("source_id", 0))
        camera_id = str(context.get("camera_id", f"source_{source_id}"))

        now_ns = time.monotonic_ns()
        gst_buffer = info.get_buffer()

        with self._lock:
            self._source_camera_map[source_id] = camera_id
            window = self._source_windows[source_id]
            window.input_frames += 1

            pts = self._extract_pts(gst_buffer)
            if pts is not None:
                self._remember_input_pts(source_id=source_id, pts_ns=pts, seen_ns=now_ns)

            self._maybe_emit_locked(now_ns)

        return Gst.PadProbeReturn.OK

    def _on_infer_buffer_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        _user_data: object,
    ) -> Gst.PadProbeReturn:
        now_ns = time.monotonic_ns()
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        with self._lock:
            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                except StopIteration:
                    break

                source_id = int(frame_meta.source_id)
                window = self._source_windows[source_id]
                window.infer_frames += 1
                self._last_infer_seen_ns_by_source[source_id] = now_ns

                pts_raw = int(frame_meta.buf_pts)
                if pts_raw != CLOCK_TIME_NONE:
                    input_seen_ns = self._consume_input_pts(source_id, pts_raw)
                    if input_seen_ns is not None:
                        latency_ms = (now_ns - input_seen_ns) / 1e6
                        if latency_ms >= 0:
                            window.latency_samples_ms.append(latency_ms)

                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break

            self._maybe_emit_locked(now_ns)

        return Gst.PadProbeReturn.OK

    def _on_output_buffer_probe(
        self,
        _pad: Gst.Pad,
        _info: Gst.PadProbeInfo,
        _user_data: object,
    ) -> Gst.PadProbeReturn:
        now_ns = time.monotonic_ns()

        with self._lock:
            self._window_output_buffers += 1

            if self.settings.pipeline.sink == "rtmp":
                for source_id, last_infer_ns in self._last_infer_seen_ns_by_source.items():
                    rtmp_delay_ms = (now_ns - last_infer_ns) / 1e6
                    if rtmp_delay_ms >= 0:
                        self._source_windows[source_id].rtmp_delay_samples_ms.append(
                            rtmp_delay_ms
                        )

            self._maybe_emit_locked(now_ns)

        return Gst.PadProbeReturn.OK

    def _maybe_emit_locked(self, now_ns: int) -> None:
        if (now_ns - self._window_started_ns) < int(self.log_interval_sec * 1e9):
            return
        self._emit_locked(force=False)

    def _emit_locked(self, force: bool) -> None:
        now_ns = time.monotonic_ns()
        elapsed_sec = max((now_ns - self._window_started_ns) / 1e9, 1e-6)
        if not force and elapsed_sec < self.log_interval_sec:
            return

        source_ids = self._collect_source_ids()
        if not source_ids:
            self._window_started_ns = now_ns
            return

        output_fps = self._window_output_buffers / elapsed_sec
        queue_level, queue_level_detail = self._snapshot_queue_levels()
        system_metrics = self._sample_system_metrics()
        event_stats = self.publisher.consume_window_stats(elapsed_sec)

        for source_id in source_ids:
            camera_id = self._source_camera_map.get(source_id, f"source_{source_id}")
            window = self._source_windows.get(source_id, _SourceWindow())

            input_fps = window.input_frames / elapsed_sec
            infer_fps = window.infer_frames / elapsed_sec
            latency_ms = self._average(window.latency_samples_ms)
            dropped_frames = max(0, window.input_frames - window.infer_frames)
            rtmp_delay_ms = self._average(window.rtmp_delay_samples_ms)

            payload = {
                "window_sec": round(elapsed_sec, 3),
                "source_id": int(source_id),
                "camera_id": camera_id,
                "input_fps": round(input_fps, 3),
                "infer_fps": round(infer_fps, 3),
                "output_fps": round(output_fps, 3),
                "latency_ms": round(latency_ms, 3),
                "dropped_frames": int(dropped_frames),
                "queue_level": int(queue_level),
                "queue_level_detail": queue_level_detail,
                "event_rate": round(float(event_stats["event_rate"]), 3),
                "events_in_window": int(event_stats["events_in_window"]),
                "events_by_type": event_stats["events_by_type"],
                "total_events": int(event_stats["total_events"]),
                "cpu_utilization_pct": system_metrics["cpu_utilization_pct"],
                "gpu_utilization_pct": system_metrics["gpu_utilization_pct"],
                "ram_utilization_pct": system_metrics["ram_utilization_pct"],
                "vram_utilization_pct": system_metrics["vram_utilization_pct"],
                "cpu_temp_celsius": system_metrics["cpu_temp_celsius"],
                "gpu_temp_celsius": system_metrics["gpu_temp_celsius"],
                "camera_active": 1 if input_fps > 0 else 0,
                "rtmp_delay_ms": round(rtmp_delay_ms, 3),
            }

            self.publisher.publish(event_type="pipeline_debug_metrics", payload=payload)

        logger.info(
            "Pipeline metrics | window=%.2fs | sources=%d | output_fps=%.2f | queue_level=%d",
            elapsed_sec,
            len(source_ids),
            output_fps,
            queue_level,
        )

        self._source_windows = defaultdict(_SourceWindow)
        self._window_output_buffers = 0
        self._window_started_ns = now_ns

    def _collect_source_ids(self) -> List[int]:
        observed = set(self._source_camera_map)
        observed.update(self._source_windows.keys())
        observed.update(self._last_infer_seen_ns_by_source.keys())
        return sorted(observed)

    def _snapshot_queue_levels(self) -> tuple[int, Dict[str, int]]:
        detail: Dict[str, int] = {}

        for label, queue in self.queue_elements.items():
            level = -1
            try:
                if queue.find_property("current-level-buffers") is not None:
                    level = int(queue.get_property("current-level-buffers"))
            except Exception:
                level = -1
            detail[label] = level

        valid_levels = [value for value in detail.values() if value >= 0]
        queue_level = max(valid_levels) if valid_levels else -1
        return queue_level, detail

    def _remember_input_pts(self, *, source_id: int, pts_ns: int, seen_ns: int) -> None:
        pending_map = self._pending_input_pts_ns[source_id]
        pending_order = self._pending_input_pts_order[source_id]

        if pts_ns not in pending_map:
            pending_order.append(pts_ns)
        pending_map[pts_ns] = seen_ns

        while len(pending_order) > MAX_PENDING_PTS_PER_SOURCE:
            old_pts = pending_order.popleft()
            pending_map.pop(old_pts, None)

    def _consume_input_pts(self, source_id: int, pts_ns: int) -> Optional[int]:
        pending_map = self._pending_input_pts_ns.get(source_id)
        if not pending_map:
            return None

        pending_order = self._pending_input_pts_order.get(source_id)
        if pending_order is None:
            return None

        seen_ns = pending_map.pop(pts_ns, None)
        if seen_ns is None:
            return None

        while pending_order and pending_order[0] not in pending_map:
            pending_order.popleft()
        return seen_ns

    def _extract_pts(self, gst_buffer: Optional[Gst.Buffer]) -> Optional[int]:
        if gst_buffer is None:
            return None

        pts = int(gst_buffer.pts)
        if pts == CLOCK_TIME_NONE:
            dts = int(gst_buffer.dts)
            if dts == CLOCK_TIME_NONE:
                return None
            pts = dts
        return pts

    def _average(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _sample_system_metrics(self) -> Dict[str, Optional[float]]:
        cpu_util = self._sample_cpu_percent()
        ram_util = self._sample_ram_percent()
        gpu_util, vram_util, gpu_temp = self._sample_gpu_metrics()
        cpu_temp = self._sample_cpu_temperature()

        return {
            "cpu_utilization_pct": self._round_or_none(cpu_util),
            "gpu_utilization_pct": self._round_or_none(gpu_util),
            "ram_utilization_pct": self._round_or_none(ram_util),
            "vram_utilization_pct": self._round_or_none(vram_util),
            "cpu_temp_celsius": self._round_or_none(cpu_temp),
            "gpu_temp_celsius": self._round_or_none(gpu_temp),
        }

    def _sample_cpu_percent(self) -> Optional[float]:
        current = self._read_cpu_total_idle()
        previous = self._cpu_last_total_idle
        self._cpu_last_total_idle = current

        if current is None or previous is None:
            return None

        total_diff = current[0] - previous[0]
        idle_diff = current[1] - previous[1]
        if total_diff <= 0:
            return None

        busy_ratio = (total_diff - idle_diff) / float(total_diff)
        return max(0.0, min(100.0, busy_ratio * 100.0))

    def _read_cpu_total_idle(self) -> Optional[tuple[int, int]]:
        try:
            with Path("/proc/stat").open("r", encoding="utf-8") as file:
                first_line = file.readline().strip()
        except OSError:
            return None

        if not first_line.startswith("cpu "):
            return None

        parts = first_line.split()[1:]
        if len(parts) < 4:
            return None

        try:
            values = [int(part) for part in parts]
        except ValueError:
            return None

        total = int(sum(values))
        idle = int(values[3] + (values[4] if len(values) > 4 else 0))
        return total, idle

    def _sample_ram_percent(self) -> Optional[float]:
        try:
            with Path("/proc/meminfo").open("r", encoding="utf-8") as file:
                meminfo_lines = file.readlines()
        except OSError:
            return None

        mem_total_kb = None
        mem_available_kb = None

        for line in meminfo_lines:
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available_kb = int(line.split()[1])

        if mem_total_kb is None or mem_total_kb <= 0 or mem_available_kb is None:
            return None

        used_kb = max(mem_total_kb - mem_available_kb, 0)
        return (used_kb * 100.0) / float(mem_total_kb)

    def _sample_gpu_metrics(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        cmd = [
            "nvidia-smi",
            f"--id={self.settings.app.gpu_id}",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=0.8,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None, None, None

        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if not first_line:
            return None, None, None

        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) < 4:
            return None, None, None

        try:
            gpu_util = float(parts[0])
            mem_used_mb = float(parts[1])
            mem_total_mb = float(parts[2])
            gpu_temp = float(parts[3])
        except ValueError:
            return None, None, None

        vram_util = None
        if mem_total_mb > 0:
            vram_util = (mem_used_mb * 100.0) / mem_total_mb

        return gpu_util, vram_util, gpu_temp

    def _sample_cpu_temperature(self) -> Optional[float]:
        zones_path = Path("/sys/class/thermal")
        if not zones_path.exists():
            return None

        cpu_temps = []
        fallback_temps = []

        for zone_dir in zones_path.glob("thermal_zone*"):
            try:
                type_file = zone_dir / "type"
                temp_file = zone_dir / "temp"
                if type_file.exists() and temp_file.exists():
                    z_type = type_file.read_text(encoding="utf-8").strip().lower()
                    z_temp = float(temp_file.read_text(encoding="utf-8").strip()) / 1000.0

                    if "x86_pkg_temp" in z_type or "cpu" in z_type or "core" in z_type:
                        cpu_temps.append(z_temp)
                    else:
                        fallback_temps.append(z_temp)
            except Exception:
                continue

        if cpu_temps:
            return sum(cpu_temps) / len(cpu_temps)
        if fallback_temps:
            return sum(fallback_temps) / len(fallback_temps)
        return None

    def _round_or_none(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return round(float(value), 3)
