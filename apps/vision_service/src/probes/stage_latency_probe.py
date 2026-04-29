from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import pyds

from apps.vision_service.src.logger import get_logger


logger = get_logger(__name__)

CLOCK_TIME_NONE = int(getattr(Gst, "CLOCK_TIME_NONE", 18446744073709551615))

FrameKey = Tuple[int, int]
PtsKey = Tuple[int, int]

MODE_SOURCE_PTS = "source_pts"
MODE_META_FRAME = "meta_frame"
MODE_PTS_VIA_BUFFER = "pts_via_buffer"


@dataclass
class _WindowStats:
    samples_ms: List[float] = field(default_factory=list)

    def add(self, value_ms: float) -> None:
        if value_ms >= 0:
            self.samples_ms.append(value_ms)

    def consume(self) -> Tuple[int, float, float, float]:
        if not self.samples_ms:
            return 0, 0.0, 0.0, 0.0

        values = self.samples_ms
        self.samples_ms = []

        values_sorted = sorted(values)
        count = len(values_sorted)
        avg = sum(values_sorted) / float(count)
        max_value = values_sorted[-1]
        p95_idx = max(0, min(count - 1, math.ceil(count * 0.95) - 1))
        p95 = values_sorted[p95_idx]
        return count, avg, p95, max_value


@dataclass
class _IntervalWindowStats:
    samples_ms: List[float] = field(default_factory=list)

    def add(self, value_ms: float) -> None:
        if value_ms >= 0:
            self.samples_ms.append(value_ms)

    def consume(self) -> Tuple[int, float, float, float, float]:
        if not self.samples_ms:
            return 0, 0.0, 0.0, 0.0, 0.0

        values = self.samples_ms
        self.samples_ms = []

        values_sorted = sorted(values)
        count = len(values_sorted)
        min_value = values_sorted[0]
        avg = sum(values_sorted) / float(count)
        max_value = values_sorted[-1]
        p95_idx = max(0, min(count - 1, math.ceil(count * 0.95) - 1))
        p95 = values_sorted[p95_idx]
        return count, min_value, avg, p95, max_value


class StageLatencyProbe:
    """
    Probe đo độ trễ theo từng mốc trong pipeline.

    Mốc hỗ trợ:
    - Sau decodebin (qua source queue src)
    - Sau nvstreammux
    - Sau nvinfer
    - Sau nvtracker
    - Sau nvdsosd
    - Sau nvv4l2h264enc
    - Trước udpsink
    """

    def __init__(self, log_interval_sec: float = 5.0) -> None:
        self.log_interval_sec = max(log_interval_sec, 1.0)
        self.stale_after_ns = int(max(30.0, self.log_interval_sec * 4.0) * 1e9)

        self.started_at = time.monotonic()
        self.last_log_at = self.started_at

        self.stage_order: List[str] = []
        self.stage_label: Dict[str, str] = {}
        self.stage_mode: Dict[str, str] = {}

        self.frame_stage_ts: Dict[str, Dict[FrameKey, int]] = {}
        self.pts_stage_ts: Dict[str, Dict[PtsKey, int]] = {}

        self.frame_to_pts: Dict[FrameKey, PtsKey] = {}
        self.pts_to_frame: Dict[PtsKey, FrameKey] = {}
        self.source_age_origin: Dict[int, Tuple[int, int]] = {}

        self.last_seen_frame_ns: Dict[FrameKey, int] = {}
        self.last_seen_pts_ns: Dict[PtsKey, int] = {}

        self.transition_stats: Dict[Tuple[str, str], _WindowStats] = {}
        self.stage_window_units: Dict[str, int] = {}
        self.stage_window_buffers: Dict[str, int] = {}
        self.stage_window_unique_pts: Dict[str, Set[PtsKey]] = {}
        self.stage_frame_age_stats: Dict[str, _WindowStats] = {}
        self.stage_event_queue_ns: Dict[str, Deque[int]] = {}
        self.timestamp_probe_stages: Set[str] = {
            "after_decodebin",
            "after_h264parse",
            "before_rtmp_sink",
        }
        self.timestamp_sample_every = 100
        self.stage_pts_delta_stats: Dict[str, _IntervalWindowStats] = {}
        self.stage_wall_delta_stats: Dict[str, _IntervalWindowStats] = {}
        self.stage_duration_stats: Dict[str, _IntervalWindowStats] = {}
        self.stage_last_pts_ns: Dict[str, Dict[int, int]] = {}
        self.stage_last_wall_ns: Dict[str, Dict[int, int]] = {}
        self.stage_timestamp_sample_count: Dict[str, Dict[int, int]] = {}
        self.max_stage_queue_size = 2048

    def attach_stage(
        self,
        element: Gst.Element,
        *,
        stage_key: str,
        stage_label: str,
        mode: str,
        pad_name: str = "src",
        source_id: int = 0,
    ) -> None:
        if mode not in (MODE_SOURCE_PTS, MODE_META_FRAME, MODE_PTS_VIA_BUFFER):
            raise ValueError(f"Unsupported stage mode: {mode}")

        pad = element.get_static_pad(pad_name)
        if pad is None:
            raise RuntimeError(
                f"Failed to get pad '{pad_name}' from element '{element.get_name()}'."
            )

        self.stage_order.append(stage_key)
        self.stage_label[stage_key] = stage_label
        self.stage_mode[stage_key] = mode

        self.frame_stage_ts.setdefault(stage_key, {})
        self.pts_stage_ts.setdefault(stage_key, {})
        self.stage_window_units.setdefault(stage_key, 0)
        self.stage_window_buffers.setdefault(stage_key, 0)
        self.stage_window_unique_pts.setdefault(stage_key, set())
        self.stage_event_queue_ns.setdefault(stage_key, deque())

        context = {
            "stage_key": stage_key,
            "mode": mode,
            "source_id": int(source_id),
        }

        pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer_probe, context)
        logger.info(
            "StageLatencyProbe attached | stage=%s | element=%s | pad=%s | mode=%s",
            stage_key,
            element.get_name(),
            pad_name,
            mode,
        )

    def _on_buffer_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        user_data: object,
    ) -> Gst.PadProbeReturn:
        context = user_data if isinstance(user_data, dict) else {}
        stage_key = str(context.get("stage_key", ""))
        mode = str(context.get("mode", ""))
        source_id = int(context.get("source_id", 0))

        if not stage_key:
            return Gst.PadProbeReturn.OK

        now_ns = time.monotonic_ns()
        gst_buffer = info.get_buffer()
        duration_ns = self._extract_duration_ns_from_buffer(gst_buffer)
        observed_buffers = 1 if gst_buffer is not None else 0
        observed_units = 0
        observed_pts_keys: List[PtsKey] = []

        if mode == MODE_SOURCE_PTS:
            if gst_buffer is not None:
                observed_units = 1
            pts_key = self._extract_pts_key_from_buffer(info, source_id)
            if pts_key is not None:
                observed_pts_keys.append(pts_key)
                self._record_stage_hit(
                    stage_key,
                    frame_key=None,
                    pts_key=pts_key,
                    duration_ns=duration_ns,
                    now_ns=now_ns,
                )

        elif mode == MODE_META_FRAME:
            entries = self._extract_frame_entries(info)
            observed_units = len(entries) if entries else (1 if gst_buffer is not None else 0)
            for frame_key, pts_key in entries:
                if pts_key is not None:
                    observed_pts_keys.append(pts_key)
                self._record_stage_hit(
                    stage_key,
                    frame_key=frame_key,
                    pts_key=pts_key,
                    duration_ns=duration_ns,
                    now_ns=now_ns,
                )

        elif mode == MODE_PTS_VIA_BUFFER:
            if gst_buffer is not None:
                observed_units = 1
            pts_key = self._extract_pts_key_from_buffer(info, source_id)
            if pts_key is not None:
                observed_pts_keys.append(pts_key)
                frame_key = self.pts_to_frame.get(pts_key)
                self._record_stage_hit(
                    stage_key,
                    frame_key=frame_key,
                    pts_key=pts_key,
                    duration_ns=duration_ns,
                    now_ns=now_ns,
                )

        self._record_stage_observation(
            stage_key,
            observed_units=observed_units,
            observed_buffers=observed_buffers,
            observed_pts_keys=observed_pts_keys,
        )
        self._maybe_log(now_ns)
        return Gst.PadProbeReturn.OK

    def _record_stage_observation(
        self,
        stage_key: str,
        *,
        observed_units: int,
        observed_buffers: int,
        observed_pts_keys: List[PtsKey],
    ) -> None:
        self.stage_window_units[stage_key] = self.stage_window_units.get(stage_key, 0) + max(
            observed_units,
            0,
        )
        self.stage_window_buffers[stage_key] = self.stage_window_buffers.get(stage_key, 0) + max(
            observed_buffers,
            0,
        )
        if observed_pts_keys:
            self.stage_window_unique_pts.setdefault(stage_key, set()).update(observed_pts_keys)

    def _record_stage_hit(
        self,
        stage_key: str,
        *,
        frame_key: Optional[FrameKey],
        pts_key: Optional[PtsKey],
        duration_ns: Optional[int],
        now_ns: int,
    ) -> None:
        if frame_key is not None:
            self.frame_stage_ts[stage_key][frame_key] = now_ns
            self.last_seen_frame_ns[frame_key] = now_ns

        if pts_key is not None:
            self.pts_stage_ts[stage_key][pts_key] = now_ns
            self.last_seen_pts_ns[pts_key] = now_ns

        if frame_key is not None and pts_key is not None:
            self.frame_to_pts[frame_key] = pts_key
            self.pts_to_frame[pts_key] = frame_key

        self._record_stage_intervals(
            stage_key=stage_key,
            pts_key=pts_key,
            duration_ns=duration_ns,
            now_ns=now_ns,
        )
        self._record_frame_age(stage_key=stage_key, pts_key=pts_key, now_ns=now_ns)
        self._push_stage_event(stage_key, now_ns)

        prev_stage = self._get_prev_stage(stage_key)
        if prev_stage is None:
            return

        prev_mode = self.stage_mode.get(prev_stage, "")
        curr_mode = self.stage_mode.get(stage_key, "")

        prev_ts_ns: Optional[int] = None

        if curr_mode == MODE_META_FRAME and prev_mode == MODE_SOURCE_PTS:
            if pts_key is not None:
                prev_ts_ns = self.pts_stage_ts[prev_stage].get(pts_key)

        elif curr_mode == MODE_PTS_VIA_BUFFER and prev_mode == MODE_META_FRAME:
            if frame_key is not None:
                prev_ts_ns = self.frame_stage_ts[prev_stage].get(frame_key)

        elif curr_mode == MODE_META_FRAME and prev_mode == MODE_META_FRAME:
            if frame_key is not None:
                prev_ts_ns = self.frame_stage_ts[prev_stage].get(frame_key)

        elif curr_mode == MODE_PTS_VIA_BUFFER and prev_mode == MODE_PTS_VIA_BUFFER:
            if pts_key is not None:
                prev_ts_ns = self.pts_stage_ts[prev_stage].get(pts_key)

        else:
            if frame_key is not None:
                prev_ts_ns = self.frame_stage_ts[prev_stage].get(frame_key)
            if prev_ts_ns is None and pts_key is not None:
                prev_ts_ns = self.pts_stage_ts[prev_stage].get(pts_key)

        if prev_ts_ns is None:
            prev_ts_ns = self._pop_prev_stage_event(prev_stage)
        else:
            self._pop_prev_stage_event(prev_stage)

        if prev_ts_ns is None:
            return

        delta_ms = (now_ns - prev_ts_ns) / 1e6
        if delta_ms < 0:
            return

        transition_key = (prev_stage, stage_key)
        stats = self.transition_stats.get(transition_key)
        if stats is None:
            stats = _WindowStats()
            self.transition_stats[transition_key] = stats
        stats.add(delta_ms)

    def _record_frame_age(
        self,
        *,
        stage_key: str,
        pts_key: Optional[PtsKey],
        now_ns: int,
    ) -> None:
        if pts_key is None:
            return

        source_id, pts_ns = pts_key
        origin = self.source_age_origin.get(source_id)
        if origin is None or pts_ns < origin[0]:
            self.source_age_origin[source_id] = (pts_ns, now_ns)
            age_ms = 0.0
        else:
            origin_pts_ns, origin_wall_ns = origin
            media_elapsed_ns = pts_ns - origin_pts_ns
            wall_elapsed_ns = now_ns - origin_wall_ns
            age_ms = (wall_elapsed_ns - media_elapsed_ns) / 1e6

        if age_ms < 0:
            return

        stats = self.stage_frame_age_stats.get(stage_key)
        if stats is None:
            stats = _WindowStats()
            self.stage_frame_age_stats[stage_key] = stats
        stats.add(age_ms)

    def _record_stage_intervals(
        self,
        *,
        stage_key: str,
        pts_key: Optional[PtsKey],
        duration_ns: Optional[int],
        now_ns: int,
    ) -> None:
        if stage_key not in self.timestamp_probe_stages or pts_key is None:
            return

        source_id, pts_ns = pts_key
        last_pts_by_source = self.stage_last_pts_ns.setdefault(stage_key, {})
        last_wall_by_source = self.stage_last_wall_ns.setdefault(stage_key, {})
        sample_count_by_source = self.stage_timestamp_sample_count.setdefault(stage_key, {})

        prev_pts_ns = last_pts_by_source.get(source_id)
        prev_wall_ns = last_wall_by_source.get(source_id)
        pts_delta_ms: Optional[float] = None
        wall_delta_ms: Optional[float] = None

        if prev_pts_ns is not None:
            pts_delta_ms = (pts_ns - prev_pts_ns) / 1e6
            if pts_delta_ms >= 0:
                stats = self.stage_pts_delta_stats.get(stage_key)
                if stats is None:
                    stats = _IntervalWindowStats()
                    self.stage_pts_delta_stats[stage_key] = stats
                stats.add(pts_delta_ms)
            else:
                pts_delta_ms = None

        if prev_wall_ns is not None:
            wall_delta_ms = (now_ns - prev_wall_ns) / 1e6
            if wall_delta_ms >= 0:
                stats = self.stage_wall_delta_stats.get(stage_key)
                if stats is None:
                    stats = _IntervalWindowStats()
                    self.stage_wall_delta_stats[stage_key] = stats
                stats.add(wall_delta_ms)
            else:
                wall_delta_ms = None

        duration_ms: Optional[float] = None
        if duration_ns is not None:
            duration_ms = duration_ns / 1e6
            if duration_ms >= 0:
                stats = self.stage_duration_stats.get(stage_key)
                if stats is None:
                    stats = _IntervalWindowStats()
                    self.stage_duration_stats[stage_key] = stats
                stats.add(duration_ms)
            else:
                duration_ms = None

        last_pts_by_source[source_id] = pts_ns
        last_wall_by_source[source_id] = now_ns

        sample_count = sample_count_by_source.get(source_id, 0) + 1
        sample_count_by_source[source_id] = sample_count
        if sample_count % self.timestamp_sample_every == 0:
            self._log_stage_timestamp_sample(
                stage_key=stage_key,
                source_id=source_id,
                sample_count=sample_count,
                pts_ns=pts_ns,
                pts_delta_ms=pts_delta_ms,
                wall_ns=now_ns,
                wall_delta_ms=wall_delta_ms,
                duration_ms=duration_ms,
            )

    def _log_stage_timestamp_sample(
        self,
        *,
        stage_key: str,
        source_id: int,
        sample_count: int,
        pts_ns: int,
        pts_delta_ms: Optional[float],
        wall_ns: int,
        wall_delta_ms: Optional[float],
        duration_ms: Optional[float],
    ) -> None:
        stage_label = self.stage_label.get(stage_key, stage_key)
        pts_ms = pts_ns / 1e6
        wall_ms = wall_ns / 1e6
        logger.info(
            "Stage timestamp sample | stage=%s | source=%d | sample=%d | "
            "pts_ms=%.2f | delta_pts_ms=%s | wall_ms=%.2f | delta_wall_ms=%s | duration_ms=%s",
            stage_label,
            source_id,
            sample_count,
            pts_ms,
            f"{pts_delta_ms:.2f}" if pts_delta_ms is not None else "None",
            wall_ms,
            f"{wall_delta_ms:.2f}" if wall_delta_ms is not None else "None",
            f"{duration_ms:.2f}" if duration_ms is not None else "None",
        )

    def _push_stage_event(self, stage_key: str, now_ns: int) -> None:
        queue = self.stage_event_queue_ns.setdefault(stage_key, deque())
        queue.append(now_ns)
        if len(queue) > self.max_stage_queue_size:
            queue.popleft()

    def _pop_prev_stage_event(self, stage_key: str) -> Optional[int]:
        queue = self.stage_event_queue_ns.get(stage_key)
        if not queue:
            return None
        return queue.popleft()

    def _extract_pts_key_from_buffer(
        self,
        info: Gst.PadProbeInfo,
        source_id: int,
    ) -> Optional[PtsKey]:
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return None

        pts_raw = int(gst_buffer.pts)
        if pts_raw == CLOCK_TIME_NONE:
            dts_raw = int(gst_buffer.dts)
            if dts_raw == CLOCK_TIME_NONE:
                return None
            pts_raw = dts_raw

        return source_id, pts_raw

    def _extract_duration_ns_from_buffer(self, gst_buffer: Optional[Gst.Buffer]) -> Optional[int]:
        if gst_buffer is None:
            return None

        duration_raw = int(gst_buffer.duration)
        if duration_raw == CLOCK_TIME_NONE:
            return None
        return duration_raw

    def _extract_frame_entries(
        self,
        info: Gst.PadProbeInfo,
    ) -> List[Tuple[FrameKey, Optional[PtsKey]]]:
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return []

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return []

        entries: List[Tuple[FrameKey, Optional[PtsKey]]] = []

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            source_id = int(frame_meta.source_id)
            frame_num = int(frame_meta.frame_num)
            frame_key: FrameKey = (source_id, frame_num)

            pts_raw = int(frame_meta.buf_pts)
            pts_key: Optional[PtsKey] = None
            if pts_raw != CLOCK_TIME_NONE:
                pts_key = (source_id, pts_raw)

            entries.append((frame_key, pts_key))

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return entries

    def _get_prev_stage(self, stage_key: str) -> Optional[str]:
        if stage_key not in self.stage_order:
            return None
        idx = self.stage_order.index(stage_key)
        if idx == 0:
            return None
        return self.stage_order[idx - 1]

    def _maybe_log(self, now_ns: int) -> None:
        now = now_ns / 1e9
        if (now - self.last_log_at) < self.log_interval_sec:
            return

        self._cleanup_stale(now_ns)

        summary_parts: List[str] = []
        for prev_stage, curr_stage in self._iter_transitions_in_order():
            stats = self.transition_stats.get((prev_stage, curr_stage))
            if stats is None:
                continue

            count, avg, p95, max_value = stats.consume()
            if count == 0:
                continue

            prev_label = self.stage_label.get(prev_stage, prev_stage)
            curr_label = self.stage_label.get(curr_stage, curr_stage)
            summary_parts.append(
                f"{prev_label} -> {curr_label}: n={count}, avg={avg:.2f}ms, p95={p95:.2f}ms, max={max_value:.2f}ms"
            )

        if summary_parts:
            logger.info("Stage latency | %s", " | ".join(summary_parts))

        rate_parts: List[str] = []
        elapsed_window = max(now - self.last_log_at, 1e-6)
        for stage_key in self.stage_order:
            stage_label = self.stage_label.get(stage_key, stage_key)
            units = self.stage_window_units.get(stage_key, 0)
            buffers = self.stage_window_buffers.get(stage_key, 0)
            unique_pts = len(self.stage_window_unique_pts.get(stage_key, set()))
            unit_rate = units / elapsed_window
            buffer_rate = buffers / elapsed_window
            frame_rate_by_pts = unique_pts / elapsed_window
            rate_parts.append(
                f"{stage_label}: units={units}, unit_rate={unit_rate:.2f}/s, "
                f"buffers={buffers}, buf_rate={buffer_rate:.2f}/s, "
                f"unique_pts={unique_pts}, frame_rate_pts={frame_rate_by_pts:.2f}/s"
            )

        if rate_parts:
            logger.info("Stage rate | %s", " | ".join(rate_parts))

        self._log_stage_timestamp_delta()
        self._log_output_frame_rate(elapsed_window)
        self._log_tracker_to_osd_drop()
        self._log_frame_age()

        self._reset_window_counters()
        self.last_log_at = now

    def _log_output_frame_rate(self, elapsed_window: float) -> None:
        parser_key = "after_h264parse"
        pay_sink_key = "before_rtph264pay"

        parser_pts = len(self.stage_window_unique_pts.get(parser_key, set()))
        pay_sink_pts = len(self.stage_window_unique_pts.get(pay_sink_key, set()))
        if parser_pts == 0 and pay_sink_pts == 0:
            return

        parser_fps = parser_pts / elapsed_window
        pay_sink_fps = pay_sink_pts / elapsed_window
        logger.info(
            "Output frame-rate (unique PTS) | h264parse_src=%.2f/s (%d) | rtph264pay_sink=%.2f/s (%d)",
            parser_fps,
            parser_pts,
            pay_sink_fps,
            pay_sink_pts,
        )

    def _log_stage_timestamp_delta(self) -> None:
        delta_parts: List[str] = []
        for stage_key in self.stage_order:
            if stage_key not in self.timestamp_probe_stages:
                continue

            pts_stats = self.stage_pts_delta_stats.get(stage_key)
            wall_stats = self.stage_wall_delta_stats.get(stage_key)
            duration_stats = self.stage_duration_stats.get(stage_key)
            pts_count, pts_min, pts_avg, pts_p95, pts_max = (
                pts_stats.consume() if pts_stats is not None else (0, 0.0, 0.0, 0.0, 0.0)
            )
            wall_count, wall_min, wall_avg, wall_p95, wall_max = (
                wall_stats.consume()
                if wall_stats is not None
                else (0, 0.0, 0.0, 0.0, 0.0)
            )
            duration_count, duration_min, duration_avg, duration_p95, duration_max = (
                duration_stats.consume()
                if duration_stats is not None
                else (0, 0.0, 0.0, 0.0, 0.0)
            )

            if pts_count == 0 and wall_count == 0 and duration_count == 0:
                continue

            stage_label = self.stage_label.get(stage_key, stage_key)
            delta_parts.append(
                f"{stage_label}: "
                f"pts_delta_ms(n={pts_count}, min={pts_min:.2f}, avg={pts_avg:.2f}, p95={pts_p95:.2f}, max={pts_max:.2f}) | "
                f"wall_delta_ms(n={wall_count}, min={wall_min:.2f}, avg={wall_avg:.2f}, p95={wall_p95:.2f}, max={wall_max:.2f}) | "
                f"duration_ms(n={duration_count}, min={duration_min:.2f}, avg={duration_avg:.2f}, p95={duration_p95:.2f}, max={duration_max:.2f})"
            )

        if delta_parts:
            logger.info("Stage timestamp delta | %s", " | ".join(delta_parts))

    def _log_tracker_to_osd_drop(self) -> None:
        self._log_drop_between("after_nvtracker", "after_pre_osd_queue")
        self._log_drop_between("after_pre_osd_queue", "after_nvdsosd")
        self._log_drop_between("after_nvtracker", "after_nvdsosd")

    def _log_drop_between(self, in_stage: str, out_stage: str) -> None:
        if in_stage not in self.stage_window_units or out_stage not in self.stage_window_units:
            return

        in_count = self.stage_window_units.get(in_stage, 0)
        out_count = self.stage_window_units.get(out_stage, 0)
        if in_count <= 0:
            return

        drop_count = max(0, in_count - out_count)
        drop_ratio = (drop_count * 100.0) / float(in_count)
        logger.info(
            "Stage drop | %s -> %s | in=%d | out=%d | drop=%d | drop_ratio=%.2f%%",
            self.stage_label.get(in_stage, in_stage),
            self.stage_label.get(out_stage, out_stage),
            in_count,
            out_count,
            drop_count,
            drop_ratio,
        )

    def _log_frame_age(self) -> None:
        age_parts: List[str] = []
        for stage_key in self.stage_order:
            stats = self.stage_frame_age_stats.get(stage_key)
            if stats is None:
                continue
            count, avg, p95, max_value = stats.consume()
            if count == 0:
                continue
            stage_label = self.stage_label.get(stage_key, stage_key)
            age_parts.append(
                f"{stage_label}: n={count}, avg={avg:.2f}ms, p95={p95:.2f}ms, max={max_value:.2f}ms"
            )

        if age_parts:
            logger.info("Frame age | %s", " | ".join(age_parts))

    def _reset_window_counters(self) -> None:
        for stage_key in self.stage_window_units:
            self.stage_window_units[stage_key] = 0
        for stage_key in self.stage_window_buffers:
            self.stage_window_buffers[stage_key] = 0
        for stage_key in self.stage_window_unique_pts:
            self.stage_window_unique_pts[stage_key].clear()

    def _iter_transitions_in_order(self) -> List[Tuple[str, str]]:
        transitions: List[Tuple[str, str]] = []
        if len(self.stage_order) < 2:
            return transitions
        for idx in range(1, len(self.stage_order)):
            transitions.append((self.stage_order[idx - 1], self.stage_order[idx]))
        return transitions

    def _cleanup_stale(self, now_ns: int) -> None:
        stale_frames = [
            frame_key
            for frame_key, last_seen in self.last_seen_frame_ns.items()
            if (now_ns - last_seen) > self.stale_after_ns
        ]
        for frame_key in stale_frames:
            self.last_seen_frame_ns.pop(frame_key, None)
            pts_key = self.frame_to_pts.pop(frame_key, None)
            if pts_key is not None:
                self.pts_to_frame.pop(pts_key, None)
            for stage_map in self.frame_stage_ts.values():
                stage_map.pop(frame_key, None)

        stale_pts = [
            pts_key
            for pts_key, last_seen in self.last_seen_pts_ns.items()
            if (now_ns - last_seen) > self.stale_after_ns
        ]
        for pts_key in stale_pts:
            self.last_seen_pts_ns.pop(pts_key, None)
            frame_key = self.pts_to_frame.pop(pts_key, None)
            if frame_key is not None:
                self.frame_to_pts.pop(frame_key, None)
            for stage_map in self.pts_stage_ts.values():
                stage_map.pop(pts_key, None)
