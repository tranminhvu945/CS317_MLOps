from __future__ import annotations

import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import pyds

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.pipeline.infer import inspect_nvinfer_assets
from apps.vision_service.src.pipeline.osd_draw import (
    apply_safe_style,
    attach_fps_label,
    attach_roi_polygon,
    apply_tracking_label,
    apply_violation_style,
)
from apps.vision_service.src.services.event_publisher import JsonlEventPublisher
from apps.vision_service.src.services.redis_alert_publisher import RedisAlertPublisher
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.geometry import (
    bbox_anchor_bottom_center,
    point_in_polygon,
)


logger = get_logger(__name__)

UINT64_MAX = 18446744073709551615

PGIE_UNIQUE_ID = 1

NO_HELMET_LABEL = "no_helmet"
SAFE_OSD_LABEL = "helmet"


class InferProbe:
    def __init__(
        self,
        settings: RootSettings,
        publisher: JsonlEventPublisher,
        redis_alert_publisher: RedisAlertPublisher | None = None,
    ) -> None:
        self.settings = settings
        self.publisher = publisher
        self.redis_alert_publisher = redis_alert_publisher
        self.pgie_labels = self._load_labels(settings.infer.config_file)

        if 0 not in self.pgie_labels or 1 not in self.pgie_labels:
            raise RuntimeError(
                "YOLOv8 helmet model requires labels at index 0 (Safe_Motorcycle) "
                f"and 1 (Violation_Motorcycle). Found: {self.pgie_labels}"
            )

        self.started_at = time.monotonic()
        self.last_summary_at = self.started_at
        self.window_frames = 0
        self.window_objects = 0
        self.window_counts_by_label: Counter[str] = Counter()

        self.fps_last_at: dict[int, float] = {}
        self.fps_frames: dict[int, int] = {}
        self.current_fps: dict[int, float] = {}
        self.window_probe_callback_ms: List[float] = []

        # Telegram alert anti-noise state:
        import os
        min_consec_env = os.environ.get("MIN_CONSECUTIVE_NO_HELMET_FRAMES") or os.environ.get("TELEGRAM_MIN_CONSEC_NO_HELMET_FRAMES")
        if min_consec_env is not None:
            self.telegram_min_consecutive_frames = max(1, int(min_consec_env))
        else:
            self.telegram_min_consecutive_frames = max(
                1, int(self.settings.telegram.min_consecutive_no_helmet_frames)
            )
        snapshot_src = getattr(self.settings.telegram, "snapshot_source", "")
        if snapshot_src != "probe":
            logger.warning("[LATENCY][WARNING] snapshot_source=%s may add RTMP/HLS buffering delay", snapshot_src)

        self._violation_streak_by_track: Dict[str, int] = {}
        self._alerted_track_keys: set[str] = set()
        self._last_seen_frame_by_track: Dict[str, int] = {}
        self._streak_gc_after_frames = 120
        self._last_snapshot_extract_warn_at = 0.0
        self._snapshot_extract_warn_interval_sec = 10.0

    def _update_fps(self, source_id: int) -> float:
        """Tính FPS riêng biệt cho từng camera (source_id)."""
        now = time.monotonic()

        if source_id not in self.fps_last_at:
            self.fps_last_at[source_id] = now
            self.fps_frames[source_id] = 0
            self.current_fps[source_id] = 0.0

        self.fps_frames[source_id] += 1
        elapsed = now - self.fps_last_at[source_id]

        if elapsed >= 1.0:
            self.current_fps[source_id] = self.fps_frames[source_id] / elapsed
            self.fps_frames[source_id] = 0
            self.fps_last_at[source_id] = now

        return self.current_fps[source_id]


    def attach(self, element: Gst.Element, pad_name: str = "src") -> None:
        pad = element.get_static_pad(pad_name)
        if pad is None:
            raise RuntimeError(
                f"Failed to get pad '{pad_name}' from element '{element.get_name()}'."
            )
        pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer_probe, None)
        logger.info(
            "InferProbe attached to element='%s', pad='%s'",
            element.get_name(),
            pad_name,
        )

    def _load_labels(self, infer_config_file: str) -> Dict[int, str]:
        assets = inspect_nvinfer_assets(infer_config_file)
        label_path = assets.resolved_paths.get("labelfile-path")
        if label_path is None:
            raise RuntimeError(
                f"'labelfile-path' is missing from nvinfer config: {infer_config_file}"
            )

        labels: Dict[int, str] = {}
        for idx, line in enumerate(
            Path(label_path).read_text(encoding="utf-8").splitlines()
        ):
            label = line.strip()
            if not label:
                continue
            labels[idx] = label

        if not labels:
            raise RuntimeError(f"No labels found in: {label_path}")

        logger.info("Loaded %d labels from %s", len(labels), label_path)
        return labels

    def _get_frame_resolution(self, frame_meta: Any) -> Tuple[float, float]:
        pad_width = float(getattr(frame_meta, "pad_width", 0) or 0)
        pad_height = float(getattr(frame_meta, "pad_height", 0) or 0)

        if pad_width > 0 and pad_height > 0:
            return pad_width, pad_height

        return (
            float(self.settings.pipeline.streammux_width),
            float(self.settings.pipeline.streammux_height),
        )

    def _get_roi_polygon(
        self,
        camera_id: str,
        canvas_w: float,
        canvas_h: float,
        source_w: float,
        source_h: float,
    ) -> List[Tuple[float, float]]:
        """
        Transform ROI polygon from camera.yaml space
        to canvas space (nvstreammux output).

        If source_w/source_h are unavailable, caller should already have
        fallen back to canvas size, so scaling still stays safe.
        """
        for camera in self.settings.cameras:
            if camera.camera_id != camera_id:
                continue

            detection = getattr(camera, "detection", None)
            if detection is None:
                return []

            roi = getattr(detection, "roi", None)
            if roi is None or not getattr(roi, "enabled", False):
                return []

            polygon = getattr(roi, "polygon", None)
            if not polygon:
                return []

            if source_w <= 0 or source_h <= 0:
                return [(float(x), float(y)) for x, y in polygon]

            return [
                (float(x) * canvas_w / source_w, float(y) * canvas_h / source_h)
                for x, y in polygon
            ]

        return []

    def _get_min_confidence(self, camera_id: str) -> float:
        for camera in self.settings.cameras:
            if camera.camera_id != camera_id:
                continue
            detection = getattr(camera, "detection", None)
            if detection is not None:
                return float(getattr(detection, "min_confidence", 0.25))
        return 0.25

    def _resolve_camera_id(self, frame_meta: Any) -> str:
        source_id = int(frame_meta.source_id)
        if 0 <= source_id < len(self.settings.cameras):
            return self.settings.cameras[source_id].camera_id
        return f"source_{source_id}"

    def _extract_probe_snapshot_frame(
        self,
        gst_buffer: Gst.Buffer,
        frame_meta: Any,
    ) -> Any | None:
        """
        Extract a copy of current frame surface from pipeline buffer.
        Returned frame is used by RedisAlertPublisher thread to write snapshot.
        """
        now = time.monotonic()
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            if (now - self._last_snapshot_extract_warn_at) >= self._snapshot_extract_warn_interval_sec:
                logger.warning("numpy not installed — cannot extract probe snapshot frame")
                self._last_snapshot_extract_warn_at = now
            return None

        try:
            surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            if surface is None:
                return None
            return np.array(surface, copy=True, order="C")
        except Exception as exc:  # noqa: BLE001
            if (now - self._last_snapshot_extract_warn_at) >= self._snapshot_extract_warn_interval_sec:
                logger.warning(
                    "Failed to extract probe snapshot frame (likely non-RGBA pad format): %s",
                    exc,
                )
                self._last_snapshot_extract_warn_at = now
            return None

    def _build_track_key(
        self,
        camera_id: str,
        track_id: Optional[int],
        left: float,
        top: float,
        width: float,
        height: float,
    ) -> str:
        if track_id is not None:
            return f"{camera_id}|track:{track_id}"

        # Fallback for anonymous objects (rare when tracker is enabled):
        # use quantized bbox center to keep a stable key across nearby frames.
        center_x = left + (width * 0.5)
        center_y = top + (height * 0.5)
        qx = int(center_x / 32.0)
        qy = int(center_y / 32.0)
        return f"{camera_id}|anon:{qx}:{qy}"

    def _cleanup_stale_track_state(self, camera_id: str, frame_num: int) -> None:
        prefix = f"{camera_id}|"
        stale_keys: List[str] = []
        for key, last_seen in self._last_seen_frame_by_track.items():
            if not key.startswith(prefix):
                continue
            if (frame_num - last_seen) > self._streak_gc_after_frames:
                stale_keys.append(key)

        for key in stale_keys:
            self._last_seen_frame_by_track.pop(key, None)
            self._violation_streak_by_track.pop(key, None)
            self._alerted_track_keys.discard(key)

    def _should_emit_telegram_alert(
        self,
        *,
        camera_id: str,
        frame_num: int,
        class_id: int,
        track_id: Optional[int],
        left: float,
        top: float,
        width: float,
        height: float,
    ) -> bool:
        if self.redis_alert_publisher is None:
            return False

        key = self._build_track_key(
            camera_id=camera_id,
            track_id=track_id,
            left=left,
            top=top,
            width=width,
            height=height,
        )
        self._last_seen_frame_by_track[key] = frame_num

        # Any non-violation on this track resets alert state.
        if class_id != 1:
            self._violation_streak_by_track[key] = 0
            self._alerted_track_keys.discard(key)
            return False

        streak = self._violation_streak_by_track.get(key, 0) + 1
        self._violation_streak_by_track[key] = streak

        if streak < self.telegram_min_consecutive_frames:
            return False
        if key in self._alerted_track_keys:
            return False

        self._alerted_track_keys.add(key)
        return True

    def _on_buffer_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        _user_data: object,
    ) -> Gst.PadProbeReturn:
        callback_started_ns = time.perf_counter_ns()
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            self._record_callback_time(callback_started_ns)
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            self._record_callback_time(callback_started_ns)
            return Gst.PadProbeReturn.OK

        canvas_w = float(self.settings.pipeline.streammux_width)
        canvas_h = float(self.settings.pipeline.streammux_height)

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            camera_id = self._resolve_camera_id(frame_meta)
            frame_num = int(frame_meta.frame_num)
            self._cleanup_stale_track_state(camera_id=camera_id, frame_num=frame_num)
            source_id = int(frame_meta.source_id)
            fps = self._update_fps(source_id)
            if self.settings.visualization.enabled:
                attach_fps_label(batch_meta, frame_meta, fps)


            source_w, source_h = self._get_frame_resolution(frame_meta)

            roi_polygon = self._get_roi_polygon(
                camera_id, canvas_w, canvas_h, source_w, source_h
            )
            min_confidence = self._get_min_confidence(camera_id)
            window_counts: Counter[str] = Counter()

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                rect_params = obj_meta.rect_params
                text_params = obj_meta.text_params

                # Always clear previous OSD state before filtering object metadata.
                rect_params.border_width = 0
                if hasattr(rect_params, "has_bg_color"):
                    rect_params.has_bg_color = 0

                text_params.display_text = ""
                if hasattr(text_params, "set_bg_clr"):
                    text_params.set_bg_clr = 0

                unique_id = int(obj_meta.unique_component_id)
                if unique_id != PGIE_UNIQUE_ID:
                    l_obj = l_obj.next
                    continue

                class_id = int(obj_meta.class_id)
                confidence = float(obj_meta.confidence)

                rect = obj_meta.rect_params
                left = float(rect.left)
                top = float(rect.top)
                width = float(rect.width)
                height = float(rect.height)

                if confidence < min_confidence:
                    l_obj = l_obj.next
                    continue

                track_id: Optional[int] = None
                if int(obj_meta.object_id) != UINT64_MAX:
                    track_id = int(obj_meta.object_id)

                in_roi = True
                if roi_polygon:
                    anchor = bbox_anchor_bottom_center(left, top, width, height)
                    in_roi = point_in_polygon(anchor, roi_polygon)

                if not in_roi:
                    l_obj = l_obj.next
                    continue

                track_str = f" #{track_id}" if track_id is not None else ""

                if class_id == 1:
                    ts_detect = time.time()
                    ts_ms = int(ts_detect * 1000)
                    event_id = f"violation_{camera_id}_{track_id if track_id is not None else 'anon'}_{ts_ms}"
                    label = f"{NO_HELMET_LABEL}{track_str}"
                    apply_violation_style(obj_meta)
                    apply_tracking_label(obj_meta, NO_HELMET_LABEL, track_id)

                    event_payload = {
                        "event_id": event_id,
                        "camera_id": camera_id,
                        "track_id": track_id,
                        "confidence": confidence,
                        "frame_num": frame_num,
                        "bbox": [left, top, width, height],
                    }
                    self.publisher.publish(
                        event_type="helmet_violation",
                        payload=event_payload,
                    )
                    if self._should_emit_telegram_alert(
                        camera_id=camera_id,
                        frame_num=frame_num,
                        class_id=class_id,
                        track_id=track_id,
                        left=left,
                        top=top,
                        width=width,
                        height=height,
                    ):
                        snapshot_frame = self._extract_probe_snapshot_frame(
                            gst_buffer=gst_buffer,
                            frame_meta=frame_meta,
                        )
                        event = {
                            "event_type": "helmet_violation",
                            "event_id": event_id,
                            "camera_id": camera_id,
                            "track_id": track_id,
                            "timestamp": ts_detect,
                            "confidence": confidence,
                            "frame_num": frame_num,
                            "bbox": [left, top, width, height],
                        }
                        event["ts_detect"] = ts_detect
                        event["ts_detect_readable"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_detect))

                        streak_key = self._build_track_key(
                            camera_id=camera_id,
                            track_id=track_id,
                            left=left,
                            top=top,
                            width=width,
                            height=height,
                        )
                        current_streak = self._violation_streak_by_track.get(streak_key, 0)
                        logger.info(
                            f"[LATENCY][DETECT] "
                            f"event_id={event_id} "
                            f"camera_id={camera_id} "
                            f"track_id={track_id} "
                            f"frame_num={frame_num} "
                            f"ts_detect={ts_detect:.6f} "
                            f"streak={current_streak} "
                            f"min_consec={self.telegram_min_consecutive_frames}"
                        )
                        
                        min_frames = self.telegram_min_consecutive_frames
                        fps_val = fps if fps > 0 else 30.0
                        estimated_delay = min_frames / fps_val
                        logger.info(
                            f"[LATENCY][STREAK_FILTER] min_frames={min_frames} fps={fps_val:.2f} estimated_delay_sec={estimated_delay:.2f}"
                        )

                        t_before_enqueue = time.time()
                        event["ts_before_enqueue"] = t_before_enqueue
                        self.redis_alert_publisher.enqueue_violation(
                            event,
                            snapshot_frame=snapshot_frame,
                        )
                        t_after_enqueue = time.time()
                        event["ts_after_enqueue"] = t_after_enqueue

                        logger.info(
                            f"[LATENCY][ENQUEUE_CALL] "
                            f"event_id={event_id} "
                            f"enqueue_cost_ms={(t_after_enqueue - t_before_enqueue) * 1000:.2f}"
                        )
                else:
                    label = f"{SAFE_OSD_LABEL}{track_str}"
                    apply_safe_style(obj_meta)
                    apply_tracking_label(obj_meta, SAFE_OSD_LABEL, track_id)
                    self._should_emit_telegram_alert(
                        camera_id=camera_id,
                        frame_num=frame_num,
                        class_id=class_id,
                        track_id=track_id,
                        left=left,
                        top=top,
                        width=width,
                        height=height,
                    )

                window_counts[label] += 1
                self.window_objects += 1

                l_obj = l_obj.next

            self.window_frames += 1
            self.window_counts_by_label.update(window_counts)

            now = time.monotonic()
            if (now - self.last_summary_at) >= self.settings.infer.summary_interval_sec:
                self._emit_summary(now)

            if self.settings.visualization.enabled and roi_polygon:
                attach_roi_polygon(batch_meta, frame_meta, roi_polygon)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        self._record_callback_time(callback_started_ns)
        return Gst.PadProbeReturn.OK

    def _record_callback_time(self, started_ns: int) -> None:
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
        if elapsed_ms >= 0:
            self.window_probe_callback_ms.append(elapsed_ms)

    def _summarize_window_ms(self, values: List[float]) -> Tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0

        sorted_values = sorted(values)
        count = len(sorted_values)
        avg_ms = sum(sorted_values) / float(count)
        p95_idx = max(0, min(count - 1, ((count * 95 + 99) // 100) - 1))
        p95_ms = sorted_values[p95_idx]
        max_ms = sorted_values[-1]
        return avg_ms, p95_ms, max_ms

    def _emit_summary(self, now: float) -> None:
        elapsed = max(now - self.last_summary_at, 1e-6)
        fps_like = self.window_frames / elapsed
        cb_avg_ms, cb_p95_ms, cb_max_ms = self._summarize_window_ms(
            self.window_probe_callback_ms
        )

        self.publisher.publish(
            event_type="detection_window_summary",
            payload={
                "window_sec": round(elapsed, 3),
                "frames": self.window_frames,
                "objects": self.window_objects,
                "counts_by_label": dict(self.window_counts_by_label),
                "buffer_rate": round(fps_like, 3),
                "probe_callback_ms": {
                    "avg": round(cb_avg_ms, 4),
                    "p95": round(cb_p95_ms, 4),
                    "max": round(cb_max_ms, 4),
                },
            },
        )

        logger.info(
            "Detection summary | frames=%d | objects=%d | counts=%s | rate=%.2f/s | probe_cb_ms(avg/p95/max)=%.3f/%.3f/%.3f",
            self.window_frames,
            self.window_objects,
            dict(self.window_counts_by_label),
            fps_like,
            cb_avg_ms,
            cb_p95_ms,
            cb_max_ms,
        )

        self.last_summary_at = now
        self.window_frames = 0
        self.window_objects = 0
        self.window_counts_by_label = Counter()
        self.window_probe_callback_ms = []
