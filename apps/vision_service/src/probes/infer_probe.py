from __future__ import annotations

import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from apps.vision_service.src.pipeline.osd_draw import (
    apply_safe_style,
    apply_tracking_label,
    apply_violation_style,
    attach_roi_polygon,
    attach_fps_label,
)

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import pyds

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.pipeline.infer import inspect_nvinfer_assets
from apps.vision_service.src.pipeline.osd_draw import (
    apply_safe_style,
    apply_tracking_label,
    apply_violation_style,
    attach_roi_polygon,
)
from apps.vision_service.src.services.event_publisher import JsonlEventPublisher
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
    ) -> None:
        self.settings = settings
        self.publisher = publisher
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

        self.fps_last_at = time.monotonic()
        self.fps_frames = 0
        self.current_fps = 0.0
        self.window_probe_callback_ms: List[float] = []

    def _update_fps(self) -> float:
        self.fps_frames += 1
        now = time.monotonic()
        elapsed = now - self.fps_last_at

        if elapsed >= 1.0:
            self.current_fps = self.fps_frames / elapsed
            self.fps_frames = 0
            self.fps_last_at = now

        return self.current_fps


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
            fps = self._update_fps()
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
                    label = f"{NO_HELMET_LABEL}{track_str}"
                    apply_violation_style(obj_meta)
                    apply_tracking_label(obj_meta, NO_HELMET_LABEL, track_id)

                    self.publisher.publish(
                        event_type="helmet_violation",
                        payload={
                            "event_id": str(uuid.uuid4()),
                            "camera_id": camera_id,
                            "track_id": track_id,
                            "confidence": confidence,
                            "bbox": [left, top, width, height],
                        },
                    )
                else:
                    label = f"{SAFE_OSD_LABEL}{track_str}"
                    apply_safe_style(obj_meta)
                    apply_tracking_label(obj_meta, SAFE_OSD_LABEL, track_id)

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
