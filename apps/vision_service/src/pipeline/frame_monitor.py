from __future__ import annotations

import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import pyds

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.pipeline.osd_draw import attach_roi_polygon


logger = get_logger(__name__)


class FrameFlowMonitor:
    """
    Monitor dòng buffer đi qua một pad trong pipeline.

    Ghi chú quan trọng:
    - Probe này được gắn sau nvstreammux.
    - Vì vậy, mỗi Gst.Buffer ở đây là một "batch buffer".
    - Với 1 source, batch buffer rate gần như tương đương FPS.
    - Với nhiều source, đây chưa phải FPS riêng từng camera.
    """

    def __init__(self, log_interval_sec: float = 5.0, roi_polygon=None) -> None:
        self.log_interval_sec = max(log_interval_sec, 0.5)
        self._roi_polygon = roi_polygon  # None = no ROI display

        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self.last_buffer_at: float | None = None

        self.total_buffers = 0
        self.buffers_at_last_log = 0

    def attach(self, element: Gst.Element, pad_name: str = "src") -> None:
        pad = element.get_static_pad(pad_name)
        if pad is None:
            raise RuntimeError(
                f"Failed to get pad '{pad_name}' from element '{element.get_name()}'."
            )

        # Use ROI-aware probe if polygon is configured
        if self._roi_polygon:
            pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer_probe_with_roi, None)
            logger.info(
                "FrameFlowMonitor attached (with ROI) to element='%s', pad='%s'",
                element.get_name(),
                pad_name,
            )
        else:
            pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer_probe, None)
            logger.info(
                "Attached frame-flow monitor to element='%s', pad='%s'",
                element.get_name(),
                pad_name,
            )

    def _on_buffer_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        _user_data: object,
    ) -> Gst.PadProbeReturn:
        now = time.monotonic()

        self.total_buffers += 1
        self.last_buffer_at = now

        if self._should_log(now):
            self._log_status(now)

        return Gst.PadProbeReturn.OK

    def _on_buffer_probe_with_roi(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        _user_data: object,
    ) -> Gst.PadProbeReturn:
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            # Attach ROI polygon if configured
            if self._roi_polygon:
                attach_roi_polygon(batch_meta, frame_meta, self._roi_polygon)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _should_log(self, now: float) -> bool:
        return (now - self.last_log_at) >= self.log_interval_sec

    def _log_status(self, now: float) -> None:
        elapsed_total = max(now - self.started_at, 1e-6)
        elapsed_window = max(now - self.last_log_at, 1e-6)

        buffers_in_window = self.total_buffers - self.buffers_at_last_log

        interval_rate = buffers_in_window / elapsed_window
        overall_rate = self.total_buffers / elapsed_total

        logger.info(
            "Frame flow alive | total_buffers=%d | interval_rate=%.2f buf/s | overall_rate=%.2f buf/s",
            self.total_buffers,
            interval_rate,
            overall_rate,
        )

        self.last_log_at = now
        self.buffers_at_last_log = self.total_buffers