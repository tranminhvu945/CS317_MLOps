from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.pipeline.bus_handler import BusHandler
from apps.vision_service.src.pipeline.frame_monitor import FrameFlowMonitor
from apps.vision_service.src.pipeline.infer import create_primary_infer
from apps.vision_service.src.pipeline.rtmp_output import create_rtmp_output_chain
from apps.vision_service.src.pipeline.rtsp_output import create_rtsp_output_chain
from apps.vision_service.src.pipeline.source_hls import create_hls_source_bin
from apps.vision_service.src.pipeline.source_file import create_file_source_bin
from apps.vision_service.src.pipeline.tiler import create_tiler
from apps.vision_service.src.pipeline.tracker import create_tracker
from apps.vision_service.src.probes.infer_probe import InferProbe
from apps.vision_service.src.probes.stage_latency_probe import (
    MODE_META_FRAME,
    MODE_PTS_VIA_BUFFER,
    MODE_SOURCE_PTS,
    StageLatencyProbe,
)
from apps.vision_service.src.services.event_publisher import JsonlEventPublisher
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.gst_utils import make_element, request_streammux_sink_pad


logger = get_logger(__name__)


def _set_property_if_exists(element: Gst.Element, property_name: str, value: object) -> None:
    if element.find_property(property_name) is not None:
        element.set_property(property_name, value)


def _configure_live_queue(queue: Gst.Element, *, silent: bool = True) -> None:
    # Prefer smoother playback over lowest-possible latency.
    queue.set_property("max-size-buffers", 8)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    _set_property_if_exists(queue, "flush-on-eos", True)
    _set_property_if_exists(queue, "silent", silent)


@dataclass
class SourceBinding:
    camera_id: str
    source_bin: Gst.Bin
    mux_sink_pad: Gst.Pad


@dataclass
class QueueSignalStats:
    overrun_count: int = 0
    underrun_count: int = 0
    last_overrun_log_at: float = 0.0
    last_underrun_log_at: float = 0.0


class PipelineBuilder:
    def __init__(self, settings: RootSettings, main_loop: GLib.MainLoop) -> None:
        self.settings = settings
        self.main_loop = main_loop
        self.pipeline: Gst.Pipeline | None = None
        self.streammux: Gst.Element | None = None
        self.infer: Gst.Element | None = None
        self.tracker: Gst.Element | None = None
        self.queue: Gst.Element | None = None
        self.identity: Gst.Element | None = None
        self.source_bindings: List[SourceBinding] = []
        self.queue_signal_stats: Dict[str, QueueSignalStats] = {}

        self.bus_handler = BusHandler(main_loop)

        self.frame_monitor = FrameFlowMonitor(
            log_interval_sec=self.settings.pipeline.frame_log_interval_sec
        )
        self.event_publisher = JsonlEventPublisher(
            output_file=self.settings.events.output_file,
        )
        self.infer_probe: InferProbe | None = None
        self.stage_latency_probe: StageLatencyProbe | None = None

        self._infer_assets = None
        self.tiler: Gst.Element | None = None
        self.pre_osd_convert: Gst.Element | None = None
        self.pre_osd_capsfilter: Gst.Element | None = None
        self.osd: Gst.Element | None = None
        self.post_osd_convert: Gst.Element | None = None
        self.post_osd_capsfilter: Gst.Element | None = None
        self.pre_osd_queue: Gst.Element | None = None
        self.pre_encoder_queue: Gst.Element | None = None
        self.encoder: Gst.Element | None = None
        self.parser: Gst.Element | None = None
        self.rtmp_muxer: Gst.Element | None = None
        self.rtmp_sink: Gst.Element | None = None
        self.post_parser_tee: Gst.Element | None = None
        self.debug_h264_file_queue: Gst.Element | None = None
        self.debug_h264_file_sink: Gst.Element | None = None
        self.payloader: Gst.Element | None = None
        self.udpsink: Gst.Element | None = None
        self.rtsp_server = None

        Gst.init(None)

    def _queue_level_snapshot(self, queue: Gst.Element) -> tuple[int, int, int]:
        level_buffers = (
            int(queue.get_property("current-level-buffers"))
            if queue.find_property("current-level-buffers") is not None
            else -1
        )
        level_time = (
            int(queue.get_property("current-level-time"))
            if queue.find_property("current-level-time") is not None
            else -1
        )
        level_bytes = (
            int(queue.get_property("current-level-bytes"))
            if queue.find_property("current-level-bytes") is not None
            else -1
        )
        return level_buffers, level_time, level_bytes

    def _attach_queue_diagnostics(self, queue: Gst.Element, label: str) -> None:
        stats = QueueSignalStats()
        self.queue_signal_stats[label] = stats

        def _log_event(event_name: str) -> None:
            now = time.monotonic()
            level_buffers, level_time_ns, level_bytes = self._queue_level_snapshot(queue)
            level_time_ms = (level_time_ns / 1e6) if level_time_ns >= 0 else -1.0
            leaky_mode = (
                int(queue.get_property("leaky"))
                if queue.find_property("leaky") is not None
                else -1
            )
            log_fn = logger.warning if event_name == "overrun" else logger.info
            log_fn(
                "Queue %s | queue=%s | event=%s | overrun=%d | underrun=%d | "
                "level_buffers=%d | level_time_ms=%.2f | level_bytes=%d | leaky=%d",
                label,
                queue.get_name(),
                event_name,
                stats.overrun_count,
                stats.underrun_count,
                level_buffers,
                level_time_ms,
                level_bytes,
                leaky_mode,
            )
            if event_name == "overrun":
                stats.last_overrun_log_at = now
            elif event_name == "underrun":
                stats.last_underrun_log_at = now

        def _on_overrun(_queue: Gst.Element) -> None:
            stats.overrun_count += 1
            now = time.monotonic()
            should_log = (
                stats.overrun_count <= 3
                or (now - stats.last_overrun_log_at) >= 1.0
                or (stats.overrun_count % 20) == 0
            )
            if should_log:
                _log_event("overrun")

        def _on_underrun(_queue: Gst.Element) -> None:
            stats.underrun_count += 1
            now = time.monotonic()
            should_log = (
                stats.underrun_count <= 3
                or (now - stats.last_underrun_log_at) >= 2.0
                or (stats.underrun_count % 100) == 0
            )
            if should_log:
                _log_event("underrun")

        queue.connect("overrun", _on_overrun)
        queue.connect("underrun", _on_underrun)
        logger.info("Attached queue diagnostics | queue=%s | label=%s", queue.get_name(), label)

    def build(self) -> Gst.Pipeline:
        logger.info("Building DeepStream pipeline...")

        cameras = self.settings.cameras
        if not cameras:
            raise RuntimeError("No enabled cameras found in configs.")
        n_cameras = len(cameras)

        pipeline = Gst.Pipeline.new("helmet-violation-pipeline")
        if pipeline is None:
            raise RuntimeError("Failed to create Gst.Pipeline.")

        streammux = make_element("nvstreammux", "stream-muxer")
        infer = create_primary_infer(self.settings) if self.settings.infer.enabled else None
        tracker = create_tracker(self.settings) if self.settings.tracker.enabled else None
        queue = make_element("queue", "post-tracker-queue")
        _configure_live_queue(queue, silent=False)
        pre_osd_queue = make_element("queue", "pre-osd-queue")
        _configure_live_queue(pre_osd_queue, silent=False)
        identity = make_element("identity", "flow-identity")
        if identity.find_property("silent") is not None:
            identity.set_property("silent", True)

        if self.settings.pipeline.sink == "rtsp":
            output_chain = create_rtsp_output_chain(self.settings)
            self.payloader = output_chain.payloader
            self.udpsink = output_chain.udpsink
            self.rtsp_server = output_chain.server
            self.rtmp_muxer = None
            self.rtmp_sink = None
        elif self.settings.pipeline.sink == "rtmp":
            output_chain = create_rtmp_output_chain(self.settings)
            self.payloader = None
            self.udpsink = None
            self.rtsp_server = None
            self.rtmp_muxer = output_chain.muxer
            self.rtmp_sink = output_chain.sink
        else:
            raise RuntimeError(
                f"Unsupported pipeline sink: {self.settings.pipeline.sink}. Supported: rtsp, rtmp."
            )

        self.pre_osd_convert = output_chain.pre_osd_convert
        self.pre_osd_capsfilter = output_chain.pre_osd_capsfilter
        self.osd = output_chain.osd
        self.post_osd_convert = output_chain.post_osd_convert
        self.post_osd_capsfilter = output_chain.post_osd_capsfilter
        self.pre_encoder_queue = make_element("queue", "pre-encoder-queue")
        _configure_live_queue(self.pre_encoder_queue, silent=False)
        self.encoder = output_chain.encoder
        self.parser = output_chain.parser
        self.pre_osd_queue = pre_osd_queue
        self.post_parser_tee = None
        self.debug_h264_file_queue = None
        self.debug_h264_file_sink = None

        # --- Tiler: thêm nvmultistreamtiler cho multi-camera ---
        if n_cameras > 1 and self.settings.tiler.enabled:
            self.tiler = create_tiler(self.settings, n_cameras)
        else:
            self.tiler = None


        debug_h264_output_file = self.settings.rtsp.debug_h264_output_file.strip()
        if debug_h264_output_file:
            debug_output_path = Path(debug_h264_output_file).expanduser().resolve()
            debug_output_path.parent.mkdir(parents=True, exist_ok=True)

            self.post_parser_tee = make_element("tee", "post-parser-tee")
            self.debug_h264_file_queue = make_element("queue", "debug-h264-file-queue")
            _set_property_if_exists(self.debug_h264_file_queue, "flush-on-eos", True)
            _set_property_if_exists(self.debug_h264_file_queue, "silent", True)
            self.debug_h264_file_sink = make_element("filesink", "debug-h264-file-sink")
            self.debug_h264_file_sink.set_property("location", str(debug_output_path))
            self.debug_h264_file_sink.set_property("sync", False)
            self.debug_h264_file_sink.set_property("async", False)
            _set_property_if_exists(self.debug_h264_file_sink, "append", False)

            logger.info(
                "Debug H264 recording enabled | output=%s",
                debug_output_path,
            )

        self._attach_queue_diagnostics(queue, "post-tracker")
        self._attach_queue_diagnostics(pre_osd_queue, "pre-osd")
        self._attach_queue_diagnostics(self.pre_encoder_queue, "pre-encoder")

        streammux.set_property("batch-size", n_cameras)
        streammux.set_property("width", self.settings.pipeline.streammux_width)
        streammux.set_property("height", self.settings.pipeline.streammux_height)
        streammux.set_property(
            "batched-push-timeout",
            self.settings.pipeline.batched_push_timeout_usec,
        )

        is_live = any(c.stream.type in ("hls", "rtsp") for c in self.settings.cameras)
        streammux.set_property("live-source", 1 if is_live else 0)
        if is_live:
            _set_property_if_exists(streammux, "sync-inputs", False)
            _set_property_if_exists(streammux, "attach-sys-ts", True)
            _set_property_if_exists(streammux, "cache-buffer", False)
            _set_property_if_exists(streammux, "max-latency", 0)

        pipeline.add(streammux)
        if infer is not None:
            pipeline.add(infer)
        if tracker is not None:
            pipeline.add(tracker)
        pipeline.add(queue)
        pipeline.add(identity)
        if self.tiler is not None:
            pipeline.add(self.tiler)
        pipeline.add(pre_osd_queue)
        pipeline.add(self.pre_osd_convert)
        pipeline.add(self.pre_osd_capsfilter)
        pipeline.add(self.osd)
        pipeline.add(self.post_osd_convert)
        pipeline.add(self.post_osd_capsfilter)
        pipeline.add(self.pre_encoder_queue)
        pipeline.add(self.encoder)
        pipeline.add(self.parser)
        if self.rtmp_muxer is not None:
            pipeline.add(self.rtmp_muxer)
        if self.rtmp_sink is not None:
            pipeline.add(self.rtmp_sink)
        if self.post_parser_tee is not None:
            pipeline.add(self.post_parser_tee)
        if self.debug_h264_file_queue is not None:
            pipeline.add(self.debug_h264_file_queue)
        if self.debug_h264_file_sink is not None:
            pipeline.add(self.debug_h264_file_sink)
        if self.payloader is not None:
            pipeline.add(self.payloader)
        if self.udpsink is not None:
            pipeline.add(self.udpsink)

        if infer is not None:
            if not streammux.link(infer):
                raise RuntimeError("Failed to link streammux -> nvinfer.")
            if tracker is not None:
                if not infer.link(tracker):
                    raise RuntimeError("Failed to link nvinfer -> nvtracker.")
                if not tracker.link(queue):
                    raise RuntimeError("Failed to link nvtracker -> queue.")
            else:
                if not infer.link(queue):
                    raise RuntimeError("Failed to link nvinfer -> queue.")
        else:
            if tracker is not None:
                if not streammux.link(tracker):
                    raise RuntimeError("Failed to link streammux -> nvtracker.")
                if not tracker.link(queue):
                    raise RuntimeError("Failed to link nvtracker -> queue.")
            else:
                if not streammux.link(queue):
                    raise RuntimeError("Failed to link streammux -> queue.")

        if not queue.link(identity):
            raise RuntimeError("Failed to link queue -> identity.")

        # Nếu có tiler, chèn vào giữa identity và pre_osd_convert
        if self.tiler is not None:
            if not identity.link(self.tiler):
                raise RuntimeError("Failed to link identity -> nvmultistreamtiler.")
            if not self.tiler.link(self.pre_osd_convert):
                raise RuntimeError("Failed to link nvmultistreamtiler -> pre-osd-convert.")
        else:
            if not identity.link(self.pre_osd_convert):
                raise RuntimeError("Failed to link identity -> pre-osd-convert.")
        if not self.pre_osd_convert.link(self.pre_osd_capsfilter):
            raise RuntimeError("Failed to link pre-osd-convert -> pre-osd-capsfilter.")
        if not self.pre_osd_capsfilter.link(pre_osd_queue):
            raise RuntimeError("Failed to link pre-osd-capsfilter -> pre-osd-queue.")
        if not pre_osd_queue.link(self.osd):
            raise RuntimeError("Failed to link pre-osd-queue -> nvdsosd.")
        if not self.osd.link(self.post_osd_convert):
            raise RuntimeError("Failed to link nvdsosd -> post-osd-convert.")
        if not self.post_osd_convert.link(self.post_osd_capsfilter):
            raise RuntimeError("Failed to link post-osd-convert -> post-osd-capsfilter.")
        if not self.post_osd_capsfilter.link(self.pre_encoder_queue):
            raise RuntimeError("Failed to link post-osd-capsfilter -> pre-encoder-queue.")
        if not self.pre_encoder_queue.link(self.encoder):
            raise RuntimeError("Failed to link pre-encoder-queue -> encoder.")
        if not self.encoder.link(self.parser):
            raise RuntimeError("Failed to link encoder -> parser.")
        parser_output = self.parser
        if self.post_parser_tee is not None:
            if not self.parser.link(self.post_parser_tee):
                raise RuntimeError("Failed to link parser -> post-parser-tee.")
            parser_output = self.post_parser_tee
            if self.debug_h264_file_queue is None or self.debug_h264_file_sink is None:
                raise RuntimeError("Debug H264 branch elements are missing.")
            if not self.post_parser_tee.link(self.debug_h264_file_queue):
                raise RuntimeError("Failed to link post-parser-tee -> debug-h264-file-queue.")
            if not self.debug_h264_file_queue.link(self.debug_h264_file_sink):
                raise RuntimeError("Failed to link debug-h264-file-queue -> debug-h264-file-sink.")

        if self.settings.pipeline.sink == "rtsp":
            if self.payloader is None or self.udpsink is None:
                raise RuntimeError("RTSP output chain is incomplete.")
            if not parser_output.link(self.payloader):
                raise RuntimeError(f"Failed to link {parser_output.get_name()} -> rtph264pay.")
            if not self.payloader.link(self.udpsink):
                raise RuntimeError("Failed to link rtph264pay -> udpsink.")
        elif self.settings.pipeline.sink == "rtmp":
            if not hasattr(output_chain, "link_from"):
                raise RuntimeError("RTMP output chain does not support parser linking.")
            output_chain.link_from(parser_output)

        # --- Tạo và kết nối tất cả sources ---
        for idx, camera in enumerate(cameras):
            if camera.stream.type == "hls" or camera.stream.type == "rtsp":
                source_bin = create_hls_source_bin(
                    index=idx,
                    uri=str(camera.stream.uri),
                    drop_frame_interval=camera.stream.decoder_drop_frame_interval,
                )
            elif camera.stream.type == "file":
                source_bin = create_file_source_bin(
                    index=idx,
                    uri=str(camera.stream.uri),
                    loop=camera.stream.loop,
                )
            else:
                raise RuntimeError(f"Unsupported stream type: {camera.stream.type}")

            pipeline.add(source_bin)

            src_pad = source_bin.get_static_pad("src")
            if src_pad is None:
                raise RuntimeError(
                    f"Failed to get src pad from source bin: {camera.camera_id}"
                )

            mux_sink_pad = request_streammux_sink_pad(streammux, index=idx)
            result = src_pad.link(mux_sink_pad)
            if result != Gst.PadLinkReturn.OK:
                raise RuntimeError(
                    f"Failed to link source bin -> nvstreammux "
                    f"for camera={camera.camera_id}, result={result}"
                )

            self.source_bindings.append(
                SourceBinding(
                    camera_id=camera.camera_id,
                    source_bin=source_bin,
                    mux_sink_pad=mux_sink_pad,
                )
            )
            logger.info(
                "Source linked | idx=%d | camera_id=%s | uri=%s",
                idx,
                camera.camera_id,
                camera.stream.uri,
            )

        # Gắn stage latency probe cho source đầu tiên (representative)
        first_source_bin = self.source_bindings[0].source_bin
        source_decode_probe_target = first_source_bin.get_by_name("source-queue-00")
        source_decode_probe_pad = "src"
        if source_decode_probe_target is None:
            logger.warning(
                "source-queue-00 not found in source bin, fallback to source bin src pad for decode stage monitoring"
            )
            source_decode_probe_target = first_source_bin

        self.stage_latency_probe = StageLatencyProbe(
            log_interval_sec=self.settings.pipeline.frame_log_interval_sec,
        )

        self.stage_latency_probe.attach_stage(
            source_decode_probe_target,
            stage_key="after_decodebin",
            stage_label="Sau decodebin",
            mode=MODE_SOURCE_PTS,
            pad_name=source_decode_probe_pad,
            source_id=0,
        )

        self.stage_latency_probe.attach_stage(
            streammux,
            stage_key="after_nvstreammux",
            stage_label="Sau nvstreammux",
            mode=MODE_META_FRAME,
            pad_name="src",
        )

        if infer is not None:
            self.stage_latency_probe.attach_stage(
                infer,
                stage_key="after_nvinfer",
                stage_label="Sau nvinfer",
                mode=MODE_META_FRAME,
                pad_name="src",
            )

        if tracker is not None:
            self.stage_latency_probe.attach_stage(
                tracker,
                stage_key="after_nvtracker",
                stage_label="Sau nvtracker",
                mode=MODE_META_FRAME,
                pad_name="src",
            )

        self.stage_latency_probe.attach_stage(
            pre_osd_queue,
            stage_key="after_pre_osd_queue",
            stage_label="Sau pre-osd-queue",
            mode=MODE_META_FRAME,
            pad_name="src",
        )

        self.stage_latency_probe.attach_stage(
            self.osd,
            stage_key="after_nvdsosd",
            stage_label="Sau nvdsosd",
            mode=MODE_META_FRAME,
            pad_name="src",
        )

        self.stage_latency_probe.attach_stage(
            self.post_osd_convert,
            stage_key="after_post_osd_convert",
            stage_label="Sau post-osd-convert",
            mode=MODE_META_FRAME,
            pad_name="src",
        )

        self.stage_latency_probe.attach_stage(
            self.post_osd_capsfilter,
            stage_key="after_post_osd_caps",
            stage_label="Sau post-osd-capsfilter",
            mode=MODE_META_FRAME,
            pad_name="src",
        )

        self.stage_latency_probe.attach_stage(
            self.pre_encoder_queue,
            stage_key="after_pre_encoder_queue",
            stage_label="Sau pre-encoder-queue",
            mode=MODE_PTS_VIA_BUFFER,
            pad_name="src",
            source_id=0,
        )

        self.stage_latency_probe.attach_stage(
            self.encoder,
            stage_key="before_nvv4l2h264enc",
            stage_label="Truoc nvv4l2h264enc",
            mode=MODE_PTS_VIA_BUFFER,
            pad_name="sink",
            source_id=0,
        )

        self.stage_latency_probe.attach_stage(
            self.encoder,
            stage_key="after_nvv4l2h264enc",
            stage_label="Sau nvv4l2h264enc",
            mode=MODE_PTS_VIA_BUFFER,
            pad_name="src",
            source_id=0,
        )

        self.stage_latency_probe.attach_stage(
            self.parser,
            stage_key="after_h264parse",
            stage_label="Sau h264parse",
            mode=MODE_PTS_VIA_BUFFER,
            pad_name="src",
            source_id=0,
        )

        if self.payloader is not None:
            self.stage_latency_probe.attach_stage(
                self.payloader,
                stage_key="before_rtph264pay",
                stage_label="Truoc rtph264pay",
                mode=MODE_PTS_VIA_BUFFER,
                pad_name="sink",
                source_id=0,
            )

            self.stage_latency_probe.attach_stage(
                self.payloader,
                stage_key="after_rtph264pay",
                stage_label="Sau rtph264pay",
                mode=MODE_PTS_VIA_BUFFER,
                pad_name="src",
                source_id=0,
            )

        if self.udpsink is not None:
            self.stage_latency_probe.attach_stage(
                self.udpsink,
                stage_key="before_udpsink",
                stage_label="Trước udpsink",
                mode=MODE_PTS_VIA_BUFFER,
                pad_name="sink",
                source_id=0,
            )

        if self.rtmp_sink is not None and self.rtmp_muxer is not None:
            self.stage_latency_probe.attach_stage(
                self.rtmp_sink,
                stage_key="before_rtmp_sink",
                stage_label="Truoc RTMP sink",
                mode=MODE_PTS_VIA_BUFFER,
                pad_name="sink",
                source_id=0,
            )

        probe_element = tracker if tracker is not None else infer
        if probe_element is not None:
            self.infer_probe = InferProbe(
                settings=self.settings,
                publisher=self.event_publisher,
            )
            self.infer_probe.attach(probe_element, pad_name="src")

        self.frame_monitor.attach(identity, pad_name="src")
        self.bus_handler.attach(pipeline)

        # Đăng ký các file sources cần loop với BusHandler
        file_source_bins = [
            sb.source_bin
            for sb in self.source_bindings
            if getattr(sb.source_bin, "_loop", False)
        ]
        if file_source_bins:
            self.bus_handler.register_loop_sources(pipeline, file_source_bins)

        self.pipeline = pipeline
        self.streammux = streammux
        self.infer = infer
        self.tracker = tracker
        self.queue = queue
        self.identity = identity

        logger.info(
            "Pipeline built | n_cameras=%d | infer=%s | tracker=%s | tiler=%s | sink=%s",
            n_cameras,
            self.settings.infer.enabled,
            self.settings.tracker.enabled,
            self.tiler is not None,
            self.settings.pipeline.sink,
        )
        return pipeline

    def start(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been built yet.")

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set pipeline to PLAYING state.")

    def stop(self) -> None:
        if self.pipeline is None:
            logger.info("Pipeline was never created; nothing to stop.")
            return

        logger.info("Stopping pipeline...")
        try:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline.get_state(5 * Gst.SECOND)
        except Exception as exc:
            logger.warning("Error while stopping pipeline: %s", exc)
        finally:
            self.rtsp_server = None
            self.udpsink = None
            self.payloader = None
            self.rtmp_sink = None
            self.rtmp_muxer = None
            self.debug_h264_file_sink = None
            self.debug_h264_file_queue = None
            self.post_parser_tee = None
            self.parser = None
            self.encoder = None
            self.post_osd_capsfilter = None
            self.post_osd_convert = None
            self.osd = None
            self.pre_osd_capsfilter = None
            self.pre_osd_convert = None
            self.pre_osd_queue = None
            self.pre_encoder_queue = None
            self.identity = None
            self.queue = None
            self.tracker = None
            self.infer = None
            self.streammux = None
            self.source_bindings = []
            self.tiler = None
            self.pipeline = None
