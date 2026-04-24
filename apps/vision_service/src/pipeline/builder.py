from __future__ import annotations

from dataclasses import dataclass
from typing import List

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.pipeline.bus_handler import BusHandler
from apps.vision_service.src.pipeline.frame_monitor import FrameFlowMonitor
from apps.vision_service.src.pipeline.infer import create_primary_infer
from apps.vision_service.src.pipeline.rtsp_output import create_rtsp_output_chain
from apps.vision_service.src.pipeline.source_hls import create_hls_source_bin
from apps.vision_service.src.pipeline.source_file import create_file_source_bin
from apps.vision_service.src.pipeline.tracker import create_tracker
from apps.vision_service.src.probes.infer_probe import InferProbe
from apps.vision_service.src.services.event_publisher import JsonlEventPublisher
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.gst_utils import make_element, request_streammux_sink_pad


logger = get_logger(__name__)


@dataclass
class SourceBinding:
    camera_id: str
    source_bin: Gst.Bin
    mux_sink_pad: Gst.Pad


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

        self.bus_handler = BusHandler(main_loop)

        self.frame_monitor = FrameFlowMonitor(
            log_interval_sec=self.settings.pipeline.frame_log_interval_sec
        )
        self.event_publisher = JsonlEventPublisher(
            output_file=self.settings.events.output_file,
        )
        self.infer_probe: InferProbe | None = None

        self._infer_assets = None
        self.pre_osd_convert: Gst.Element | None = None
        self.pre_osd_capsfilter: Gst.Element | None = None
        self.osd: Gst.Element | None = None
        self.post_osd_convert: Gst.Element | None = None
        self.post_osd_capsfilter: Gst.Element | None = None
        self.encoder: Gst.Element | None = None
        self.parser: Gst.Element | None = None
        self.payloader: Gst.Element | None = None
        self.udpsink: Gst.Element | None = None
        self.rtsp_server = None

        Gst.init(None)

    def build(self) -> Gst.Pipeline:
        logger.info("Building DeepStream pipeline...")

        if not self.settings.cameras:
            raise RuntimeError("No enabled cameras found in configs.")
        if len(self.settings.cameras) > 1:
            raise RuntimeError(
                f"Only single-source is supported. Found {len(self.settings.cameras)} cameras."
            )

        pipeline = Gst.Pipeline.new("helmet-violation-pipeline")
        if pipeline is None:
            raise RuntimeError("Failed to create Gst.Pipeline.")

        streammux = make_element("nvstreammux", "stream-muxer")
        infer = create_primary_infer(self.settings) if self.settings.infer.enabled else None
        tracker = create_tracker(self.settings) if self.settings.tracker.enabled else None
        queue = make_element("queue", "post-tracker-queue")
        identity = make_element("identity", "flow-identity")
        if identity.find_property("silent") is not None:
            identity.set_property("silent", True)

        output_chain = create_rtsp_output_chain(self.settings)
        self.pre_osd_convert = output_chain.pre_osd_convert
        self.pre_osd_capsfilter = output_chain.pre_osd_capsfilter
        self.osd = output_chain.osd
        self.post_osd_convert = output_chain.post_osd_convert
        self.post_osd_capsfilter = output_chain.post_osd_capsfilter
        self.encoder = output_chain.encoder
        self.parser = output_chain.parser
        self.payloader = output_chain.payloader
        self.udpsink = output_chain.udpsink
        self.rtsp_server = output_chain.server

        streammux.set_property("batch-size", 1)
        streammux.set_property("width", self.settings.pipeline.streammux_width)
        streammux.set_property("height", self.settings.pipeline.streammux_height)
        streammux.set_property(
            "batched-push-timeout",
            self.settings.pipeline.batched_push_timeout_usec,
        )

        is_live = any(c.stream.type in ("hls", "rtsp") for c in self.settings.cameras)
        streammux.set_property("live-source", 1 if is_live else 0)

        pipeline.add(streammux)
        if infer is not None:
            pipeline.add(infer)
        if tracker is not None:
            pipeline.add(tracker)
        pipeline.add(queue)
        pipeline.add(identity)
        pipeline.add(self.pre_osd_convert)
        pipeline.add(self.pre_osd_capsfilter)
        pipeline.add(self.osd)
        pipeline.add(self.post_osd_convert)
        pipeline.add(self.post_osd_capsfilter)
        pipeline.add(self.encoder)
        pipeline.add(self.parser)
        pipeline.add(self.payloader)
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
        if not identity.link(self.pre_osd_convert):
            raise RuntimeError("Failed to link identity -> pre-osd-convert.")
        if not self.pre_osd_convert.link(self.pre_osd_capsfilter):
            raise RuntimeError("Failed to link pre-osd-convert -> pre-osd-capsfilter.")
        if not self.pre_osd_capsfilter.link(self.osd):
            raise RuntimeError("Failed to link pre-osd-capsfilter -> nvdsosd.")
        if not self.osd.link(self.post_osd_convert):
            raise RuntimeError("Failed to link nvdsosd -> post-osd-convert.")
        if not self.post_osd_convert.link(self.post_osd_capsfilter):
            raise RuntimeError("Failed to link post-osd-convert -> post-osd-capsfilter.")
        if not self.post_osd_capsfilter.link(self.encoder):
            raise RuntimeError("Failed to link post-osd-capsfilter -> encoder.")
        if not self.encoder.link(self.parser):
            raise RuntimeError("Failed to link encoder -> parser.")
        if not self.parser.link(self.payloader):
            raise RuntimeError("Failed to link parser -> rtph264pay.")
        if not self.payloader.link(self.udpsink):
            raise RuntimeError("Failed to link rtph264pay -> udpsink.")

        camera = self.settings.cameras[0]
        if camera.stream.type == "hls":
            source_bin = create_hls_source_bin(index=0, uri=str(camera.stream.uri))
        elif camera.stream.type == "file":
            source_bin = create_file_source_bin(index=0, uri=str(camera.stream.uri))
        else:
            raise RuntimeError(f"Unsupported stream type: {camera.stream.type}")

        pipeline.add(source_bin)

        src_pad = source_bin.get_static_pad("src")
        if src_pad is None:
            raise RuntimeError(f"Failed to get src pad from source bin: {camera.camera_id}")

        mux_sink_pad = request_streammux_sink_pad(streammux, index=0)
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

        probe_element = tracker if tracker is not None else infer
        if probe_element is not None:
            self.infer_probe = InferProbe(
                settings=self.settings,
                publisher=self.event_publisher,
            )
            self.infer_probe.attach(probe_element, pad_name="src")

        self.frame_monitor.attach(identity, pad_name="src")
        self.bus_handler.attach(pipeline)

        self.pipeline = pipeline
        self.streammux = streammux
        self.infer = infer
        self.tracker = tracker
        self.queue = queue
        self.identity = identity

        logger.info(
            "Pipeline built | single-source | infer=%s | tracker=%s | sink=%s",
            self.settings.infer.enabled,
            self.settings.tracker.enabled,
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
            self.parser = None
            self.encoder = None
            self.post_osd_capsfilter = None
            self.post_osd_convert = None
            self.osd = None
            self.pre_osd_capsfilter = None
            self.pre_osd_convert = None
            self.identity = None
            self.queue = None
            self.tracker = None
            self.infer = None
            self.streammux = None
            self.source_bindings = []
            self.pipeline = None