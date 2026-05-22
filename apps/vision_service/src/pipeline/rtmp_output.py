from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.gst_utils import make_element


logger = get_logger(__name__)


class RtmpOutputChain:
    def __init__(
        self,
        pre_osd_convert: Gst.Element,
        pre_osd_capsfilter: Gst.Element,
        osd: Gst.Element,
        post_osd_convert: Gst.Element,
        post_osd_capsfilter: Gst.Element,
        encoder: Gst.Element,
        parser: Gst.Element,
        muxer: Gst.Element,
        sink: Gst.Element,
        *,
        sink_factory: str,
    ) -> None:
        self.pre_osd_convert = pre_osd_convert
        self.pre_osd_capsfilter = pre_osd_capsfilter
        self.osd = osd
        self.post_osd_convert = post_osd_convert
        self.post_osd_capsfilter = post_osd_capsfilter
        self.encoder = encoder
        self.parser = parser
        self.muxer = muxer
        self.sink = sink
        self.sink_factory = sink_factory

    def link_from(self, upstream: Gst.Element) -> None:
        src_pad = upstream.get_static_pad("src")
        if src_pad is None:
            src_pad = upstream.request_pad_simple("src_%u")
        if src_pad is None:
            src_pad = upstream.get_request_pad("src_%u")
        if src_pad is None:
            raise RuntimeError(
                f"Failed to get src pad from element '{upstream.get_name()}'."
            )

        # Step 1: Link muxer -> sink FIRST (downstream must be ready before upstream)
        if not self.muxer.link(self.sink):
            raise RuntimeError(
                f"Failed to link {self.muxer.get_name()} -> {self.sink.get_name()}."
            )

        # Step 2: Request sink pad from muxer, then link upstream -> muxer
        sink_pad = self.muxer.get_compatible_pad(src_pad, None)
        if sink_pad is None:
            sink_pad = self.muxer.request_pad_simple("video")
        if sink_pad is None:
            sink_pad = self.muxer.get_request_pad("video")
        if sink_pad is None:
            raise RuntimeError(
                f"Failed to request video sink pad from element '{self.muxer.get_name()}'."
            )

        result = src_pad.link(sink_pad)
        if result != Gst.PadLinkReturn.OK:
            raise RuntimeError(
                f"Failed to link {upstream.get_name()} src -> {self.muxer.get_name()}, result={result}"
            )


def _make_caps(
    width: int,
    height: int,
    memory: str = "NVMM",
    format_str: str = "I420",
) -> Gst.Caps:
    return Gst.Caps.from_string(
        f"video/x-raw(memory:{memory}), width={width}, height={height}, format={format_str}"
    )


def _set_property_if_exists(element: Gst.Element, property_name: str, value: object) -> None:
    if element.find_property(property_name) is not None:
        element.set_property(property_name, value)


def _configure_encoder_and_parser(
    settings: RootSettings,
    *,
    canvas_w: int,
    canvas_h: int,
    encoder: Gst.Element,
    parser: Gst.Element,
) -> None:
    _set_property_if_exists(encoder, "maxperf-enable", True)
    _set_property_if_exists(encoder, "preset-level", 1)
    _set_property_if_exists(encoder, "preset-id", 1)
    _set_property_if_exists(encoder, "tuning-info-id", 2)
    _set_property_if_exists(encoder, "control-rate", 1)
    _set_property_if_exists(encoder, "num-B-Frames", 0)
    _set_property_if_exists(encoder, "poc-type", 2)
    _set_property_if_exists(encoder, "insert-aud", 1)
    _set_property_if_exists(encoder, "insert-vui", 1)
    _set_property_if_exists(encoder, "MeasureEncoderLatency", True)
    _set_property_if_exists(encoder, "measure-encoder-latency", True)
    _set_property_if_exists(encoder, "copy-timestamp", True)
    _set_property_if_exists(encoder, "bufapi-version", True)

    if encoder.find_property("bitrate") is not None:
        encoder.set_property("bitrate", settings.rtsp.bitrate)
    if encoder.find_property("width") is not None:
        encoder.set_property("width", canvas_w)
    if encoder.find_property("height") is not None:
        encoder.set_property("height", canvas_h)
    if encoder.find_property("iframeinterval") is not None:
        encoder.set_property("iframeinterval", max(1, settings.rtsp.iframe_interval))
    if encoder.find_property("idrinterval") is not None:
        encoder.set_property("idrinterval", max(1, settings.rtsp.iframe_interval))
    if encoder.find_property("insert-sps-pps") is not None:
        encoder.set_property("insert-sps-pps", 1)

    if parser.find_property("config-interval") is not None:
        parser.set_property("config-interval", settings.rtsp.sps_pps_interval)
    _set_property_if_exists(parser, "disable-passthrough", True)


def _resolve_rtmp_sink_factory() -> str:
    if Gst.ElementFactory.find("rtmpsink") is not None:
        return "rtmpsink"
    if Gst.ElementFactory.find("rtmp2sink") is not None:
        return "rtmp2sink"
    raise RuntimeError(
        "Missing RTMP sink factory. Need either 'rtmpsink' or 'rtmp2sink' "
        "in the runtime plugin environment."
    )


def _configure_rtmp_mux_and_sink(
    settings: RootSettings,
    *,
    muxer: Gst.Element,
    sink: Gst.Element,
) -> None:
    location = settings.rtmp.location.strip()
    if not location:
        raise RuntimeError("RTMP output location is empty. Please set rtmp.location in app.yaml.")

    _set_property_if_exists(muxer, "streamable", settings.rtmp.streamable_mux)
    sink.set_property("location", location)
    _set_property_if_exists(sink, "sync", settings.rtmp.sink_sync)
    _set_property_if_exists(sink, "async", settings.rtmp.sink_async)


def create_rtmp_output_chain(settings: RootSettings) -> RtmpOutputChain:
    canvas_w = settings.pipeline.streammux_width
    canvas_h = settings.pipeline.streammux_height

    pre_osd_convert = make_element("nvvideoconvert", "pre-osd-convert")
    pre_osd_capsfilter = make_element("capsfilter", "pre-osd-capsfilter")
    osd = make_element("nvdsosd", "on-screen-display")
    post_osd_convert = make_element("nvvideoconvert", "post-osd-convert")
    post_osd_capsfilter = make_element("capsfilter", "post-osd-caps")
    encoder = make_element("nvv4l2h264enc", "rtmp-h264-encoder")
    parser = make_element("h264parse", "rtmp-h264-parser")
    muxer = make_element("flvmux", "rtmp-flv-muxer")
    sink_factory = _resolve_rtmp_sink_factory()
    sink = make_element(sink_factory, "rtmp-network-sink")

    osd.set_property("gpu-id", settings.tracker.gpu_id)
    osd.set_property("process-mode", settings.visualization.osd_process_mode)
    osd.set_property("display-text", settings.visualization.display_text)
    osd.set_property("display-bbox", settings.visualization.display_bbox)
    osd.set_property("display-clock", settings.visualization.display_clock)

    pre_osd_capsfilter.set_property("caps", _make_caps(canvas_w, canvas_h, "NVMM", "RGBA"))
    post_osd_capsfilter.set_property("caps", _make_caps(canvas_w, canvas_h, "NVMM", "NV12"))

    _configure_encoder_and_parser(
        settings,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        encoder=encoder,
        parser=parser,
    )
    _configure_rtmp_mux_and_sink(settings, muxer=muxer, sink=sink)

    logger.info(
        "RTMP output ready | sink=%s | location=%s | bitrate=%d | iframe_interval=%d"
        " | sink_sync=%s | sink_async=%s | streamable_mux=%s",
        sink_factory,
        settings.rtmp.location,
        settings.rtsp.bitrate,
        settings.rtsp.iframe_interval,
        settings.rtmp.sink_sync,
        settings.rtmp.sink_async,
        settings.rtmp.streamable_mux,
    )

    return RtmpOutputChain(
        pre_osd_convert=pre_osd_convert,
        pre_osd_capsfilter=pre_osd_capsfilter,
        osd=osd,
        post_osd_convert=post_osd_convert,
        post_osd_capsfilter=post_osd_capsfilter,
        encoder=encoder,
        parser=parser,
        muxer=muxer,
        sink=sink,
        sink_factory=sink_factory,
    )
