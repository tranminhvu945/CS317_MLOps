from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.gst_utils import make_element


logger = get_logger(__name__)


class FakeOutputChain:
    def __init__(
        self,
        pre_osd_convert: Gst.Element,
        pre_osd_capsfilter: Gst.Element,
        osd: Gst.Element,
        post_osd_convert: Gst.Element,
        post_osd_capsfilter: Gst.Element,
        encoder: Gst.Element,
        parser: Gst.Element,
        sink: Gst.Element,
    ) -> None:
        self.pre_osd_convert = pre_osd_convert
        self.pre_osd_capsfilter = pre_osd_capsfilter
        self.osd = osd
        self.post_osd_convert = post_osd_convert
        self.post_osd_capsfilter = post_osd_capsfilter
        self.encoder = encoder
        self.parser = parser
        self.sink = sink


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


def create_fake_output_chain(settings: RootSettings) -> FakeOutputChain:
    canvas_w = settings.pipeline.streammux_width
    canvas_h = settings.pipeline.streammux_height

    pre_osd_convert = make_element("nvvideoconvert", "pre-osd-convert")
    pre_osd_capsfilter = make_element("capsfilter", "pre-osd-capsfilter")
    osd = make_element("nvdsosd", "on-screen-display")
    post_osd_convert = make_element("nvvideoconvert", "post-osd-convert")
    post_osd_capsfilter = make_element("capsfilter", "post-osd-caps")
    encoder = make_element("nvv4l2h264enc", "fake-h264-encoder")
    parser = make_element("h264parse", "fake-h264-parser")
    sink = make_element("fakesink", "fake-output-sink")

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

    _set_property_if_exists(sink, "sync", False)
    _set_property_if_exists(sink, "async", False)
    _set_property_if_exists(sink, "qos", False)
    _set_property_if_exists(sink, "enable-last-sample", False)

    logger.info(
        "Fake output ready | sink=fakesink | bitrate=%d | iframe_interval=%d",
        settings.rtsp.bitrate,
        settings.rtsp.iframe_interval,
    )

    return FakeOutputChain(
        pre_osd_convert=pre_osd_convert,
        pre_osd_capsfilter=pre_osd_capsfilter,
        osd=osd,
        post_osd_convert=post_osd_convert,
        post_osd_capsfilter=post_osd_capsfilter,
        encoder=encoder,
        parser=parser,
        sink=sink,
    )
