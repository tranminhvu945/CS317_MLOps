from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer  # noqa: E402

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.gst_utils import make_element


logger = get_logger(__name__)


class RtspOutputChain:
    def __init__(
        self,
        pre_osd_convert: Gst.Element,
        pre_osd_capsfilter: Gst.Element,
        osd: Gst.Element,
        post_osd_convert: Gst.Element,
        post_osd_capsfilter: Gst.Element,
        encoder: Gst.Element,
        parser: Gst.Element,
        payloader: Gst.Element,
        udpsink: Gst.Element,
        server: GstRtspServer.RTSPServer,
    ) -> None:
        self.pre_osd_convert = pre_osd_convert
        self.pre_osd_capsfilter = pre_osd_capsfilter
        self.osd = osd
        self.post_osd_convert = post_osd_convert
        self.post_osd_capsfilter = post_osd_capsfilter
        self.encoder = encoder
        self.parser = parser
        self.payloader = payloader
        self.udpsink = udpsink
        self.server = server


def _make_caps(
    width: int,
    height: int,
    memory: str = "NVMM",
    format_str: str = "I420",
) -> Gst.Caps:
    return Gst.Caps.from_string(
        f"video/x-raw(memory:{memory}), width={width}, height={height}, format={format_str}"
    )


def create_rtsp_output_chain(settings: RootSettings) -> RtspOutputChain:
    canvas_w = settings.pipeline.streammux_width
    canvas_h = settings.pipeline.streammux_height

    pre_osd_convert = make_element("nvvideoconvert", "pre-osd-convert")
    pre_osd_capsfilter = make_element("capsfilter", "pre-osd-capsfilter")
    osd = make_element("nvdsosd", "on-screen-display")
    post_osd_convert = make_element("nvvideoconvert", "post-osd-convert")
    post_osd_capsfilter = make_element("capsfilter", "post-osd-caps")
    encoder = make_element("nvv4l2h264enc", "rtsp-h264-encoder")
    parser = make_element("h264parse", "rtsp-h264-parser")
    payloader = make_element("rtph264pay", "rtsp-payloader")
    udpsink = make_element("udpsink", "rtsp-udpsink")

    # ── nvdsosd ────────────────────────────────────────────────────────────────
    osd.set_property("gpu-id", settings.tracker.gpu_id)
    osd.set_property("process-mode", settings.visualization.osd_process_mode)
    osd.set_property("display-text", settings.visualization.display_text)
    osd.set_property("display-bbox", settings.visualization.display_bbox)
    osd.set_property("display-clock", settings.visualization.display_clock)

    # pre_osd_capsfilter.set_property("caps", _make_caps(canvas_w, canvas_h, "NVMM", "I420"))
    # post_osd_capsfilter.set_property("caps", _make_caps(canvas_w, canvas_h, "NVMM", "I420"))
    
    pre_osd_capsfilter.set_property("caps", _make_caps(canvas_w, canvas_h, "NVMM", "RGBA"))
    post_osd_capsfilter.set_property("caps", _make_caps(canvas_w, canvas_h, "NVMM", "NV12"))

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
        parser.set_property("config-interval", -1)

    payloader.set_property("pt", 96)
    if payloader.find_property("config-interval") is not None:
        payloader.set_property("config-interval", 1)

    udpsink.set_property("host", settings.rtsp.host)
    udpsink.set_property("port", settings.rtsp.udp_port)
    udpsink.set_property("async", False)
    udpsink.set_property("sync", False)

    server = GstRtspServer.RTSPServer.new()
    server.set_service(str(settings.rtsp.rtsp_port))

    mount_points = server.get_mount_points()
    factory = GstRtspServer.RTSPMediaFactory.new()
    launch_description = (
        f"( udpsrc name=pay0 port={settings.rtsp.udp_port} "
        f'caps="application/x-rtp, media=video, clock-rate=90000, '
        f'encoding-name=(string)H264, payload=96" )'
    )
    factory.set_launch(launch_description)
    factory.set_shared(True)
    mount_points.add_factory(settings.rtsp.mount_point, factory)
    server.attach(None)

    logger.info(
        "RTSP output ready at rtsp://<host>:%d%s",
        settings.rtsp.rtsp_port,
        settings.rtsp.mount_point,
    )

    return RtspOutputChain(
        pre_osd_convert=pre_osd_convert,
        pre_osd_capsfilter=pre_osd_capsfilter,
        osd=osd,
        post_osd_convert=post_osd_convert,
        post_osd_capsfilter=post_osd_capsfilter,
        encoder=encoder,
        parser=parser,
        payloader=payloader,
        udpsink=udpsink,
        server=server,
    )


