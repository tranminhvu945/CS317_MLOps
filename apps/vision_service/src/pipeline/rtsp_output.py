from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtsp, GstRtspServer  # noqa: E402

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


def _set_property_if_exists(element: Gst.Element, property_name: str, value: object) -> None:
    if element.find_property(property_name) is not None:
        element.set_property(property_name, value)


def _make_rtp_caps(payload_type: int) -> str:
    return (
        "application/x-rtp, "
        "media=(string)video, "
        "clock-rate=(int)90000, "
        "encoding-name=(string)H264, "
        f"payload=(int){payload_type}"
    )


def _build_rtsp_factory_launch(settings: RootSettings, rtp_caps: str) -> str:
    payload_type = settings.rtsp.payload_type
    mtu = settings.rtsp.rtp_mtu
    sps_pps_interval = settings.rtsp.sps_pps_interval
    jitter = (
        f"! rtpjitterbuffer latency={settings.rtsp.rtsp_repay_jitter_latency_ms} "
        f"drop-on-latency={'true' if settings.rtsp.rtsp_repay_jitter_drop_on_latency else 'false'} "
        if settings.rtsp.rtsp_repay_enabled
        else ""
    )
    post_parse_leaky_queue = (
        "! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream "
        if settings.rtsp.rtsp_repay_leaky_queue_enabled
        else ""
    )

    return (
        f"( udpsrc name=rtsp_udp_src port={settings.rtsp.udp_port} "
        f"buffer-size={settings.rtsp.udp_buffer_size} "
        f'caps="{rtp_caps}" '
        f"{jitter}"
        "! rtph264depay "
        f"! h264parse config-interval={sps_pps_interval} "
        f"{post_parse_leaky_queue}"
        f"! rtph264pay name=pay0 pt={payload_type} "
        f"mtu={mtu} config-interval={sps_pps_interval} )"
    )


def _get_rtsp_transport_flags(settings: RootSettings) -> GstRtsp.RTSPLowerTrans:
    mode = settings.rtsp.rtsp_transport
    if mode == "tcp":
        return GstRtsp.RTSPLowerTrans.TCP
    if mode == "udp":
        return GstRtsp.RTSPLowerTrans.UDP
    return GstRtsp.RTSPLowerTrans.UDP | GstRtsp.RTSPLowerTrans.TCP


def create_rtsp_output_chain(settings: RootSettings) -> RtspOutputChain:
    canvas_w = settings.pipeline.streammux_width
    canvas_h = settings.pipeline.streammux_height
    payload_type = settings.rtsp.payload_type
    mtu = settings.rtsp.rtp_mtu
    sps_pps_interval = settings.rtsp.sps_pps_interval
    rtp_caps = _make_rtp_caps(payload_type)
    rtp_caps_obj = Gst.Caps.from_string(rtp_caps)
    if rtp_caps_obj is None or rtp_caps_obj.is_empty():
        raise RuntimeError(f"Invalid RTSP RTP caps: {rtp_caps}")

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
        parser.set_property("config-interval", sps_pps_interval)
    _set_property_if_exists(parser, "disable-passthrough", True)

    _set_property_if_exists(payloader, "pt", payload_type)
    _set_property_if_exists(payloader, "payload", payload_type)
    _set_property_if_exists(payloader, "mtu", mtu)
    _set_property_if_exists(payloader, "aggregate-mode", 0)  # RTSP compatibility
    if payloader.find_property("config-interval") is not None:
        payloader.set_property("config-interval", sps_pps_interval)

    udpsink.set_property("host", settings.rtsp.host)
    udpsink.set_property("port", settings.rtsp.udp_port)
    _set_property_if_exists(udpsink, "buffer-size", settings.rtsp.udp_buffer_size)
    udpsink.set_property("async", settings.rtsp.udpsink_async)
    udpsink.set_property("sync", settings.rtsp.udpsink_sync)
    _set_property_if_exists(udpsink, "qos", settings.rtsp.udpsink_qos)

    server = GstRtspServer.RTSPServer.new()
    server.set_service(str(settings.rtsp.rtsp_port))

    mount_points = server.get_mount_points()
    factory = GstRtspServer.RTSPMediaFactory.new()
    launch_description = _build_rtsp_factory_launch(settings, rtp_caps)
    factory.set_launch(launch_description)
    factory.set_shared(True)
    if hasattr(factory, "set_protocols"):
        factory.set_protocols(_get_rtsp_transport_flags(settings))
    if hasattr(factory, "set_latency"):
        factory.set_latency(settings.rtsp.rtsp_repay_jitter_latency_ms)
    mount_points.add_factory(settings.rtsp.mount_point, factory)
    server.attach(None)

    logger.info(
        "RTSP output ready at rtsp://<host>:%d%s | payload=%d | mtu=%d | parser/pay config-interval=%d | "
        "repack=%s | transport=%s | udpsink(sync=%s,async=%s,qos=%s)",
        settings.rtsp.rtsp_port,
        settings.rtsp.mount_point,
        payload_type,
        mtu,
        sps_pps_interval,
        settings.rtsp.rtsp_repay_enabled,
        settings.rtsp.rtsp_transport,
        settings.rtsp.udpsink_sync,
        settings.rtsp.udpsink_async,
        settings.rtsp.udpsink_qos,
    )
    logger.info("RTSP RTP caps: %s", rtp_caps_obj.to_string())
    logger.info("RTSP server launch: %s", launch_description)

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
