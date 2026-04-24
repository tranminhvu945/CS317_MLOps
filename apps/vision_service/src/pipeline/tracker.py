from __future__ import annotations

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.file_utils import ensure_file_exists
from apps.vision_service.src.utils.gst_utils import make_element


logger = get_logger(__name__)


def create_tracker(settings: RootSettings) -> Gst.Element:
    if not settings.tracker.enabled:
        raise RuntimeError("Tracker is disabled in settings.")

    ll_lib_file = ensure_file_exists(
        settings.tracker.ll_lib_file,
        description="tracker low-level library",
    )
    ll_config_file = ensure_file_exists(
        settings.tracker.ll_config_file,
        description="tracker config file",
    )

    tracker = make_element("nvtracker", "object-tracker")

    tracker.set_property("tracker-width", settings.tracker.tracker_width)
    tracker.set_property("tracker-height", settings.tracker.tracker_height)
    tracker.set_property("gpu_id", settings.tracker.gpu_id)
    tracker.set_property("ll-lib-file", str(ll_lib_file))
    tracker.set_property("ll-config-file", str(ll_config_file))

    if tracker.find_property("display-tracking-id") is not None:
        tracker.set_property("display-tracking-id", settings.tracker.display_tracking_id)

    return tracker