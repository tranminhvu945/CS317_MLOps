"""
osd_draw.py — nvdsosd styling helpers.
"""
from __future__ import annotations

from typing import List, Tuple

import pyds

from apps.vision_service.src.logger import get_logger

logger = get_logger(__name__)

# ── Color Definitions ─────────────────────────────────────────────────────────

VIOLATION_COLOR_RGBA = (1.0, 0.0, 0.0, 1.0)   # Red
SAFE_COLOR_RGBA = (0.0, 1.0, 0.0, 1.0)        # Green
ROI_LINE_COLOR_RGBA = (0.0, 1.0, 0.0, 1.0)    # Green
CAM_LABEL_COLOR_RGBA = (1.0, 1.0, 1.0, 1.0)   # White
TEXT_BG_RGBA = (0.0, 0.0, 0.0, 0.7)           # Semi-transparent black

VIOLATION_BORDER_WIDTH = 3
SAFE_BORDER_WIDTH = 2
ROI_LINE_WIDTH = 2
TEXT_FONT_SIZE = 14


def _apply_rgba(
    color_params: pyds.NvOSD_ColorParams,
    rgba: Tuple[float, float, float, float],
) -> None:
    color_params.set(rgba[0], rgba[1], rgba[2], rgba[3])


# ── Object-meta styling ───────────────────────────────────────────────────────

def apply_violation_style(obj_meta: pyds.NvDsObjectMeta) -> None:
    """Style for helmet violation objects."""
    rect = obj_meta.rect_params
    rect.border_width = VIOLATION_BORDER_WIDTH
    _apply_rgba(rect.border_color, VIOLATION_COLOR_RGBA)

    if hasattr(rect, "has_bg_color"):
        rect.has_bg_color = 0


def apply_safe_style(obj_meta: pyds.NvDsObjectMeta) -> None:
    """Style for safe objects."""
    rect = obj_meta.rect_params
    rect.border_width = SAFE_BORDER_WIDTH
    _apply_rgba(rect.border_color, SAFE_COLOR_RGBA)

    if hasattr(rect, "has_bg_color"):
        rect.has_bg_color = 0


def apply_tracking_label(
    obj_meta: pyds.NvDsObjectMeta,
    label: str,
    track_id: int | None,
) -> None:
    """Attach visible text label to object."""
    display = f"{label} #{track_id}" if track_id is not None else label

    rect = obj_meta.rect_params
    text = obj_meta.text_params

    text.display_text = display
    text.x_offset = max(0, int(rect.left))
    text.y_offset = max(0, int(rect.top) - 10)

    text.font_params.font_name = "Sans"
    text.font_params.font_size = TEXT_FONT_SIZE
    _apply_rgba(text.font_params.font_color, CAM_LABEL_COLOR_RGBA)

    text.set_bg_clr = 1
    _apply_rgba(text.text_bg_clr, TEXT_BG_RGBA)


# ── Display-meta helpers ──────────────────────────────────────────────────────

def attach_camera_label(
    batch_meta: pyds.NvDsBatchMeta,
    frame_meta: pyds.NvDsFrameMeta,
    camera_id: str,
    x_offset: int = 20,
    y_offset: int = 30,
) -> None:
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    display_meta.num_labels = 1

    tp = display_meta.text_params[0]
    tp.display_text = f"cam={camera_id}"
    tp.x_offset = x_offset
    tp.y_offset = y_offset
    tp.font_params.font_name = "Sans"
    tp.font_params.font_size = 14
    _apply_rgba(tp.font_params.font_color, CAM_LABEL_COLOR_RGBA)

    tp.set_bg_clr = 1
    _apply_rgba(tp.text_bg_clr, TEXT_BG_RGBA)

    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def attach_roi_polygon(
    batch_meta: pyds.NvDsBatchMeta,
    frame_meta: pyds.NvDsFrameMeta,
    polygon: List[Tuple[float, float]],
    max_lines: int = 16,
) -> None:
    n = min(len(polygon), max_lines)
    if n < 3:
        return

    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    display_meta.num_lines = n

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        line = display_meta.line_params[i]
        line.x1 = int(x1)
        line.y1 = int(y1)
        line.x2 = int(x2)
        line.y2 = int(y2)
        line.line_width = ROI_LINE_WIDTH
        _apply_rgba(line.line_color, ROI_LINE_COLOR_RGBA)

    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def attach_fps_label(
    batch_meta: pyds.NvDsBatchMeta,
    frame_meta: pyds.NvDsFrameMeta,
    fps: float,
    x_offset: int = 20,
    y_offset: int = 30,
) -> None:
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    display_meta.num_labels = 1

    tp = display_meta.text_params[0]
    tp.display_text = f"FPS: {fps:.1f}"
    tp.x_offset = x_offset
    tp.y_offset = y_offset

    tp.font_params.font_name = "Sans"
    tp.font_params.font_size = 18
    _apply_rgba(tp.font_params.font_color, CAM_LABEL_COLOR_RGBA)

    tp.set_bg_clr = 1
    _apply_rgba(tp.text_bg_clr, TEXT_BG_RGBA)

    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
