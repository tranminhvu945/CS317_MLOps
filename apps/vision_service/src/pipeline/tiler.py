"""
tiler.py — Tạo và cấu hình element nvmultistreamtiler cho multi-camera pipeline.

Layout tự động:
  - Nếu rows/cols = 0 trong config → tính theo ceil(sqrt(n_cameras))
  - Ví dụ: 3 cameras → 2×2 grid; 4 cameras → 2×2; 5 cameras → 3×3
"""
from __future__ import annotations

import math

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.gst_utils import make_element


logger = get_logger(__name__)


def _compute_grid(n: int, rows_cfg: int, cols_cfg: int) -> tuple[int, int]:
    """
    Tính số hàng và cột cho tiled layout.

    Nếu rows_cfg và cols_cfg đều > 0: dùng giá trị config trực tiếp.
    Nếu chỉ một trong hai > 0: tính cái còn lại.
    Nếu cả hai = 0: tự động tính theo ceil(sqrt(n)).

    Returns:
        (rows, cols) sao cho rows * cols >= n.
    """
    if n <= 0:
        return 1, 1

    if rows_cfg > 0 and cols_cfg > 0:
        return rows_cfg, cols_cfg

    if rows_cfg > 0:
        cols = math.ceil(n / rows_cfg)
        return rows_cfg, cols

    if cols_cfg > 0:
        rows = math.ceil(n / cols_cfg)
        return rows, cols_cfg

    # Auto-sqrt: tính số cột trước (≥ rows để layout ngang hơn)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def create_tiler(settings: RootSettings, num_sources: int) -> Gst.Element:
    """
    Tạo và cấu hình element `nvmultistreamtiler`.

    Args:
        settings:     Cấu hình toàn cục (RootSettings).
        num_sources:  Số camera/sources đang active.

    Returns:
        Element GStreamer đã được cấu hình, chưa được add vào pipeline.
    """
    tiler_cfg = settings.tiler

    rows, cols = _compute_grid(
        num_sources,
        rows_cfg=tiler_cfg.rows,
        cols_cfg=tiler_cfg.cols,
    )

    tiler = make_element("nvmultistreamtiler", "multi-stream-tiler")

    tiler.set_property("rows", rows)
    tiler.set_property("columns", cols)
    tiler.set_property("width", tiler_cfg.width)
    tiler.set_property("height", tiler_cfg.height)

    logger.info(
        "Tiler configured | sources=%d | grid=%dx%d | output=%dx%d",
        num_sources,
        rows,
        cols,
        tiler_cfg.width,
        tiler_cfg.height,
    )

    return tiler
