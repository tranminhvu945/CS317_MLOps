from __future__ import annotations

from typing import Sequence, Tuple

Point = Tuple[float, float]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False

    n = len(polygon)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside

        j = i

    return inside


def bbox_anchor_bottom_center(
    left: float,
    top: float,
    width: float,
    height: float,
) -> Point:
    return (left + width / 2.0, top + height)