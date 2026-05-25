"""
camera_config.py
────────────────
Helper module để đọc / ghi / xóa cấu hình camera trong thư mục
`apps/vision_service/configs/camera/`.

Mỗi camera tương ứng với một file `.yaml` độc lập. Module này
đảm bảo tính đồng nhất giữa file cấu hình và trạng thái runtime.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ─── Đường dẫn mặc định ────────────────────────────────────────────────────────
DEFAULT_CAMERA_DIR = Path(
    os.getenv("CAMERA_CONFIG_DIR", "/workspace/apps/vision_service/configs/camera")
)


def get_camera_dir() -> Path:
    return DEFAULT_CAMERA_DIR


def _parse_camera_file(path: Path) -> Dict[str, Any]:
    """Đọc và parse một file camera YAML, thêm trường `source_file` vào kết quả."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["source_file"] = path.name
    # Đảm bảo luôn có trường camera_id lấy từ tên file nếu thiếu
    if "camera_id" not in data:
        data["camera_id"] = path.stem
    return data


def list_cameras() -> List[Dict[str, Any]]:
    """Trả về danh sách toàn bộ cấu hình camera (cả enabled và disabled)."""
    camera_dir = get_camera_dir()
    cameras = []
    for yaml_file in sorted(camera_dir.glob("*.yaml")):
        try:
            cameras.append(_parse_camera_file(yaml_file))
        except Exception as exc:
            cameras.append({"source_file": yaml_file.name, "error": str(exc)})
    return cameras


def get_camera(camera_id: str) -> Optional[Dict[str, Any]]:
    """Tìm kiếm và trả về cấu hình của một camera theo ID."""
    for cam in list_cameras():
        if cam.get("camera_id") == camera_id:
            return cam
    return None


def _find_camera_file(camera_id: str) -> Optional[Path]:
    """Tìm file YAML của camera theo camera_id."""
    camera_dir = get_camera_dir()
    for yaml_file in camera_dir.glob("*.yaml"):
        try:
            with yaml_file.open("r") as f:
                data = yaml.safe_load(f) or {}
            if data.get("camera_id") == camera_id:
                return yaml_file
        except Exception:
            continue
    return None


def _next_camera_filename() -> str:
    """Sinh tên file `.yaml` tiếp theo theo dạng camera_NNN.yaml."""
    camera_dir = get_camera_dir()
    existing = list(camera_dir.glob("camera*.yaml"))
    used_ids = set()
    for f in existing:
        m = re.search(r"camera_?(\d+)", f.stem)
        if m:
            used_ids.add(int(m.group(1)))
    idx = 1
    while idx in used_ids:
        idx += 1
    return f"camera_{idx:03d}.yaml"


def create_camera(
    camera_id: str,
    name: str,
    stream_type: str,
    uri: str,
    enabled: bool = True,
    min_confidence: float = 0.15,
    reconnect_interval_sec: int = 10,
    timeout_sec: int = 15,
) -> Dict[str, Any]:
    """Tạo file cấu hình YAML cho camera mới.

    Returns:
        Cấu hình camera vừa tạo.
    Raises:
        ValueError: nếu camera_id đã tồn tại.
    """
    if _find_camera_file(camera_id) is not None:
        raise ValueError(f"Camera ID '{camera_id}' already exists.")

    camera_dir = get_camera_dir()
    camera_dir.mkdir(parents=True, exist_ok=True)

    filename = _next_camera_filename()
    file_path = camera_dir / filename

    config: Dict[str, Any] = {
        "camera_id": camera_id,
        "name": name,
        "enabled": enabled,
        "stream": {
            "type": stream_type,
            "uri": uri,
            "reconnect_interval_sec": reconnect_interval_sec,
            "timeout_sec": timeout_sec,
            "decoder_drop_frame_interval": 0,
        },
        "detection": {
            "min_confidence": min_confidence,
            "roi": {
                "enabled": False,
                "polygon": [],
            },
        },
    }

    with file_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    return _parse_camera_file(file_path)


def set_camera_enabled(camera_id: str, enabled: bool) -> Dict[str, Any]:
    """Bật hoặc tắt một camera và lưu lại vào file YAML.

    Returns:
        Cấu hình camera sau khi cập nhật.
    Raises:
        KeyError: nếu không tìm thấy camera_id.
    """
    file_path = _find_camera_file(camera_id)
    if file_path is None:
        raise KeyError(f"Camera '{camera_id}' not found.")

    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data["enabled"] = enabled

    with file_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    return _parse_camera_file(file_path)


def delete_camera(camera_id: str) -> bool:
    """Xóa file cấu hình YAML của camera.

    Returns:
        True nếu xóa thành công, False nếu không tìm thấy.
    """
    file_path = _find_camera_file(camera_id)
    if file_path is None:
        return False
    file_path.unlink()
    return True
