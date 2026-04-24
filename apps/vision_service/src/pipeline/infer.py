from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from apps.vision_service.src.logger import get_logger
from apps.vision_service.src.settings import RootSettings
from apps.vision_service.src.utils.file_utils import ensure_file_exists, resolve_path


logger = get_logger(__name__)


ENGINE_PATH_KEY = "model-engine-file"

SOURCE_MODEL_KEYS = [
    "onnx-file",
    "tlt-encoded-model",
    "uff-file",
    "model-file",
    "caffemodel",
]

OPTIONAL_PATH_KEYS = [
    "labelfile-path",
    "custom-lib-path",
    "int8-calib-file",
]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def _parse_infer_dims(value: str) -> tuple[int, int, int]:
    """Parse infer-dims=3;640;640 → (channels, height, width)."""
    parts = [p.strip() for p in value.split(";")]
    if len(parts) != 3:
        raise ValueError(f"infer-dims must have 3 components, got: {value!r}")
    channels, height, width = int(parts[0]), int(parts[1]), int(parts[2])
    return channels, height, width


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class NvInferAssets:
    config_file: Path
    property_map: dict[str, str]
    resolved_paths: dict[str, Path]
    net_channels: int
    net_height: int
    net_width: int


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Config parsing
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def parse_nvinfer_config(config_path: Path) -> dict[str, str]:
    properties: dict[str, str] = {}
    in_property_section = False

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_property_section = line[1:-1].strip().lower() == "property"
            continue
        if not in_property_section or "=" not in line:
            continue
        key, value = line.split("=", 1)
        properties[key.strip()] = value.strip()

    if not properties:
        raise ValueError(
            f"No [property] section found in nvinfer config: {config_path}"
        )
    return properties


def inspect_nvinfer_assets(config_file: str) -> NvInferAssets:
    config_path = ensure_file_exists(config_file, description="nvinfer config")
    properties = parse_nvinfer_config(config_path)
    base_dir = config_path.parent

    resolved_paths: dict[str, Path] = {}
    for key in [ENGINE_PATH_KEY] + SOURCE_MODEL_KEYS + OPTIONAL_PATH_KEYS:
        value = properties.get(key)
        if not value:
            continue
        resolved_paths[key] = resolve_path(base_dir, value)

    # Parse network input dimensions from infer-dims
    infer_dims_raw = properties.get("infer-dims", "3;640;640")
    net_channels, net_height, net_width = _parse_infer_dims(infer_dims_raw)
    logger.debug(
        "Network input: %dx%d channels=%d",
        net_width, net_height, net_channels,
    )

    return NvInferAssets(
        config_file=config_path,
        property_map=properties,
        resolved_paths=resolved_paths,
        net_channels=net_channels,
        net_height=net_height,
        net_width=net_width,
    )


def validate_nvinfer_assets(config_file: str) -> NvInferAssets:
    assets = inspect_nvinfer_assets(config_file)

    engine_path = assets.resolved_paths.get(ENGINE_PATH_KEY)
    existing_source: list[str] = []

    for key in SOURCE_MODEL_KEYS:
        if key not in assets.property_map:
            continue
        path = assets.resolved_paths.get(key)
        if path is not None and path.exists():
            existing_source.append(key)

    # Has ONNX / model file → valid
    if existing_source:
        if engine_path is not None:
            engine_path.parent.mkdir(parents=True, exist_ok=True)
        for key in OPTIONAL_PATH_KEYS:
            path = assets.resolved_paths.get(key)
            if path is not None and not path.exists():
                raise FileNotFoundError(
                    f"Optional file declared but missing: {path} (key={key})"
                )
        logger.info(
            "Validated assets | config=%s | sources=%s | engine=%s",
            assets.config_file,
            existing_source,
            engine_path,
        )
        return assets

    # Has engine → valid
    if engine_path is not None and engine_path.exists():
        logger.info(
            "Validated assets (engine only) | config=%s | engine=%s",
            assets.config_file,
            engine_path,
        )
        return assets

    # Neither → error
    raise FileNotFoundError(
        f"No valid model artifact found. "
        f"Provide onnx-file or model-engine-file in: {config_file}"
    )

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# nvinfer element factory
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def create_primary_infer(settings: RootSettings) -> Gst.Element:
    assets = validate_nvinfer_assets(settings.infer.config_file)

    infer = make_element("nvinfer", "primary-infer")
    infer.set_property("config-file-path", str(assets.config_file))

    if infer.find_property("batch-size") is not None:
        infer.set_property("batch-size", 1)   # single-source
    if infer.find_property("unique-id") is not None:
        infer.set_property("unique-id", settings.infer.unique_id)
    if infer.find_property("gpu-id") is not None:
        infer.set_property("gpu-id", settings.app.gpu_id)

    logger.info(
        "Created primary nvinfer | config=%s | unique_id=%d | gpu=%d",
        assets.config_file,
        settings.infer.unique_id,
        settings.app.gpu_id,
    )
    return infer


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers (deferred import to avoid circular dep)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def make_element(factory_name: str, name: str) -> Gst.Element:
    from apps.vision_service.src.utils.gst_utils import make_element as _make
    return _make(factory_name, name)
