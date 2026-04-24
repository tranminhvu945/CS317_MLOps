from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target

def ensure_file_exists(path: str | Path, description: str = "file") -> Path:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"{description.capitalize()} not found: {target}")
    return target


def resolve_path(base_dir: str | Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (Path(base_dir) / candidate).resolve()