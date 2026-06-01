#!/usr/bin/env python3
"""
sync_storage_to_data_new.py
───────────────────────────
Đồng bộ dữ liệu detect từ:
  - storage/logs/events.jsonl (metadata bbox/class)
  - storage/snapshots/*.jpg   (ảnh snapshot)
thành raw dataset format mà format_data_new_yolo.py đang dùng:

  <output_dir>/
    cam_001/
      *.jpg
    annotations/
      obj_train_data/
        cam_001/
          *.txt   (YOLO labels: class_id cx cy w h)
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync detect data từ storage sang dataset/data_new cho pipeline prepare-data"
    )
    parser.add_argument(
        "--events-file",
        default="storage/logs/events.jsonl",
        help="Đường dẫn events.jsonl (chứa payload bbox/class)",
    )
    parser.add_argument(
        "--snapshots-dir",
        default="storage/snapshots",
        help="Thư mục snapshot ảnh vi phạm",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/data_new",
        help="Thư mục raw output theo layout cam*/annotations/obj_train_data",
    )
    parser.add_argument(
        "--mode",
        choices=["replace", "append"],
        default="replace",
        help="replace: xóa output cũ; append: giữ output cũ và thêm dữ liệu mới",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Bỏ qua event có confidence thấp hơn ngưỡng này",
    )
    return parser.parse_args()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def bbox_to_yolo(
    bbox: List[float],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    if len(bbox) != 4 or img_w <= 0 or img_h <= 0:
        return None

    left, top, width, height = [float(v) for v in bbox]
    x1 = clamp(left, 0.0, float(img_w))
    y1 = clamp(top, 0.0, float(img_h))
    x2 = clamp(left + width, 0.0, float(img_w))
    y2 = clamp(top + height, 0.0, float(img_h))

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + (bw / 2.0)
    cy = y1 + (bh / 2.0)

    return (
        cx / float(img_w),
        cy / float(img_h),
        bw / float(img_w),
        bh / float(img_h),
    )


def iter_helmet_events(events_file: Path) -> Iterable[Dict]:
    with events_file.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] JSON lỗi tại line {lineno}, bỏ qua")
                continue

            event_type = record.get("event_type")
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
            if event_type != "helmet_violation":
                continue
            if not isinstance(payload, dict):
                continue
            yield payload


def resolve_snapshot_path(payload: Dict, snapshots_dir: Path) -> Optional[Path]:
    snapshot_path = payload.get("snapshot_path")
    if isinstance(snapshot_path, str) and snapshot_path.strip():
        candidate = Path(snapshot_path.strip())
        if candidate.exists():
            return candidate
        alt = snapshots_dir / candidate.name
        if alt.exists():
            return alt

    camera_id = payload.get("camera_id")
    event_id = payload.get("event_id")
    if not isinstance(camera_id, str) or not isinstance(event_id, str):
        return None

    base = f"violation_{camera_id}_{event_id}"
    for ext in IMAGE_EXTENSIONS:
        p = snapshots_dir / f"{base}{ext}"
        if p.exists():
            return p

    matches = sorted(snapshots_dir.glob(f"{base}.*"))
    return matches[0] if matches else None


def extract_objects(payload: Dict) -> List[Tuple[int, List[float]]]:
    objects: List[Tuple[int, List[float]]] = []

    all_objects = payload.get("all_objects")
    if isinstance(all_objects, list) and all_objects:
        for obj in all_objects:
            if not isinstance(obj, dict):
                continue
            class_id = obj.get("class_id")
            bbox = obj.get("bbox")
            if isinstance(class_id, int) and isinstance(bbox, list) and len(bbox) == 4:
                objects.append((class_id, bbox))
        if objects:
            return objects

    class_id = payload.get("class_id")
    bbox = payload.get("bbox")
    if isinstance(class_id, int) and isinstance(bbox, list) and len(bbox) == 4:
        objects.append((class_id, bbox))

    return objects


def camera_folder_name(camera_id: str) -> str:
    return camera_id if camera_id.startswith("cam_") else f"cam_{camera_id}"


def write_label_file(label_path: Path, rows: List[Tuple[int, float, float, float, float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for class_id, cx, cy, bw, bh in rows:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main() -> None:
    args = parse_args()
    events_file = Path(args.events_file)
    snapshots_dir = Path(args.snapshots_dir)
    output_dir = Path(args.output_dir)

    if not events_file.exists():
        raise FileNotFoundError(
            f"Không thấy events file: {events_file}. "
            "Hãy chạy vision service và đảm bảo events được ghi vào storage/logs/events.jsonl"
        )
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"Không thấy snapshots dir: {snapshots_dir}")

    if args.mode == "replace" and output_dir.exists():
        shutil.rmtree(output_dir)

    annotations_root = output_dir / "annotations" / "obj_train_data"
    annotations_root.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    written_labels = 0
    skipped = 0
    class_counter: Counter = Counter()
    processed_snapshots = set()

    for payload in iter_helmet_events(events_file):
        confidence = payload.get("confidence", 0.0)
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < args.min_confidence:
            skipped += 1
            continue

        camera_id = payload.get("camera_id")
        if not isinstance(camera_id, str) or not camera_id:
            skipped += 1
            continue

        snapshot_path = resolve_snapshot_path(payload, snapshots_dir)
        if snapshot_path is None or not snapshot_path.exists():
            skipped += 1
            continue

        snapshot_key = str(snapshot_path.resolve())
        if snapshot_key in processed_snapshots:
            continue

        img = cv2.imread(str(snapshot_path))
        if img is None:
            skipped += 1
            continue
        img_h, img_w = img.shape[:2]

        objects = extract_objects(payload)
        if not objects:
            skipped += 1
            continue

        label_rows: List[Tuple[int, float, float, float, float]] = []
        for class_id, bbox in objects:
            yolo_box = bbox_to_yolo(bbox, img_w=img_w, img_h=img_h)
            if yolo_box is None:
                continue
            cx, cy, bw, bh = yolo_box
            label_rows.append((class_id, cx, cy, bw, bh))
            class_counter[class_id] += 1

        if not label_rows:
            skipped += 1
            continue

        cam_dir = output_dir / camera_folder_name(camera_id)
        cam_dir.mkdir(parents=True, exist_ok=True)
        image_dst = cam_dir / snapshot_path.name
        shutil.copy2(snapshot_path, image_dst)
        copied_images += 1

        label_dst = annotations_root / camera_folder_name(camera_id) / f"{snapshot_path.stem}.txt"
        write_label_file(label_dst, label_rows)
        written_labels += 1

        processed_snapshots.add(snapshot_key)

    print("=" * 70)
    print("SYNC STORAGE -> DATA_NEW HOÀN TẤT")
    print("=" * 70)
    print(f"Events file    : {events_file}")
    print(f"Snapshots dir  : {snapshots_dir}")
    print(f"Output raw dir : {output_dir}")
    print("-" * 70)
    print(f"Copied images  : {copied_images}")
    print(f"Written labels : {written_labels}")
    print(f"Skipped events : {skipped}")
    print(f"Class counts   : {dict(class_counter)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
