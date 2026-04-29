#!/usr/bin/env python3

import os
import shutil
import sys
from ultralytics import YOLO


def main() -> None:
    # ── Paths ────────────────────────────────────────────────────────────────
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

    SRC_PT = "/mmlab_students/storageStudents/nguyenvd/uit_medseg/datasetMLOps/TrainYolo/runs/detect/helmet_training/run_12/weights/best.pt"

    DST_ONNX = os.path.join(
        PROJECT_DIR,
        "apps/vision_service/models/yolov8/yolov8_helmet.onnx"
    )

    ENGINE = DST_ONNX + "_b1_gpu0_fp16.engine"

    # ── Verify source ────────────────────────────────────────────────────────
    if not os.path.exists(SRC_PT):
        print(f"[ERROR] Source weight not found:\n  {SRC_PT}")
        sys.exit(1)

    print(f"[INFO] Source: {SRC_PT} ({os.path.getsize(SRC_PT) / 1024 / 1024:.1f} MB)")

    # ── Ensure output dir exists ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(DST_ONNX), exist_ok=True)

    # ── Backup old ONNX ─────────────────────────────────────────────────────
    if os.path.exists(DST_ONNX):
        backup_path = DST_ONNX + ".bak"
        shutil.copy2(DST_ONNX, backup_path)
        print(f"[INFO] Backed up old ONNX -> {backup_path}")

    # ── Export YOLOv8 to ONNX ────────────────────────────────────────────────
    print(f"[INFO] Loading YOLOv8 model from:\n  {SRC_PT}")
    model = YOLO(SRC_PT)

    print("[INFO] Exporting to ONNX ...")

    exported = model.export(
        format="onnx",
        imgsz=640,
        keras=False,
        optimize=False,

        # Khuyên dùng False khi export ONNX.
        # FP16 nên để TensorRT xử lý ở bước build engine.
        half=False,

        dynamic=False,

        # Nếu muốn nhanh hơn khi debug, để False.
        # Khi cần tối ưu ONNX thì đổi lại True.
        simplify=False,

        opset=11,
        verbose=True,
    )

    print(f"[INFO] Ultralytics exported file:\n  {exported}")

    # ── Copy exported ONNX to project path ───────────────────────────────────
    shutil.copy2(exported, DST_ONNX)

    size_mb = os.path.getsize(DST_ONNX) / 1024 / 1024
    print(f"[OK] ONNX exported and copied to:\n  {DST_ONNX} ({size_mb:.1f} MB)")

    # ── Remove stale TensorRT engine ─────────────────────────────────────────
    if os.path.exists(ENGINE):
        os.remove(ENGINE)
        print(f"[INFO] Removed stale engine -> {os.path.basename(ENGINE)}")
    else:
        print("[INFO] No stale TensorRT engine found.")

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Export complete!")
    print()
    print("ONNX:", DST_ONNX)
    print()
    print("Next steps:")
    print("  1. Rebuild TensorRT engine:")
    print("     docker compose run --rm vision-service \\")
    print("       /workspace/scripts/build_engine.sh")
    print()
    print("  2. Restart the service:")
    print("     docker compose up -d vision-service")
    print("=" * 60)


if __name__ == "__main__":
    main()