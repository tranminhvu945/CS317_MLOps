#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
import tempfile
import venv


def _run_venv_command(venv_dir: str, pip_packages: list[str], python_code: str) -> None:
    """
    Create a temporary venv, install `pip_packages`, then run `python_code`
    inside it.  Captures and re-raises any failure.
    """
    print(f"[INFO] Creating temporary venv at {venv_dir} ...")
    venv.create(venv_dir, with_pip=True, clear=False)

    pip_exe = os.path.join(venv_dir, "bin", "pip")
    python_exe = os.path.join(venv_dir, "bin", "python")

    subprocess.check_call(
        [pip_exe, "install", "-q"] + pip_packages,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    print("[INFO] Running export inside venv ...")
    subprocess.check_call(
        [python_exe, "-c", python_code],
    )


def main() -> None:
    # ── Paths ────────────────────────────────────────────────────────────────
    SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

    SRC_PT   = "/mmlab_students/storageStudents/nguyenvd/uit_medseg/datasetMLOps/TrainYolo/runs/detect/helmet_training/run_13/weights/best.pt"
    DST_ONNX = os.path.join(PROJECT_DIR, "apps/vision_service/models/yolov8/yolov8_helmet.onnx")
    ENGINE   = DST_ONNX + "_b1_gpu0_fp16.engine"

    # ── Verify source ─────────────────────────────────────────────────────────
    if not os.path.exists(SRC_PT):
        print(f"[ERROR] Source weight not found:\n  {SRC_PT}")
        sys.exit(1)
    print(f"[INFO] Source: {SRC_PT}  ({os.path.getsize(SRC_PT)/1024/1024:.1f} MB)")

    # ── Backup old ONNX ────────────────────────────────────────────────────────
    if os.path.exists(DST_ONNX):
        shutil.copy2(DST_ONNX, DST_ONNX + ".bak")
        print(f"[INFO] Backed up old ONNX → {DST_ONNX}.bak")

    # ── Build export script (escaped for embedding in -c argument) ────────────
    # We use a tempdir so the venv is auto-deleted by the OS.
    export_code = (
        f"import shutil, sys\n"
        f"sys.path.insert(0, {repr(SCRIPT_DIR)!r})\n"
        "import ultralytics\n"
        f"SRC_PT   = {SRC_PT!r}\n"
        f"DST_ONNX = {DST_ONNX!r}\n"
        "print(f'[INFO] Loading YOLOv8 model from {{SRC_PT}} ...')\n"
        "model = ultralytics.YOLO(SRC_PT)\n"
        "print('[INFO] Exporting to ONNX (FP16, opset=11, dynamic=False) ...')\n"
        "exported = model.export(\n"
        "    format   = 'onnx',\n"
        "    imgsz    = 640,\n"
        "    keras    = False,\n"
        "    optimize = False,\n"
        "    half     = True,\n"
        "    dynamic  = False,\n"
        "    simplify = True,\n"
        "    opset    = 11,\n"
        "    verbose  = True,\n"
        ")\n"
        "shutil.copy2(exported, DST_ONNX)\n"
        "sz = os.path.getsize(DST_ONNX)\n"
        "print(f'[OK] ONNX exported & copied to: {{DST_ONNX}}  ({{sz/1024/1024:.1f}} MB)')\n"
    )

    with tempfile.TemporaryDirectory(prefix="yolo_export_venv_") as tmpdir:
        _run_venv_command(
            venv_dir=tmpdir,
            pip_packages=["ultralytics>=8.0"],
            python_code=export_code,
        )

    # ── Remove stale engine (must be rebuilt after weight change) ──────────────
    if os.path.exists(ENGINE):
        os.remove(ENGINE)
        print(f"[INFO] Removed stale engine → {os.path.basename(ENGINE)}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Export complete!")
    print()
    print("  ONNX :", DST_ONNX)
    print("  Next steps:")
    print("    1. Rebuild TensorRT engine:")
    print("       docker compose run --rm vision-service \\")
    print("         /workspace/scripts/build_engine.sh")
    print()
    print("    2. Restart the service:")
    print("       docker compose up -d vision-service")
    print("=" * 60)


if __name__ == "__main__":
    main()
