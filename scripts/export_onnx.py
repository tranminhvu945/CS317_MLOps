#!/usr/bin/env python3
"""
export_onnx.py
──────────────
Tải model best.pt từ MLflow Model Registry (alias Production),
sau đó export ra ONNX để sẵn sàng build TensorRT engine.

Không còn hardcode đường dẫn file — mọi thứ được đọc từ params.yaml
và tải tự động từ MLflow server.

Usage:
    python scripts/export_onnx.py [--alias ALIAS]

Options:
    --alias   Alias trong MLflow Registry cần tải (mặc định: Production)
"""

import argparse
import os
import shutil
import sys
import tempfile
import yaml
from pathlib import Path

from ultralytics import YOLO


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_params(path: str = "params.yaml") -> dict:
    params_path = Path(path)
    if not params_path.exists():
        # Thử tìm từ thư mục gốc project (khi chạy từ scripts/)
        alt = Path(__file__).parent.parent / "params.yaml"
        if alt.exists():
            params_path = alt
        else:
            print(f"[ERROR] Không tìm thấy params.yaml tại: {path}")
            sys.exit(1)
    with open(params_path) as f:
        return yaml.safe_load(f)


def download_model_from_registry(
    tracking_uri: str,
    registry_name: str,
    alias: str,
) -> str:
    """
    Kết nối MLflow Registry, tìm version có alias cho trước,
    tải file .pt về thư mục tạm và trả về đường dẫn tuyệt đối.
    """
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError:
        print("[ERROR] Thiếu thư viện mlflow. Chạy: pip install mlflow")
        sys.exit(1)

    print(f"[INFO] Kết nối MLflow server: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # ── Lấy thông tin version theo alias ─────────────────────────────────────
    print(f"[INFO] Tìm model '{registry_name}' với alias '{alias}' ...")
    try:
        model_version = client.get_model_version_by_alias(
            name=registry_name,
            alias=alias,
        )
    except Exception as e:
        print(f"[ERROR] Không tìm thấy model '{registry_name}@{alias}'")
        print(f"        Chi tiết: {e}")
        print()
        print("  Hãy chắc chắn:")
        print(f"  1. MLflow server đang chạy tại {tracking_uri}")
        print(f"  2. Đã train model và đăng ký vào Registry (dvc repro / make dvc-train)")
        print(f"  3. Model được gắn alias '{alias}'")
        sys.exit(1)

    version = model_version.version
    source = model_version.source  # ví dụ: mlflow-artifacts:/1/abc123/artifacts/weights/best.pt
    print(f"[OK]   Tìm thấy: version={version}, source={source}")

    # ── Tải artifact về thư mục tạm ──────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="mlflow_model_")
    print(f"[INFO] Đang tải artifact về: {tmp_dir} ...")

    # Xây dựng model URI dạng models:/Name@Alias
    model_uri = f"models:/{registry_name}@{alias}"
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=source,
        dst_path=tmp_dir,
    )

    # Tìm file .pt trong thư mục đã tải
    pt_files = list(Path(local_path).parent.glob("**/*.pt")) if Path(local_path).is_file() else list(Path(local_path).glob("**/*.pt"))
    if Path(local_path).is_file() and local_path.endswith(".pt"):
        pt_path = local_path
    elif pt_files:
        pt_path = str(pt_files[0])
    else:
        print(f"[ERROR] Không tìm thấy file .pt trong artifact đã tải: {local_path}")
        sys.exit(1)

    size_mb = os.path.getsize(pt_path) / 1024 / 1024
    print(f"[OK]   Đã tải: {pt_path} ({size_mb:.1f} MB)")
    return pt_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model từ MLflow Registry sang ONNX"
    )
    parser.add_argument(
        "--alias",
        default=None,
        help="Alias trong MLflow Registry (mặc định lấy từ params.yaml → mlflow.deploy_alias)",
    )
    parser.add_argument(
        "--local-pt",
        default=None,
        metavar="PATH",
        help="(Tùy chọn) Bỏ qua MLflow, dùng file .pt local tại đường dẫn này",
    )
    args = parser.parse_args()

    # ── Đọc config ───────────────────────────────────────────────────────────
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_DIR = SCRIPT_DIR.parent
    params = load_params(str(PROJECT_DIR / "params.yaml"))
    mlflow_cfg = params.get("mlflow", {})

    tracking_uri  = mlflow_cfg.get("tracking_uri", "http://localhost:5001")
    registry_name = mlflow_cfg.get("registry_name", "YOLOv8_Helmet_Model")
    deploy_alias  = args.alias or mlflow_cfg.get("deploy_alias", "Production")

    DST_ONNX = PROJECT_DIR / "apps/vision_service/models/yolov8/yolov8_helmet.onnx"
    ENGINE   = Path(str(DST_ONNX) + "_b1_gpu0_fp16.engine")

    # ── Lấy đường dẫn file .pt ───────────────────────────────────────────────
    if args.local_pt:
        # Chế độ thủ công (debug)
        SRC_PT = args.local_pt
        print(f"[INFO] Chế độ local — dùng file: {SRC_PT}")
        if not os.path.exists(SRC_PT):
            print(f"[ERROR] Không tìm thấy file: {SRC_PT}")
            sys.exit(1)
    else:
        # Chế độ tự động — tải từ MLflow Model Registry
        SRC_PT = download_model_from_registry(
            tracking_uri=tracking_uri,
            registry_name=registry_name,
            alias=deploy_alias,
        )

    print(f"\n[INFO] Source .pt: {SRC_PT} ({os.path.getsize(SRC_PT) / 1024 / 1024:.1f} MB)")

    # ── Tạo thư mục output ────────────────────────────────────────────────────
    os.makedirs(DST_ONNX.parent, exist_ok=True)

    # ── Backup ONNX cũ nếu có ────────────────────────────────────────────────
    if DST_ONNX.exists():
        backup_path = Path(str(DST_ONNX) + ".bak")
        shutil.copy2(DST_ONNX, backup_path)
        print(f"[INFO] Backup ONNX cũ → {backup_path.name}")

    # ── Export YOLOv8 → ONNX ─────────────────────────────────────────────────
    print(f"\n[INFO] Đang load model từ: {SRC_PT}")
    model = YOLO(SRC_PT)

    print("[INFO] Đang export sang ONNX ...")
    exported = model.export(
        format="onnx",
        imgsz=640,
        keras=False,
        optimize=False,
        half=False,       # FP16 để TensorRT xử lý ở bước build engine
        dynamic=False,
        simplify=False,
        opset=11,
        verbose=True,
    )

    print(f"[INFO] Ultralytics exported file: {exported}")

    # ── Copy sang thư mục project ─────────────────────────────────────────────
    shutil.copy2(exported, DST_ONNX)
    size_mb = os.path.getsize(DST_ONNX) / 1024 / 1024
    print(f"[OK]   ONNX đã lưu tại: {DST_ONNX} ({size_mb:.1f} MB)")

    # ── Xoá TensorRT engine cũ (stale) ───────────────────────────────────────
    if ENGINE.exists():
        ENGINE.unlink()
        print(f"[INFO] Đã xoá engine cũ: {ENGINE.name}")
    else:
        print("[INFO] Không có engine cũ cần xoá.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Export hoàn tất!")
    print()
    print(f"  Model Registry : {registry_name}@{deploy_alias}")
    print(f"  ONNX output    : {DST_ONNX}")
    print()
    print("Bước tiếp theo:")
    print("  1. Build TensorRT engine:")
    print("       make build-engine")
    print()
    print("  2. Khởi động lại vision-service:")
    print("       docker compose up -d vision-service")
    print("=" * 60)


if __name__ == "__main__":
    main()