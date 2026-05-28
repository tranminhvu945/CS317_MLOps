#!/usr/bin/env python3
"""
export_onnx.py
──────────────
Tải model best.pt từ MLflow Model Registry (alias Production),
sau đó export ra ONNX để sẵn sàng build TensorRT engine.

Stage này đọc evaluate_status.json:
  - promote=false  -> skip export, giữ nguyên ONNX/engine cũ, ghi export_status.json
  - promote=true   -> export Production mới sang ONNX, ghi export_status.json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Optional, Tuple

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


def write_export_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Ghi export_status.json: {path}")


def download_model_from_registry(
    tracking_uri: str,
    registry_name: str,
    alias: str,
) -> Tuple[str, str, str]:
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
    return pt_path, str(version), str(source)


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
    mlflow_cfg  = params.get("mlflow", {})
    train_cfg   = params.get("train", {})

    tracking_uri  = mlflow_cfg.get("tracking_uri", "http://localhost:5001")
    registry_name = mlflow_cfg.get("registry_name", "YOLOv8_Helmet_Model")
    deploy_alias  = args.alias or mlflow_cfg.get("deploy_alias", "Production")

    DST_ONNX = PROJECT_DIR / "apps/vision_service/models/yolov8/yolov8_helmet.onnx"
    ENGINE   = Path(str(DST_ONNX) + "_b1_gpu0_fp16.engine")

    run_dir = (
        PROJECT_DIR
        / "runs" / "detect"
        / train_cfg.get("project", "Helmet_Detection_Project")
        / train_cfg.get("name", "yolov8_binary_class")
    )
    eval_status_path = run_dir / "evaluate_status.json"
    export_status_path = run_dir / "export_status.json"

    # ── Đọc evaluate_status.json — bỏ qua export nếu Candidate bị reject ────
    promote = True
    gate_reason = "evaluate_status_missing"

    if eval_status_path.exists():
        try:
            with open(eval_status_path) as f:
                eval_status = json.load(f)
        except Exception as e:
            print(f"[ERROR] evaluate_status.json không hợp lệ: {e}")
            sys.exit(1)

        promote = bool(eval_status.get("promote", True))
        gate_reason = str(eval_status.get("reason", "unknown"))
        if not promote:
            payload = {
                "status": "skipped",
                "reason": "candidate_rejected_by_quality_gate",
                "evaluate_reason": gate_reason,
                "promote": False,
                "onnx_path": str(DST_ONNX),
                "onnx_exists": DST_ONNX.exists(),
                "engine_path": str(ENGINE),
                "engine_exists": ENGINE.exists(),
            }
            write_export_status(export_status_path, payload)

            print("=" * 60)
            print("Export ONNX — SKIPPED")
            print("=" * 60)
            print(f"[INFO] Quality Gate đã reject Candidate (reason: {gate_reason}).")
            print("[INFO] Giữ nguyên ONNX/Engine Production hiện tại.")
            print(f"[INFO] ONNX hiện tại: {DST_ONNX}")
            print("=" * 60)
            if not DST_ONNX.exists():
                print("[ERROR] Không thể skip export vì ONNX Production hiện tại không tồn tại.")
                sys.exit(1)
            return
        print(f"[INFO] Quality Gate PASS (reason: {gate_reason}) → tiến hành export.")
    else:
        print("[WARN] Không tìm thấy evaluate_status.json, mặc định promote=true và tiếp tục export.")

    model_version: Optional[str] = None
    model_source: Optional[str] = None

    try:
        # ── Lấy đường dẫn file .pt ───────────────────────────────────────────
        if args.local_pt:
            SRC_PT = args.local_pt
            model_source = "local-file"
            print(f"[INFO] Chế độ local — dùng file: {SRC_PT}")
            if not os.path.exists(SRC_PT):
                print(f"[ERROR] Không tìm thấy file: {SRC_PT}")
                sys.exit(1)
        else:
            SRC_PT, model_version, model_source = download_model_from_registry(
                tracking_uri=tracking_uri,
                registry_name=registry_name,
                alias=deploy_alias,
            )

        print(f"\n[INFO] Source .pt: {SRC_PT} ({os.path.getsize(SRC_PT) / 1024 / 1024:.1f} MB)")

        # ── Tạo thư mục output ────────────────────────────────────────────────
        os.makedirs(DST_ONNX.parent, exist_ok=True)

        # ── Backup ONNX cũ nếu có ────────────────────────────────────────────
        if DST_ONNX.exists():
            backup_path = Path(str(DST_ONNX) + ".bak")
            shutil.copy2(DST_ONNX, backup_path)
            print(f"[INFO] Backup ONNX cũ → {backup_path.name}")

        # ── Export YOLOv8 → ONNX ─────────────────────────────────────────────
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

        # ── Copy sang thư mục project ─────────────────────────────────────────
        shutil.copy2(exported, DST_ONNX)
        size_mb = os.path.getsize(DST_ONNX) / 1024 / 1024
        print(f"[OK]   ONNX đã lưu tại: {DST_ONNX} ({size_mb:.1f} MB)")

        # ── Xoá TensorRT engine cũ (stale) ───────────────────────────────────
        removed_engine = False
        if ENGINE.exists():
            ENGINE.unlink()
            removed_engine = True
            print(f"[INFO] Đã xoá engine cũ: {ENGINE.name}")
        else:
            print("[INFO] Không có engine cũ cần xoá.")

        payload = {
            "status": "exported",
            "reason": "production_model_exported",
            "evaluate_reason": gate_reason,
            "promote": True,
            "model_registry": registry_name,
            "deploy_alias": deploy_alias,
            "model_version": model_version,
            "model_source": model_source,
            "onnx_path": str(DST_ONNX),
            "onnx_size_mb": round(size_mb, 2),
            "engine_removed": removed_engine,
        }
        write_export_status(export_status_path, payload)
    except Exception as e:
        payload = {
            "status": "failed",
            "reason": "export_failed",
            "evaluate_reason": gate_reason,
            "promote": True,
            "error": str(e),
            "onnx_path": str(DST_ONNX),
        }
        write_export_status(export_status_path, payload)
        raise

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
