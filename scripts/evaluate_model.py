#!/usr/bin/env python3
"""
evaluate_model.py
─────────────────
Kiểm định chất lượng mô hình (Model Quality Gate) trước khi đưa vào Production.
So sánh mAP50 trên tập Gold Standard giữa Candidate model mới và Production model hiện tại.

Gold Standard là tập ảnh cố định (300 ảnh, không thuộc tập train/val),
đảm bảo việc so sánh công bằng và nhất quán giữa các lần train khác nhau.
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path
from ultralytics import YOLO


def load_params(path: str = "params.yaml") -> dict:
    params_path = Path(path)
    if not params_path.exists():
        alt = Path(__file__).parent.parent / "params.yaml"
        if alt.exists():
            params_path = alt
        else:
            print(f"[ERROR] Không tìm thấy params.yaml tại: {path}")
            sys.exit(1)
    with open(params_path) as f:
        return yaml.safe_load(f)


def main():
    params = load_params("params.yaml")
    train_cfg = params.get("train", {})
    mlflow_cfg = params.get("mlflow", {})

    tracking_uri = mlflow_cfg.get("tracking_uri", "http://localhost:5001")
    registry_name = mlflow_cfg.get("registry_name", "YOLOv8_Helmet_Model")

    # Dùng Gold Standard dataset cố định để đánh giá Quality Gate
    # → đảm bảo so sánh công bằng giữa Candidate và Production
    gold_standard_yaml = "dataset/gold_standard.yaml"
    if not os.path.exists(gold_standard_yaml):
        print(f"[ERROR] Không tìm thấy file Gold Standard dataset: {gold_standard_yaml}")
        sys.exit(1)

    device = train_cfg.get("device", [0])
    if isinstance(device, list) and len(device) > 0:
        device = device[0]

    best_pt_path = os.path.join(
        "runs", "detect", train_cfg["project"], train_cfg["name"], "weights", "best.pt"
    )

    if not os.path.exists(best_pt_path):
        print(f"[ERROR] Không tìm thấy file Candidate model tại: {best_pt_path}")
        sys.exit(1)

    print("=" * 60)
    print("Mô hình kiểm định chất lượng (Model Quality Gate)")
    print("=" * 60)

    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError:
        print("[ERROR] Thiếu thư viện mlflow. Chạy: pip install mlflow")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # 1. Đánh giá Candidate Model trên Gold Standard
    print(f"[INFO] Đang đánh giá Candidate Model tại: {best_pt_path}")
    print(f"[INFO] Dùng Gold Standard dataset: {gold_standard_yaml}")
    candidate_yolo = YOLO(best_pt_path)
    candidate_metrics = candidate_yolo.val(
        data=gold_standard_yaml, split="test", device=device, verbose=False
    )
    candidate_map = candidate_metrics.box.map50
    print(f"[OK]   Candidate mAP50 (Gold Standard): {candidate_map:.4f}")

    # 2. Tìm và đánh giá Production Model hiện tại
    prod_map = 0.0
    prod_version = None
    has_production_model = False

    print(f"[INFO] Tìm model '{registry_name}' với alias 'Production'...")
    try:
        prod_model_info = client.get_model_version_by_alias(
            name=registry_name, alias="Production"
        )
        has_production_model = True
        prod_version = prod_model_info.version
        source = prod_model_info.source
        print(f"[OK]   Tìm thấy Production Model: version {prod_version}")

        # Tải Production Model từ registry về
        tmp_dir = tempfile.mkdtemp(prefix="mlflow_prod_")
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=source,
            dst_path=tmp_dir,
        )
        # Tìm file .pt
        pt_files = (
            list(Path(local_path).parent.glob("**/*.pt"))
            if Path(local_path).is_file()
            else list(Path(local_path).glob("**/*.pt"))
        )
        if Path(local_path).is_file() and local_path.endswith(".pt"):
            prod_pt_path = local_path
        elif pt_files:
            prod_pt_path = str(pt_files[0])
        else:
            print(
                f"[ERROR] Không tìm thấy file .pt trong artifact Production đã tải: {local_path}"
            )
            sys.exit(1)

        print(f"[INFO] Đang đánh giá Production Model tại: {prod_pt_path}")
        prod_yolo = YOLO(prod_pt_path)
        prod_metrics = prod_yolo.val(
            data=gold_standard_yaml, split="test", device=device, verbose=False
        )
        prod_map = prod_metrics.box.map50
        print(f"[OK]   Production mAP50 (Gold Standard): {prod_map:.4f}")
    except Exception as e:
        print(f"[INFO] Không có Production Model hoạt động hoặc không tải được: {e}")
        print(
            "[INFO] Candidate Model sẽ tự động được chấp nhận làm Production Model đầu tiên."
        )

    # 3. So sánh Quality Gate
    print("-" * 60)
    print(f"Candidate mAP50  : {candidate_map:.4f}")
    print(f"Production mAP50 : {prod_map:.4f}")
    print("-" * 60)

    if candidate_map >= prod_map:
        print(
            "[PASS] Candidate Model đạt chất lượng tốt hơn hoặc bằng Production hiện tại!"
        )

        # Lấy thông tin Candidate version từ registry để promote
        try:
            cand_model_info = client.get_model_version_by_alias(
                name=registry_name, alias="Candidate"
            )
            cand_version = cand_model_info.version
            print(
                f"[INFO] Thăng cấp model version {cand_version} thành 'Production'..."
            )

            # Gắn alias Production
            client.set_registered_model_alias(
                name=registry_name, alias="Production", version=cand_version
            )
            print(
                f"[OK]   Thăng cấp thành công! Version {cand_version} giờ là 'Production'"
            )

            # Ghi file evaluate_status.json để DVC nhận diện đầu ra thành công
            eval_status_parent = os.path.dirname(os.path.dirname(best_pt_path))
            eval_status_path = os.path.join(eval_status_parent, "evaluate_status.json")
            import json
            with open(eval_status_path, "w") as f:
                json.dump({
                    "status": "passed",
                    "candidate_map50": float(candidate_map),
                    "production_map50": float(prod_map)
                }, f, indent=2)
            print(f"[INFO] Ghi file trạng thái chất lượng: {eval_status_path}")

            # Xóa alias Candidate
            try:
                client.delete_registered_model_alias(
                    name=registry_name, alias="Candidate"
                )
            except Exception:
                pass
        except Exception as e:
            print(f"[ERROR] Không thể thăng cấp model trên Registry: {e}")
            sys.exit(1)
    else:
        print("[FAIL] Candidate Model có hiệu năng kém hơn Production hiện tại.")
        print("[FAIL] Từ chối thăng cấp model.")
        sys.exit(1)


if __name__ == "__main__":
    main()
