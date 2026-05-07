"""
train_yolo.py
─────────────
Huấn luyện mô hình YOLOv8 trên dataset đã được giải nén từ DVC pipeline.
"""

import os
import yaml
import mlflow
from ultralytics import YOLO

def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    params = load_params("params.yaml")
    train_cfg = params["train"]
    model_cfg = params["model"]
    mlflow_cfg = params["mlflow"]
    
    # MLflow Tracking Configuration
    if mlflow_cfg.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        # YOLO cũng hay dùng biến môi trường này:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_cfg["tracking_uri"]


    print("=" * 60)
    print("Huấn luyện YOLOv8")
    print("=" * 60)
    print(f"  Model  : {model_cfg['name']}")
    print(f"  Epochs : {train_cfg['epochs']}")
    print(f"  Batch  : {train_cfg['batch']}")
    print(f"  imgsz  : {train_cfg['imgsz']}")
    print(f"  Device : {train_cfg['device']}")

    model = YOLO(model_cfg["name"])
    
    # Bắt đầu huấn luyện YOLO (YOLO sẽ tự động log vào MLflow theo project và name)
    results = model.train(
        data=params["dataset"]["yaml"],
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        patience=train_cfg["patience"],
        project=train_cfg["project"],
        name=train_cfg["name"],
        exist_ok=True,  # Bắt buộc YOLO ghi đè vào thư mục cũ
    )

    # Đánh giá sau khi train xong
    print("\nĐánh giá trên tập Test...")
    metrics = model.val(split="test")
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
