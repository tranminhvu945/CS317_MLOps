from ultralytics import YOLO
import os
import yaml

# ─── Đọc hyperparameters từ params.yaml (DVC quản lý) ────────────────────────
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_cfg = params["train"]
model_cfg = params["model"]
mlflow_cfg = params["mlflow"]

# ─── Cấu hình MLflow Tracking ────────────────────────────────────────────────
# Ultralytics tự động phát hiện biến môi trường MLFLOW_* và ghi log
os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_cfg["experiment_name"]
os.environ["MLFLOW_RUN"] = mlflow_cfg["run_name"]
if mlflow_cfg.get("tracking_uri"):
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_cfg["tracking_uri"]


def main():
    print(f"Khởi tạo mô hình {model_cfg['name']}...")
    model = YOLO(model_cfg["name"])

    print("Bắt đầu quá trình huấn luyện...")
    results = model.train(
        data=params["dataset"]["yaml"],
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        patience=train_cfg["patience"],
        project=train_cfg["project"],
        name=train_cfg["name"],
    )

    print(f"Huấn luyện hoàn tất! Weights tốt nhất lưu tại: {train_cfg['project']}/{train_cfg['name']}/weights/best.pt")

    # ─── Đánh giá trên tập Test ───────────────────────────────────────────────
    print("\nTiến hành đánh giá trên tập Test...")
    metrics = model.val(split="test")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
