"""
train_yolo.py
─────────────
Huấn luyện mô hình YOLOv8 trên dataset đã được giải nén từ DVC pipeline.

Sau khi train xong, tự động:
  1. Log file best.pt lên MLflow server dưới dạng artifact.
  2. Đăng ký model vào MLflow Model Registry.
  3. Gắn alias Candidate cho version mới nhất (chưa promote Production).
"""

import os
import yaml
import mlflow
from mlflow import MlflowClient
from ultralytics import YOLO


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class YOLOv8Wrapper(mlflow.pyfunc.PythonModel):
    """Một wrapper rỗng để trick MLflow hiểu đây là một model hợp lệ."""
    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        pass

def register_best_model(
    run_id: str,
    best_pt_path: str,
    registry_name: str,
    deploy_alias: str,
    client: MlflowClient,
) -> None:
    """
    Đóng gói file best.pt thành MLflow Model và đăng ký vào Registry.
    """
    if not os.path.exists(best_pt_path):
        print(f"[WARN] best.pt không tìm thấy tại: {best_pt_path}")
        print("       Bỏ qua bước đăng ký Model Registry.")
        return

    print(f"\n{'='*60}")
    print("Đăng ký Model vào MLflow Model Registry")
    print(f"{'='*60}")

    # ── 1. Log file best.pt dưới dạng MLflow Model ─────────────────────────
    print(f"[INFO] Logging model artifact từ: {best_pt_path}")
    
    with mlflow.start_run(run_id=run_id):
        mlflow.pyfunc.log_model(
            artifact_path="yolo_model",
            python_model=YOLOv8Wrapper(),
            artifacts={"best.pt": best_pt_path}
        )

    artifact_uri = f"runs:/{run_id}/yolo_model"
    print(f"[INFO] Artifact URI: {artifact_uri}")

    # ── 2. Đăng ký vào Model Registry ───────────────────────────────────────
    print(f"[INFO] Đăng ký model '{registry_name}' vào Registry...")
    registered_model = mlflow.register_model(
        model_uri=artifact_uri,
        name=registry_name,
    )
    version = registered_model.version
    print(f"[OK]   Đã tạo version: {version}")

    # ── 3. Gắn alias cho version vừa đăng ký ────────────────────────────────
    print(f"[INFO] Gắn alias '{deploy_alias}' cho version {version}...")
    client.set_registered_model_alias(
        name=registry_name,
        alias=deploy_alias,
        version=version,
    )
    print(f"[OK]   Model '{registry_name}' version {version} → alias '{deploy_alias}'")
    print(f"       Dùng lệnh: mlflow models download --model-uri 'models:/{registry_name}@{deploy_alias}'")


def main():
    params = load_params("params.yaml")
    train_cfg = params["train"]
    model_cfg = params["model"]
    mlflow_cfg = params["mlflow"]

    registry_name = mlflow_cfg.get("registry_name", "YOLOv8_Helmet_Model")
    deploy_alias = "Candidate"

    # ── MLflow Tracking Configuration ────────────────────────────────────────
    tracking_uri = mlflow_cfg.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    client = MlflowClient()

    workers = train_cfg.get("workers", 4)

    print("=" * 60)
    print("Huấn luyện YOLOv8")
    print("=" * 60)
    print(f"  Model   : {model_cfg['name']}")
    print(f"  Epochs  : {train_cfg['epochs']}")
    print(f"  Batch   : {train_cfg['batch']}")
    print(f"  Workers : {workers}")
    print(f"  imgsz   : {train_cfg['imgsz']}")
    print(f"  Device  : {train_cfg['device']}")

    model = YOLO(model_cfg["name"])

    # ── Bắt đầu huấn luyện YOLO ──────────────────────────────────────────────
    # YOLO sẽ tự động tạo/dùng MLflow run và log metrics, params theo project+name
    results = model.train(
        data=params["dataset"]["yaml"],
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        workers=workers,
        device=train_cfg["device"],
        patience=train_cfg["patience"],
        project=train_cfg["project"],
        name=train_cfg["name"],
        exist_ok=True,  # Ghi đè vào thư mục cũ
    )

    # ── Đánh giá sau khi train xong ──────────────────────────────────────────
    print("\nĐánh giá trên tập Test...")
    metrics = model.val(split="test")
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")

    # ── Lấy run_id của experiment vừa chạy để đăng ký model ──────────────────
    # YOLO log vào experiment có tên giống với train_cfg["project"]
    experiment = client.get_experiment_by_name(train_cfg["project"])
    run_id = None
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.`mlflow.runName` = '{train_cfg['name']}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if runs:
            run_id = runs[0].info.run_id
            print(f"\n[INFO] Tìm thấy MLflow run_id: {run_id}")

    # ── Đường dẫn file best.pt ────────────────────────────────────────────────
    best_pt_path = os.path.join(
        "runs", "detect", train_cfg["project"], train_cfg["name"], "weights", "best.pt"
    )

    # ── Đăng ký model vào MLflow Model Registry ───────────────────────────────
    if run_id:
        register_best_model(
            run_id=run_id,
            best_pt_path=best_pt_path,
            registry_name=registry_name,
            deploy_alias=deploy_alias,
            client=client,
        )
    else:
        print("\n[WARN] Không tìm thấy MLflow run_id — bỏ qua bước Model Registry.")
        print("       Hãy chắc chắn MLflow server đang chạy tại:", tracking_uri)


if __name__ == "__main__":
    main()
