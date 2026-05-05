from ultralytics import YOLO
import os

# 1. Cấu hình MLflow Tracking (YOLOv8 tự động nhận diện nếu đã cài thư viện mlflow)
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Helmet_Detection_Project"
os.environ["MLFLOW_RUN"] = "yolov8_binary_class"
# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000" # Mở comment dòng này nếu bạn có tracking server riêng

def main():
    print("Khởi tạo mô hình YOLOv8 Nano...")
    # Tải pre-trained model (nên bắt đầu với bản 'n' - nano để test trước cho nhẹ)
    model = YOLO("yolov8n.pt") 

    print("Bắt đầu quá trình huấn luyện...")
    # Cấu hình các tham số train
    results = model.train(
        data="/mmlab_students/storageStudents/nguyenvd/uit_medseg/datasetMLOps/osfstorage/yolo_helmet/dataset.yaml",
        epochs=15,                  # Số vòng lặp qua toàn bộ dataset
        imgsz=640,                  # Kích thước ảnh đầu vào
        batch=320,                   # Số ảnh đưa vào GPU mỗi lần (Giảm xuống 8 nếu GPU báo lỗi Out Of Memory)
        device=[4,5,6,7],                   # Sử dụng GPU đầu tiên (cuda:0)
        patience=4,                # Early stopping: Dừng sớm nếu sau 10 epoch không cải thiện
        project="helmet_training",  # Thư mục lưu kết quả model (weights/best.pt)
        name="run_1"                # Tên thư mục con lưu kết quả của lần chạy này
    )

    print("Huấn luyện hoàn tất! Weights tốt nhất được lưu trong thư mục 'helmet_training/run_1/weights/best.pt'")

    # Chạy kiểm thử tự động trên tập Test sau khi train xong
    print("\nTiến hành đánh giá trên tập Test...")
    metrics = model.val(split='test')
    print(f"mAP50-95 trên tập Test: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()


