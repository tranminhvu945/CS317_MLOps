from ultralytics import YOLO

def evaluate_on_test():
    # Đường dẫn tới file trọng số tốt nhất sau khi train
    best_weight_path = "/mmlab_students/storageStudents/nguyenvd/uit_medseg/datasetMLOps/TrainYolo/runs/detect/helmet_training/run_13/weights/best.pt"
    
    # Khởi tạo mô hình
    print("Đang load mô hình để đánh giá...")
    model = YOLO(best_weight_path)

    # Chạy kiểm thử trên tập test
    print("Bắt đầu đánh giá trên tập test...")
    metrics = model.val(
        data="/mmlab_students/storageStudents/nguyenvd/uit_medseg/datasetMLOps/osfstorage/yolo_helmet/dataset.yaml",
        split='test',          # Chỉ định rõ là chạy trên tập test
        device=[4, 5, 6, 7],   # Chạy trên các GPU giống khi train
        batch=128              # Batch size khi test (có thể để lớn hơn khi train)
    )

    # In kết quả
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")

if __name__ == "__main__":
    evaluate_on_test()