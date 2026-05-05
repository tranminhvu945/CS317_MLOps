from ultralytics import YOLO

def main():
    # 1. Khai báo đường dẫn đến file best.pt của bạn
    # Ví dụ: lấy từ run_13 hoặc run_14 của bạn
    weight_path = "/mmlab_students/storageStudents/nguyenvd/uit_medseg/datasetMLOps/TrainYolo/runs/detect/helmet_training/run_13/weights/best.pt"
    
    print(f"Đang tải mô hình từ: {weight_path}")
    model = YOLO(weight_path)

    # 2. Thực hiện Export sang ONNX
    print("Bắt đầu chuyển đổi sang định dạng ONNX...")
    
    # Các tham số quan trọng:
    # format="onnx": Chỉ định định dạng xuất ra
    # imgsz=640: Khóa cứng kích thước ảnh đầu vào (Nên giống với lúc train)
    # half=False: Nếu để True, mô hình sẽ nén xuống FP16 (chạy nhanh hơn, nhẹ hơn, nhưng có thể giảm mAP xíu)
    # dynamic=False: Nếu để True, bạn có thể truyền ảnh kích thước khác 640x640 lúc inference, nhưng model sẽ chạy chậm hơn.
    
    exported_path = model.export(
        format="onnx", 
        imgsz=640, 
        half=False,    # Đổi thành True nếu muốn mô hình nhẹ và chạy nhanh hơn trên GPU lúc deploy
        dynamic=False 
    )

    print(f"✅ Chuyển đổi thành công! File ONNX được lưu tại: {exported_path}")

if __name__ == "__main__":
    main()