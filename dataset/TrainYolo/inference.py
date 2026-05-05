from ultralytics import YOLO
import os

def predict_on_video():
    # 1. Đường dẫn tới file trọng số tốt nhất của bạn
    best_weight_path = "/mmlab_students/storageStudents/nguyenvd/uit_medseg/datasetMLOps/TrainYolo/runs/detect/helmet_training/run_13/weights/best.pt"
    
    # 2. Đường dẫn tới file video bạn muốn test
    # (Bạn nhớ thay đổi đường dẫn này trỏ tới file video thực tế của bạn nhé)
    video_path = "/mmlab_students/storageStudents/nguyenvd/uit_medseg/MLOps/data/test.mp4" 
    
    print("Đang load mô hình để nhận diện video...")
    model = YOLO(best_weight_path)

    # 3. Chạy dự đoán
    print(f"Bắt đầu xử lý video: {video_path}...")
    results = model.predict(
        source=video_path,
        save=True,          # Bắt buộc để True nếu bạn muốn lưu lại video kết quả có vẽ box
        conf=0.5,           # Ngưỡng tự tin (chỉ lấy các box có độ tự tin >= 50%)
        iou=0.45,           # Ngưỡng NMS
        device=[4, 5, 6, 7],# Chạy trên GPU
        project="helmet_test_results", 
        name="predict_video"           
    )
    
    print(f"\nXử lý video hoàn tất!")
    print(f"Video kết quả đã được lưu tại: helmet_test_results/predict_video")

if __name__ == "__main__":
    predict_on_video()