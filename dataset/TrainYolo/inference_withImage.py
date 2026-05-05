from ultralytics import YOLO
import os
import cv2
import numpy as np

def predict_on_video():
    # 1. Đường dẫn tới file trọng số tốt nhất
    best_weight_path = "/mmlab_students/storageStudents/nguyenvd/uit_medseg/datasetMLOps/TrainYolo/runs/detect/helmet_training/run_13/weights/best.pt"
    
    # 2. Đường dẫn tới file video test
    video_path = "/mmlab_students/storageStudents/nguyenvd/uit_medseg/MLOps/data/test.mp4"
    
    # 3. Thư mục output
    project_dir = "helmet_test_results"
    run_name = "predict_video"
    output_dir = os.path.join(project_dir, run_name)
    samples_dir = os.path.join(output_dir, "sample_frames")
    before_dir = os.path.join(samples_dir, "before")
    after_dir = os.path.join(samples_dir, "after")
    compare_dir = os.path.join(samples_dir, "compare")

    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    print("Đang load mô hình để nhận diện video...")
    model = YOLO(best_weight_path)

    # =========================
    # PHẦN 1: Inference toàn bộ video
    # =========================
    print(f"Bắt đầu xử lý video: {video_path}...")
    results = model.predict(
        source=video_path,
        save=True,               # Lưu video kết quả
        conf=0.5,
        iou=0.45,
        device=4,                # Nên dùng 1 GPU. Nếu muốn multi-GPU phải xử lý kiểu khác
        project=project_dir,
        name=run_name
    )

    print("\nXử lý video hoàn tất!")
    print(f"Video kết quả đã được lưu tại: {output_dir}")

    # =========================
    # PHẦN 2: Trích vài frame trước và sau inference
    # =========================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video để trích frame.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Tổng số frame trong video: {total_frames}")

    # Chọn một vài frame đại diện
    num_samples = 4
    if total_frames < num_samples:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()

    print(f"Sẽ lưu các frame mẫu tại vị trí: {frame_indices}")

    for idx, frame_id in enumerate(frame_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            print(f"Không đọc được frame {frame_id}")
            continue

        # Lưu ảnh gốc
        before_path = os.path.join(before_dir, f"before_{idx:02d}.jpg")
        cv2.imwrite(before_path, frame)

        # Chạy inference trên frame
        pred = model.predict(
            source=frame,
            conf=0.5,
            iou=0.45,
            device=4,
            verbose=False
        )

        # Vẽ kết quả lên ảnh
        annotated_frame = pred[0].plot()

        # Lưu ảnh sau inference
        after_path = os.path.join(after_dir, f"after_{idx:02d}.jpg")
        cv2.imwrite(after_path, annotated_frame)

        # Ghép ảnh trước / sau
        h1, w1 = frame.shape[:2]
        h2, w2 = annotated_frame.shape[:2]

        # Resize nếu kích thước khác nhau
        if h1 != h2 or w1 != w2:
            annotated_frame = cv2.resize(annotated_frame, (w1, h1))

        # Ghi nhãn text
        before_show = frame.copy()
        after_show = annotated_frame.copy()
        cv2.putText(before_show, "Before Inference", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(after_show, "After Inference", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        compare_img = np.hstack((before_show, after_show))
        compare_path = os.path.join(compare_dir, f"compare_{idx:02d}.jpg")
        cv2.imwrite(compare_path, compare_img)

        print(f"Đã lưu frame mẫu {idx}:")
        print(f"  - Before : {before_path}")
        print(f"  - After  : {after_path}")
        print(f"  - Compare: {compare_path}")

    cap.release()
    print("\nĐã xuất xong các ảnh trước và sau inference.")

if __name__ == "__main__":
    predict_on_video()