#!/usr/bin/env python3
"""
deploy_model.py
───────────────
Đọc evaluate_status.json để kiểm tra xem Candidate có đạt Quality Gate và được
chấp nhận (accepted/promote=true) hay không.
Nếu đạt:
  1. Cập nhật symlink 'yolov8_helmet_active.engine' trỏ vào engine vừa compile.
  2. Khởi động lại container 'uit_medseg_vision' hoặc compose service 'vision-service'.
Nếu không đạt:
  - Bỏ qua bước cập nhật và in thông báo, giữ nguyên model cũ đang chạy.
"""

import argparse
import json
import os
import subprocess
import sys
import yaml
from pathlib import Path

def load_params(path: str = "params.yaml") -> dict:
    params_path = Path(path)
    if not params_path.exists():
        print(f"[ERROR] Không tìm thấy params.yaml tại: {path}")
        sys.exit(1)
    with open(params_path) as f:
        return yaml.safe_load(f)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart CD: Tự động deploy model mới nếu đạt Quality Gate"
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Đường dẫn đến file params.yaml",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Chế độ rollback khẩn cấp: bỏ qua kiểm tra evaluate_status.json, buộc deploy engine hiện tại",
    )
    args = parser.parse_args()

    # 1. Đọc config
    params = load_params(args.params)
    train_cfg = params.get("train", {})
    project = train_cfg.get("project", "Helmet_Detection_Project")
    name = train_cfg.get("name", "yolov8_binary_class")

    # Đường dẫn file trạng thái evaluate
    eval_status_path = Path("runs/detect") / project / name / "evaluate_status.json"

    if not eval_status_path.exists():
        print(f"[WARN] Không tìm thấy {eval_status_path}.")
        print("[WARN] Sẽ bỏ qua deploy tự động. Hãy đảm bảo pipeline evaluate đã chạy thành công.")
        sys.exit(0)

    # 2. Đọc kết quả Quality Gate (bỏ qua khi rollback)
    status = "accepted"
    promote = True

    if args.rollback:
        print("[INFO] Chế độ ROLLBACK — bỏ qua kiểm tra evaluate_status.json")
        print("[INFO] Buộc deploy engine hiện tại và khởi động lại dịch vụ.")
    else:
        if not eval_status_path.exists():
            print(f"[WARN] Không tìm thấy {eval_status_path}.")
            print("[WARN] Sẽ bỏ qua deploy tự động. Hãy đảm bảo pipeline evaluate đã chạy thành công.")
            sys.exit(0)

        try:
            with open(eval_status_path, "r", encoding="utf-8") as f:
                status_data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Không đọc được evaluate_status.json: {e}")
            sys.exit(1)

        status = status_data.get("status", "rejected")
        promote = status_data.get("promote", False)

    print("=" * 60)
    print("Smart CD Deployment Engine")
    print("=" * 60)
    if args.rollback:
        print("Mode    : ROLLBACK")
    else:
        print(f"Status  : {status.upper()}")
        print(f"Promote : {promote}")
        print(f"Reason  : {status_data.get('reason', 'unknown')}")
    print("-" * 60)

    # 3. Kiểm tra điều kiện deploy
    if status == "accepted" and promote:
        print("[INFO] Model mới đã đạt Quality Gate! Bắt đầu deploy...")

        # Cấu hình đường dẫn
        model_dir = Path("apps/vision_service/models/yolov8")
        target_engine = "yolov8_helmet.onnx_b1_gpu0_fp16.engine"
        active_symlink = model_dir / "yolov8_helmet_active.engine"

        # Kiểm tra file engine gốc có tồn tại không
        if not (model_dir / target_engine).exists():
            print(f"[ERROR] Không tìm thấy engine biên dịch tại: {model_dir / target_engine}")
            sys.exit(1)

        # Cập nhật symlink
        try:
            if active_symlink.is_symlink() or active_symlink.exists():
                active_symlink.unlink()
            active_symlink.symlink_to(target_engine)
            print(f"[OK]   Đã cập nhật symlink: {active_symlink.name} -> {target_engine}")
        except Exception as e:
            print(f"[ERROR] Lỗi khi tạo symlink: {e}")
            sys.exit(1)

        # Khởi động lại container/service để nạp model mới
        print("[INFO] Khởi động lại container dịch vụ để nạp model mới...")
        
        # Thử restart container standalone
        res = subprocess.run(
            ["docker", "restart", "uit_medseg_vision"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if res.returncode == 0:
            print("[OK]   Đã restart container 'uit_medseg_vision' thành công.")
        else:
            # Nếu không tìm thấy container standalone, thử qua Docker Compose
            print("[INFO] Không thể restart container standalone. Thử qua Docker Compose...")
            compose_cmd = ["docker", "compose"]
            if os.path.exists("apps/vision_service/.env"):
                compose_cmd.extend(["--env-file", "apps/vision_service/.env"])
            compose_cmd.extend(["restart", "vision-service"])
            
            res_compose = subprocess.run(
                compose_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if res_compose.returncode == 0:
                print("[OK]   Đã restart service 'vision-service' qua Compose thành công.")
            else:
                print("[ERROR] Restart thất bại cả standalone và compose.")
                print(f"Stdout: {res_compose.stdout}")
                print(f"Stderr: {res_compose.stderr}")
                sys.exit(1)

        print("[SUCCESS] Triển khai model mới thành công!")
    else:
        print("[SKIP] Model mới không đạt Quality Gate (hoặc chưa được đánh giá).")
        print("[SKIP] Giữ nguyên model cũ đang chạy để tránh gián đoạn dịch vụ.")
    print("=" * 60)

if __name__ == "__main__":
    main()
