#!/usr/bin/env python3
"""
pack_shards.py
──────────────
Công cụ Ingestion Script dùng để "dọn dẹp" và chuẩn hoá dữ liệu thô (ví dụ từ CVAT)
thành chuẩn WebDataset Shards (.tar) của project MLOps này.

Quy trình:
1. Quét tìm tất cả các file ảnh (.jpg, .png) trong thư mục input.
2. Tìm file nhãn (.txt) tương ứng trong thư mục `annotations/obj_train_data/...`.
3. Xáo trộn toàn bộ tập dữ liệu (Shuffle).
4. Cắt thành 3 tập train, val, test theo tỷ lệ trong `params.yaml` (vd: 80/10/10).
5. Đóng gói thành các file `.tar` (chứa 1000 ảnh/shard).
6. Tự động tính toán để đặt tên shard nối tiếp (vd shard-003.tar) để không ghi đè dữ liệu cũ.
"""

import os
import glob
import random
import tarfile
import argparse
import yaml
import shutil
from pathlib import Path


def load_params(path: str = "params.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[WARN] Không thể đọc {path}. Dùng cấu hình mặc định.")
        return {}


def get_next_shard_id(shard_dir: str) -> int:
    """Tìm ID tiếp theo cho file shard trong thư mục chỉ định."""
    os.makedirs(shard_dir, exist_ok=True)
    existing_tars = glob.glob(os.path.join(shard_dir, "*.tar"))
    if not existing_tars:
        return 1
    
    # Giả sử tên file là "shard-00001.tar"
    max_id = 0
    for tar in existing_tars:
        basename = os.path.basename(tar)
        # Bỏ ".tar" và cắt chuỗi sau "shard-"
        try:
            num_str = basename.replace(".tar", "").split("-")[-1]
            num = int(num_str)
            if num > max_id:
                max_id = num
        except ValueError:
            pass
    return max_id + 1


def create_shard(data_pairs: list, output_tar: str):
    """Đóng gói một danh sách các cặp (image, label) vào file .tar."""
    print(f"  → Tạo {os.path.basename(output_tar)} ({len(data_pairs)} files)")
    with tarfile.open(output_tar, "w") as tar:
        for img_path, txt_path in data_pairs:
            # Lưu vào .tar với tên gốc (không giữ cấu trúc thư mục)
            tar.add(img_path, arcname=os.path.basename(img_path))
            tar.add(txt_path, arcname=os.path.basename(txt_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Thư mục chứa dữ liệu thô (vd: dataset/data_new)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[ERROR] Không tìm thấy thư mục đầu vào: {input_dir}")
        return

    # ── 1. Đọc cấu hình từ params.yaml ─────────────────────────────────────────
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_DIR = SCRIPT_DIR.parent
    params = load_params(str(PROJECT_DIR / "params.yaml"))
    
    sharding_cfg = params.get("sharding", {})
    output_base_dir = PROJECT_DIR / sharding_cfg.get("output_dir", "dataset/shards")
    shard_size = sharding_cfg.get("shard_size", 1000)
    split_ratios = sharding_cfg.get("split_ratios", [0.8, 0.1, 0.1])

    # ── 2. Quét & ghép cặp ảnh + nhãn ──────────────────────────────────────────
    print(f"\n[1/4] Đang quét tìm ảnh và nhãn trong: {input_dir}")
    all_images = list(input_dir.glob("cam*/*.jpg")) + list(input_dir.glob("cam*/*.png"))
    
    valid_pairs = []
    missing_labels = 0

    for img_path in all_images:
        # Đường dẫn nhãn tương ứng (dựa vào cấu trúc bạn mô tả)
        # Ảnh: dataset/data_new/cam1_20260507_163002/frame_000001.jpg
        # Nhãn: dataset/data_new/annotations/obj_train_data/cam1_20260507_163002/frame_000001.txt
        cam_dir_name = img_path.parent.name
        label_filename = img_path.stem + ".txt"
        
        txt_path = input_dir / "annotations" / "obj_train_data" / cam_dir_name / label_filename
        
        if txt_path.exists():
            valid_pairs.append((str(img_path), str(txt_path)))
        else:
            missing_labels += 1

    print(f"      → Tìm thấy {len(valid_pairs)} cặp (ảnh + nhãn) hợp lệ.")
    if missing_labels > 0:
        print(f"      → [WARN] Có {missing_labels} ảnh không có nhãn (.txt), đã bị loại bỏ.")

    if not valid_pairs:
        print("[ERROR] Không tìm thấy cặp dữ liệu nào. Dừng đóng gói.")
        return

    # ── 3. Xáo trộn ngẫu nhiên ────────────────────────────────────────────────
    print("\n[2/4] Xáo trộn dữ liệu (Shuffle)...")
    random.seed(42)  # Seed cố định để có thể tái lập (nếu cần)
    random.shuffle(valid_pairs)

    # ── 4. Chia tỷ lệ (Split) ─────────────────────────────────────────────────
    print(f"\n[3/4] Cắt dữ liệu theo tỷ lệ {split_ratios}...")
    total = len(valid_pairs)
    r_train, r_val, r_test = split_ratios
    
    num_train = int(total * r_train)
    num_val   = int(total * r_val)
    # Phần còn lại dành cho test để tránh sai số làm tròn
    
    splits = {
        "train": valid_pairs[:num_train],
        "val": valid_pairs[num_train : num_train + num_val],
        "test": valid_pairs[num_train + num_val:]
    }
    
    for k, v in splits.items():
        print(f"      → {k.upper()}: {len(v)} ảnh")

    # ── 5. Đóng gói thành Shards ───────────────────────────────────────────────
    print(f"\n[4/4] Đóng gói thành WebDataset Shards (Kích thước: {shard_size} ảnh/shard)...")
    
    for split_name, data in splits.items():
        if not data:
            continue
            
        split_dir = os.path.join(output_base_dir, split_name)
        start_id = get_next_shard_id(split_dir)
        
        print(f"  [{split_name.upper()}] Bắt đầu từ ID: {start_id}")
        
        # Cắt thành các chunk nhỏ (shard_size)
        chunks = [data[i : i + shard_size] for i in range(0, len(data), shard_size)]
        
        for i, chunk in enumerate(chunks):
            shard_id = start_id + i
            # Format tên có nhiều số 0 (padding), ví dụ shard-00003.tar
            tar_name = f"shard-{shard_id:05d}.tar"
            tar_path = os.path.join(split_dir, tar_name)
            
            create_shard(chunk, tar_path)

    print("\n" + "="*60)
    print("🎉 HOÀN TẤT ĐÓNG GÓI DỮ LIỆU MỚI THÀNH SHARD!")
    print("="*60)
    print("Bạn có thể chạy các lệnh sau để huấn luyện:")
    print("  1. dvc add dataset/shards")
    print("  2. make dvc-train")
    print("="*60)


if __name__ == "__main__":
    main()
