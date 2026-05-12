"""
extract_shards.py
─────────────────
Script giải nén các file WebDataset (.tar) từ thư mục shards
trở lại định dạng YOLO (images/ và labels/) để phục vụ huấn luyện.
"""

import os
import glob
import tarfile
import argparse
import time
import yaml
from pathlib import Path


def extract_shards(shards_base: str, output_base: str, splits: list[str], limit_shards: int = 0):
    print("=" * 60)
    print("Giải nén WebDataset Shards sang định dạng YOLO")
    print("=" * 60)
    print(f"Shards dir : {shards_base}")
    print(f"Output dir : {output_base}")
    print(f"Splits     : {splits}")
    if limit_shards > 0:
        print(f"Limit      : Tối đa {limit_shards} shards/split (CHẾ ĐỘ TEST)")
    print("-" * 60)

    overall_start = time.time()
    
    for split in splits:
        split_shards_dir = os.path.join(shards_base, split)
        if not os.path.isdir(split_shards_dir):
            print(f"[SKIP] Không tìm thấy thư mục: {split_shards_dir}")
            continue
            
        images_dir = os.path.join(output_base, "images", split)
        labels_dir = os.path.join(output_base, "labels", split)
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        tar_files = sorted(glob.glob(os.path.join(split_shards_dir, "*.tar")))
        if not tar_files:
            print(f"[SKIP] Không có file .tar nào trong: {split_shards_dir}")
            continue
            
        # ─── MỚI: Lọc theo danh sách selected_shards (dạng dictionary) ──
        try:
            with open("params.yaml", "r") as f:
                params = yaml.safe_load(f)
                selected_shards_dict = params.get("sharding", {}).get("selected_shards", {})
                
                # Nếu người dùng cấu hình dạng dict: { "train": ["..."], "val": ["..."] }
                if isinstance(selected_shards_dict, dict) and split in selected_shards_dict:
                    allowed_files = selected_shards_dict[split]
                    if allowed_files:  # Nếu danh sách không rỗng
                        tar_files = [t for t in tar_files if os.path.basename(t) in allowed_files]
                        if not tar_files:
                            print(f"[SKIP] Có cấu hình selected_shards cho '{split}' nhưng không file nào khớp.")
                            continue
        except Exception:
            pass # Bỏ qua nếu không đọc được params.yaml
        # ────────────────────────────────────────────────────────────────
        
        if limit_shards > 0:
            tar_files = tar_files[:limit_shards]
            
        print(f"[{split.upper()}] Giải nén {len(tar_files)} shards...")
        start_time = time.time()
        
        extracted_images = 0
        extracted_labels = 0
        
        for tar_path in tar_files:
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".txt"):
                        # Extract vào thư mục labels
                        member.name = os.path.basename(member.name)
                        tar.extract(member, path=labels_dir)
                        extracted_labels += 1
                    elif member.name.endswith((".jpg", ".png", ".jpeg")):
                        # Extract vào thư mục images
                        member.name = os.path.basename(member.name)
                        tar.extract(member, path=images_dir)
                        extracted_images += 1
                        
        elapsed = time.time() - start_time
        print(f"  → Đã giải nén {extracted_images} ảnh và {extracted_labels} nhãn ({elapsed:.1f}s)")

    total_elapsed = time.time() - overall_start
    print("=" * 60)
    print(f"✅ Hoàn tất giải nén trong {total_elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", required=True, help="Thư mục chứa shards")
    parser.add_argument("--output-dir", required=True, help="Thư mục đích YOLO")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Các split cần giải nén")
    parser.add_argument("--limit", type=int, default=0, help="Số shards tối đa mỗi split")
    args = parser.parse_args()
    
    extract_shards(args.shards_dir, args.output_dir, args.splits, args.limit)
