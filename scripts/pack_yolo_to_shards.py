#!/usr/bin/env python3
"""
pack_yolo_to_shards.py
──────────────────────
Đóng gói một YOLO dataset (images/ + labels/) thành WebDataset shards (.tar).

Quy trình:
  1. Quét tìm toàn bộ cặp (ảnh, nhãn) trong thư mục input-dir.
  2. Shuffle ngẫu nhiên (seed cố định để tái lập được).
  3. Chia train / val / test theo tỉ lệ trong params.yaml.
  4. Đóng gói mỗi split thành các file shard-XXXXX.tar (nối tiếp ID cũ).
  5. Cập nhật DVC tracking qua 'dvc commit dataset/shards.dvc'.

Cách dùng:
  python scripts/pack_yolo_to_shards.py \
      --input-dir dataset/extracted/yolo_helmet_dataset_new

  # Tuỳ chọn override tỉ lệ split
  python scripts/pack_yolo_to_shards.py \
      --input-dir dataset/extracted/yolo_helmet_dataset_new \
      --split-ratios 0.8 0.1 0.1

  # Chỉ đóng gói một split cụ thể (skip split kia)
  python scripts/pack_yolo_to_shards.py \
      --input-dir dataset/extracted/yolo_helmet_dataset_new \
      --splits train val
"""

import io
import os
import glob
import random
import tarfile
import argparse
import yaml
from pathlib import Path


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_params(path: str = "params.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        print(f"[WARN] Không thể đọc {path}. Dùng cấu hình mặc định.")
        return {}


def get_next_shard_id(shard_dir: str) -> int:
    """
    Trả về ID tiếp theo (1-indexed) để đặt tên shard không ghi đè file cũ.
    Hỗ trợ cả 2 định dạng: 'shard-00052.tar' và 'train-000051.tar'.
    """
    os.makedirs(shard_dir, exist_ok=True)
    existing = glob.glob(os.path.join(shard_dir, "*.tar"))
    if not existing:
        return 1
    max_id = 0
    for tar_path in existing:
        name = os.path.basename(tar_path).replace(".tar", "")
        try:
            num = int(name.split("-")[-1])
            if num > max_id:
                max_id = num
        except ValueError:
            pass
    return max_id + 1


def collect_pairs(input_dir: Path) -> list[tuple[str, str]]:
    """
    Quét đệ quy toàn bộ ảnh trong input_dir/images/**/
    và ghép cặp với nhãn tương ứng trong input_dir/labels/**/
    """
    images_root = input_dir / "images"
    labels_root = input_dir / "labels"

    if not images_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_root}")

    image_extensions = {".jpg", ".jpeg", ".png"}
    all_images = sorted(
        p for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    if not all_images:
        print(f"[INFO] Không tìm thấy ảnh nào trong: {images_root}")
        return []

    pairs = []
    missing = 0
    for img_path in all_images:
        # Giữ đường dẫn tương đối từ images_root để map sang labels_root
        rel = img_path.relative_to(images_root)
        lbl_path = (labels_root / rel).with_suffix(".txt")
        if lbl_path.exists():
            pairs.append((str(img_path), str(lbl_path)))
        else:
            pairs.append((str(img_path), None))
            missing += 1

    print(f"  Tổng số ảnh tìm thấy : {len(pairs)}")
    if missing:
        print(f"  ⚠️  Ảnh thiếu nhãn    : {missing} (sẽ dùng nhãn rỗng)")
    return pairs


def create_shard(pairs: list, output_tar: str):
    """Đóng gói danh sách cặp (img, lbl) vào một file .tar."""
    print(f"    → {os.path.basename(output_tar)}  ({len(pairs)} mẫu)")
    with tarfile.open(output_tar, "w") as tar:
        for img_path, lbl_path in pairs:
            # Ảnh
            tar.add(img_path, arcname=os.path.basename(img_path))
            # Nhãn
            if lbl_path and os.path.exists(lbl_path):
                tar.add(lbl_path, arcname=os.path.basename(lbl_path))
            else:
                # Nhãn trống (ảnh không có object)
                empty_name = Path(img_path).stem + ".txt"
                info = tarfile.TarInfo(name=empty_name)
                info.size = 0
                tar.addfile(info, io.BytesIO(b""))


def pack_split(split_name: str, pairs: list, output_base: Path, shard_size: int):
    """Đóng gói một split thành các shard file, nối tiếp ID cũ."""
    if not pairs:
        print(f"  [SKIP] {split_name}: không có dữ liệu.")
        return

    split_dir = output_base / split_name
    start_id = get_next_shard_id(str(split_dir))
    chunks = [pairs[i : i + shard_size] for i in range(0, len(pairs), shard_size)]
    num_shards = len(chunks)

    print(f"\n[{split_name.upper()}]  {len(pairs)} ảnh  →  {num_shards} shard(s)  (bắt đầu từ ID {start_id})")
    for i, chunk in enumerate(chunks):
        shard_id = start_id + i
        tar_path = split_dir / f"shard-{shard_id:05d}.tar"
        create_shard(chunk, str(tar_path))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chia train/val/test rồi đóng gói YOLO dataset sang WebDataset shards"
    )
    parser.add_argument(
        "--input-dir",
        default="dataset/extracted/yolo_helmet_dataset_new",
        help="Thư mục YOLO dataset đầu vào (chứa images/ và labels/)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Danh sách split cần tạo shard (mặc định: train val test)",
    )
    parser.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Tỉ lệ chia train/val/test (mặc định lấy từ params.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed để shuffle (mặc định: 42)",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Đường dẫn đến file params.yaml",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[ERROR] Không tìm thấy thư mục: {input_dir}")
        return

    # 1. Đọc cấu hình params.yaml
    params = load_params(args.params)
    sharding_cfg = params.get("sharding", {})
    output_base = Path(sharding_cfg.get("output_dir", "dataset/shards"))
    shard_size = sharding_cfg.get("shard_size", 1000)

    # Tỉ lệ split: ưu tiên CLI arg → params.yaml → mặc định 0.8/0.1/0.1
    if args.split_ratios:
        split_ratios = args.split_ratios
    else:
        split_ratios = sharding_cfg.get("split_ratios", [0.8, 0.1, 0.1])

    r_train, r_val, r_test = split_ratios

    print("=" * 60)
    print("WebDataset Sharding Pipeline  (với Train/Val/Test Split)")
    print("=" * 60)
    print(f"Input dir    : {input_dir}")
    print(f"Output dir   : {output_base}")
    print(f"Shard size   : {shard_size} ảnh/shard")
    print(f"Split ratios : train={r_train}  val={r_val}  test={r_test}")
    print(f"Random seed  : {args.seed}")
    print("-" * 60)

    # 2. Thu thập toàn bộ cặp ảnh-nhãn
    print("\n[1/3] Quét dữ liệu...")
    pairs = collect_pairs(input_dir)
    if not pairs:
        print("[INFO] Không có dữ liệu mới để đóng gói thành shards. Bỏ qua.")
        return
    total = len(pairs)

    # 3. Shuffle
    print("\n[2/3] Shuffle...")
    random.seed(args.seed)
    random.shuffle(pairs)

    # 4. Chia split
    num_train = int(total * r_train)
    num_val   = int(total * r_val)
    # Phần còn lại vào test để tránh sai số làm tròn
    splits_data = {
        "train": pairs[:num_train],
        "val"  : pairs[num_train : num_train + num_val],
        "test" : pairs[num_train + num_val :],
    }
    print(f"\n  train : {len(splits_data['train'])} ảnh")
    print(f"  val   : {len(splits_data['val'])} ảnh")
    print(f"  test  : {len(splits_data['test'])} ảnh")

    # 5. Đóng gói
    print("\n[3/3] Đóng gói thành Shards...")
    for split_name in args.splits:
        pack_split(split_name, splits_data.get(split_name, []), output_base, shard_size)

    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT!")
    print("=" * 60)
    print("Các bước tiếp theo để cập nhật DVC:")
    print("  dvc commit dataset/shards.dvc")
    print("  git add dataset/shards.dvc")
    print("  git commit -m 'chore: add new shards from yolo_helmet_dataset_new'")
    print("  dvc push")
    print("=" * 60)


if __name__ == "__main__":
    main()
