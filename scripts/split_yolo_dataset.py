#!/usr/bin/env python3
"""
split_yolo_dataset.py
─────────────────────
Chia một YOLO dataset (chưa hoặc đã có split) thành 3 tập train / val / test
theo cấu trúc chuẩn YOLO:

  <dataset>/
    images/
      train/  val/  test/
    labels/
      train/  val/  test/

Quy trình:
  1. Quét đệ quy toàn bộ ảnh trong <dataset>/images/**.
  2. Ghép cặp với nhãn tương ứng trong <dataset>/labels/**.
  3. Shuffle (seed cố định).
  4. Chia theo tỉ lệ trong params.yaml (hoặc CLI arg).
  5. Move (hoặc copy) file vào đúng thư mục split.

Cách dùng:
  # Dùng tỉ lệ trong params.yaml (mặc định 0.8 / 0.1 / 0.1)
  python scripts/split_yolo_dataset.py \
      --dataset-dir dataset/extracted/yolo_helmet_dataset_new

  # Override tỉ lệ
  python scripts/split_yolo_dataset.py \
      --dataset-dir dataset/extracted/yolo_helmet_dataset_new \
      --split-ratios 0.7 0.15 0.15

  # Copy thay vì move (giữ nguyên file gốc)
  python scripts/split_yolo_dataset.py \
      --dataset-dir dataset/extracted/yolo_helmet_dataset_new \
      --copy
"""

import argparse
import random
import shutil
import yaml
from pathlib import Path


# ─── Helpers ──────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_params(path: str = "params.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        print(f"[WARN] Không đọc được {path}. Dùng cấu hình mặc định.")
        return {}


def collect_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Thu thập toàn bộ cặp (ảnh, nhãn) từ dataset_dir/images/**/."""
    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"

    if not images_root.exists():
        raise FileNotFoundError(f"Không tìm thấy: {images_root}")

    all_images = sorted(
        p for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not all_images:
        raise ValueError(f"Không có ảnh nào trong: {images_root}")

    pairs = []
    missing = 0
    for img in all_images:
        rel = img.relative_to(images_root)
        lbl = (labels_root / rel).with_suffix(".txt")
        pairs.append((img, lbl if lbl.exists() else None))
        if not lbl.exists():
            missing += 1

    print(f"  Tổng ảnh   : {len(pairs)}")
    if missing:
        print(f"  ⚠️  Thiếu nhãn : {missing} ảnh (sẽ tạo file nhãn rỗng)")
    return pairs


def transfer(src: Path, dst: Path, use_copy: bool):
    """Copy hoặc Move file từ src → dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def place_pair(
    img_src: Path,
    lbl_src: Path,
    images_root: Path,
    split_name: str,
    dataset_dir: Path,
    use_copy: bool,
):
    """Đặt một cặp ảnh/nhãn vào đúng split."""
    images_split_dir = dataset_dir / "images" / split_name
    labels_split_dir = dataset_dir / "labels" / split_name

    # Tên file phẳng (bỏ cấu trúc thư mục con trung gian như "train/")
    img_dst = images_split_dir / img_src.name
    lbl_dst = labels_split_dir / img_src.stem + ".txt" if False else labels_split_dir / (img_src.stem + ".txt")

    transfer(img_src, img_dst, use_copy)

    if lbl_src is not None:
        transfer(lbl_src, lbl_dst, use_copy)
    else:
        # Tạo file nhãn rỗng
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.write_text("")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chia YOLO dataset thành train/val/test"
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset/extracted/yolo_helmet_dataset_new",
        help="Thư mục YOLO dataset (chứa images/ và labels/)",
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
        help="Random seed (mặc định: 42)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy thay vì move (giữ nguyên file gốc)",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Đường dẫn đến file params.yaml",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"[ERROR] Không tìm thấy thư mục: {dataset_dir}")
        return

    # Đọc params
    params = load_params(args.params)
    sharding_cfg = params.get("sharding", {})

    if args.split_ratios:
        r_train, r_val, r_test = args.split_ratios
    else:
        ratios = sharding_cfg.get("split_ratios", [0.8, 0.1, 0.1])
        r_train, r_val, r_test = ratios

    total_ratio = r_train + r_val + r_test
    if abs(total_ratio - 1.0) > 0.01:
        print(f"[WARN] Tổng tỉ lệ = {total_ratio:.2f} (không bằng 1.0). Tự động chuẩn hóa.")
        r_train /= total_ratio
        r_val   /= total_ratio
        r_test  /= total_ratio

    print("=" * 60)
    print("YOLO Dataset Splitter")
    print("=" * 60)
    print(f"Dataset dir  : {dataset_dir}")
    print(f"Split ratios : train={r_train:.0%}  val={r_val:.0%}  test={r_test:.0%}")
    print(f"Seed         : {args.seed}")
    print(f"Mode         : {'copy' if args.copy else 'move'}")
    print("-" * 60)

    # 1. Thu thập cặp
    print("\n[1/3] Quét dữ liệu...")
    pairs = collect_pairs(dataset_dir)
    total = len(pairs)

    # 2. Shuffle
    print("\n[2/3] Shuffle...")
    random.seed(args.seed)
    random.shuffle(pairs)

    # 3. Chia split
    num_train = int(total * r_train)
    num_val   = int(total * r_val)

    splits = {
        "train": pairs[:num_train],
        "val"  : pairs[num_train : num_train + num_val],
        "test" : pairs[num_train + num_val :],
    }

    print(f"\n  train : {len(splits['train'])} ảnh  ({len(splits['train'])/total:.1%})")
    print(f"  val   : {len(splits['val'])} ảnh  ({len(splits['val'])/total:.1%})")
    print(f"  test  : {len(splits['test'])} ảnh  ({len(splits['test'])/total:.1%})")

    # 4. Move/Copy vào đúng split
    print("\n[3/3] Phân phối file vào các split...")
    images_root = dataset_dir / "images"

    for split_name, split_pairs in splits.items():
        print(f"\n  [{split_name.upper()}] {len(split_pairs)} ảnh")
        for img_src, lbl_src in split_pairs:
            # Bỏ qua nếu file đã ở đúng vị trí
            expected_img = dataset_dir / "images" / split_name / img_src.name
            if img_src.resolve() == expected_img.resolve():
                continue
            place_pair(img_src, lbl_src, images_root, split_name, dataset_dir, args.copy)

    # Xoá thư mục "train" trung gian (nếu move và đã rỗng)
    if not args.copy:
        for old_split_dir in (dataset_dir / "images").iterdir():
            if old_split_dir.is_dir() and old_split_dir.name not in ("train", "val", "test"):
                pass
        # Thử xoá thư mục nguồn gốc nếu rỗng (ví dụ images/train/ cũ nếu đã move hết)
        for candidate in [dataset_dir / "images" / "train",
                          dataset_dir / "labels" / "train"]:
            if candidate.exists():
                remaining = list(candidate.iterdir())
                if not remaining:
                    candidate.rmdir()
                    print(f"  [CLEAN] Đã xóa thư mục rỗng: {candidate}")

    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT! Cấu trúc dataset mới:")
    print("=" * 60)
    for split_name in ("train", "val", "test"):
        img_dir = dataset_dir / "images" / split_name
        lbl_dir = dataset_dir / "labels" / split_name
        img_count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        lbl_count = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
        print(f"  {split_name:5s}  → images: {img_count}  labels: {lbl_count}")
    print("=" * 60)
    print("\nBước tiếp theo: tạo shards từ dataset đã chia:")
    print("  python scripts/pack_yolo_to_shards.py \\")
    print(f"      --input-dir {dataset_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
