"""
create_shards.py
────────────────
Chuyển đổi dataset YOLO format sang WebDataset shards (.tar).

Layout đầu vào (YOLO):
    dataset/yolo_helmet_dataset/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt

Layout đầu ra (WebDataset shards):
    dataset/shards/
        train/  train-000000.tar  train-000001.tar ...
        val/    val-000000.tar    val-000001.tar   ...
        test/   test-000000.tar   test-000001.tar  ...

Mỗi sample trong shard gồm 2 key:
    {stem}.jpg  → nội dung ảnh (bytes)
    {stem}.txt  → nội dung nhãn YOLO (bytes), rỗng nếu không có nhãn
"""

import os
import sys
import glob
import yaml
import json
import argparse
import time
from pathlib import Path

import webdataset as wds


# ─── Đọc cấu hình từ params.yaml ──────────────────────────────────────────────
def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def collect_samples(images_dir: str, labels_dir: str) -> list[dict]:
    """Thu thập tất cả cặp (ảnh, nhãn) từ thư mục YOLO."""
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(images_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(images_dir, "**", "*.jpeg"), recursive=True)
    )

    samples = []
    for img_path in image_paths:
        stem = Path(img_path).stem
        ext = Path(img_path).suffix.lstrip(".")  # "jpg" hoặc "png"
        label_path = os.path.join(labels_dir, stem + ".txt")
        samples.append(
            {
                "key": stem,
                "image_path": img_path,
                "image_ext": ext,
                "label_path": label_path if os.path.exists(label_path) else None,
            }
        )
    return samples


def create_split_shards(
    split: str,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    shard_size: int,
) -> dict:
    """Tạo WebDataset shards cho một split (train/val/test)."""
    os.makedirs(output_dir, exist_ok=True)

    samples = collect_samples(images_dir, labels_dir)
    if not samples:
        print(f"[WARN] Không tìm thấy ảnh nào trong: {images_dir}")
        return {}

    total = len(samples)
    pattern = os.path.join(output_dir, f"{split}-%06d.tar")

    print(f"\n[{split.upper()}] {total} ảnh → shards (mỗi shard {shard_size} ảnh)")
    print(f"  Pattern: {pattern}")

    start = time.time()
    written = 0
    missing_labels = 0

    with wds.ShardWriter(pattern, maxcount=shard_size) as sink:
        for i, sample in enumerate(samples):
            # Đọc ảnh
            with open(sample["image_path"], "rb") as f:
                image_bytes = f.read()

            # Đọc nhãn (hoặc bytes rỗng nếu không có file nhãn)
            if sample["label_path"]:
                with open(sample["label_path"], "rb") as f:
                    label_bytes = f.read()
            else:
                label_bytes = b""
                missing_labels += 1

            sink.write(
                {
                    "__key__": sample["key"],
                    sample["image_ext"]: image_bytes,
                    "txt": label_bytes,
                }
            )
            written += 1

            # Progress log mỗi 5000 ảnh
            if (i + 1) % 5000 == 0 or (i + 1) == total:
                elapsed = time.time() - start
                speed = written / elapsed
                print(f"  [{i+1}/{total}] {speed:.0f} ảnh/giây", flush=True)

    elapsed = time.time() - start
    num_shards = len(glob.glob(os.path.join(output_dir, f"{split}-*.tar")))

    stats = {
        "split": split,
        "total_images": total,
        "missing_labels": missing_labels,
        "num_shards": num_shards,
        "shard_size": shard_size,
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  ✅ Tạo {num_shards} shards trong {elapsed:.1f}s")
    if missing_labels:
        print(f"  ⚠️  {missing_labels} ảnh không có file nhãn (đã dùng nhãn rỗng)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Tạo WebDataset shards từ YOLO dataset")
    parser.add_argument(
        "--params", default="params.yaml", help="Đường dẫn đến file params.yaml"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Các split cần xử lý",
    )
    args = parser.parse_args()

    # ─── Load params ──────────────────────────────────────────────────────────
    params = load_params(args.params)
    dataset_root = params["dataset"]["root"]
    output_base = params["sharding"]["output_dir"]
    shard_size = params["sharding"]["shard_size"]

    print("=" * 60)
    print("WebDataset Sharding Pipeline")
    print("=" * 60)
    print(f"Dataset root : {dataset_root}")
    print(f"Output dir   : {output_base}")
    print(f"Shard size   : {shard_size} ảnh/shard")
    print(f"Splits       : {args.splits}")

    all_stats = {}
    overall_start = time.time()

    for split in args.splits:
        images_dir = os.path.join(dataset_root, "images", split)
        labels_dir = os.path.join(dataset_root, "labels", split)
        output_dir = os.path.join(output_base, split)

        if not os.path.isdir(images_dir):
            print(f"[SKIP] Không tìm thấy thư mục: {images_dir}")
            continue

        stats = create_split_shards(
            split=split,
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_dir,
            shard_size=shard_size,
        )
        all_stats[split] = stats

    # ─── Lưu metadata shards để DVC và các script khác tham chiếu ────────────
    metadata_path = os.path.join(output_base, "shards_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Metadata đã lưu: {metadata_path}")

    # ─── Tổng kết ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"✅ Hoàn tất trong {total_elapsed:.1f}s")
    for split, s in all_stats.items():
        print(
            f"  {split:5s}: {s['total_images']:6d} ảnh → {s['num_shards']:3d} shards"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
