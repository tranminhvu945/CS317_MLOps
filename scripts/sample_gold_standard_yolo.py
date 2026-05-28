import random
import shutil
import csv
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = Path(
    "/mmlab_students/storageStudents/nguyenvd/uit_medseg/CS317_MLOps/dataset/extracted/yolo_helmet_dataset_new"
)

OUT_DIR = Path(
    "/mmlab_students/storageStudents/nguyenvd/uit_medseg/CS317_MLOps/dataset/gold_standard"
)

TOTAL_SAMPLES = 300
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_images_labels_dirs(dataset_root: Path):
    """
    Tìm thư mục images và labels trong YOLO dataset chưa chia split.

    Kỳ vọng:
        dataset_root/images
        dataset_root/labels
    """

    img_dir = dataset_root / "images"
    lbl_dir = dataset_root / "labels"

    if img_dir.exists() and lbl_dir.exists():
        return img_dir, lbl_dir

    raise FileNotFoundError(
        f"Không tìm thấy thư mục images/labels.\n"
        f"Đã thử:\n"
        f"  {img_dir}\n"
        f"  {lbl_dir}"
    )


def list_images(img_dir: Path):
    return sorted([
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def get_label_path(image_path: Path, img_dir: Path, lbl_dir: Path):
    """
    Tìm label tương ứng với ảnh.

    Ví dụ:
        images/abc.jpg -> labels/abc.txt

    Nếu ảnh nằm trong subfolder:
        images/cam1/abc.jpg -> labels/cam1/abc.txt
    """

    rel = image_path.relative_to(img_dir)
    return (lbl_dir / rel).with_suffix(".txt")


def validate_yolo_label(label_path: Path):
    """
    Kiểm tra label YOLO đơn giản:
        class_id x_center y_center width height
    """

    if not label_path.exists():
        return False, "missing_label"

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()

    # Label rỗng vẫn có thể hợp lệ nếu ảnh không có object
    if text == "":
        return True, "empty_label"

    for line in text.splitlines():
        parts = line.strip().split()

        if len(parts) < 5:
            return False, f"invalid_columns: {line}"

        try:
            _class_id = int(float(parts[0]))
            values = [float(x) for x in parts[1:5]]
        except ValueError:
            return False, f"invalid_number: {line}"

        for v in values:
            if v < 0 or v > 1:
                return False, f"value_out_of_range: {line}"

    return True, "valid"


def safe_output_name(image_path: Path, img_dir: Path):
    """
    Tránh trùng tên nếu ảnh nằm trong nhiều thư mục con.
    Ví dụ:
        images/cam1/001.jpg -> cam1_001.jpg
    """

    rel = image_path.relative_to(img_dir)
    parts = rel.with_suffix("").parts
    new_stem = "_".join(parts)

    return f"{new_stem}{image_path.suffix.lower()}"


# =========================================================
# MAIN
# =========================================================

def main():
    random.seed(SEED)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT không tồn tại: {DATASET_ROOT}")

    img_dir, lbl_dir = find_images_labels_dirs(DATASET_ROOT)

    print(f"Image dir: {img_dir}")
    print(f"Label dir: {lbl_dir}")

    all_images = list_images(img_dir)

    valid_pairs = []
    invalid_pairs = []

    for img_path in all_images:
        label_path = get_label_path(img_path, img_dir, lbl_dir)
        is_valid, status = validate_yolo_label(label_path)

        if is_valid:
            valid_pairs.append((img_path, label_path, status))
        else:
            invalid_pairs.append((img_path, label_path, status))

    print(f"Total images found: {len(all_images)}")
    print(f"Valid YOLO pairs: {len(valid_pairs)}")
    print(f"Invalid/missing label pairs: {len(invalid_pairs)}")

    if len(valid_pairs) < TOTAL_SAMPLES:
        raise ValueError(
            f"Dataset chỉ có {len(valid_pairs)} cặp ảnh-label hợp lệ, "
            f"nhưng bạn yêu cầu {TOTAL_SAMPLES}."
        )

    selected_pairs = random.sample(valid_pairs, TOTAL_SAMPLES)

    out_images_dir = OUT_DIR / "images"
    out_labels_dir = OUT_DIR / "labels"

    if OUT_DIR.exists():
        print(f"[INFO] Cleaning old gold standard dir: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for img_path, label_path, label_status in selected_pairs:
        out_img_name = safe_output_name(img_path, img_dir)
        out_lbl_name = Path(out_img_name).with_suffix(".txt").name

        out_img_path = out_images_dir / out_img_name
        out_lbl_path = out_labels_dir / out_lbl_name

        # MOVE thay vì COPY
        shutil.move(str(img_path), str(out_img_path))
        shutil.move(str(label_path), str(out_lbl_path))

        manifest_rows.append({
            "gold_image_path": str(out_img_path),
            "gold_label_path": str(out_lbl_path),
            "source_image_path": str(img_path),
            "source_label_path": str(label_path),
            "label_status": label_status,
        })

    manifest_path = OUT_DIR / "manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gold_image_path",
                "gold_label_path",
                "source_image_path",
                "source_label_path",
                "label_status",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nDone.")
    print(f"Gold Standard Dataset saved to: {OUT_DIR}")
    print(f"Images: {out_images_dir}")
    print(f"Labels: {out_labels_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total moved samples: {len(selected_pairs)}")


if __name__ == "__main__":
    main()