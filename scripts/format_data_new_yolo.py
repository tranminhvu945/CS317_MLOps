import glob
import shutil
from pathlib import Path

# Paths
DATA_NEW_DIR = Path("dataset/data_new")
OUTPUT_DIR = Path("dataset/extracted/yolo_helmet_dataset_new")

# Chưa sampling, chưa split train/val/test
# Copy toàn bộ data hợp lệ vào split train
SPLIT_NAME = "train"

IMAGE_EXTENSIONS = (
    "*.jpg", "*.jpeg", "*.png",
    "*.JPG", "*.JPEG", "*.PNG"
)


def main():
    # 1. Scan raw images trong các thư mục cam*
    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(str(DATA_NEW_DIR / "cam*" / ext)))

    images = sorted(images)

    print(f"[INFO] Found {len(images)} raw images.")

    # 2. Match image với label YOLO tương ứng
    valid_pairs = []
    missing_labels = []

    for img_path in images:
        img_path = Path(img_path)

        cam_folder = img_path.parent.name
        filename = img_path.stem

        label_path = (
            DATA_NEW_DIR
            / "annotations"
            / "obj_train_data"
            / cam_folder
            / f"{filename}.txt"
        )

        if label_path.exists():
            valid_pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path)

    print(f"[INFO] Matched pairs: {len(valid_pairs)} image-label pairs.")
    print(f"[INFO] Missing labels: {len(missing_labels)} images.")

    if not valid_pairs:
        print("[ERROR] No valid image-label pairs found.")
        return

    # 3. Clear output directory
    if OUTPUT_DIR.exists():
        print(f"[INFO] Cleaning existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # 4. Tạo YOLO folder structure
    img_dest_dir = OUTPUT_DIR / "images" / SPLIT_NAME
    lbl_dest_dir = OUTPUT_DIR / "labels" / SPLIT_NAME

    img_dest_dir.mkdir(parents=True, exist_ok=True)
    lbl_dest_dir.mkdir(parents=True, exist_ok=True)

    # 5. Copy toàn bộ ảnh-label, không sampling
    print(f"[INFO] Copying all valid pairs to YOLO format split: {SPLIT_NAME}")

    for img_path, lbl_path in valid_pairs:
        cam_prefix = img_path.parent.name

        # Đổi tên để tránh trùng file giữa các camera
        new_stem = f"{cam_prefix}_{img_path.stem}"

        new_img_name = f"{new_stem}{img_path.suffix.lower()}"
        new_lbl_name = f"{new_stem}.txt"

        shutil.copy2(img_path, img_dest_dir / new_img_name)
        shutil.copy2(lbl_path, lbl_dest_dir / new_lbl_name)

    print("\n[OK] YOLO formatted dataset saved at:")
    print(f"  {OUTPUT_DIR}")

    print("\nDataset Summary:")
    print(f"  Images: {len(valid_pairs)}")
    print(f"  Labels: {len(valid_pairs)}")
    print(f"  Split : {SPLIT_NAME}")

    print("\nOutput layout:")
    print(f"  {img_dest_dir}")
    print(f"  {lbl_dest_dir}")


if __name__ == "__main__":
    main()