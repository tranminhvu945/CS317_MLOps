import argparse
import glob
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = (
    "*.jpg", "*.jpeg", "*.png",
    "*.JPG", "*.JPEG", "*.PNG"
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Format raw YOLO data mới thành cấu trúc images/labels cho bước split."
    )
    parser.add_argument(
        "--raw-dir",
        default="dataset/data_new",
        help="Thư mục data mới (chứa cam*/ và annotations/obj_train_data/).",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/extracted/yolo_helmet_dataset_new",
        help="Thư mục output dataset YOLO tạm trước khi split/shard.",
    )
    parser.add_argument(
        "--split-name",
        default="train",
        help="Split đích ban đầu trước khi chạy split_yolo_dataset.py (mặc định: train).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_new_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    split_name = args.split_name

    # 1. Scan raw images trong các thư mục cam*
    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(str(data_new_dir / "cam*" / ext)))

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
            data_new_dir
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

    # 3. Clear output directory
    if output_dir.exists():
        print(f"[INFO] Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    if not valid_pairs:
        print("[INFO] Không có ảnh/nhãn mới nào hợp lệ để xử lý.")
        return

    # 4. Tạo YOLO folder structure
    img_dest_dir = output_dir / "images" / split_name
    lbl_dest_dir = output_dir / "labels" / split_name

    img_dest_dir.mkdir(parents=True, exist_ok=True)
    lbl_dest_dir.mkdir(parents=True, exist_ok=True)

    # 5. Copy toàn bộ ảnh-label, không sampling
    print(f"[INFO] Copying all valid pairs to YOLO format split: {split_name}")

    for img_path, lbl_path in valid_pairs:
        cam_prefix = img_path.parent.name

        # Đổi tên để tránh trùng file giữa các camera
        new_stem = f"{cam_prefix}_{img_path.stem}"

        new_img_name = f"{new_stem}{img_path.suffix.lower()}"
        new_lbl_name = f"{new_stem}.txt"

        shutil.copy2(img_path, img_dest_dir / new_img_name)
        shutil.copy2(lbl_path, lbl_dest_dir / new_lbl_name)

    print("\n[OK] YOLO formatted dataset saved at:")
    print(f"  {output_dir}")

    print("\nDataset Summary:")
    print(f"  Images: {len(valid_pairs)}")
    print(f"  Labels: {len(valid_pairs)}")
    print(f"  Split : {split_name}")

    print("\nOutput layout:")
    print(f"  {img_dest_dir}")
    print(f"  {lbl_dest_dir}")


if __name__ == "__main__":
    main()
