#!/usr/bin/env python3
"""
evaluate_model.py
─────────────────
Quality Gate: So sánh Candidate model mới với Production model hiện tại
trên tập Gold Standard cố định (300 ảnh, không thuộc train/val).

Thiết kế:
  - KHÔNG bao giờ exit 1 (trừ lỗi kỹ thuật thực sự).
  - Kết quả PASS/FAIL được ghi vào evaluate_status.json để pipeline tiếp tục.
  - Stage export đọc file này để quyết định có deploy hay không.

Criteria Quality Gate (theo thứ tự ưu tiên):
  1. recall_no_helmet_candidate >= recall_no_helmet_production        (bắt buộc)
  2. map50_candidate >= map50_production - MAP50_TOLERANCE            (bắt buộc)
  3. precision_no_helmet_candidate >= precision_no_helmet_production - PREC_TOLERANCE
"""

import json
import os
import sys
import tempfile
import yaml
from pathlib import Path

# ─── Ngưỡng Quality Gate ──────────────────────────────────────────────────────
MAP50_TOLERANCE  = 0.01   # Cho phép mAP50 giảm tối đa 1% so với Production
PREC_TOLERANCE   = 0.02   # Cho phép precision(no_helmet) giảm tối đa 2%


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_params(path: str = "params.yaml") -> dict:
    params_path = Path(path)
    if not params_path.exists():
        alt = Path(__file__).parent.parent / "params.yaml"
        if alt.exists():
            params_path = alt
        else:
            print(f"[ERROR] Không tìm thấy params.yaml tại: {path}")
            sys.exit(1)
    with open(params_path) as f:
        return yaml.safe_load(f)


def extract_class_metrics(metrics, class_names: list) -> dict:
    """
    Trích xuất metrics theo từng class từ kết quả YOLO val.
    Trả về dict: { class_name: { "precision", "recall", "map50", "fn_count" } }
    fn_count = số lượng False Negative (bỏ sót) = GT_count * (1 - recall)
    """
    result = {}
    try:
        ap_class = metrics.ap_class_index        # list of class indices evaluated
        names    = metrics.names                  # {idx: name}
        prec     = metrics.box.p                  # per-class precision
        rec      = metrics.box.r                  # per-class recall
        ap50     = metrics.box.ap50               # per-class AP50

        # Lấy số lượng GT instances mỗi class (nt = number of targets)
        nt_per_class = getattr(metrics.box, "nt_per_class", None)
        if nt_per_class is None:
            # Fallback: thử lấy từ confusion matrix hoặc bỏ
            nt_per_class = {}

        for i, cls_idx in enumerate(ap_class):
            cls_name = names.get(int(cls_idx), f"class_{cls_idx}")
            recall_i = float(rec[i])

            # Tính FN: nếu biết GT count thì FN = GT * (1 - recall)
            if hasattr(nt_per_class, '__getitem__'):
                try:
                    gt_count = int(nt_per_class[int(cls_idx)])
                    fn_count = int(round(gt_count * (1 - recall_i)))
                except Exception:
                    gt_count = None
                    fn_count = None
            else:
                gt_count = None
                fn_count = None

            result[cls_name] = {
                "precision": float(prec[i]),
                "recall":    recall_i,
                "map50":     float(ap50[i]),
                "gt_count":  gt_count,
                "fn_count":  fn_count,
            }
    except Exception as e:
        print(f"[WARN] Không trích xuất được per-class metrics: {e}")
    return result


def run_evaluation(model_path: str, gold_yaml: str, device) -> dict:
    """Chạy model.val() và trả về dict metrics đầy đủ."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    metrics = model.val(data=gold_yaml, split="test", device=device, verbose=False)
    class_metrics = extract_class_metrics(metrics, ["helmet", "no_helmet"])
    return {
        "map50":       float(metrics.box.map50),
        "map50_95":    float(metrics.box.map),
        "per_class":   class_metrics,
    }


def write_status(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Ghi evaluate_status.json: {path}")


def check_quality_gate(cand: dict, prod: dict) -> tuple:
    """
    So sánh Candidate vs Production theo multi-metric criteria.
    Trả về (passed: bool, failures: list[str], notes: list[str])

    Rules (theo thứ tự ưu tiên):
      1. recall_no_helmet_candidate >= recall_no_helmet_production          (bắt buộc)
      2. mAP50_candidate >= mAP50_production - MAP50_TOLERANCE              (bắt buộc)
      3. precision_no_helmet_candidate >= precision_no_helmet_production - PREC_TOLERANCE
      4. fn_no_helmet_candidate <= fn_no_helmet_production  (không tăng FN) (bắt buộc)
    """
    failures = []
    notes    = []

    cand_map      = cand["map50"]
    prod_map      = prod["map50"]
    cand_nh       = cand["per_class"].get("no_helmet", {})
    prod_nh       = prod["per_class"].get("no_helmet", {})

    cand_recall_nh = cand_nh.get("recall",    0.0)
    prod_recall_nh = prod_nh.get("recall",    0.0)
    cand_prec_nh   = cand_nh.get("precision", 0.0)
    prod_prec_nh   = prod_nh.get("precision", 0.0)
    cand_fn_nh     = cand_nh.get("fn_count",  None)
    prod_fn_nh     = prod_nh.get("fn_count",  None)

    # Rule 1: recall(no_helmet) không được thấp hơn Production (bắt buộc)
    if cand_recall_nh < prod_recall_nh:
        failures.append(
            f"recall_no_helmet: {cand_recall_nh:.4f} < production {prod_recall_nh:.4f}"
        )
    else:
        notes.append(f"recall_no_helmet OK: {cand_recall_nh:.4f} >= {prod_recall_nh:.4f}")

    # Rule 2: mAP50 không giảm quá MAP50_TOLERANCE (bắt buộc)
    if cand_map < prod_map - MAP50_TOLERANCE:
        failures.append(
            f"mAP50: {cand_map:.4f} < production {prod_map:.4f} - tolerance {MAP50_TOLERANCE}"
        )
    else:
        notes.append(f"mAP50 OK: {cand_map:.4f} >= {prod_map:.4f} - {MAP50_TOLERANCE}")

    # Rule 3: precision(no_helmet) không giảm quá PREC_TOLERANCE
    if cand_prec_nh < prod_prec_nh - PREC_TOLERANCE:
        failures.append(
            f"precision_no_helmet: {cand_prec_nh:.4f} < production {prod_prec_nh:.4f} - tolerance {PREC_TOLERANCE}"
        )
    else:
        notes.append(f"precision_no_helmet OK: {cand_prec_nh:.4f} >= {prod_prec_nh:.4f} - {PREC_TOLERANCE}")

    # Rule 4: FN(no_helmet) không được tăng (bắt buộc nếu có dữ liệu)
    if cand_fn_nh is not None and prod_fn_nh is not None:
        if cand_fn_nh > prod_fn_nh:
            failures.append(
                f"fn_no_helmet: {cand_fn_nh} > production {prod_fn_nh} "
                f"(bỏ sót thêm {cand_fn_nh - prod_fn_nh} trường hợp không đội mũ)"
            )
        else:
            notes.append(
                f"fn_no_helmet OK: {cand_fn_nh} <= production {prod_fn_nh} "
                f"(đảm bảo không bỏ sót thêm)"
            )
    else:
        notes.append("fn_no_helmet: không có dữ liệu GT count, bỏ qua Rule 4")

    return len(failures) == 0, failures, notes


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    params      = load_params("params.yaml")
    train_cfg   = params.get("train", {})
    mlflow_cfg  = params.get("mlflow", {})

    tracking_uri  = mlflow_cfg.get("tracking_uri", "http://localhost:5001")
    registry_name = mlflow_cfg.get("registry_name", "YOLOv8_Helmet_Model")

    gold_yaml = "dataset/gold_standard.yaml"
    if not os.path.exists(gold_yaml):
        print(f"[ERROR] Không tìm thấy Gold Standard dataset: {gold_yaml}")
        sys.exit(1)

    device = train_cfg.get("device", [0])
    if isinstance(device, list) and len(device) > 0:
        device = device[0]

    best_pt_path = os.path.join(
        "runs", "detect", train_cfg["project"], train_cfg["name"], "weights", "best.pt"
    )
    eval_status_path = os.path.join(
        "runs", "detect", train_cfg["project"], train_cfg["name"], "evaluate_status.json"
    )

    if not os.path.exists(best_pt_path):
        print(f"[ERROR] Không tìm thấy Candidate model: {best_pt_path}")
        sys.exit(1)

    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError:
        print("[ERROR] Thiếu thư viện mlflow. Chạy: pip install mlflow")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    print("=" * 60)
    print("Model Quality Gate")
    print("=" * 60)
    print(f"Gold Standard : {gold_yaml}")
    print(f"Ngưỡng mAP50  : tolerance = {MAP50_TOLERANCE}")
    print(f"Ngưỡng prec   : tolerance = {PREC_TOLERANCE}")
    print("-" * 60)

    # ── 1. Đánh giá Candidate ─────────────────────────────────────────────────
    print(f"\n[1/3] Đánh giá Candidate: {best_pt_path}")
    cand_result = run_evaluation(best_pt_path, gold_yaml, device)
    cand_nh = cand_result["per_class"].get("no_helmet", {})
    print(f"  mAP50          : {cand_result['map50']:.4f}")
    print(f"  recall_no_helmet    : {cand_nh.get('recall', 0):.4f}")
    print(f"  precision_no_helmet : {cand_nh.get('precision', 0):.4f}")

    # ── 2. Đánh giá Production (nếu có) ──────────────────────────────────────
    prod_result  = {"map50": 0.0, "map50_95": 0.0, "per_class": {}}
    prod_version = None
    has_prod     = False

    print(f"\n[2/3] Đánh giá Production model (alias 'Production')...")
    try:
        prod_info = client.get_model_version_by_alias(name=registry_name, alias="Production")
        has_prod     = True
        prod_version = prod_info.version
        source       = prod_info.source
        print(f"  Tìm thấy Production: version {prod_version}")

        tmp_dir    = tempfile.mkdtemp(prefix="mlflow_prod_")
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=source, dst_path=tmp_dir)
        pt_files   = (
            list(Path(local_path).parent.glob("**/*.pt"))
            if Path(local_path).is_file()
            else list(Path(local_path).glob("**/*.pt"))
        )
        if Path(local_path).is_file() and local_path.endswith(".pt"):
            prod_pt = local_path
        elif pt_files:
            prod_pt = str(pt_files[0])
        else:
            raise FileNotFoundError(f"Không tìm thấy .pt trong artifact: {local_path}")

        prod_result = run_evaluation(prod_pt, gold_yaml, device)
        prod_nh = prod_result["per_class"].get("no_helmet", {})
        print(f"  mAP50          : {prod_result['map50']:.4f}")
        print(f"  recall_no_helmet    : {prod_nh.get('recall', 0):.4f}")
        print(f"  precision_no_helmet : {prod_nh.get('precision', 0):.4f}")

    except Exception as e:
        print(f"  [INFO] Không có Production model: {e}")
        print("  → Candidate sẽ được chấp nhận làm Production đầu tiên (baseline).")

    # ── 3. Quality Gate ───────────────────────────────────────────────────────
    print(f"\n[3/3] So sánh Quality Gate...")

    if not has_prod:
        # Không có production → chấp nhận candidate làm baseline
        passed   = True
        failures = []
        reason   = "first_model_baseline"
    else:
        passed, failures, notes = check_quality_gate(cand_result, prod_result)
        reason = "candidate_passed_quality_gate" if passed else "candidate_metric_lower_than_production"
        for note in notes:
            print(f"  ✅ {note}")
        for fail in failures:
            print(f"  ❌ {fail}")

    print("-" * 60)

    cand_nh = cand_result["per_class"].get("no_helmet", {})
    prod_nh = prod_result["per_class"].get("no_helmet", {})

    status_payload = {
        "status":   "accepted" if passed else "rejected",
        "promote":  passed,
        "reason":   reason,
        "failures": failures if not passed else [],
        # Flat fields để tiện đọc nhanh trong pipeline / dashboard
        "candidate_map50":              cand_result["map50"],
        "production_map50":             prod_result["map50"],
        "candidate_no_helmet_recall":   cand_nh.get("recall",    0.0),
        "production_no_helmet_recall":  prod_nh.get("recall",    0.0),
        "candidate_no_helmet_precision":  cand_nh.get("precision", 0.0),
        "production_no_helmet_precision": prod_nh.get("precision", 0.0),
        "candidate_no_helmet_fn":       cand_nh.get("fn_count", None),
        "production_no_helmet_fn":      prod_nh.get("fn_count", None),
        "candidate": {
            "map50":                cand_result["map50"],
            "map50_95":             cand_result["map50_95"],
            "recall_no_helmet":     cand_nh.get("recall",    0.0),
            "precision_no_helmet":  cand_nh.get("precision", 0.0),
            "map50_no_helmet":      cand_nh.get("map50",     0.0),
        },
        "production": {
            "version":              prod_version,
            "map50":                prod_result["map50"],
            "map50_95":             prod_result["map50_95"],
            "recall_no_helmet":     prod_nh.get("recall",    0.0),
            "precision_no_helmet":  prod_nh.get("precision", 0.0),
            "map50_no_helmet":      prod_nh.get("map50",     0.0),
        },
        "thresholds": {
            "map50_tolerance":      MAP50_TOLERANCE,
            "precision_tolerance":  PREC_TOLERANCE,
        },
    }

    if passed:
        print("[PASS] ✅ Candidate đạt Quality Gate — tiến hành promote lên Production!")

        # Promote Candidate → Production
        try:
            cand_info    = client.get_model_version_by_alias(name=registry_name, alias="Candidate")
            cand_version = cand_info.version
            client.set_registered_model_alias(name=registry_name, alias="Production", version=cand_version)
            print(f"[OK]   Version {cand_version} → alias 'Production'")
            status_payload["promoted_version"] = cand_version
            status_payload["production"]["version"] = cand_version
        except Exception as e:
            print(f"[WARN] Không promote được trên Registry: {e}")
            status_payload["status"] = "rejected"
            status_payload["promote"] = False
            status_payload["reason"] = "promotion_failed_after_quality_gate"
            status_payload["failures"] = [f"promotion_failed: {e}"]
            status_payload["promotion_error"] = str(e)
    else:
        print("[FAIL] ❌ Candidate bị reject — giữ nguyên Production hiện tại.")
        print("       Pipeline tiếp tục với status='rejected'.")

    write_status(eval_status_path, status_payload)
    print("\n" + "=" * 60)
    print(f"  Status  : {status_payload['status'].upper()}")
    print(f"  Promote : {status_payload['promote']}")
    print(f"  Reason  : {status_payload['reason']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
