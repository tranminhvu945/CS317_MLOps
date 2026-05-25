#!/usr/bin/env bash
set -euo pipefail

COUNT="${1:-}"
if [[ -z "${COUNT}" ]]; then
  echo "Usage: $0 <count>" >&2
  echo "Example: $0 1  # enable cam01 only" >&2
  exit 1
fi

if ! [[ "${COUNT}" =~ ^[0-9]+$ ]]; then
  echo "Invalid count: ${COUNT}" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAM_DIR="${ROOT_DIR}/apps/vision_service/configs/camera"

# Camera order for scale tests.
CAM_FILES=(
  "${CAM_DIR}/camera.yaml"
  "${CAM_DIR}/camera_002.yaml"
  "${CAM_DIR}/camera_003.yaml"
  "${CAM_DIR}/camera_004.yaml"
)

TOTAL=${#CAM_FILES[@]}
if (( COUNT < 1 || COUNT > TOTAL )); then
  echo "count must be in [1, ${TOTAL}]" >&2
  exit 1
fi

idx=0
for file in "${CAM_FILES[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Missing camera config: ${file}" >&2
    exit 1
  fi

  idx=$((idx + 1))
  if (( idx <= COUNT )); then
    sed -i -E '0,/^enabled:[[:space:]]*(true|false)[[:space:]]*$/s//enabled: true/' "${file}"
  else
    sed -i -E '0,/^enabled:[[:space:]]*(true|false)[[:space:]]*$/s//enabled: false/' "${file}"
  fi

done

echo "[camera-scale] Enabled ${COUNT}/${TOTAL} camera configs"
for file in "${CAM_FILES[@]}"; do
  cid=$(grep -E '^camera_id:' "${file}" | head -1 | awk '{print $2}')
  ena=$(grep -E '^enabled:' "${file}" | head -1 | awk '{print $2}')
  uri=$(grep -E '^  uri:' "${file}" | head -1 | sed -E 's/^[[:space:]]*uri:[[:space:]]*//')
  echo "  - ${cid}: enabled=${ena} uri=${uri}"
done
