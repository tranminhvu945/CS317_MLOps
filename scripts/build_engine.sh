#!/bin/bash
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

ONNX="/workspace/apps/vision_service/models/yolov8/yolov8_helmet.onnx"
ENGINE="/workspace/apps/vision_service/models/yolov8/yolov8_helmet.onnx_b1_gpu0_fp16.engine"

# TensorRT trtexec
for TRT_EXE in \
    "/usr/src/tensorrt/bin/trtexec" \
    "/usr/local/tensorrt/bin/trtexec" \
    "$(which trtexec)"; do
    [ -f "$TRT_EXE" ] && break
done

if [ ! -f "$TRT_EXE" ]; then
    echo -e "${RED}[ERROR] TensorRT (trtexec) not found${NC}"
    exit 1
fi

if [ ! -f "$ONNX" ]; then
    echo -e "${RED}[ERROR] ONNX not found: ${ONNX}${NC}"
    echo -e "${YELLOW}[HINT]${NC} Run 'make export-onnx' first to convert best.pt → ONNX"
    exit 1
fi

if [ -f "$ENGINE" ]; then
    echo -e "  Engine exists: $ENGINE (skipping)"
else
    echo -e "${YELLOW}[build_engine]${NC} Compiling ONNX → TensorRT FP16 engine..."
    $TRT_EXE \
        --onnx="$ONNX" \
        --explicitBatch=1 \
        --fp16 \
        --saveEngine="$ENGINE" \
        --workspace=4096 \
        2>&1 | tee "/workspace/apps/vision_service/models/yolov8/build.log"

    if [ ! -f "$ENGINE" ]; then
        echo -e "${RED}[FAIL]${NC} Engine build failed — check build.log"
        exit 1
    fi
    echo -e "${GREEN}[OK]${NC} Engine saved: $ENGINE"
fi

echo -e "${GREEN}[build_engine]${NC} Done."
