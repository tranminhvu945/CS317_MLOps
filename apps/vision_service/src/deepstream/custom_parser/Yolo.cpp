#include "Yolo.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

// ============================================================
// Tùy chọn
// ============================================================
// 1 = bbox là [cx, cy, w, h]
// 0 = bbox là [x1, y1, x2, y2]
#define YOLO_BBOX_XYWH 1

// 1 = class score là logits, cần sigmoid
// 0 = class score đã là probability [0,1]
#define YOLO_APPLY_SIGMOID 0

// NMS trong parser (config pgie_yolov8_helmet.txt dùng nms-iou-threshold=0.55)
#define YOLO_NMS_IOU 0.55f
#define YOLO_PRE_NMS_TOPK 300
#define YOLO_MAX_OUTPUT 256

static inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ============================================================
// NMS per class
// ============================================================
static void nmsPerClass(std::vector<NvDsInferObjectDetectionInfo>& proposals,
                        std::vector<NvDsInferObjectDetectionInfo>& output) {
    std::sort(proposals.begin(), proposals.end(),
              [](const auto& a, const auto& b) {
                  return a.detectionConfidence > b.detectionConfidence;
              });

    std::vector<bool> suppressed(proposals.size(), false);
    int kept = 0;

    for (size_t i = 0; i < proposals.size() && kept < YOLO_MAX_OUTPUT; ++i) {
        if (suppressed[i]) continue;

        output.push_back(proposals[i]);
        ++kept;

        const float ax1 = proposals[i].left;
        const float ay1 = proposals[i].top;
        const float ax2 = proposals[i].left + proposals[i].width;
        const float ay2 = proposals[i].top + proposals[i].height;

        for (size_t j = i + 1; j < proposals.size(); ++j) {
            if (suppressed[j]) continue;

            const float bx1 = proposals[j].left;
            const float by1 = proposals[j].top;
            const float bx2 = proposals[j].left + proposals[j].width;
            const float by2 = proposals[j].top + proposals[j].height;

            const float ix1 = std::max(ax1, bx1);
            const float iy1 = std::max(ay1, by1);
            const float ix2 = std::min(ax2, bx2);
            const float iy2 = std::min(ay2, by2);

            const float iw = std::max(0.0f, ix2 - ix1);
            const float ih = std::max(0.0f, iy2 - iy1);
            const float inter = iw * ih;

            const float areaA = proposals[i].width * proposals[i].height;
            const float areaB = proposals[j].width * proposals[j].height;
            const float uni = areaA + areaB - inter;

            if (uni > 0.0f && (inter / uni) > YOLO_NMS_IOU) {
                suppressed[j] = true;
            }
        }
    }
}

// ============================================================
// NvDsInferParseYolo — YOLOv8 custom bbox parser
//
// Supported ONNX output: [1, 6, 8400] → [6, 8400] after batch squeeze
//   Layout: [4 bbox + 2 class, 8400 proposals]
//   BBox format: [cx, cy, w, h] (YOLOv8 standard)
//
// Config:
//   parse-bbox-func-name=NvDsInferParseYolo
//   custom-lib-path=.../libnvdsinfer_custom_impl_Yolo.so
// ============================================================
bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {

    if (outputLayersInfo.empty()) {
        std::fprintf(stderr, "[YOLO] ERROR: outputLayersInfo is empty\n");
        return false;
    }

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* data = static_cast<const float*>(layer.buffer);
    if (!data) {
        std::fprintf(stderr, "[YOLO] ERROR: output buffer is null\n");
        return false;
    }

    const auto& dims = layer.inferDims;
    if (dims.numDims != 2) {
        std::fprintf(stderr,
            "[YOLO] ERROR: expected 2 dims after batch squeeze, got numDims=%u\n",
            dims.numDims);
        return false;
    }

    const unsigned int d0 = dims.d[0];
    const unsigned int d1 = dims.d[1];

    unsigned int C = 0;  // channels = 4 + num_classes
    unsigned int N = 0;  // num proposals
    bool channelFirst = false;

    // Ưu tiên case DeepStream thường thấy: [C, N] = [6, 8400]
    if (d0 >= 6 && d1 > d0) {
        C = d0;
        N = d1;
        channelFirst = true;
    }
    // Fallback: [N, C]
    else if (d1 >= 6 && d0 > d1) {
        C = d1;
        N = d0;
        channelFirst = false;
    } else {
        std::fprintf(stderr,
            "[YOLO] ERROR: unsupported dims [%u, %u]. Expected [C,N] or [N,C], with C >= 6\n",
            d0, d1);
        return false;
    }

    unsigned int numClasses = C - 4;

    if (detectionParams.numClassesConfigured > 0 &&
        detectionParams.numClassesConfigured != numClasses) {
        std::fprintf(stderr,
            "[YOLO] WARN: tensor implies numClasses=%u but config says numClassesConfigured=%u. "
            "Will use tensor value.\n",
            numClasses, detectionParams.numClassesConfigured);
    }

    std::vector<float> perClassThreshold(numClasses, 0.25f);
    if (!detectionParams.perClassPreclusterThreshold.empty()) {
        for (unsigned int c = 0; c < numClasses; ++c) {
            if (c < detectionParams.perClassPreclusterThreshold.size()) {
                perClassThreshold[c] = detectionParams.perClassPreclusterThreshold[c];
            } else {
                perClassThreshold[c] = detectionParams.perClassPreclusterThreshold[0];
            }
        }
    }

    std::vector<std::vector<NvDsInferObjectDetectionInfo>> classBoxes(numClasses);

    for (unsigned int p = 0; p < N; ++p) {
        float v0, v1, v2, v3;

        if (channelFirst) {
            v0 = data[0 * N + p];
            v1 = data[1 * N + p];
            v2 = data[2 * N + p];
            v3 = data[3 * N + p];
        } else {
            v0 = data[p * C + 0];
            v1 = data[p * C + 1];
            v2 = data[p * C + 2];
            v3 = data[p * C + 3];
        }

        int bestCls = -1;
        float bestScore = -1.0f;

        for (unsigned int c = 0; c < numClasses; ++c) {
            float score = channelFirst ? data[(4 + c) * N + p] : data[p * C + (4 + c)];

#if YOLO_APPLY_SIGMOID
            score = sigmoidf(score);
#endif

            if (score > bestScore) {
                bestScore = score;
                bestCls = static_cast<int>(c);
            }
        }

        if (bestCls < 0) continue;
        if (bestScore < perClassThreshold[bestCls]) continue;

        float left, top, width, height;

#if YOLO_BBOX_XYWH
        // tensor = [cx, cy, w, h]
        const float cx = v0;
        const float cy = v1;
        const float w  = v2;
        const float h  = v3;

        left   = cx - w * 0.5f;
        top    = cy - h * 0.5f;
        width  = w;
        height = h;
#else
        // tensor = [x1, y1, x2, y2]
        const float x1 = v0;
        const float y1 = v1;
        const float x2 = v2;
        const float y2 = v3;

        left   = x1;
        top    = y1;
        width  = x2 - x1;
        height = y2 - y1;
#endif

        left   = clampf(left, 0.0f, static_cast<float>(networkInfo.width  - 1));
        top    = clampf(top,  0.0f, static_cast<float>(networkInfo.height - 1));
        width  = clampf(width,  1.0f, static_cast<float>(networkInfo.width  - left));
        height = clampf(height, 1.0f, static_cast<float>(networkInfo.height - top));

        if (width < 1.0f || height < 1.0f) continue;

        NvDsInferObjectDetectionInfo obj{};
        obj.classId = bestCls;
        obj.detectionConfidence = bestScore;
        obj.left = left;
        obj.top = top;
        obj.width = width;
        obj.height = height;

        classBoxes[bestCls].push_back(obj);
    }

    // Pre-NMS top-k per class
    for (unsigned int c = 0; c < numClasses; ++c) {
        auto& boxes = classBoxes[c];
        if (boxes.size() > YOLO_PRE_NMS_TOPK) {
            std::partial_sort(
                boxes.begin(),
                boxes.begin() + YOLO_PRE_NMS_TOPK,
                boxes.end(),
                [](const auto& a, const auto& b) {
                    return a.detectionConfidence > b.detectionConfidence;
                });
            boxes.resize(YOLO_PRE_NMS_TOPK);
        }
    }

    // NMS per class
    objectList.clear();
    for (unsigned int c = 0; c < numClasses; ++c) {
        if (classBoxes[c].empty()) continue;
        std::vector<NvDsInferObjectDetectionInfo> classOutput;
        nmsPerClass(classBoxes[c], classOutput);
        objectList.insert(objectList.end(), classOutput.begin(), classOutput.end());
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);