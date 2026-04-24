#ifndef __YOLO_H__
#define __YOLO_H__

#include <vector>
#include "nvdsinfer.h"
// CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE macro comes from here
#include "nvdsinfer_custom_impl.h"

#if defined(_WIN32) || defined(__CYGWIN__)
  #define DSY_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
  #define DSY_EXPORT __attribute__((visibility("default")))
#else
  #define DSY_EXPORT
#endif

/**
 * Custom YOLOv8 bounding-box parsing function.
 * Converts raw TensorRT output to NvDsInferObjectDetectionInfo objects.
 *
 * NOTE: Must be exported as a C-linkage symbol so dlopen/dlsym can find it.
 */
extern "C" DSY_EXPORT
bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

#endif /* __YOLO_H__ */
