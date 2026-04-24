// Debug TensorRT engine output memory layout
// Compile: nvcc -o debug_trt debug_trt_engine.cu -lnvinfer
// Run: ./debug_trt <engine_path>

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

using namespace nvinfer1;

static Logger gLogger;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <engine_path>" << std::endl;
        return 1;
    }
    std::string enginePath = argv[1];

    // Load engine
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Cannot open engine: " << enginePath << std::endl;
        return 1;
    }
    file.seekg(0, std::ifstream::end);
    size_t fsize = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> engineData(fsize);
    file.read(engineData.data(), fsize);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fsize);

    std::cout << "\n=== ENGINE BINDINGS ===" << std::endl;
    std::cout << "Num bindings: " << engine->getNbBindings() << std::endl;
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        const char* name = engine->getBindingName(i);
        Dims dims = engine->getBindingDimensions(i);
        DataType dtype = engine->getBindingDataType(i);
        bool input = engine->bindingIsInput(i);
        std::cout << "  [" << i << "] " << (input ? "INPUT" : "OUTPUT")
                  << " name=" << name
                  << " dtype=" << (int)dtype
                  << " dims: ";
        for (int d = 0; d < dims.nbDims; ++d)
            std::cout << dims.d[d] << (d < dims.nbDims-1 ? "x" : "");
        std::cout << " (nbDims=" << dims.nbDims << ")"
                  << " numElements=" << (dims.nbDims > 0 ? 1 : 0);
        for (int d = 0; d < dims.nbDims; ++d)
            std::cout << "*" << dims.d[d];
        std::cout << std::endl;
    }

    // Find output binding
    int outIdx = -1;
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        if (!engine->bindingIsInput(i)) { outIdx = i; break; }
    }
    if (outIdx < 0) { std::cerr << "No output binding found\n"; return 1; }

    Dims outDims = engine->getBindingDimensions(outIdx);
    size_t numElements = 1;
    for (int d = 0; d < outDims.nbDims; ++d)
        numElements *= outDims.d[d];

    std::cout << "\n=== OUTPUT TENSOR ===" << std::endl;
    std::cout << "Binding index: " << outIdx << std::endl;
    std::cout << "Dims: ";
    for (int d = 0; d < outDims.nbDims; ++d)
        std::cout << outDims.d[d] << (d < outDims.nbDims-1 ? ", " : "");
    std::cout << std::endl;
    std::cout << "nbDims: " << outDims.nbDims << std::endl;
    std::cout << "numElements: " << numElements << std::endl;

    // Run inference with dummy input
    IExecutionContext* context = engine->createExecutionContext();

    float* d_input; float* d_output;
    CHECK(cudaMalloc(&d_input, 1*3*640*640 * sizeof(float)));
    CHECK(cudaMalloc(&d_output, numElements * sizeof(float)));
    CHECK(cudaMemset(d_input, 0, 1*3*640*640 * sizeof(float)));

    void* bindings[] = { d_input, d_output };
    bool ok = context->executeV2(bindings);
    if (!ok) { std::cerr << "Inference failed\n"; return 1; }

    std::vector<float> h_output(numElements);
    CHECK(cudaMemcpy(h_output.data(), d_output,
                     numElements * sizeof(float), cudaMemcpyDeviceToHost));

    // Dump key values
    unsigned int C = outDims.nbDims >= 1 ? outDims.d[0] : 1;
    unsigned int N = outDims.nbDims >= 2 ? outDims.d[1] :
                     outDims.nbDims >= 1 ? outDims.d[0] : numElements;
    if (outDims.nbDims == 1) N = outDims.d[0];

    std::cout << "\n=== MEMORY DUMP (first 10 floats) ===" << std::endl;
    for (unsigned int i = 0; i < std::min(10u, (unsigned int)numElements); ++i)
        std::cout << "  flat[" << i << "] = " << h_output[i] << std::endl;

    std::cout << "\n=== HYPOTHESIS TEST ===" << std::endl;
    // Hypothesis 1: [C,N]=[6,8400], flat[c*N+p]
    //   cx_p0 = flat[0]
    //   cy_p0 = flat[N]
    //   w_p0  = flat[2*N]
    //   h_p0  = flat[3*N]
    //   cls0_p0 = flat[4*N]
    //   cls1_p0 = flat[5*N]
    // Hypothesis 2: [N,C]=[8400,6], flat[p*C+c]
    //   cx_p0 = flat[0]
    //   cy_p0 = flat[1]
    //   w_p0  = flat[2]
    //   h_p0  = flat[3]
    //   cls0_p0 = flat[4]
    //   cls1_p0 = flat[5]

    unsigned int stride2 = (outDims.nbDims >= 2) ? outDims.d[1] : 1;
    unsigned long long stride1 = (outDims.nbDims >= 2) ?
        (unsigned long long)outDims.d[1] * outDims.d[0] : numElements;

    std::cout << "\nHypothesis 1 — [C,N]=[6,8400], data[c*N+p]:\n";
    if (outDims.nbDims >= 2) {
        unsigned int N2 = outDims.d[1];
        std::cout << "  N = " << N2 << std::endl;
        std::cout << "  flat[0]     = " << h_output[0] << "  ← cx_p0\n";
        std::cout << "  flat[1]     = " << h_output[1] << "  ← cx_p1 (H1) or cy_p0 (H2)\n";
        std::cout << "  flat[N]     = " << h_output[N2] << "  ← cy_p0 (H1)\n";
        std::cout << "  flat[2*N]   = " << h_output[2*N2] << "  ← w_p0 (H1)\n";
        std::cout << "  flat[3*N]   = " << h_output[3*N2] << "  ← h_p0 (H1)\n";
        std::cout << "  flat[4*N]   = " << h_output[4*N2] << "  ← cls0_p0 (H1)\n";
        std::cout << "  flat[5*N]   = " << h_output[5*N2] << "  ← cls1_p0 (H1)\n";

        // Check: flat[4*N] should be ~0.000018 if H1 is correct
        if (std::abs(h_output[4*N2] - 0.000018f) < 0.0001f)
            std::cout << "  ✓ flat[4*N] ≈ 0.000018 → H1 [C,N] CORRECT!\n";
        else if (std::abs(h_output[4] - 0.000018f) < 0.0001f)
            std::cout << "  ✓ flat[4] ≈ 0.000018 → H2 [N,C] CORRECT!\n";
        else {
            std::cout << "  ✗ Neither flat[4*N] nor flat[4] matches ONNX ref 0.000018\n";
            std::cout << "  → Check if engine was built from different ONNX!\n";
        }

        // Also check adjacent proposals
        std::cout << "\n  Adjacent proposals:\n";
        std::cout << "    flat[0]  (cx_p0)  = " << h_output[0] << std::endl;
        std::cout << "    flat[1]  (cx_p1?)  = " << h_output[1] << std::endl;
        std::cout << "    flat[N]  (cy_p0?)  = " << h_output[N2] << std::endl;
        std::cout << "    flat[N+1](cy_p1?)  = " << h_output[N2+1] << std::endl;
    }

    std::cout << "\nHypothesis 2 — [N,C]=[8400,6], data[p*C+c] (stride=" << C << "):\n";
    if (outDims.nbDims >= 2) {
        std::cout << "  flat[0] = " << h_output[0] << " ← cx_p0\n";
        std::cout << "  flat[1] = " << h_output[1] << " ← cy_p0\n";
        std::cout << "  flat[2] = " << h_output[2] << " ← w_p0\n";
        std::cout << "  flat[3] = " << h_output[3] << " ← h_p0\n";
        std::cout << "  flat[4] = " << h_output[4] << " ← cls0_p0\n";
        std::cout << "  flat[5] = " << h_output[5] << " ← cls1_p0\n";
    }

    // Extra: check total memory size
    std::cout << "\n=== MEMORY SIZE ===" << std::endl;
    std::cout << "numElements = " << numElements << std::endl;
    std::cout << "H1: 6*8400 = " << 6*8400 << std::endl;
    std::cout << "H2: 8400*6 = " << 8400*6 << std::endl;
    if (numElements == 6*8400)
        std::cout << "  numElements matches both H1 and H2" << std::endl;

    cudaFree(d_input); cudaFree(d_output);
    delete context; delete engine; delete runtime;
    return 0;
}
