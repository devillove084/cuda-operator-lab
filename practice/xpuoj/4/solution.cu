#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* probs,
    const int32_t* top_k,
    float* renorm_probs,
    int64_t batch_size,
    int64_t num_classes
) {
    // TODO: 实现并启动 CUDA kernel。
}
