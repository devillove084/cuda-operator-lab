#include <stdint.h>
#include <stdbool.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* logits,
    const bool* bitmask,
    const int32_t* indices,
    float* output,
    int64_t batch_size,
    int64_t vocab_size,
    int64_t has_indices
) {
    // TODO: 实现并启动 CUDA kernel。
}
