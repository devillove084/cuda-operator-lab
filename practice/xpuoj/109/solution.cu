#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* logits,
    const float* target,
    float* loss,
    float smooth,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W
) {
    // TODO: 实现并启动 CUDA kernel。
}
