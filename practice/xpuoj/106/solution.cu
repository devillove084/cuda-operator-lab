#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* anchor,
    const float* positive,
    const float* negative,
    float* loss,
    float margin,
    int64_t N,
    int64_t D
) {
    // TODO: 实现并启动 CUDA kernel。
}
