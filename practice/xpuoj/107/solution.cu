#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* x1,
    const float* x2,
    const int64_t* y,
    float* loss,
    float margin,
    int64_t N,
    int64_t D
) {
    // TODO: 实现并启动 CUDA kernel。
}
