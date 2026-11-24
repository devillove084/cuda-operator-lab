#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* a,
    const float* b,
    const float* bias,
    float* c,
    int64_t M,
    int64_t N,
    int64_t K
) {
    // TODO: 实现并启动 CUDA kernel。
}
