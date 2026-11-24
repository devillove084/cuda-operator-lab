#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t N,
    int64_t K
) {
    // TODO: 实现并启动 CUDA kernel。
}
