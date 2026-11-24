#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* A,
    const float* B,
    const float* C,
    float* D,
    float alpha,
    float beta,
    int64_t N,
    int64_t K
) {
    // TODO: 实现并启动 CUDA kernel。
}
