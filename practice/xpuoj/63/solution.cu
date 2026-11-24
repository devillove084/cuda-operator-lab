#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* A,
    const float* B,
    const float* bias,
    float* output,
    float alpha,
    int64_t M,
    int64_t K,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
