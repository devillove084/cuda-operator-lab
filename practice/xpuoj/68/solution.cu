#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* A,
    const float* C,
    float* output,
    float alpha,
    float beta,
    int64_t N,
    int64_t K
) {
    // TODO: 实现并启动 CUDA kernel。
}
