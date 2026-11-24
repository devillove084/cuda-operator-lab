#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* x,
    float* y,
    float scale,
    int64_t M,
    int64_t K
) {
    // TODO: 实现并启动 CUDA kernel。
}
