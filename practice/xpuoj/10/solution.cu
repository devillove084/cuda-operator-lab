#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_fp8_e4m3* A,
    const float* A_scale,
    const __nv_fp8_e4m3* B,
    const float* B_scale,
    __nv_bfloat16* D,
    int64_t M,
    int64_t K,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
