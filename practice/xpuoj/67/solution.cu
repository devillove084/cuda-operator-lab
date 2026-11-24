#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_bfloat16* y,
    int64_t N,
    int64_t C_in,
    int64_t C_out,
    int64_t D,
    int64_t H,
    int64_t W,
    int64_t Kd,
    int64_t Kh,
    int64_t Kw,
    int64_t stride,
    int64_t padding
) {
    // TODO: 实现并启动 CUDA kernel。
}
