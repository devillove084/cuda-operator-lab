#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    const __nv_bfloat16* broadcast,
    __nv_bfloat16* output,
    int64_t N,
    int64_t C_in,
    int64_t C_out,
    int64_t D,
    int64_t H,
    int64_t W,
    int64_t T,
    int64_t R,
    int64_t S,
    int64_t stride,
    int64_t padding
) {
    // TODO: 实现并启动 CUDA kernel。
}
