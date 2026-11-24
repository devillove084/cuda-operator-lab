#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w,
    const __nv_bfloat16* ln_weight,
    const __nv_bfloat16* ln_bias,
    __nv_bfloat16* out,
    int64_t M,
    int64_t K,
    int64_t N,
    float eps
) {
    // TODO: 实现并启动 CUDA kernel。
}
