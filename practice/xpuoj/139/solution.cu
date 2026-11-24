#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int64_t B,
    int64_t I,
    int64_t J
) {
    // TODO: 实现并启动 CUDA kernel。
}
