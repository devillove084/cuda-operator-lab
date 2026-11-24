#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* output,
    int64_t pattern_id,
    int64_t B,
    int64_t M,
    int64_t N,
    int64_t K
) {
    // TODO: 实现并启动 CUDA kernel。
}
