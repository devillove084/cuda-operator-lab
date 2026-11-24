#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w1,
    const __nv_bfloat16* w3,
    const int32_t* group_sizes,
    __nv_bfloat16* out,
    int64_t total_tokens,
    int64_t in_features,
    int64_t out_features,
    int64_t num_groups
) {
    // TODO: 实现并启动 CUDA kernel。
}
