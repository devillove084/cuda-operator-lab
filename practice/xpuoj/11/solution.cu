#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    const int32_t* m_indices,
    __nv_bfloat16* D,
    int64_t M_total,
    int64_t K,
    int64_t N,
    int64_t num_groups
) {
    // TODO: 实现并启动 CUDA kernel。
}
