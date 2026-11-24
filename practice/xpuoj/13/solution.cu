#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_fp8_e4m3* a_data,
    const float* a_sf,
    const __nv_fp8_e4m3* b_data,
    const float* b_sf,
    const int32_t* m_indices,
    __nv_bfloat16* D,
    int64_t M_total,
    int64_t K,
    int64_t N,
    int64_t num_groups
) {
    // TODO: 实现并启动 CUDA kernel。
}
