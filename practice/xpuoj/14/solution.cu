#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_fp8_e4m3* a_data,
    const float* a_sf,
    const __nv_fp8_e4m3* b_data,
    const float* b_sf,
    const int32_t* masked_m,
    __nv_bfloat16* d,
    int64_t num_groups,
    int64_t max_m,
    int64_t n,
    int64_t k
) {
    // TODO: 实现并启动 CUDA kernel。
}
