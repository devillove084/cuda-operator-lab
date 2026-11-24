#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const void* a, const int8_t* b, const float* a_scales, const __nv_bfloat16* b_scales, const int32_t* experts_offsets, const int32_t* problem_sizes, const int64_t* a_strides, const int64_t* b_strides, const int64_t* d_strides, const int64_t* s_strides, __nv_bfloat16* out, int64_t M_total, int64_t K, int64_t N, int64_t E, int64_t chunk_size
) {
    // TODO: 实现并启动 CUDA kernel。
}
