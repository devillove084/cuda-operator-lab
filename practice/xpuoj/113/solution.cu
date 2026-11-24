#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* q_rope, const __nv_bfloat16* k_rope, const __nv_bfloat16* q_nope, const __nv_bfloat16* k_nope, const float* cos_sin_cache, const int32_t* pos_ids, float* q_rope_out, float* k_rope_out, float* q_nope_out, float* k_nope_out, int64_t nnz, int64_t qh, int64_t kh, int64_t rope_dim, int64_t nope_dim, int64_t is_neox
) {
    // TODO: 实现并启动 CUDA kernel。
}
