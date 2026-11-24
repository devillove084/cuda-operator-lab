#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* q, const __nv_bfloat16* k, const int32_t* pos_ids, __nv_bfloat16* q_out, __nv_bfloat16* k_out, int64_t nnz, int64_t qh, int64_t kh, int64_t head_dim, int64_t rotary_dim, int64_t interleave
) {
    // TODO: 实现并启动 CUDA kernel。
}
