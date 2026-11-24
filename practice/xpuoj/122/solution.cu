#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* q,
const __nv_bfloat16* k,
const __nv_bfloat16* v,
const __nv_bfloat16* grad_out,
const int32_t* cu_seqlens_q,
const int32_t* cu_seqlens_k,
__nv_bfloat16* dq,
__nv_bfloat16* dk,
__nv_bfloat16* dv,
int64_t batch_size,
int64_t num_heads,
int64_t num_kv_heads,
int64_t head_dim
) {
    // TODO: 实现并启动 CUDA kernel。
}
