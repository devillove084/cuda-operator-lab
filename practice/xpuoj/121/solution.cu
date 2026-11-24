#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* q,
const __nv_bfloat16* k,
const __nv_bfloat16* v,
const int32_t* cu_q,
const int32_t* cu_k,
__nv_bfloat16* out,
int64_t total_tokens,
int64_t num_heads,
int64_t num_kv_heads,
int64_t head_dim,
int64_t max_seqlen,
int64_t causal
) {
    // TODO: 实现并启动 CUDA kernel。
}
