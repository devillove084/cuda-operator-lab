#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* q,
__nv_bfloat16* k_cache,
__nv_bfloat16* v_cache,
const __nv_bfloat16* k_new,
const __nv_bfloat16* v_new,
__nv_bfloat16* out,
int64_t batch_size,
int64_t seqlen_q,
int64_t max_cache_len,
int64_t num_heads,
int64_t num_kv_heads,
int64_t head_dim,
int64_t cache_seqlen
) {
    // TODO: 实现并启动 CUDA kernel。
}
