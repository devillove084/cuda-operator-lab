#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* q,
const __nv_bfloat16* k_cache,
const int32_t* block_table,
const int32_t* cache_seqlens,
const int32_t* tile_scheduler_metadata,
const int32_t* num_splits,
float softmax_scale,
int64_t causal,
int64_t is_fp8_kvcache,
const int32_t* indices,
__nv_bfloat16* out,
float* softmax_lse,
int64_t batch_size,
int64_t seq_len_q,
int64_t num_heads_q,
int64_t head_dim,
int64_t num_blocks,
int64_t page_block_size,
int64_t num_heads_k,
int64_t max_num_blocks_per_seq,
int64_t indices_topk
) {
    // TODO: 实现并启动 CUDA kernel。
}
