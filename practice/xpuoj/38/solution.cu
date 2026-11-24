#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* query_ptr,
    const __nv_bfloat16* kv_cache_ptr,
    const void* workspace_buffer_ptr,
    const int32_t* block_tables_ptr,
    const int32_t* seq_lens_ptr,
    const int32_t* cum_seq_lens_q_ptr,
    const int32_t* cum_seq_lens_kv_ptr,
    __nv_bfloat16* out_ptr,
    float bmm1_scale,
    float bmm2_scale,
    long batch_size,
    long num_heads,
    long head_dim,
    long page_size,
    long max_pages_per_seq,
    long total_tokens_q,
    long max_q_len,
    long max_kv_len,
    long window_left
) {
    // TODO: 实现并启动 CUDA kernel。
}
