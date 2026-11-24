#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_cache_ptr,
    const __nv_bfloat16* v_cache_ptr,
    __nv_bfloat16* out_ptr,
    float scale,
    const void* workspace_buffer_ptr,
    long max_sequence_kv,
    const int32_t* actual_seq_lens_kv_ptr,
    const int32_t* block_tables_ptr,
    long batch_size,
    long num_qo_heads,
    long num_kv_heads,
    long head_dim,
    long page_size,
    long max_pages_per_seq
) {
    // TODO: 实现并启动 CUDA kernel。
}
