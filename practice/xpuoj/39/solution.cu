#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* query_ptr,
    const __nv_bfloat16* kv_cache_ptr,
    const int32_t* block_tables_ptr,
    const int32_t* seq_lens_ptr,
    __nv_bfloat16* out_ptr,
    long batch_size,
    long num_qo_heads,
    long num_kv_heads,
    long head_dim,
    long page_size,
    long max_num_pages,
    long max_seq_len
) {
    // TODO: 实现并启动 CUDA kernel。
}
