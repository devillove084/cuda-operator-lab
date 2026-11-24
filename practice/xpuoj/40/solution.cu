#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_cache_ptr,
    const __nv_bfloat16* v_cache_ptr,
    __nv_bfloat16* out_ptr,
    const int32_t* page_table_ptr,
    const int32_t* seq_lens_ptr,
    long num_kv_heads,
    long page_size,
    long batch_size,
    long num_qo_heads,
    long head_dim,
    long max_num_pages_per_req
) {
    // TODO: 实现并启动 CUDA kernel。
}
