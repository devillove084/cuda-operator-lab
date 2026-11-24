#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr,
    const __nv_bfloat16* cos_cache_ptr,
    const __nv_bfloat16* sin_cache_ptr,
    __nv_bfloat16* q_out_ptr,
    __nv_bfloat16* k_out_ptr,
    long batch_size,
    long seq_len,
    long num_heads,
    long num_kv_heads,
    long head_dim
) {
    // TODO: 实现并启动 CUDA kernel。
}
