#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr,
    const __nv_bfloat16* v_ptr,
    __nv_bfloat16* out_ptr,
    float* lse_ptr,
    long qo_len,
    long kv_len,
    long num_qo_heads,
    long num_kv_heads,
    long head_dim
) {
    // TODO: 实现并启动 CUDA kernel。
}
