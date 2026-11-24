#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* qkv_ptr,
    const int32_t* cu_seqlens_ptr,
    long max_s,
    long num_heads,
    long head_dim,
    __nv_bfloat16* out_ptr,
    long batch_size
) {
    // TODO: 实现并启动 CUDA kernel。
}
