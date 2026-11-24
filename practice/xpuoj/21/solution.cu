#include <stdint.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_fp8_e4m3* q,
    const __nv_fp8_e4m3* kv,
    const float* kv_sf,
    const float* weights,
    const int32_t* cu_seqlen_ks,
    const int32_t* cu_seqlen_ke,
    float* logits,
    int64_t M,
    int64_t seq_len_kv,
    int64_t num_heads,
    int64_t head_dim
) {
    // TODO: 实现并启动 CUDA kernel。
}
