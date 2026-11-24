#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* hidden_states,
    const float* weight,
    const int32_t* cu_seqlens,
    float* y,
    int64_t num_tokens,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dim,
    int64_t n_gram_in,
    int64_t n_gram_out
) {
    // TODO: 实现并启动 CUDA kernel。
}
