#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    const float* conv_states,
    float* output,
    int64_t batch_size,
    int64_t dim,
    int64_t seq_len,
    int64_t kernel_size,
    int64_t has_initial_state,
    int64_t silu_activation
) {
    // TODO: 实现并启动 CUDA kernel。
}
