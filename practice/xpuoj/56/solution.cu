#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* grad_output,
    const float* weight,
    float* grad_input,
    int64_t N,
    int64_t C_in,
    int64_t C_out,
    int64_t H_in,
    int64_t W_in,
    int64_t R,
    int64_t S,
    int64_t stride,
    int64_t padding
) {
    // TODO: 实现并启动 CUDA kernel。
}
