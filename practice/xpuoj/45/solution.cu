#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int64_t N,
    int64_t C_in,
    int64_t C_out,
    int64_t H,
    int64_t W,
    int64_t R,
    int64_t S,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    // TODO: 实现并启动 CUDA kernel。
}
