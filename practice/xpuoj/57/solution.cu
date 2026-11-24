#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* grad_output,
    const __nv_bfloat16* weight,
    __nv_bfloat16* grad_input,
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
