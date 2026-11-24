#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const half* x,
    const half* weight,
    const half* bias,
    half* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t R,
    int64_t S,
    int64_t stride,
    int64_t padding
) {
    // TODO: 实现并启动 CUDA kernel。
}
