#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const half* a,
    const int8_t* b_quant,
    const half* b_scales,
    half* out,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_size
) {
    // TODO: 实现并启动 CUDA kernel。
}
