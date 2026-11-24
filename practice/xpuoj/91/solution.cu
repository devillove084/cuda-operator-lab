#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int64_t M,
    int64_t N,
    float dropout_rate,
    int64_t dropout_seed
) {
    // TODO: 实现并启动 CUDA kernel。
}
