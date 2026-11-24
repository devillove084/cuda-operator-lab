#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int64_t T,
    int64_t H
) {
    // TODO: 实现并启动 CUDA kernel。
}
