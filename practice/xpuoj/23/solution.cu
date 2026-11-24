#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* input,
    const int64_t* indices,
    __nv_bfloat16* output,
    int64_t N,
    int64_t M,
    int64_t H
) {
    // TODO: 实现并启动 CUDA kernel。
}
