#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* X,
    const int32_t* indices,
    __nv_bfloat16* Y,
    int64_t M,
    int64_t K,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
