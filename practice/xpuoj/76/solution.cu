#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* pred,
    const __nv_bfloat16* target,
    float* loss,
    float beta,
    int64_t M,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
