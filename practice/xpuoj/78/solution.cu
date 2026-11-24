#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* input,
    const int64_t* target,
    float* loss,
    float smoothing,
    int64_t N,
    int64_t C
) {
    // TODO: 实现并启动 CUDA kernel。
}
