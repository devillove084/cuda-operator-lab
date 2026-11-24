#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* input,
    const float* scale,
    int8_t* output,
    float* amax,
    int64_t M,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
