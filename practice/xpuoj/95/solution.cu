#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* in1,
    const float* in2,
    const int32_t* idx1,
    float* out,
    int64_t M,
    int64_t K,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
