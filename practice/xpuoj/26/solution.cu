#include <stdint.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const int8_t* a,
    const __nv_fp8_e4m3* b,
    float* output,
    int64_t G,
    int64_t M,
    int64_t N,
    int64_t K
) {
    // TODO: 实现并启动 CUDA kernel。
}
