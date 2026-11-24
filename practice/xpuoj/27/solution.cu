#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const int8_t* a,
    const int8_t* b,
    int32_t* output,
    int64_t M,
    int64_t N,
    int64_t K
) {
    // TODO: 实现并启动 CUDA kernel。
}
