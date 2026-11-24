#include <stdint.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const void* input,
    __nv_fp8_storage_t* output_q,
    float* scale,
    int64_t M,
    int64_t N,
    int64_t dtype_code
) {
    // TODO: 实现并启动 CUDA kernel。
}
