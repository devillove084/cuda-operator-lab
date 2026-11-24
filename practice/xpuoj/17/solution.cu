#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* input,
    __nv_fp8_storage_t* output_q,
    float* scale,
    int64_t is_static,
    int64_t M,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
