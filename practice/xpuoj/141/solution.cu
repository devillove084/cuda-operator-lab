#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* a,
    __nv_bfloat16* out,
    int64_t B,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
