#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* X,
__nv_bfloat16* residual,
const __nv_bfloat16* W,
float* Y,
int64_t M,
int64_t N,
float scale
) {
    // TODO: 实现并启动 CUDA kernel。
}
