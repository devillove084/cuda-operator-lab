#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* X,
const float* W,
const float* B,
__nv_bfloat16* Y,
int64_t batch_size,
int64_t hidden_size
) {
    // TODO: 实现并启动 CUDA kernel。
}
