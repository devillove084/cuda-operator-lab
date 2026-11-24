#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* x,
const __nv_bfloat16* weight,
const __nv_bfloat16* bias,
const long* target,
float* loss,
long B_L,
long d,
long V
) {
    // TODO: 实现并启动 CUDA kernel。
}
