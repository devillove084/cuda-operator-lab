#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* x_ptr,
    const __nv_bfloat16* bias_ptr,
    __nv_bfloat16* out_ptr,
    long batch_size,
    long hidden_size
) {
    // TODO: 实现并启动 CUDA kernel。
}
