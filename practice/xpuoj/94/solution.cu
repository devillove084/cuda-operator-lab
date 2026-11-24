#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* X,
    const int32_t* cu_seqlens,
    __nv_bfloat16* Y,
    int32_t* cu_seqlens_out,
    int32_t* max_t_out,
    int64_t batch_size,
    int64_t w_in,
    int64_t channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding
) {
    // TODO: 实现并启动 CUDA kernel。
}
