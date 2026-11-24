#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w_all,
    __nv_bfloat16* out,
    const int64_t* out_dims,
    int64_t batch_size,
    int64_t in_features,
    int64_t K
) {
    // TODO: 实现并启动 CUDA kernel。
}
