#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* X, const __nv_bfloat16* W, const __nv_bfloat16* B, __nv_bfloat16* Y, int64_t Bsz, int64_t S, int64_t InF, int64_t OutF
) {
    // TODO: 实现并启动 CUDA kernel。
}
