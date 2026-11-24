#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* A, const __nv_bfloat16* B, const __nv_bfloat16* PreAct, __nv_bfloat16* DX, __nv_bfloat16* PostAct, int64_t M, int64_t N, int64_t K, int64_t act_id
) {
    // TODO: 实现并启动 CUDA kernel。
}
