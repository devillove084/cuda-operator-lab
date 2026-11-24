#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* b, const __nv_bfloat16* ln_w, const __nv_bfloat16* ln_b, const bool* mask, __nv_bfloat16* out, int64_t B, int64_t T, int64_t H, int64_t ksize
) {
    // TODO: 实现并启动 CUDA kernel。
}
