#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* q,
const __nv_bfloat16* k,
const __nv_bfloat16* v,
__nv_bfloat16* out,
int64_t Bsz,
int64_t Seq,
int64_t H,
int64_t HK,
int64_t D,
int64_t causal,
float scale
) {
    // TODO: 实现并启动 CUDA kernel。
}
