#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* qkv, __nv_bfloat16* out, int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim
) {
    // TODO: 实现并启动 CUDA kernel。
}
