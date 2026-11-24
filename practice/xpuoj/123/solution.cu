#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void run_kernel(
const __half* q,
const __half* k,
const __half* v,
const __half* g,
const __half* level_scales,
__half* out,
int64_t batch_size,
int64_t seq_len,
int64_t num_heads,
int64_t head_dim,
int64_t num_levels
) {
    // TODO: 实现并启动 CUDA kernel。
}
