#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void run_kernel(
half* qkv, const half* query_bias, const half* pos_bias, half* pos_query, int64_t B, int64_t T, int64_t H, int64_t D, int64_t max_seqlen
) {
    // TODO: 实现并启动 CUDA kernel。
}
