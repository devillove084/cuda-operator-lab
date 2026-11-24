#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void run_kernel(
const half* logits, const int64_t* labels, float* losses, int64_t M, int64_t V, float smoothing, int64_t padding_idx
) {
    // TODO: 实现并启动 CUDA kernel。
}
