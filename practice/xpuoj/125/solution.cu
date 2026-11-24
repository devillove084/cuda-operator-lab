#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
const float* logits, const float* top_p, int32_t* out, int64_t batch_size, int64_t vocab_size
) {
    // TODO: 实现并启动 CUDA kernel。
}
