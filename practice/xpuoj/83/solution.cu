#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
const float* gating_output, float* topk_weights, int32_t* topk_ids, int64_t num_tokens, int64_t num_experts, int64_t topk
) {
    // TODO: 实现并启动 CUDA kernel。
}
