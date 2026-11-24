#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* input_tokens,
const __nv_bfloat16* expert_weights,
const int32_t* expert_indices,
__nv_bfloat16* output,
int64_t total_tokens,
int64_t hidden_dim,
int64_t up_hidden_dim,
int64_t num_experts,
int64_t topk
) {
    // TODO: 实现并启动 CUDA kernel。
}
