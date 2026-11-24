#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* hidden_states, const bool* attention_mask, __nv_bfloat16* out, int64_t batch_size, int64_t seq_len, int64_t hidden_size, int64_t nnz
) {
    // TODO: 实现并启动 CUDA kernel。
}
