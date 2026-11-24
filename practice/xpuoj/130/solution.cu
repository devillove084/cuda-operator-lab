#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" void run_kernel(
const __nv_bfloat16* logits, const int64_t* labels, __nv_bfloat16* loss_out, int64_t batch_size, int64_t seq_len, int64_t vocab_size
) {
    // TODO: 实现并启动 CUDA kernel。
}
