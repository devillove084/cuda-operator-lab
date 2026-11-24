#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(const __nv_bfloat16* input, __nv_bfloat16* out, int64_t batch_size, int64_t hidden_size) {
    // TODO: 实现并启动 CUDA kernel。
}
