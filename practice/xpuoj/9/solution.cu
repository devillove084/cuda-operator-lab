#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* logits,
    const int64_t* labels,
    float* loss,
    int64_t M,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
