#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const int32_t* b_q_weight,
    int32_t* repacked_weight,
    int64_t size_k,
    int64_t size_n,
    int64_t num_bits
) {
    // TODO: 实现并启动 CUDA kernel。
}
