#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* A_data,
    const float* B_data,
    float* C_data,
    const int32_t* group_m,
    const int32_t* group_k,
    const int32_t* group_n,
    const int32_t* a_offsets,
    const int32_t* b_offsets,
    const int32_t* c_offsets,
    int64_t group_count
) {
    // TODO: 实现并启动 CUDA kernel。
}
