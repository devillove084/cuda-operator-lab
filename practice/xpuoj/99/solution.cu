#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const int8_t* a_q4_main,
    const half* a_s4_main,
    const int8_t* b_q4_main,
    const half* b_s4_main,
    const int8_t* a_q8_keeper,
    const half* a_s8_keeper,
    const int8_t* b_q8_keeper,
    const half* b_s8_keeper,
    half* out,
    int64_t M,
    int64_t N,
    int64_t K_main,
    int64_t K_keeper,
    int64_t group_size
) {
    // TODO: 实现并启动 CUDA kernel。
}
