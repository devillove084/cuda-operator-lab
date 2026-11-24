#include <stdint.h>
#include <stdbool.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const uint8_t* bits,
    uint8_t* packed,
    int64_t num_bits
) {
    // TODO: 实现并启动 CUDA kernel。
}
