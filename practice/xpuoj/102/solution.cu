#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* input_flat,
    float* output_flat,
    int64_t rows,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
