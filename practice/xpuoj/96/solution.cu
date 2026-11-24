#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const float* X_real,
    const float* X_imag,
    float* Y_real,
    float* Y_imag,
    int64_t batch,
    int64_t n
) {
    // TODO: 实现并启动 CUDA kernel。
}
