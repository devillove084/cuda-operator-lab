#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const double* a,
    const double* b,
    double* output,
    double alpha,
    int64_t upper,
    int64_t M,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
