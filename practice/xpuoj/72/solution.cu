#include <stdint.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const cuComplex* A,
    const cuComplex* B,
    cuComplex* C,
    int64_t M,
    int64_t N,
    int64_t K
) {
    // TODO: 实现并启动 CUDA kernel。
}
