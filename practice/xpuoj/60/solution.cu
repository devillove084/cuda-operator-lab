#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const double2* A,
    const double2* B,
    double2* output,
    float alpha,
    int64_t upper_flag,
    int64_t M,
    int64_t N
) {
    // TODO: 实现并启动 CUDA kernel。
}
