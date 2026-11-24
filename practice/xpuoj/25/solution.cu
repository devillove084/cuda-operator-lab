#include <stdint.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(
    const void* values,
    const int64_t* indices,
    void* output,
    int64_t first_axis_dim,
    int64_t M,
    int64_t H,
    int64_t dtype_code
) {
    // TODO: 实现并启动 CUDA kernel。
}
