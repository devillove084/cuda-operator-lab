#include <cuda_runtime.h>
#include <cuda_fp8.h>

extern "C" void run_kernel(const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const float* a_scale,
                           const float* b_scale, int m, int n, int k, int block_k, float* out) {
    // TODO: Launch your CUDA kernel on the default stream.
}
