#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void run_kernel(const __half* a, const __half* b, const float* bias, const int* groups,
                           int count, float* out) {
    // TODO: Launch your CUDA kernel on the default stream.
}
