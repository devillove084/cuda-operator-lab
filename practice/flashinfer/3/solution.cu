#include <cuda_runtime.h>

extern "C" void run_kernel(const float* values, const float* lse, int batch, int heads,
                           int splits, int dim, float* out, float* out_lse) {
    // TODO: Launch your CUDA kernel on the default stream.
}
