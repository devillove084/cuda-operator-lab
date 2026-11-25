#include <cuda_runtime.h>

extern "C" void run_kernel(const float* q, const float* k, const float* v, const float* g,
                           const float* beta, const float* initial, const int* offsets,
                           int batch, int tokens, int heads, int key_dim, int value_dim,
                           float scale, float* out, float* final) {
    // TODO: Launch your CUDA kernel on the default stream.
}
