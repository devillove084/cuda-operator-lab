#include <cuda_runtime.h>

extern "C" void run_kernel(const float* q, const float* k, const float* v, const float* g,
                           const float* initial, const float* grad_out, const float* grad_final,
                           int batch, int tokens, int heads, int key_dim, int value_dim, float scale,
                           float* d_q, float* d_k, float* d_v, float* d_g, float* d_initial) {
    // TODO: Launch your CUDA kernel on the default stream.
}
