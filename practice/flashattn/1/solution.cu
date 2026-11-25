#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void run_kernel(const __half* q, const __half* k, const __half* v, int batch,
                           int q_len, int kv_len, int heads, int dim, int window_left,
                           float scale, float* out, float* out_lse) {
    // TODO: Launch your CUDA kernel on the default stream.
}
