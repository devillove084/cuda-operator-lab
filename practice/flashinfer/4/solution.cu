#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void run_kernel(const __half* q, const __half* cache_k, const __half* cache_v,
                           const int* indptr, const int* indices, const int* lengths,
                           int batch, int page_size, int kv_heads, int q_heads,
                           int dim, float scale, float* out, float* out_lse) {
    // TODO: Launch your CUDA kernel on the default stream.
}
