#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void run_kernel(const __half* cache_k, const __half* cache_v,
                           const int* indptr, const int* indices, const int* out_indptr,
                           int batch, int page_size, int heads, int dim,
                           __half* out_k, __half* out_v) {
    // TODO: Launch your CUDA kernel on the default stream.
}
