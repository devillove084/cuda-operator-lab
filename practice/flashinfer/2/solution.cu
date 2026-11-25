#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void run_kernel(const __half* append_k, const __half* append_v,
                           const int* indptr, const int* indices, const int* batch_ids,
                           const int* positions, int updates, int page_size, int heads, int dim,
                           __half* cache_k, __half* cache_v) {
    // TODO: Launch your CUDA kernel on the default stream.
}
