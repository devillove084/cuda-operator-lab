#include <cuda_runtime.h>

extern "C" void run_kernel(const int* assignments, int routes, int experts, int tile,
                           int* sorted_routes, int* block_experts, int* padded_count) {
    // TODO: Launch your CUDA kernel on the default stream.
}
