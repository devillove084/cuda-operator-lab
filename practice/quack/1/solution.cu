#include <cuda_runtime.h>

extern "C" void run_kernel(const float* logits, const int* labels, const float* grad_loss,
                           int rows, int vocab, float smoothing, float* loss, float* d_logits) {
    // TODO: Launch your CUDA kernel on the default stream.
}
