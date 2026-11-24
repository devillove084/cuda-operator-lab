"""Local vector-add comparison kernel using Triton's blocked programming model."""
import triton
import triton.language as tl


@triton.jit
def vector_add(a, b, result, length, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = indices < length
    values = tl.load(a + indices, valid, other=0) + tl.load(b + indices, valid, other=0)
    tl.store(result + indices, values, valid)
