def run_kernel(
    q,             # Tensor[float16], shape (B, seq_len, HQ, D)
    k,             # Tensor[float16], shape (B, seq_len, H, D)
    v,             # Tensor[float16], shape (B, seq_len, H, D)
    block_indices, # Tensor[int32], shape (B, seq_len, H, S)
    output,        # Tensor[float16], shape (B, seq_len, HQ, D)
    B,             # int64
    seq_len,       # int64
    H,             # int64
    HQ,            # int64
    D,             # int64
    S,             # int64
    block_size,    # int64
    is_causal,     # int64, 1 in tests
):
    ...
