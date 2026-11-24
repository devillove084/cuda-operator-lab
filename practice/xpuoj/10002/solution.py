def run_kernel(
    q,        # Tensor[float16], shape (batch, heads, dim)
    q_pe,     # Tensor[float16], shape (batch, heads, pe_dim)
    kv,       # Tensor[float16], shape (batch, kv_ctx, kv_heads, dim)
    k_pe,     # Tensor[float16], shape (batch, kv_ctx, kv_heads, pe_dim)
    output,   # Tensor[float16], shape (batch, heads, dim)
    batch,    # int64
    heads,    # int64
    kv_heads, # int64
    kv_ctx,   # int64
    dim,      # int64, fixed to 512
    pe_dim,   # int64, fixed to 64
):
    ...
