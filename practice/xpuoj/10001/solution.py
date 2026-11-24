def run_kernel(
    expert_tokens,          # Tensor[float16], shape (group_sum, d_hidden)
    routed_expert_gate,     # Tensor[float16], shape (n_routed_experts, d_expert, d_hidden)
    routed_expert_up,       # Tensor[float16], shape (n_routed_experts, d_expert, d_hidden)
    routed_expert_down,     # Tensor[float16], shape (n_routed_experts, d_hidden, d_expert)
    routed_expert_weights,  # Tensor[float16], shape (group_sum,)
    group_sizes,            # Tensor[int32], shape (n_routed_experts,)
    group_offsets,          # Tensor[int32], shape (n_routed_experts,)
    group_padded_offsets,   # Tensor[int32], shape (n_routed_experts,)
    group_idx_for_bx,       # Tensor[int32], shape (ceil(group_sum / 128) + n_routed_experts,)
    up_logits,              # Tensor[float16], shape (group_sum, d_expert), workspace
    output,                 # Tensor[float16], shape (group_sum, d_hidden)
    group_sum,              # int64
    d_hidden,               # int64
    d_expert,               # int64
    n_routed_experts,       # int64
    block_token,            # int64, fixed to 128 in tests
):
    ...
