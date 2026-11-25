"""Reference math for the broader kernel-library practice sets."""
import math

import torch

from practice_advanced_reference import Exercise, floating, integer, output


def causal_attention(workload):
    batch, q_len, kv_len, heads, dim, window_left = workload.shape
    q_shape = (batch, q_len, heads, dim)
    kv_shape = (batch, kv_len, heads, dim)
    magnitude = 17.0 if workload.variant == "extreme" else 1.0
    q = floating(q_shape, torch.float16, magnitude)
    k = floating(kv_shape, torch.float16, magnitude)
    v = floating(kv_shape, torch.float16)
    scale = dim ** -0.5
    inputs = {"q": q, "k": k, "v": v, "batch": batch, "q_len": q_len, "kv_len": kv_len,
              "heads": heads, "dim": dim, "window_left": window_left, "scale": scale}

    def reference(dtype):
        query = torch.arange(q_len, device=q.device)[:, None] + kv_len - q_len
        position = torch.arange(kv_len, device=q.device)[None, :]
        keep = position <= query
        if window_left >= 0:
            keep = keep & (position >= query - window_left)
        score = (torch.einsum("bqhd,bkhd->bhqk", q.to(dtype), k.to(dtype)) * scale)
        score = score.masked_fill(~keep, -math.inf)
        prob = torch.softmax(score, -1).masked_fill(~keep, 0.0)
        out = torch.einsum("bhqk,bkhd->bqhd", prob, v.to(dtype))
        return {"out": out.float(), "out_lse": torch.logsumexp(score, -1).transpose(1, 2).float()}

    return Exercise(inputs, {"out": output(q_shape), "out_lse": output(q_shape[:-1])},
                    reference, atol=3e-4, rtol=3e-4)


def launch_groups(workload):
    # Spacers before, between and after matrices expose writes outside group tiles.
    a_size = b_size = bias_size = out_size = 3
    groups = []
    for m, n, k in workload.shape:
        groups.append((m, n, k, a_size, b_size, bias_size, out_size))
        a_size += m * k + 3
        b_size += n * k + 5
        bias_size += n + 7
        out_size += m * n + 11
    a = floating((a_size,), torch.float16, 0.24)
    b = floating((b_size,), torch.float16, 0.24)
    bias = floating((bias_size,), scale=0.35)
    initial = floating((out_size,))
    inputs = {"a": a, "b": b, "bias": bias, "groups": integer(groups), "count": len(groups)}

    def reference(dtype):
        out = initial.clone()
        for m, n, k, off_a, off_b, off_bias, off_out in groups:
            ac = a[off_a:off_a + m * k].view(m, k).to(dtype)
            bc = b[off_b:off_b + n * k].view(n, k).to(dtype)
            result = (ac @ bc.T) + bias[off_bias:off_bias + n]
            out[off_out:off_out + m * n] = result.relu().flatten()
        return {"out": out}

    return Exercise(inputs, {"out": initial}, reference, ("out",), atol=4e-5, rtol=3e-4)


def block_scaled(workload):
    m, n, k, block_k = workload.shape
    count = (k + block_k - 1) // block_k
    a = floating((m, k), torch.float8_e4m3fn, 1.3)
    b = floating((n, k), torch.float8_e4m3fn, 1.3)
    a_scale = torch.rand((m, count), device="cuda").add_(0.125)
    b_scale = torch.rand((n, count), device="cuda").add_(0.125)
    if workload.variant == "zeros":
        a_scale[::2, 1::2].zero_()
        b_scale[::2, 2::3].zero_()
    elif workload.variant == "extreme":
        a_scale[:, 0].fill_(1e-3)
        b_scale[:, 1].fill_(32.0)
        a[:, :1] = -448.0
        b[:, :1] = 448.0
    inputs = {"a": a, "b": b, "a_scale": a_scale, "b_scale": b_scale,
              "m": m, "n": n, "k": k, "block_k": block_k}

    def reference(dtype):
        index = torch.arange(k, device=a.device) // block_k
        lhs = a.to(dtype) * a_scale[:, index]
        rhs = b.to(dtype) * b_scale[:, index]
        return {"out": (lhs @ rhs.T).float()}

    return Exercise(inputs, {"out": output((m, n))}, reference, atol=3e-4, rtol=1e-4)


def cross_entropy(workload):
    rows, vocab, smoothing = workload.shape
    logits = floating((rows, vocab), scale=4.0)
    labels = torch.randint(vocab, (rows,), dtype=torch.int32, device=logits.device)
    upstream = floating((rows,), scale=0.6)
    if workload.variant == "ignore":
        labels[1::3] = -100
    elif workload.variant == "all_ignore":
        labels.fill_(-100)
    elif workload.variant == "extreme":
        logits[0, :2] = 1000.0
        logits[1].sub_(1000.0)
        logits[2].mul_(80.0)
        labels[0] = 1
    inputs = {"logits": logits, "labels": labels, "grad_loss": upstream, "rows": rows,
              "vocab": vocab, "smoothing": smoothing}

    def reference(dtype):
        ignored = labels.eq(-100)
        chosen = labels.masked_fill(ignored, 0).view(-1, 1).long()
        lp = torch.log_softmax(logits.to(dtype), dim=-1)
        mean = lp.mean(dim=-1)
        nll = lp.gather(-1, chosen).squeeze(-1)
        loss = (-(1.0 - smoothing) * nll - smoothing * mean).masked_fill(ignored, 0)
        grad = (lp.exp() - smoothing / vocab)
        grad.scatter_add_(-1, chosen, torch.full_like(nll[:, None], smoothing - 1.0))
        grad = (upstream[:, None] * grad).masked_fill(ignored[:, None], 0)
        return {"loss": loss.float(), "d_logits": grad.float()}

    return Exercise(inputs, {"loss": output((rows,)), "d_logits": output((rows, vocab))},
                    reference, atol=3e-4, rtol=3e-4)


def moe_align(workload):
    tokens, top_k, experts, tile = workload.shape
    routes = tokens * top_k
    assignments = torch.randint(experts, (routes,), dtype=torch.int32, device="cuda")
    if workload.variant == "round_robin":
        assignments.copy_(torch.arange(routes, device="cuda") % experts)
    elif workload.variant == "hot":
        assignments.fill_(experts - 1)
    elif workload.variant == "invalid":
        assignments[0::5] = -1
    elif workload.variant == "all_invalid":
        assignments.fill_(-1)
    capacity = routes + experts * (tile - 1)
    slots = capacity // tile
    inputs = {"assignments": assignments, "routes": routes, "experts": experts, "tile": tile}

    def reference(dtype):
        valid = assignments.ge(0)
        expert = assignments.masked_fill(~valid, experts)
        counts = torch.zeros(experts + 1, device=expert.device, dtype=torch.int64)
        counts.scatter_add_(0, expert.long(), torch.ones_like(expert, dtype=counts.dtype))
        counts = counts[:-1]
        ends = counts.cumsum(dim=0)
        starts = ends - counts
        padded = (counts + tile - 1) // tile * tile
        padded_ends = padded.cumsum(dim=0)
        padded_starts = padded_ends - padded
        order = expert.argsort(stable=True)
        group = expert[order].long().clamp(max=experts - 1)
        ranks = torch.arange(routes, device=expert.device) - starts[group]
        target = (padded_starts[group] + ranks).masked_fill(~valid[order], capacity)
        sorted_ids = torch.full((capacity + 1,), routes, dtype=torch.int32, device=expert.device)
        sorted_ids.scatter_(0, target, order.int().masked_fill(~valid[order], routes))
        tile_idx = torch.arange(slots, device=expert.device) * tile
        blocks = torch.searchsorted(padded_ends, tile_idx, right=True)
        blocks = blocks.masked_fill(blocks == experts, -1)
        return {"sorted_routes": sorted_ids[:-1], "block_experts": blocks.int(),
                "padded_count": padded_ends[-1:].int()}

    outputs = {name: torch.full((size,), -777, dtype=torch.int32, device=assignments.device)
               for name, size in (("sorted_routes", capacity), ("block_experts", slots),
                                  ("padded_count", 1))}
    return Exercise(inputs, outputs, reference)


EXERCISES = {"flashattn": causal_attention, "cutlass": launch_groups, "triton": block_scaled,
             "quack": cross_entropy, "vllm": moe_align}


def exercise_for(platform, workload):
    return EXERCISES[platform](workload)
