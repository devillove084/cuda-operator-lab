"""Readable tensor oracles for the FlashInfer and FLA exercises."""
from dataclasses import dataclass, field
import math

import torch


@dataclass
class Exercise:
    inputs: dict
    outputs: dict
    reference: object
    inout: tuple = ()
    atol: float = 0.0
    rtol: float = 0.0


def floating(shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(shape, device="cuda") * scale).to(dtype)


def output(shape, dtype=torch.float32):
    return torch.full(shape, float("nan"), dtype=dtype, device="cuda")


def integer(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def offsets(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def page_map(lengths, page_size, heads, dim, shared):
    indptr = offsets((length + page_size - 1) // page_size for length in lengths)
    physical_pages = indptr[-1] + 3
    indices = torch.randperm(physical_pages, device="cuda")[:max(1, indptr[-1])].int()
    if shared:
        for base in indptr[1:-1]:
            indices[base] = indices[0]
    shape = (physical_pages, page_size, heads, dim)
    return floating(shape, torch.float16), floating(shape, torch.float16), integer(indptr), indices


def page_rows(indptr, indices, batch, position, page_size):
    page = indices[indptr[batch] + position // page_size].long()
    return page, (position % page_size).long()


def paged_io(workload, append):
    lengths, page_size, heads, dim = workload.shape
    k, v, indptr, indices = page_map(lengths, page_size, heads, dim, workload.variant == "shared")
    batches, positions, groups = [], [], []
    for sequence, length in enumerate(lengths):
        selected = list(range(length))
        if workload.variant == "suffix":
            selected = selected[length // 2:]
        elif workload.variant == "sparse":
            selected = selected[::-3]
        elif workload.variant == "no_updates":
            selected = []
        groups.append(len(selected))
        batches.extend([sequence] * len(selected))
        positions.extend(selected)
    batch = integer(batches)
    position = integer(positions)
    size = len(positions)
    shape = (size, heads, dim)
    if append:
        perm = torch.randperm(size, device="cuda")
        args = {"append_k": floating(shape, torch.float16), "append_v": floating(shape, torch.float16),
                "indptr": indptr, "indices": indices, "batch_ids": batch[perm],
                "positions": position[perm], "updates": size, "page_size": page_size,
                "heads": heads, "dim": dim}

        def apply(dtype):
            result_k, result_v = k.clone(), v.clone()
            page, slot = page_rows(indptr, indices, args["batch_ids"].long(),
                                   args["positions"], page_size)
            result_k[page, slot] = args["append_k"]
            result_v[page, slot] = args["append_v"]
            return {"cache_k": result_k, "cache_v": result_v}

        return Exercise(args, {"cache_k": k, "cache_v": v}, apply, ("cache_k", "cache_v"))

    args = {"cache_k": k, "cache_v": v, "indptr": indptr, "indices": indices,
            "out_indptr": integer(offsets(groups)), "batch": len(lengths), "page_size": page_size,
            "heads": heads, "dim": dim}

    def gather(dtype):
        page, slot = page_rows(indptr, indices, batch.long(), position, page_size)
        return {"out_k": k[page, slot], "out_v": v[page, slot]}

    return Exercise(args, {"out_k": output(shape, k.dtype), "out_v": output(shape, v.dtype)}, gather)


def merge_states(workload):
    batch, heads, splits, dim = workload.shape
    values = floating(workload.shape)
    lse = floating((batch, heads, splits), scale=6)
    if workload.variant == "equal":
        lse.zero_()
    elif workload.variant in ("missing", "all_missing"):
        lse[..., ::2] = -math.inf
        if workload.variant == "all_missing":
            lse.fill_(-math.inf)
        values.masked_fill_(lse.isneginf().unsqueeze(-1), math.nan)
    elif workload.variant == "extreme":
        lse[..., 0] = 1500.0
        lse[..., 1] = 1499.0
        lse[..., 2] = -1800.0
        lse[0, 0].sub_(2600.0)
    args = {"values": values, "lse": lse, "batch": batch, "heads": heads,
            "splits": splits, "dim": dim}

    def merge(dtype):
        partial = lse.to(dtype)
        largest = partial.amax(dim=-1, keepdim=True)
        largest = largest.masked_fill(largest.isneginf(), 0.0)
        weight = torch.exp2(partial - largest)
        total = weight.sum(dim=-1, keepdim=True)
        safe_total = total.masked_fill(total == 0, 1.0)
        valid_values = values.to(dtype).masked_fill(lse.isneginf().unsqueeze(-1), 0.0)
        merged = (valid_values * (weight / safe_total).unsqueeze(-1)).sum(dim=-2)
        return {"out": merged.float(), "out_lse": (largest + total.log2()).squeeze(-1).float()}

    return Exercise(args, {"out": output((batch, heads, dim)), "out_lse": output((batch, heads))},
                    merge, atol=3e-5, rtol=1e-5)


def paged_decode(workload):
    lengths, page_size, kv_heads, dim, q_heads = workload.shape
    batch = len(lengths)
    lens = integer(lengths)
    cache_k, cache_v, indptr, indices = page_map(lengths, page_size, kv_heads, dim,
                                                workload.variant == "shared")
    q = floating((batch, q_heads, dim), dtype=torch.float16)
    if workload.variant == "extreme":
        q.mul_(25)
        cache_k.mul_(10)
    scale = dim ** -0.5
    limit = max(1, max(lengths))
    args = {"q": q, "cache_k": cache_k, "cache_v": cache_v, "indptr": indptr, "indices": indices,
            "lengths": lens, "batch": batch, "page_size": page_size, "kv_heads": kv_heads,
            "q_heads": q_heads, "dim": dim, "scale": scale}

    def decode(dtype):
        positions = torch.arange(limit, device=q.device)
        valid = positions[None, :] < lens[:, None]
        index = (indptr[:-1, None] + positions // page_size).masked_fill(~valid, 0)
        pages = indices[index]
        k = cache_k[pages.long(), positions % page_size].to(dtype)
        v = cache_v[pages.long(), positions % page_size].to(dtype)
        grouped = q.to(dtype).view(batch, kv_heads, q_heads // kv_heads, dim)
        scores = torch.einsum("bhgd,bthd->bhgt", grouped, k) * scale
        mask = valid[:, None, None, :]
        scores = scores.masked_fill(~mask, -math.inf)
        prob = torch.softmax(scores, dim=-1).masked_fill(~mask, 0.0)
        result = torch.einsum("bhgt,bthd->bhgd", prob, v).reshape_as(q)
        lse = torch.logsumexp(scores, dim=-1).reshape(batch, q_heads) / math.log(2.0)
        return {"out": result.float(), "out_lse": lse.float()}

    return Exercise(args, {"out": output(tuple(q.shape)), "out_lse": output((batch, q_heads))},
                    decode, atol=3e-4, rtol=2e-4)


def recurrence(q, k, v, gate, state, scale, beta=None):
    result = []
    for t in range(q.shape[1]):
        key = k[:, t, ..., None]
        state = state * gate[:, t, ..., None, None].exp()
        val = v[:, t]
        if beta is not None:
            error = val - (key * state).sum(dim=-2)
            val = error * beta[:, t, ..., None]
        state = state + key * val.unsqueeze(-2)
        result.append((q[:, t, ..., None] * state).sum(dim=-2) * scale)
    return (torch.stack(result, dim=1) if result else v.clone()), state


def recurrent_inputs(shape, variant, delta, packed):
    batch, tokens, heads, key_dim, value_dim = shape
    tensor_shape = (tokens, heads) if packed else (batch, tokens, heads)
    q = floating((*tensor_shape, key_dim), scale=0.35)
    k = floating(q.shape, scale=0.2)
    if delta:
        k = torch.nn.functional.normalize(k, dim=-1)
    v = floating((*tensor_shape, value_dim), scale=0.4)
    g = floating(tensor_shape, scale=0.2).abs().add_(0.01).neg_()
    state = floating((batch, heads, key_dim, value_dim), scale=0.25)
    if variant in ("no_decay", "overwrite"):
        g.zero_()
    elif variant == "forget":
        g[:, tokens // 2].fill_(-math.inf)
    elif variant == "zero_state":
        state.zero_()

    args = {"q": q, "k": k, "v": v, "g": g}
    if delta:
        beta = torch.rand(tensor_shape, device="cuda")
        if variant in ("overwrite", "beta_zero"):
            beta.fill_(1.0 if variant == "overwrite" else 0.0)
        args["beta"] = beta
    args["initial"] = state
    return args


def recurrence_exercise(workload, kind):
    packed = kind == "packed"
    backward = kind == "backward"
    delta = kind in ("delta", "packed")
    if packed:
        lengths, heads, key_dim, value_dim = workload.shape
        batch, tokens = len(lengths), sum(lengths)
        bounds = offsets(lengths)
    else:
        batch, tokens, heads, key_dim, value_dim = workload.shape
    dims = (batch, tokens, heads, key_dim, value_dim)
    args = recurrent_inputs(dims, workload.variant, delta, packed)
    if packed:
        args["offsets"] = integer(bounds)
    if backward:
        args["grad_out"] = floating(args["v"].shape, scale=0.3)
        args["grad_final"] = floating(args["initial"].shape, scale=0.3)
        if workload.variant == "state_only":
            args["grad_out"].zero_()
        elif workload.variant == "output_only":
            args["grad_final"].zero_()
    scale = key_dim ** -0.5
    args.update(zip(("batch", "tokens", "heads", "key_dim", "value_dim"), dims))
    args["scale"] = scale

    def forward(dtype):
        q, k, v, g, initial = (args[name].to(dtype) for name in ("q", "k", "v", "g", "initial"))
        beta = args["beta"].to(dtype) if delta else None
        if packed:
            sequences = []
            states = []
            for number, (start, stop) in enumerate(zip(bounds, bounds[1:])):
                items = [x[None, start:stop] for x in (q, k, v, g)]
                o, h = recurrence(*items, initial[number:number + 1], scale, beta[None, start:stop])
                sequences.append(o[0])
                states.append(h[0])
            out, final = torch.cat(sequences), torch.stack(states)
        else:
            out, final = recurrence(q, k, v, g, initial, scale, beta)
        return {"out": out.float(), "final": final.float()}

    names = ("q", "k", "v", "g", "initial")

    def grad(dtype):
        with torch.enable_grad():
            variables = [args[name].detach().to(dtype).requires_grad_() for name in names]
            out, final = recurrence(*variables, scale)
            upstream = tuple(args[name].to(dtype) for name in ("grad_out", "grad_final"))
            gradients = torch.autograd.grad((out, final), variables, grad_outputs=upstream)
        return {f"d_{name}": value.float() for name, value in zip(names, gradients)}

    if backward:
        return Exercise(args, {f"d_{name}": output(args[name].shape) for name in names},
                        grad, atol=2e-4, rtol=3e-4)
    return Exercise(args, {"out": output(args["v"].shape), "final": output(args["initial"].shape)},
                    forward, atol=3e-5, rtol=2e-4)


def flashinfer_exercise(key, workload):
    if key in ("1", "2"):
        return paged_io(workload, append=key == "2")
    if key == "3":
        return merge_states(workload)
    return paged_decode(workload)


def make_exercise(platform, key, workload, seed):
    torch.manual_seed(seed)
    if platform == "flashinfer":
        return flashinfer_exercise(key, workload)
    if platform == "fla":
        kind = {"1": "gla", "2": "delta", "3": "packed", "4": "backward"}[key]
        return recurrence_exercise(workload, kind)
    from practice_library_reference import exercise_for

    return exercise_for(platform, workload)
