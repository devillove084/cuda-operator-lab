"""Validate the editable native FFPA extension on large-head prefill and split-KV decode."""
from statistics import median
import sys

import torch
import torch.nn.functional as F
from ffpa_attn import ffpa_attn_func
from ffpa_attn.functional import CUDABackend

key, action = sys.argv[1:]
decode = key in ("192", "193")
acc = "f16" if key == "190" else "f32"
backend = CUDABackend(forward=True, acc=acc, stages=2)
torch.manual_seed(2026)
for head_dim, length in ((320, 512), (512, 1024)):
    q_len = 1 if decode else length
    q = torch.randn((1, 4, q_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 4, length, head_dim), device=q.device, dtype=q.dtype)
    v = torch.randn_like(k)

    def evaluate():
        return ffpa_attn_func(q, k, v, forward_backend=backend)

    def reference():
        return F.scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(evaluate(), reference(), rtol=1e-2, atol=1e-2)
    print(f"PASS CUDA FFPA Nq={q_len} Nkv={length} D={head_dim} acc={acc}", flush=True)
    if action == "bench":
        for label, call in (("FFPA native CUDA", evaluate), ("PyTorch SDPA", reference)):
            for _ in range(8):
                call()
            torch.cuda.synchronize()
            times = []
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            for _ in range(20):
                start.record()
                call()
                stop.record()
                stop.synchronize()
                times.append(start.elapsed_time(stop) * 1000)
            print(f"{label} p50 {median(times):.3f} µs")
