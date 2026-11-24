"""Deterministic offline inputs for XPUOJ statements that publish a PyTorch baseline."""
from contextvars import ContextVar
import math
import re
from typing import List, Optional

import torch

from practice_workloads import BASE_SEED, Case


# Run a larger workload as well for representative bandwidth- and compute-bound families.
LARGE_CASES = {
    34, 35, 36, 61, 62, 63, 64, 67, 75, 84, 87, 100, 102, 103,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
}


def case_numbers(key):
    return (1, 2, 3) if key in LARGE_CASES else (1, 2)


input_scale = ContextVar("xpuoj_input_scale", default=0.25)


def case_plan(key):
    plan = []
    shapes = case_numbers(key)
    scales = ((0.25, "题面参考用例"), (0.015625, "小幅值"),
              (0.75, "较宽幅值"), (0.25, "独立随机种子"))
    for profile, (scale, label) in enumerate(scales):
        for variant in shapes:
            number = len(plan) + 1
            details = f"{label} · 形状方案 {variant} · 种子 {BASE_SEED + number}"
            if profile:
                details += " · 缓冲区守卫"
            def make(variant=variant, scale=scale, number=number):
                torch.manual_seed(BASE_SEED + number)
                token = input_scale.set(scale)
                try:
                    return case_inputs(key, variant)
                finally:
                    input_scale.reset(token)
            plan.append(Case(number, details, make, guarded=profile > 0))
    return plan


def baseline(problem):
    path = problem.path
    if int(problem.key) >= 10000:
        path = next(problem.pack.glob("zh_CN/06_*.md"))
        text = path.read_text()
    else:
        text = path.read_text()
        heading = list(re.finditer(r"(?m)^##[ \t]+PyTorch[ \t]+参考实现[ \t]*$", text))[-1]
        text = text[heading.end():]
    source = re.search(r"(?ms)^\x60{3}python[ \t]*\r?\n(.*?)\x60{3}", text)[1]
    scope = {
        "torch": torch, "math": math, "F": torch.nn.functional,
        "List": List, "Optional": Optional, "KernelArg": torch.Tensor | int | float,
        "_cdiv": lambda value, divisor: (value + divisor - 1) // divisor,
        "_scaled_dot_product_attention": cache_attention,
        "_rotate_half": rotate_half, "_apply_rope_with_pos_ids": apply_rope_with_pos_ids,
        "_mla_rope_quantize": mla_rope_quantize, "_log_linear_attention": log_linear_attention,
        "_ACTS": ("identity", "relu", "relu_square", "gelu"),
        "_forward_baseline": gemm_activation_derivative,
    }
    exec(compile(source, str(problem.path), "exec"), scope)
    return scope["baseline"]


def random(shape, dtype=torch.bfloat16):
    return torch.randn(shape, dtype=torch.float32, device="cuda").mul_(input_scale.get()).to(dtype)


def zeros(shape, dtype=torch.bfloat16):
    return torch.zeros(shape, dtype=dtype, device="cuda")


def indices(shape, limit, dtype=torch.int64):
    return torch.randint(limit, shape, dtype=dtype, device="cuda")


def constant(data, dtype=torch.int32):
    return torch.tensor(data, dtype=dtype, device="cuda")


def case_inputs(key, variant):
    # The two shapes vary row lengths, tile tails, broadcasting, and optional paths.
    # These are local fixtures. The statement supplies the ABI and the numerical reference.
    batch, m, n, k = ((2, 5, 11, 7), (3, 33, 65, 31), (4, 513, 1024, 768))[variant-1]
    if variant == 3:
        m, n, k = {
            34: (12288, 12288, 1), 61: (2048, 1024, 1536),
            62: (2048, 1024, 1536), 63: (2048, 1024, 1536), 64: (2048, 1024, 1536),
            75: (2049, 385, 1), 84: (16384, 1536, 1), 87: (16384, 1536, 1),
            100: (8192, 16384, 1), 102: (8193, 4097, 1), 103: (4097, 3072, 1),
        }.get(key, (m, n, k))
    float32, half, bf16 = torch.float32, torch.float16, torch.bfloat16

    if key >= 10000:
        return python_cases(key, variant)
    if key in (37, 38, 39, 40, 58):
        return paged_case(key, variant)
    if key in (121, 122):
        return varlen_case(key, variant)
    if key == 34:
        return [random((m, 2*n)), zeros((m, n)), m, n], (1,)
    if key in (35, 36, 88, 128, 129):
        q_heads, kv_heads, dim = 4, 2, 64
        kv_len = m
        if key in (35, 36) and variant > 1:
            m, kv_len = (m, m + 7) if variant == 2 else (128, 256)
            q_heads, kv_heads = (4, 1) if variant == 2 else (16, 8)
        q, kv = (m, q_heads, dim), (kv_len, kv_heads, dim)
        if key == 35:
            return [random(q), random(kv), random(kv), zeros(q), m, kv_len, q_heads, kv_heads, dim], (3,)
        if key == 36:
            return [random(q), random(kv), random(kv), zeros(q), zeros((m, q_heads), float32),
                    m, kv_len, q_heads, kv_heads, dim], (3, 4)
        if key == 88:
            return [random((batch, *q)), random((batch, *kv)), random((batch, *kv)),
                    zeros((batch, *q)), batch, m, q_heads, kv_heads, dim, variant-1, dim**-0.5], (3,)
        if key == 128:
            return [random((batch, m, 3, q_heads, dim)), zeros((batch, *q)),
                    batch, m, q_heads, dim], (1,)
        return [random((batch, *q)), random((batch, m, 2, kv_heads, dim)),
                zeros((batch, *q)), batch, m, q_heads, kv_heads, dim], (2,)
    if key in (41, 42):
        head_dim, q_heads, kv_heads = 32, 4, 2
        offset = variant if key == 42 else 0
        q, kshape = (batch, m, q_heads, head_dim), (batch, m, kv_heads, head_dim)
        angles = random((m + offset, head_dim // 2), float32).repeat(1, 2)
        args = [random(q), random(kshape), angles.cos().to(bf16), angles.sin().to(bf16),
                zeros(q), zeros(kshape), batch, m, q_heads, kv_heads, head_dim]
        if key == 42:
            args.append(offset)
        return args, (4, 5)
    if key == 43:
        return [random((m, n)), random((n,)), zeros((m, n)), m, n], (2,)
    if key == 44:
        heads, dim = 4, 32
        length = 2 * m + 1
        return [random((length, 3*heads*dim)), constant([0, m, length]), m+1,
                heads, dim, zeros((length, heads*dim)), 2], (5,)
    if key in (45, 46, 47, 48, 49, 50, 54, 55, 56, 57, 65, 66, 67):
        return convolution_case(key, variant)
    if key in (59, 60, 68, 74):
        square = random((m, m), torch.complex128 if key == 60 else float32)
        if key == 59:
            return [square + square.T, random((m, n), float32), random((m, n), float32),
                    zeros((m, n), float32), 1.25, 0.5, m, n], (3,)
        if key == 60:
            matrix = torch.complex(random((m, n), torch.float64), random((m, n), torch.float64))
            square += 1j * random((m, m), torch.float64)
            return [square, matrix, zeros((m, n), torch.complex128), 0.75, variant-1, m, n], (2,)
        rect = random((m, k), float32)
        base = random((m, m), float32)
        if key == 68:
            return [rect, base, zeros((m, m), float32), 1.25, 0.5, m, k], (2,)
        return [rect, random((m, k), float32), base, zeros((m, m), float32), 1.25, 0.5, m, k], (3,)
    if key in (61, 62, 63, 64, 69, 70, 71, 72, 79, 80, 132, 133, 134, 135, 136, 137,
               142, 143, 144):
        return matrix_case(key, variant, m, n, k)
    if key == 73:
        return [random((m, k)), random((k,)), zeros((m,), float32), 0.75, m, k], (2,)
    if key == 75:
        values = random((m, n))
        values[0] += 15
        return [values, zeros((m,)), m, n], (1,)
    if key == 76:
        return [random((m, n)), random((m, n)), zeros((), float32),
                1.0 if variant == 1 else 0.25, m, n], (2,)
    if key == 77:
        return [random((m, n)), indices((m, n), 2).to(bf16), zeros((), float32), m, n], (2,)
    if key == 78:
        return [random((m, n)), indices((m,), n), zeros((), float32), 0.1, m, n], (2,)
    if key == 81:
        return [random((m, k)), random((n, k)), None, indices((m,), n),
                zeros((), float32), m, k, n], (4,)
    if key == 82:
        labels = indices((m,), n)
        labels[0] = -100
        return [random((m, n), half), labels, zeros((m,), float32), m, n, 0.1, -100], (2,)
    if key == 83:
        return [random((m, n), float32), zeros((m, 4), float32),
                zeros((m, 4), torch.int32), m, n, 4], (1, 2)
    if key in (84, 85):
        args = [random((m, n)), random((n,), float32), random((n,), float32),
                zeros((m, n)), m, n]
        if key == 85:
            args.append(variant-1)
        return args, (3,)
    if key == 86:
        return [random((m, n)), random((m, n)), random((n,)), zeros((m, n), float32),
                m, n, 0.125], (1, 3)
    if key in (87, 111):
        return [random((m, n)), random((n,)), zeros((m, n)), m, n], (2,)
    if key == 89:
        return [random((batch, m, k)), random((n, k)), random((n,)), zeros((batch, m, n)),
                batch, m, k, n], (3,)
    if key == 90:
        length, heads, dim = 2*m+1, 4, 8
        return [random((length, heads, dim), float32), random((4, 3, heads, dim), float32),
                constant([0, m, length]), zeros((length, 3, heads, dim), float32),
                length, 2, heads, dim, 4, 3], (3,)
    if key == 91:
        return [random((m, n)), zeros((m, n)), m, n, 0.25 * (variant-1), 12345], (1,)
    if key == 92:
        return [random((m, n)), random((m, n)), random((m, n)), zeros((m, n)),
                m, n, 0.125 * (variant-1), 9876, 0.75, variant-1], (3,)
    if key == 93:
        return [random((n, k)), indices((m,), n, torch.int32), zeros((m, k)), n, m, k], (2,)
    if key == 94:
        step, padding, kernel, channels = variant, 1, 3, 4
        length, width = 2*m+1, n
        out_len = (m-1)//step + m//step + 2
        out_width = (width-1)//step + 1
        return [random((length, width, channels)), constant([0, m, length]),
                zeros((out_len, out_width, channels*kernel**2)), zeros((3,), torch.int32),
                zeros((), torch.int32), 2, width, channels, kernel, step, padding], (2, 3, 4)
    if key == 95:
        return [random((n, k), float32), random((m, k), float32), indices((m,), n, torch.int32),
                zeros((m, k), float32), n, m, k], (3,)
    if key == 96:
        length = 64 if variant == 1 else 127
        size = (batch, length)
        return [random(size, float32), random(size, float32), zeros(size, float32),
                zeros(size, float32), batch, length], (2, 3)
    if key in (97, 99):
        return quantized_case(key, variant)
    if key == 98:
        return grouped_case(key, variant)
    if key == 100:
        return [random((m, n), float32), zeros((n, m), float32), m, n], (1,)
    if key == 101:
        return [random((m, n), float32), torch.ones(n, device="cuda") * 64,
                zeros((m, n), torch.int8), zeros((n,), float32), m, n], (2, 3)
    if key == 102:
        return [random((m*n,), float32), zeros((m*n,), float32), m, n], (1,)
    if key == 103:
        return [random((m, n), float32), zeros((m, n), float32), m, n], (1,)
    if key == 104:
        return [random((m, n), float32), indices((n,), 2).bool(),
                torch.arange(0, m, dtype=torch.int32, device="cuda").flip(0),
                zeros((m, n), float32), m, n, variant-1], (3,)
    if key == 105:
        bits, size_k, size_n = (4, 64, 64) if variant == 1 else (8, 128, 128)
        pack = 32//bits
        words = torch.randint(-2147483648, 2147483647, (size_k, size_n//pack),
                              device="cuda", dtype=torch.int32)
        return [words, zeros((size_k//16, size_n*16//pack), torch.int32),
                size_k, size_n, bits], (1,)
    if key == 106:
        return [random((m, n), float32), random((m, n), float32), random((m, n), float32),
                zeros((), float32), 0.75, m, n], (3,)
    if key == 107:
        return [random((m, n), float32), random((m, n), float32), 2*indices((m,), 2)-1,
                zeros((), float32), 0.1, m, n], (3,)
    if key in (108, 109):
        shape = (batch, 3, m, n)
        return [random(shape, float32), indices(shape, 2).float(), zeros((), float32),
                1.0, *shape], (2,)
    if key == 110:
        return [random((batch, m, k)), random((n, k)), random((n,)),
                random((k, n)), random((k,)), zeros((batch, m, k)),
                batch, m, k, n, k], (5,)
    if key in (112, 113, 114, 115, 116):
        return rope_case(key, variant, batch, m)
    if key == 117:
        hidden, kernel = 16, 5
        return [random((batch, m, 2*hidden)), random((kernel, hidden)), random((hidden,)),
                random((hidden,)), random((hidden,)), indices((batch, m), 2).bool(),
                zeros((batch, m, hidden)), batch, m, hidden, kernel], (6,)
    if key == 118:
        heads, dim = 4, 16
        return [random((batch, m, 3*heads*dim), half), random((heads, dim), half),
                random((heads, dim), half), zeros((batch, heads, m, dim), half),
                batch, m, heads, dim, m], (0, 3)
    if key == 119:
        experts, width, inner, chunk = 2, 48, 32, 16
        counts = [m, m+1]
        total = sum(counts)
        return [random((total, inner), torch.float8_e4m3fn),
                (indices((experts, width, inner//2), 256, torch.int16)-128).to(torch.int8),
                constant([0.5], float32), random((experts, inner//chunk, width)).abs(),
                constant([0, m]), constant([[width, count, inner] for count in counts]),
                constant([[inner, 1, 0]]*experts, torch.int64),
                constant([[inner//2, 1, 0]]*experts, torch.int64),
                constant([[width, 1, 0]]*experts, torch.int64),
                constant([[width, 1, 0]]*experts, torch.int64),
                zeros((total, width)), total, inner, width, experts, chunk], (10,)
    if key == 120:
        return [random((m, k)), random((4, n, k)), indices((m, 2), 4, torch.int32),
                zeros((m*2, n)), m, k, n, 4, 2], (3,)
    if key == 123:
        heads, dim, levels = 4, 32, 4
        return [random((batch, m, 1, dim), half), random((batch, m, 1, dim), half),
                random((batch, m, heads, dim), half), random((batch, m, heads), half),
                random((batch, m, heads, levels), half), zeros((batch, m, heads, dim), half),
                batch, m, heads, dim, levels], (5,)
    if key in (124, 125, 127):
        threshold = (indices((m,), 4, torch.int32)+1 if key == 124 else
                     constant([0.05 if index % 2 else 0.25 for index in range(m)], float32))
        return [random((m, n), float32), threshold, zeros((m, 1), torch.int32), m, n], (2,)
    if key == 126:
        heads, kv_heads, dim, cached, query_len = 4, 2, 32, m, variant
        q = batch, query_len, heads, dim
        cache = batch, 2*cached, kv_heads, dim
        new = batch, query_len, kv_heads, dim
        return [random(q), random(cache), random(cache), random(new), random(new), zeros(q),
                batch, query_len, cache[1], heads, kv_heads, dim, cached], (1, 2, 5)
    if key == 130:
        labels = indices((batch, m), n)
        labels[0, 0] = -100
        return [random((batch, m, n)), labels, zeros(()), batch, m, n], (2,)
    if key == 131:
        mask = indices((batch, m), 2).bool()
        count = int(mask.sum().item())
        return [random((batch, m, n)), mask, zeros((count, n)), batch, m, n, count], (2,)
    if key == 138:
        return [random((batch, m, k)), random((batch, k, n)), zeros((batch, m, n)),
                batch, m, k, n], (2,)
    if key == 139:
        return [random((batch, m)), random((batch, n)), zeros((batch, m, n)), batch, m, n], (2,)
    if key == 140:
        return [random((batch, m, 3, k)), random((batch, 3, k, n)), zeros((batch, m, n)),
                batch, m, 3, k, n], (2,)
    if key == 141:
        return [random((batch, n, n)), zeros((batch,)), batch, n], (1,)
    raise ValueError(f"没有为 XPUOJ #{key} 配置本地用例。")


def convolution_case(key, variant):
    batch, channels, out_channels = 2, 4, 6
    stride = 1 if variant == 1 else 2
    padding, width = 1, 11 if variant == 1 else 19
    is_3d = key in (47, 48, 49, 67)
    transpose = key in (45, 46, 49)
    depthwise = key == 50
    gradient = key in (55, 56, 57)
    has_broadcast = key in (46, 48)
    dtype = torch.float32 if key in (45, 47, 49, 54, 56, 65, 66) else (
        torch.float16 if key in (50, 55) else torch.bfloat16
    )
    spatial = (5, 7, width) if is_3d else (7, width)
    if key == 67 and variant == 3:
        batch, channels, out_channels = 1, 32, 64
        stride, spatial = 1, (20, 40, 40)
    kernel = (3,) * len(spatial)
    output_padding = stride-1
    if transpose:
        output_spatial = tuple((size-1)*stride - 2*padding + 3 + output_padding for size in spatial)
        weight_shape = channels, out_channels, *kernel
    else:
        output_spatial = tuple((size + 2*padding - 3)//stride + 1 for size in spatial)
        weight_shape = (channels, 1, *kernel) if depthwise else (out_channels, channels, *kernel)
    if depthwise:
        out_channels = channels
    if gradient:
        return [random((batch, out_channels, *output_spatial), dtype),
                random(weight_shape, dtype), zeros((batch, channels, *spatial), dtype),
                batch, channels, out_channels, *spatial, *kernel, stride, padding], (2,)
    in_shape = batch, channels, *spatial
    out_shape = batch, out_channels, *output_spatial
    if key == 54:
        inputs = [indices(in_shape, 16, torch.int8)-8, indices(weight_shape, 16, torch.int8)-8]
    else:
        inputs = [random(in_shape, dtype), random(weight_shape, dtype)]
    inputs.append(random((out_channels,), dtype))
    if has_broadcast:
        inputs.append(random(out_shape, dtype))
    outputs = (len(inputs),)
    scalars = [batch, channels] + ([] if depthwise else [out_channels])
    scalars += [*spatial, *kernel, stride, padding]
    if transpose:
        scalars.append(output_padding)
    return [*inputs, zeros(out_shape, dtype), *scalars], outputs


def matrix_case(key, variant, m, n, k):
    dtype = {
        69: torch.complex128, 70: torch.complex128, 72: torch.complex64,
    }.get(key, torch.float32 if key <= 72 else torch.bfloat16)
    if key in (69, 70, 72):
        def complex_matrix(shape):
            real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
            return torch.complex(random(shape, real_dtype), random(shape, real_dtype))

        return [complex_matrix((m, k)), complex_matrix((k, n)), zeros((m, n), dtype),
                m, n, k], (2,)
    a, b, out = random((m, k), dtype), random((k, n), dtype), zeros((m, n), dtype)
    if key == 71:
        return [a, b, out, m, n, k], (2,)
    bias = random((n,), dtype)
    if key == 61:
        return [a, b, bias, out, m, n, k], (3,)
    if key in (62, 64, 144):
        return [a, b, bias, out, m, k, n], (3,)
    if key == 63:
        return [a, b, bias, out, 0.75, m, k, n], (3,)
    if key == 79:
        return [a, b.T.contiguous(), random((m, n)), out, m, k, n], (3,)
    if key == 80:
        return [a, b.T.contiguous(), random((m, n)), zeros((m, n)), out,
                m, n, k, variant-1], (3, 4)
    if key == 132:
        return [a, b, out, m, k, n, 8 if variant == 1 else 4], (2,)
    if key == 133:
        return [a, b, random((k,)), out, m, k, n, 1e-5], (3,)
    if key == 135:
        return [a, b, random((k,)), random((k,)), out, m, k, n, 1e-5], (4,)
    if key in (134, 136, 137):
        return [a, b, bias, random((m, n)), out, m, k, n], (4,)
    if key == 142:
        return [a, b, bias, random((n,), torch.float32), out, m, k, n], (4,)
    return [a, b.T.contiguous(), indices((m,), n), zeros((), torch.float32), m, k, n], (3,)


def quantized_case(key, variant):
    m, n, k, group = (3, 16, 32, 16) if variant == 1 else (17, 64, 128, 32)
    half = torch.float16
    def packed(shape):
        return (indices(shape, 256, torch.int16)-128).to(torch.int8)
    def scales(shape):
        return random(shape, half).abs() * 0.125
    if key == 97:
        return [random((m, k), half), packed((k, n//2)), scales((k//group, n)),
                zeros((m, n), half), m, n, k, group], (3,)
    return [packed((m, k//2)), scales((m, k//group)), packed((n, k//2)),
            scales((k//group, n)), packed((m, group)), scales((m, 1)),
            packed((group, n)), scales((1, n)), zeros((m, n), half),
            m, n, k, group, group], (8,)


def rope_case(key, variant, batch, length):
    dim, heads, kv_heads = 32, 4, 2
    torch_float16, torch_float32 = torch.float16, torch.float32
    if key in (115, 116):
        shape = length, batch, heads, dim
        out_shape = shape if variant == 1 else (batch, length, heads, dim)
        freq = random((length, 1, 1, dim//2), torch_float32).repeat(1, 1, 1, 2)
        if key == 115:
            return [random(shape, torch_float16), freq, zeros(out_shape, torch_float16),
                    *shape, variant-1], (2,)
        return [random(shape, torch_float16), freq.cos().to(torch_float16),
                freq.sin().to(torch_float16), zeros(out_shape, torch_float16),
                *shape, variant-1], (3,)
    q, k = (length, heads, dim), (length, kv_heads, dim)
    positions = indices((length,), 2*length, torch.int32)
    if key == 112:
        return [random(q), random(k), positions, zeros(q), zeros(k),
                length, heads, kv_heads, dim, dim, variant-1], (3, 4)
    if key == 114:
        k = (length, dim)
    angles = random((2*length, dim//2), torch_float32)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    args = [random(q), random(k), random(q), random(k), cache, positions,
            zeros(q, torch_float32), zeros(k, torch_float32), zeros(q, torch_float32),
            zeros(k, torch_float32), length, heads]
    if key == 113:
        args.append(kv_heads)
    return [*args, dim, dim, variant-1], (6, 7, 8, 9)

def rotate_half(value):
    left, right = value.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rotation(value, cos, sin, interleaved):
    if interleaved:
        left, right = value[..., ::2], value[..., 1::2]
        return torch.stack((left * cos - right * sin, right * cos + left * sin), dim=-1).flatten(-2)
    left, right = value.chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


def apply_rope_with_pos_ids(q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta):
    freq = rope_theta ** (-torch.arange(0, rotary_dim, 2, device=q.device).float() / rotary_dim)
    angles = pos_ids.float()[:, None, None] * freq / rope_scale
    return tuple(
        torch.cat((apply_rotation(value[..., :rotary_dim].float(),
                                  angles.cos(), angles.sin(), interleave).to(value.dtype),
                   value[..., rotary_dim:]), dim=-1)
        for value in (q, k)
    )


def mla_rope_quantize(q, k, q_nope, k_nope, cache, pos_ids, is_neox, quant_scale_q, quant_scale_kv):
    cos, sin = cache[pos_ids.long()].chunk(2, dim=-1)

    def rotate(value):
        coefficients = (cos, sin) if value.ndim == 2 else (cos[:, None], sin[:, None])
        return apply_rotation(value.float(), *coefficients, not is_neox)

    def quantize(value, scale):
        limit = torch.finfo(torch.float8_e4m3fn).max
        return (value.float() * scale).clamp(-limit, limit).to(torch.float8_e4m3fn)

    return (quantize(rotate(q), quant_scale_q), quantize(rotate(k), quant_scale_kv),
            quantize(q_nope, quant_scale_q), quantize(k_nope, quant_scale_kv))


def gemm_activation_derivative(a, b, pre_activation, activation):
    value = pre_activation.float().detach().requires_grad_(True)
    with torch.enable_grad():
        if activation == "identity":
            activated = value
        elif activation == "relu":
            activated = value.relu()
        elif activation == "relu_square":
            activated = value.relu().square()
        else:
            activated = torch.nn.functional.gelu(value, approximate="tanh")
        derivative, = torch.autograd.grad(activated.sum(), value)
    return (a.float() @ b.float().T) * derivative, activated.detach()


def cache_attention(query, kv, head_dim, causal, selected=None):
    kv = kv.repeat_interleave(query.shape[0] // kv.shape[0], dim=0).float()
    scores = (query.float() @ kv.transpose(-2, -1)) / math.sqrt(head_dim)
    if selected is not None:
        allowed = (torch.arange(kv.size(1), device=query.device) == selected[..., None]).any(dim=-2)
        scores = scores.masked_fill(~allowed, -torch.inf)
    elif causal:
        scores = scores.masked_fill(torch.ones_like(scores, dtype=torch.bool).triu(
            diagonal=1 + kv.size(1) - query.size(1)), -torch.inf)
    lse = scores.logsumexp(dim=-1)
    return scores.softmax(dim=-1).nan_to_num() @ kv, lse.masked_fill(lse == -torch.inf, torch.inf)


def log_linear_attention(q, k, v, g, level_scales):
    gate = g.float().transpose(1, 2).cumsum(dim=-1)
    positions = torch.arange(q.size(1), device=q.device)
    i, j = positions[:, None], positions[None, :]
    decay = (gate[..., :, None] - gate[..., None, :]).masked_fill(j > i, -torch.inf).exp()
    scales = level_scales.float().permute(0, 2, 3, 1)
    weights = scales[:, :, 0, :, None] * (i == j)
    for level in range(1, scales.size(2)):
        half = 1 << (level - 1)
        split = i - i % half
        mask = (i % (2 * half) >= half) & (split - half <= j) & (j < split)
        weights = weights + scales[:, :, level, :, None] * mask
    kernel = torch.einsum("bihd,bjhd->bij", q.float(), k.float())
    return torch.einsum("bhij,bjhd->bihd", decay * weights * kernel[:, None], v.float())


def paged_case(key, variant):
    batch, q_heads, kv_heads = 2, 8, 2
    page, pages = (64 if key == 58 else 16), 2
    dim = 128 if key in (37, 58) else 64
    lens = [page-1, page+variant]
    block_table = constant([[1, 0], [3, 2]])
    cache = batch * pages, kv_heads, page, dim
    q = batch, q_heads, dim
    if key == 37:
        return [random(q), random(cache), random(cache), zeros(q), dim**-0.5,
                zeros((128000,), torch.uint8), max(lens), constant(lens).reshape(batch, 1, 1, 1),
                block_table, batch, q_heads, kv_heads, dim, page, pages], (3,)
    if key == 38:
        query_lens = [variant, variant+1]
        packed = sum(query_lens), q_heads, dim
        return [random(packed), random((batch*pages, 2, q_heads, page, dim)),
                zeros((1024*1024,), torch.uint8), block_table, constant(lens),
                constant([0, query_lens[0], sum(query_lens)]),
                constant([0, lens[0], sum(lens)]), zeros(packed), dim**-0.5, 1.0,
                batch, q_heads, dim, page, pages, packed[0], max(query_lens), max(lens), -1], (7,)
    if key == 39:
        return [random(q), random((batch*pages, 2, kv_heads, page, dim)),
                block_table, constant(lens), zeros(q), batch, q_heads, kv_heads, dim,
                page, pages, max(lens)], (4,)
    transposed_cache = batch*pages, page, kv_heads, dim
    if key == 40:
        q = batch, 1, q_heads, dim
        return [random(q), random(transposed_cache), random(transposed_cache), zeros(q),
                block_table, constant(lens).reshape(batch, 1), kv_heads, page, batch,
                q_heads, dim, pages], (3,)
    q = batch, variant, q_heads, dim
    return [random(q), random(transposed_cache), block_table, constant(lens),
            zeros((1, 1), torch.int32), zeros((batch+1,), torch.int32),
            dim**-0.5, variant-1, 0, zeros((1,), torch.int32), zeros(q),
            zeros((batch, q_heads, variant), torch.float32), *q,
            batch*pages, page, kv_heads, pages, 0], (10, 11)


def varlen_case(key, variant):
    heads, kv_heads, dim = 4, (4 if key == 122 else 2), 32
    first, second = 3+variant, 7+variant
    q_lens = constant([0, first, first+second])
    kv_lens = constant([0, first+1, first+second+2])
    q = first+second, heads, dim
    kv = first+second+2, kv_heads, dim
    if key == 121:
        return [random(q), random(kv), random(kv), q_lens, kv_lens, zeros(q),
                q[0], heads, kv_heads, dim, second+1, variant-1], (5,)
    return [random(q), random(kv), random(kv), random(q), q_lens, kv_lens,
            zeros(q), zeros(kv), zeros(kv), 2, heads, kv_heads, dim], (6, 7, 8)


def grouped_case(key, variant):
    sizes = ((3, 5, 4), (7+variant, 9, 6), (2, 8, 3))
    lengths = [[m*k for m,k,n in sizes], [k*n for m,k,n in sizes], [m*n for m,k,n in sizes]]
    offsets = [constant([0, row[0], row[0]+row[1], sum(row)]) for row in lengths]
    m, k, n = [constant(column) for column in zip(*sizes)]
    return [random((sum(lengths[0]),), torch.float32), random((sum(lengths[1]),), torch.float32),
            zeros((sum(lengths[2]),), torch.float32), m, k, n, *offsets, len(sizes)], (2,)


def python_cases(key, variant):
    half = torch.float16
    if key == 10001:
        hidden, expanded, experts, tile = 32, 64, 3, 128
        final_group = 2 + variant
        starts = constant([0, 5, 5])
        count = 5 + final_group
        return [random((count, hidden), half), random((experts, expanded, hidden), half),
                random((experts, expanded, hidden), half), random((experts, hidden, expanded), half),
                random((count,), half).abs(), constant([5, 0, final_group]), starts,
                constant([0, tile, tile]), constant([0, 2, -1, -1]),
                zeros((count, expanded), half), zeros((count, hidden), half),
                count, hidden, expanded, experts, tile], (9, 10)
    if key == 10002:
        batch, heads, kv_heads, context, dim, pos_dim = 2, 8, 2, 33+variant*16, 512, 64
        return [random((batch, heads, dim), half), random((batch, heads, pos_dim), half),
                random((batch, context, kv_heads, dim), half),
                random((batch, context, kv_heads, pos_dim), half),
                zeros((batch, heads, dim), half), batch, heads, kv_heads, context, dim, pos_dim], (4,)
    batch, heads, q_heads, dim, block = 2, 2, 4, 64, 32
    length = block * variant + 7
    shape = batch, length, heads, dim
    qshape = batch, length, q_heads, dim
    block_ids = torch.zeros((batch, length, heads, 2), dtype=torch.int32, device="cuda")
    block_ids[..., 1] = -1
    block_ids[:, block:, :, 1] = 1
    return [random(qshape, half), random(shape, half), random(shape, half),
            block_ids, zeros(qshape, half), batch, length, heads, q_heads, dim, 2, block, 1], (4,)
