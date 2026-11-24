"""Build the selected local LeetCUDA binding and judge its full tensor output."""
import argparse
import ast
import os
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

from practice_leetcuda_cases import POINTWISE, leetcuda_cases
from practice_verdict import Judge, SubmissionError, fail
from practice_workloads import guarded_copy, verify_guards


def upstream_binding(source):
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "lib" for target in node.targets
        ):
            continue
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "load":
            options = {}
            for keyword in call.keywords:
                if keyword.arg == "extra_include_paths" and source.stem in ("mat_transpose", "hgemv"):
                    cutlass = os.environ.get("CUTLASS_REPO_PATH", "../../third-party/cutlass")
                    options[keyword.arg] = [str(Path(cutlass).resolve() / "include")]
                else:
                    options[keyword.arg] = ast.literal_eval(keyword.value)
            if source.stem == "sgemm":
                options["extra_ldflags"] = ["-lcublas"]
            return load(**options)
    raise ValueError(f"{source} 没有可复用的 lib = load(...) 配置")


def pointwise_input(number, shape, dtype, family):
    if number == 1:
        x = torch.tensor([-12, -4, -3, -2.999, -0.501, -0.5, -0.499, -0.001,
                          0, 0.001, 0.499, 0.5, 0.501, 2.999, 3, 12], device="cuda",
                         dtype=dtype).reshape(shape)
    elif number == 6:
        x = torch.zeros(shape, device="cuda", dtype=dtype)
    else:
        x = torch.randn(shape, device="cuda", dtype=dtype)
        if number == 7:
            x[:, ::2] *= 0.03125
    if family == "elementwise":
        y = -x if number == 2 else torch.randn_like(x)
        return (x, y), x + y
    op = {"sigmoid": torch.sigmoid, "relu": torch.relu, "elu": F.elu, "gelu": F.gelu,
          "swish": F.silu, "hardswish": F.hardswish, "hardshrink": F.hardshrink}[family]
    target = op(x, approximate="tanh") if family == "gelu" else op(x)
    return (x,), target


def row_input(number, shape, family, title, dtype):
    if number == 1:
        x = torch.zeros(shape, device="cuda", dtype=dtype)
    else:
        x = torch.randn(shape, device="cuda", dtype=dtype)
        if number == 2:
            x[::2] *= 0.125
        if number == 3:
            if family == "softmax" and title.startswith(("safe_", "online_")):
                x += 96
            elif family == "softmax":
                x += 16
            else:
                x *= 0.03125
    if number == 6:
        x.fill_(0.75)
    elif number == 7:
        x[:, ::2] *= 2
    output = torch.full_like(x, torch.nan)
    xf = x.float()
    if family == "softmax":
        return (x, output), xf.softmax(dim=-1).to(dtype), output
    gain, bias = 0.75, 0.125
    if family == "layer_norm":
        target = F.layer_norm(xf, (shape[-1],)) * gain + bias
        return (x, output, gain, bias), target.to(dtype), output
    target = xf * (xf.square().mean(dim=-1, keepdim=True) + 1e-5).rsqrt() * gain
    return (x, output, gain), target.to(dtype), output


def matrix_input(number, shape, family, dtype):
    if family == "mat_transpose":
        x = torch.arange(shape[0] * shape[1], dtype=dtype, device="cuda").reshape(shape)
        expected = x.T.contiguous()
        output = torch.full_like(expected, torch.nan)
        return (x, output), expected, output
    if family == "rope":
        tokens, width = shape
        x = torch.randn(shape, dtype=dtype, device="cuda")
        frequency = 10000 ** (-torch.arange(0, width, 2, device=x.device).float() / width)
        angles = torch.arange(tokens, device=x.device).float()[:, None] * frequency
        rotated = torch.view_as_complex(x.reshape(tokens, -1, 2)) * torch.polar(
            torch.ones_like(angles), angles)
        output = torch.full_like(x, torch.nan)
        return (x, output), torch.view_as_real(rotated).reshape(shape), output
    if family in ("hgemv", "sgemv"):
        m, k = shape
        n = 1
    else:
        m, n, k = shape
    a = torch.randint(-8, 9, (m, k), device="cuda").to(dtype) / 16
    b = torch.randint(-8, 9, (k, n), device="cuda").to(dtype) / 16
    if number == 4:
        a.zero_()
    expected = (a.float() @ b.float()).to(dtype)
    output = torch.full_like(expected, torch.nan)
    return (a, b, output), expected, output


def reduction_input(number, shape, name):
    dtype_name = name.removeprefix("block_all_reduce_sum_")
    if dtype_name.startswith("fp8_e4m3"):
        dtype = torch.float8_e4m3fn
    elif dtype_name.startswith("fp8_e5m2"):
        dtype = torch.float8_e5m2
    else:
        prefix = dtype_name.split("_")[0].split("x")[0]
        dtype = {"i8": torch.int8, "f32": torch.float32, "f16": torch.float16,
                 "bf16": torch.bfloat16}[prefix]
    out_dtype = torch.int32 if dtype == torch.int8 else torch.float32
    x = torch.randint(-4, 5, shape, device="cuda", dtype=torch.int32)
    if number == 1:
        x.zero_()
    elif number == 2:
        x[1] = -x[0]
    elif number == 4:
        x[:, 2::4] = 1
        x[:, 3::4] = -1
    if dtype != torch.int8:
        x = x.float() / 16
    values = x.to(dtype)
    expected = x.sum().to(out_dtype).reshape(1)
    return (values,), expected, None


def bindings(family, title):
    if family == "elementwise":
        return (title.replace("elementwise_", "elementwise_add_", 1),)
    if family == "softmax" and title.endswith("_per_tok"):
        return (title + "en",)
    if family == "dot_product":
        key = title.replace("dot_product_", "dot_prod_", 1)
        return (key + "_f32" if key in ("dot_prod_f32", "dot_prod_f32x4") else key,)
    if family == "block_all_reduce":
        if title.startswith("warp_"):
            return tuple("block_all_reduce_sum_" + dtype for dtype in ("f32_f32", "f16_f32", "i8_i32"))
        return (title.replace("block_all_reduce_", "block_all_reduce_sum_", 1),)
    if family == "mat_transpose":
        if "cute" in title:
            return ("mat_transpose_cute_row2col_reg", "mat_transpose_cute_col2row_reg")
        return (title.replace("mat_trans_", "mat_transpose_", 1).replace("{2d}", ""),) if "diagonal" in title else (
            title.replace("mat_trans_", "mat_transpose_", 1).replace("{2d}", "2d"),)
    if family == "hgemv":
        return (title.removesuffix("_kernel"),)
    if family == "sgemm":
        return ({
            "sgemm_t_8x8_sliced_k...bcf": "sgemm_t_8x8_sliced_k_f32x4_bcf",
            "sgemm_t_8x8_sliced_k...dbuf": "sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf",
            "sgemm_t_8x8_sliced_k16...dbuf": "sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf",
            "sgemm_t_8x8_sliced_k16...async": "sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async",
            "sgemm_wmma_m16n16k8...stages*": "sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages",
            "sgemm_wmma_m16n16k8...swizzle*": "sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem",
        }.get(title, title),)
    return (title,)


def setup(number, shape, family, title, dtype, binding):
    if family in POINTWISE:
        arguments, expected = pointwise_input(number, shape, dtype, family)
        output = torch.full_like(expected, torch.nan)
        return (*arguments, output), expected, output
    if family in ("mat_transpose", "rope", "hgemv", "sgemv", "sgemm"):
        return matrix_input(number, shape, family, dtype)
    if family == "block_all_reduce":
        return reduction_input(number, shape, binding)
    if family in ("softmax", "layer_norm", "rms_norm"):
        return row_input(number, shape, family, title, dtype)
    if family == "histogram":
        if number == 1:
            values = torch.full(shape, 17, dtype=torch.int32, device="cuda")
        elif number == 4:
            values = torch.arange(shape[0], dtype=torch.int32, device="cuda") * 53
        else:
            values = torch.randint(0, 43, shape, dtype=torch.int32, device="cuda")
            if number == 2:
                values *= 2
            if number == 5:
                values[::2] = 0
        return (values,), torch.bincount(values.to(torch.int64)).to(torch.int32), None
    if family == "embedding":
        vocab, length, width = shape
        table = torch.randn((vocab, width), device="cuda", dtype=dtype)
        ids = torch.randint(0, vocab, (length,), device="cuda", dtype=torch.int32)
        ids[:4] = torch.tensor([0, vocab-1, vocab-1, 0], device="cuda", dtype=torch.int32)
        expected = table[ids.long()]
        output = torch.full_like(expected, torch.nan)
        return (ids, table, output), expected, output

    # Quantized dyadic inputs let the FP16 multiply and small partial sums be exact.
    left = torch.randint(-8, 9, shape, device="cuda").to(dtype) / 16
    right = torch.randint(-8, 9, shape, device="cuda").to(dtype) / 16
    if number == 2:
        left[1] = left[0]
        right[1] = -right[0]
    expected = (left.float() * right.float()).sum().reshape(1)
    return (left, right), expected, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("title")
    parser.add_argument("--case", type=int)
    args = parser.parse_args()
    family = args.script.stem
    cases = list(enumerate(leetcuda_cases(family, args.title), start=1))
    if args.case is not None:
        cases = [(number, case) for number, case in cases if number == args.case]
    if not cases:
        raise ValueError("该用例编号不存在。")
    names = bindings(family, args.title)
    try:
        lib = upstream_binding(args.script)
        kernels = {name: getattr(lib, name) for name in names}
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        raise SubmissionError(f"CE · {error}") from error
    dtype = torch.float16 if family == "hgemv" or "_f16" in args.title else torch.float32
    tolerance = dict(atol=0, rtol=0)
    if family in POINTWISE - {"relu", "hardshrink", "elementwise"}:
        tolerance = dict(atol=0.001, rtol=0.004) if dtype == torch.float16 else (
            dict(atol=2e-6, rtol=2e-5)
        )
    if family == "gelu" and dtype == torch.float32:
        # CUDA fast-math tanhf and PyTorch round the approximation differently.
        tolerance = dict(atol=8e-6, rtol=1e-4)
    if family in ("softmax", "layer_norm", "rms_norm"):
        if dtype == torch.float32:
            tolerance = dict(atol=1e-5, rtol=3e-5)
        elif family == "softmax":
            tolerance = dict(atol=2e-5, rtol=0.008)
        else:
            tolerance = dict(atol=0.003, rtol=0.012)
    if family == "dot_product":
        tolerance = dict(atol=1e-4, rtol=1e-6)
    if family in ("hgemv", "sgemv", "sgemm", "block_all_reduce"):
        tolerance = dict(atol=2e-4, rtol=3e-4)
    if family == "block_all_reduce" and args.title.endswith("_bf16"):
        # The kernel reduces partial sums in BF16 before returning an FP32 scalar.
        tolerance = dict(atol=0.0625, rtol=0.008)
    if family == "rope":
        # Fast CUDA trigonometry accumulates phase error over long token positions.
        tolerance = dict(atol=0.0025, rtol=5e-4)
    judge = Judge(len(cases))
    with torch.no_grad():
        for number, (label, shape) in cases:
            for name, evaluate in kernels.items():
                torch.manual_seed(2026 + number)
                inputs, expected, output = setup(number, shape, family, args.title, dtype, name)
                if family == "sgemm" and name.startswith("sgemm_wmma_"):
                    inputs = (*inputs, 2, False, 1024)
                guards = {}
                if family == "rope":
                    guarded_input, *input_edges = guarded_copy(inputs[0], 8192)
                    output, *output_edges = guarded_copy(output, 8192)
                    guards = {"输入": input_edges, "输出": output_edges}
                    inputs = (guarded_input, output)
                result = judge.run(number, evaluate, inputs, torch.cuda.synchronize)
                verify_guards(judge, number, guards)
                actual = result if output is None else output
                try:
                    torch.testing.assert_close(actual, expected, **tolerance)
                except AssertionError as error:
                    judge.wrong(number, f"{name} · {label} {shape}\n{error}")
            print(f"{label} {shape}", flush=True)
            judge.accept(number)
    judge.finish()


if __name__ == "__main__":
    try:
        main()
    except SubmissionError as error:
        sys.exit(fail(error))
    except ValueError as error:
        sys.exit(f"评测错误 · {error}")
