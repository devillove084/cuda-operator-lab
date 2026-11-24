"""Build the selected local LeetCUDA binding and judge its full tensor output."""
import argparse
import ast
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

from practice_leetcuda_cases import POINTWISE, leetcuda_cases
from practice_verdict import Judge, SubmissionError, fail


def upstream_binding(source):
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "lib" for target in node.targets
        ):
            continue
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "load":
            return load(**{keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords})
    raise ValueError(f"{source} 没有可复用的 lib = load(...) 配置")


def pointwise_input(number, shape, dtype, family):
    if number == 1:
        x = torch.tensor([-12, -4, -3, -2.999, -0.501, -0.5, -0.499, -0.001,
                          0, 0.001, 0.499, 0.5, 0.501, 2.999, 3, 12], device="cuda",
                         dtype=dtype).reshape(shape)
    else:
        x = torch.randn(shape, device="cuda", dtype=dtype)
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


def setup(number, shape, family, title, dtype):
    if family in POINTWISE:
        arguments, expected = pointwise_input(number, shape, dtype, family)
        output = torch.full_like(expected, torch.nan)
        return (*arguments, output), expected, output
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
    kernel = args.title
    if family == "elementwise":
        kernel = kernel.replace("elementwise_", "elementwise_add_", 1)
    elif family == "softmax" and kernel.endswith("_per_tok"):
        kernel += "en"
    elif family == "dot_product":
        kernel = kernel.replace("dot_product_", "dot_prod_", 1)
        if not kernel.endswith("_f32") or kernel in ("dot_prod_f32", "dot_prod_f32x4"):
            kernel += "_f32"
    try:
        lib = upstream_binding(args.script)
        evaluate = getattr(lib, kernel)
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        raise SubmissionError(f"CE · {error}") from error
    dtype = torch.float16 if "_f16" in args.title else torch.float32
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
    judge = Judge(len(cases))
    with torch.no_grad():
        for number, (label, shape) in cases:
            torch.manual_seed(2026 + number)
            inputs, expected, output = setup(number, shape, family, args.title, dtype)
            result = judge.run(number, evaluate, inputs, torch.cuda.synchronize)
            actual = result if output is None else output
            try:
                torch.testing.assert_close(actual, expected, **tolerance)
            except AssertionError as error:
                judge.wrong(number, f"{label} {shape}\n{error}")
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
