"""Compile-time ABI consumer and offline judge for FlashInfer and FLA exercises."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import median, quantiles
import sys
from time import time_ns

import torch

from practice_advanced_cases import ADVANCED, advanced_plan
from practice_advanced_reference import make_exercise
from practice_bench import measure
from practice_catalog import ROOT
from practice_check import cuda_args, cuda_function, read_abi
from practice_verdict import Judge, SubmissionError, fail
from practice_workloads import BASE_SEED, guarded_copy, verify_guards

SIGNATURE = re.compile(r'extern\s+"C"\s+void\s+run_kernel\s*\([^;{}]*\)')


def prepare(exercise, abi):
    candidate, guards = {}, {}
    for name, value in (exercise.inputs | exercise.outputs).items():
        view = guarded_copy(value)
        if view is not None:
            tensor, before, after = view
            candidate[name] = tensor
            guards[name] = before, after
        else:
            candidate[name] = value.clone() if isinstance(value, torch.Tensor) else value
    return candidate, guards, cuda_args(candidate.values(), abi)


def validate(judge, number, exercise, candidate, expected):
    for name, value in exercise.inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        try:
            torch.testing.assert_close(candidate[name].view(torch.uint8), value.view(torch.uint8),
                                       atol=0, rtol=0)
        except AssertionError as error:
            judge.wrong(number, f"只读输入 {name} 被修改\n{error}")
    for name, value in expected.items():
        try:
            torch.testing.assert_close(candidate[name], value, atol=exercise.atol,
                                       rtol=exercise.rtol, equal_nan=False)
        except AssertionError as error:
            judge.wrong(number, f"输出 {name}\n{error}")


def measure_case(args, number, exercise, candidate, expected, invoke):
    def reset():
        for name in exercise.inout:
            candidate[name].copy_(exercise.outputs[name])

    def baseline():
        return exercise.reference(torch.float32)

    if args.mode == "profile":
        reset()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        try:
            with torch.cuda.nvtx.range(f"{args.platform}/{args.key}/{number}/solution"):
                invoke()
                torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
        return {}
    reference_fp32 = baseline()
    for name, value in expected.items():
        torch.testing.assert_close(reference_fp32[name], value, atol=exercise.atol, rtol=exercise.rtol)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    results = {}
    contenders = (("solution", reset, invoke), ("PyTorch eager FP32", lambda: None, baseline))
    for _, setup, call in contenders:
        for _ in range(args.warmup):
            setup()
            call()
    torch.cuda.synchronize()
    samples = {label: [] for label, _, _ in contenders}
    for repeat in range(args.repeats):
        order = contenders if repeat % 2 == 0 else contenders[::-1]
        for label, setup, call in order:
            samples[label].append(measure(setup, call, start, stop))
    for label, times in samples.items():
        deciles = quantiles(times, n=10, method="inclusive")
        p10, p90 = deciles[0], deciles[8]
        p50 = median(times)
        print(f"  {label:<22} p10/p50/p90 {p10:.3f} / {p50:.3f} / {p90:.3f} µs", flush=True)
        results[label] = {"p10_us": p10, "p50_us": p50, "p90_us": p90, "samples_us": times}
    return results


def run(args):
    plan = list(enumerate(advanced_plan(args.platform, args.key), start=1))
    if args.case is not None:
        if not 1 <= args.case <= len(plan):
            raise ValueError(f"用例编号为 1 到 {len(plan)}。")
        plan = [plan[args.case - 1]]
    source = ROOT / "practice" / args.platform / args.key / "solution.cu"
    source_bytes = source.read_bytes()
    signature = SIGNATURE.search(source_bytes.decode())
    if signature is None:
        raise SubmissionError("CE · solution.cu 缺少 extern C run_kernel 接口")
    abi = read_abi(signature[0])
    solve = cuda_function(args.library, "run_kernel", abi)
    torch.backends.cuda.matmul.allow_tf32 = False
    judge = Judge(len(plan))
    report = {"platform": args.platform, "problem": args.key,
              "gpu": torch.cuda.get_device_name(),
              "compute_capability": list(torch.cuda.get_device_capability()),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "pytorch": str(torch.__version__), "torch_cuda": torch.version.cuda, "tf32": False,
              "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "warmup": args.warmup, "repeats": args.repeats, "cases": []}
    for number, workload in plan:
        seed = BASE_SEED + number
        exercise = make_exercise(args.platform, args.key, workload, seed)
        expected = exercise.reference(torch.float64)
        candidate, guards, arguments = prepare(exercise, abi)
        judge.run(number, solve, arguments, torch.cuda.synchronize)
        verify_guards(judge, number, guards)
        validate(judge, number, exercise, candidate, expected)
        judge.accept(number)
        if args.mode == "test":
            continue
        print(f"用例 {number} · {workload.description()}", flush=True)

        def invoke():
            solve(*arguments)

        result = measure_case(args, number, exercise, candidate, expected, invoke)
        report["cases"].append({"case": number, "workload": workload.description(),
                                "seed": seed, "contenders": result})
    judge.finish()
    if args.mode == "bench":
        folder = ROOT / "build" / "practice" / "reports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{args.platform}-{args.key}-{time_ns()}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"性能报告 {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("platform", choices=ADVANCED)
    parser.add_argument("key")
    parser.add_argument("library", type=Path)
    parser.add_argument("--case", type=int)
    parser.add_argument("--mode", choices=("test", "bench", "profile"), default="test")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 2:
        parser.error("--warmup 至少为 0，--repeats 至少为 2。")
    try:
        with torch.no_grad():
            run(args)
    except SubmissionError as error:
        return fail(error)
    except (ValueError, OSError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    sys.exit(main())
