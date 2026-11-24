"""Compare a validated LeetGPU solution to local implementations on the same GPU."""
import hashlib
import json
import os
from statistics import median, quantiles
from time import time_ns
from types import SimpleNamespace

import torch

from practice_catalog import ENTRYPOINT, ROOT, local_test, solution, xpu_cuda_statement
from practice_check import clone, cuda_args, cuda_function, load_module, read_abi, selected_cases, xpu_public_plan


def vector_inputs(length):
    a = torch.randn(length, device="cuda")
    b = torch.randn_like(a)
    return {"A": a, "B": b, "C": torch.zeros_like(a), "N": length}


def matrix_inputs(m, n, k):
    a = torch.randn((m, n), device="cuda")
    b = torch.randn((n, k), device="cuda")
    return {"A": a, "B": b, "C": torch.zeros((m, k), device="cuda"), "M": m, "N": n, "K": k}


def workloads(key, challenge):
    yield "LeetGPU performance", challenge.generate_performance_test()
    if key == "1":
        for length in (31, 33, 255, 257, 4093, 65_537, 1_048_576, 16_777_217):
            yield "vector length " + str(length), vector_inputs(length)
    if key == "2":
        for dims in ((31, 33, 127), (1, 1024, 4096), (73, 997, 1537), (1024, 1024, 1024),
                     (1537, 769, 6145), (4096, 4096, 4096)):
            yield "GEMM " + "×".join(map(str, dims)), matrix_inputs(*dims)


def triton_vector_add(a, b, result, n):
    import triton
    from practice_triton import vector_add

    block = 1024
    vector_add[(triton.cdiv(n, block),)](a, b, result, n, block)


def measure(reset, invoke, start, stop):
    reset()
    start.record()
    invoke()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000


def compare(problem, challenge, invoke, abi, args, cases=None, validate=None):
    signature = challenge.get_solve_signature()
    outputs = [name for name, (_, role) in signature.items() if role in ("out", "inout")]
    report = {"platform": problem.platform, "problem": problem.key, "name": problem.title,
              "gpu": torch.cuda.get_device_name(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "compute_capability": list(torch.cuda.get_device_capability()),
              "pytorch": str(torch.__version__), "torch_cuda": torch.version.cuda,
              "tf32": torch.backends.cuda.matmul.allow_tf32, "seed": 2026,
              "warmup": args.warmup, "repeats": args.repeats,
              "source_sha256": hashlib.sha256(solution(problem).read_bytes()).hexdigest(),
              "cases": []}
    report_path = None
    if args.mode == "bench":
        folder = ROOT / "build" / "practice" / "reports"
        folder.mkdir(parents=True, exist_ok=True)
        report_path = folder / f"{problem.platform}-{problem.key}-{time_ns()}.json"
    if cases is None:
        cases = enumerate(workloads(problem.key, challenge))
    for index, (name, test) in cases:
        if args.case is not None and args.case != index:
            continue
        print(f"性能用例 {index} · {name}", flush=True)
        values = {field: test[field] for field in signature}
        expected = {field: clone(value) for field, value in values.items()}
        challenge.reference_impl(*expected.values())
        contenders = {"solution": invoke}
        if args.mode == "bench":
            contenders["PyTorch eager (public reference)"] = challenge.reference_impl
            if problem.platform == "leetgpu" and problem.key == "1":
                contenders["Triton vector add (1024 elements/block)"] = triton_vector_add
        prepared = {}
        for label, fn in contenders.items():
            data = {field: clone(value) for field, value in values.items()}

            def reset_outputs(data=data):
                for field in outputs:
                    if isinstance(data[field], torch.Tensor):
                        data[field].copy_(values[field])

            if label == "solution" and abi is not None:
                call_args = cuda_args(data.values(), abi)

                def call(fn=fn, call_args=call_args):
                    fn(*call_args)
            else:
                def call(fn=fn, data=data):
                    fn(*data.values())

            torch.cuda.synchronize()
            call()
            torch.cuda.synchronize()
            if validate:
                validate(index, values, data, expected)
            for field in (() if validate else outputs):
                try:
                    torch.testing.assert_close(data[field], expected[field], atol=challenge.atol,
                                               rtol=challenge.rtol, equal_nan=True)
                except AssertionError as error:
                    raise AssertionError(f"{label} · 性能用例 {index} · {field}\n{error}") from error
            for _ in range(args.warmup):
                reset_outputs()
                call()
            torch.cuda.synchronize()
            prepared[label] = reset_outputs, call

        if args.mode == "profile":
            reset, call = prepared["solution"]
            reset()
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            try:
                with torch.cuda.nvtx.range(f"{problem.platform}/{problem.key}/{index}/solution"):
                    call()
                    torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()
            return

        row = {"index": index, "label": name, "arguments": {
            field: {"shape": list(value.shape), "dtype": str(value.dtype)}
            if isinstance(value, torch.Tensor) else value
            for field, value in values.items()
            if isinstance(value, (torch.Tensor, int, float, bool))
        }, "implementations": {}}
        samples = {label: [] for label in prepared}
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        order = list(prepared)
        for step in range(args.repeats):
            for label in (order if step % 2 == 0 else reversed(order)):
                reset, call = prepared[label]
                samples[label].append(measure(reset, call, start, stop))
        for label, timings in samples.items():
            p10, *_, p90 = quantiles(timings, n=10, method="inclusive")
            entry = {"p10_us": p10, "p50_us": median(timings), "p90_us": p90,
                     "samples_us": timings}
            if entry["p50_us"] > 0 and problem.platform == "leetgpu" and problem.key == "1":
                traffic = sum(values[field].numel() * values[field].element_size()
                              for field in ("A", "B", "C"))
                entry["effective_GB_s"] = traffic / entry["p50_us"] / 1000
            if entry["p50_us"] > 0 and problem.platform == "leetgpu" and problem.key == "2":
                entry["TFLOP_s"] = 2 * values["M"] * values["N"] * values["K"] / (
                    entry["p50_us"] * 1e6
                )
            row["implementations"][label] = entry
            print(f"  {label:<39} p50 {entry['p50_us']:10.3f} µs · p90 {p90:10.3f} µs")
        report["cases"].append(row)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not report["cases"]:
        raise ValueError("没有指定编号的性能用例。")
    print(f"性能报告 {report_path}")


def compare_xpu(problem, args):
    abi = None
    mode = local_test(problem)
    if mode == "xpu-python":
        solve = load_module("learner_solution", args.implementation).run_kernel
    else:
        abi = read_abi(ENTRYPOINT.search(xpu_cuda_statement(problem))[0])
        solve = cuda_function(args.implementation, "run_kernel", abi)

    validate = None
    if mode == "cuda":
        config = load_module("public_xpuoj_config", problem.pack / "testcase_config.py")
        if problem.key == "28":
            positions = [2]  # The public checker compares only the output at position 2.
        else:
            positions = [index for index, role in enumerate(config.INPUT_CLASS)
                         if role in ("INOUT", "OUTPUT")]
        plan = selected_cases(xpu_public_plan(config), args.case)
        reference = config.baseline
        sizes_by_case = {}

        def make_inputs(workload):
            sizes, values = workload.factory()
            sizes_by_case[workload.number] = sizes
            return values

        def validate(number, values, actual, expected):
            sizes = sizes_by_case[number]
            if not config.check(sizes, list(values.values()), list(actual.values()),
                                list(expected.values())):
                raise AssertionError(f"性能用例 {number} 未通过 XPUOJ 公开校验。")
    else:
        from practice_xpuoj import baseline, case_plan

        plan = selected_cases(case_plan(int(problem.key)), args.case)
        reference = baseline(problem)
        positions = case_plan(int(problem.key))[0].factory()[1]

        def make_inputs(workload):
            return workload.factory()[0]

    first = make_inputs(plan[0])
    # Names are internal only. XPUOJ's ABI and Python reference take positional arguments.
    signature = {str(index): (None, "out" if index in positions else "in")
                 for index in range(len(first))}
    challenge = SimpleNamespace(
        get_solve_signature=lambda: signature, reference_impl=reference, atol=1e-2, rtol=1e-2,
    )

    def cases():
        for workload in plan:
            values = first if workload is plan[0] else make_inputs(workload)
            yield workload.number, (workload.label, dict(zip(signature, values)))

    compare(problem, challenge, solve, abi, args, cases(), validate)
