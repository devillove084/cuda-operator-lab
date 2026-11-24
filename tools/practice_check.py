"""Run the repo's public test data against a local CUDA library or PyTorch file."""
import argparse
import copy
import ctypes
import importlib.util
import io
import re
import sys
import types

import torch

from practice_catalog import (
    ENTRYPOINT, find_problem, leetgpu_template, local_test, xpu_cuda_statement,
)
from practice_verdict import Judge, SubmissionError, fail
from practice_workloads import BASE_SEED, Case, guarded_copy, published_replays, verify_guards

SCALARS = {
    "bool": ctypes.c_bool,
    "int": ctypes.c_int,
    "unsigned": ctypes.c_uint,
    "unsigned int": ctypes.c_uint,
    "long": ctypes.c_long,
    "unsigned long": ctypes.c_ulong,
    "long long": ctypes.c_longlong,
    "size_t": ctypes.c_size_t,
    "int32_t": ctypes.c_int32,
    "uint32_t": ctypes.c_uint32,
    "int64_t": ctypes.c_int64,
    "uint64_t": ctypes.c_uint64,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
}
SOLVE = re.compile(r'extern\s+"C"\s+void\s+solve\s*\([^)]*\)')
C_COMMENTS = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def python_function(path, symbol):
    try:
        return getattr(load_module("learner_solution", path), symbol)
    except (SyntaxError, ImportError, AttributeError) as error:
        raise SubmissionError(f"CE · 无法加载接口 {symbol}\n{error}") from error
    except Exception as error:
        raise SubmissionError(f"RE · 加载 Python 实现失败\n{error}") from error


def read_abi(signature):
    signature = C_COMMENTS.sub("", signature)
    parameters = signature.split("(", 1)[1].rsplit(")", 1)[0]
    types = []
    for parameter in parameters.split(","):
        parameter = C_COMMENTS.sub("", parameter).strip()
        match = re.fullmatch(r"(.+?)\s*([A-Za-z_]\w*)", parameter)
        if not match:
            raise ValueError(f"无法读取 CUDA 参数 {parameter}")
        c_type = " ".join(match[1].split())
        if c_type.endswith("*") and c_type.count("*") == 1:
            types.append(ctypes.c_void_p)
        elif c_type in SCALARS:
            types.append(SCALARS[c_type])
        else:
            raise ValueError(f"CUDA 参数类型暂未接入 {c_type}")
    return types


def cuda_function(path, symbol, abi):
    try:
        library = ctypes.CDLL(path)
        evaluate = getattr(library, symbol)
    except (OSError, AttributeError) as error:
        raise SubmissionError(f"CE · 无法加载接口 {symbol}\n{error}") from error
    evaluate.argtypes = abi
    evaluate.restype = None
    return evaluate


def cuda_args(values, abi):
    if len(values) != len(abi):
        raise ValueError("CUDA 接口和用例的参数数量不一致。")
    arguments = []
    for value, c_type in zip(values, abi):
        if c_type is ctypes.c_void_p:
            if value is None:
                arguments.append(None)
                continue
            if not isinstance(value, torch.Tensor) or not value.is_cuda:
                raise ValueError("公开用例的指针参数必须是 CUDA 张量。")
            arguments.append(value.data_ptr())
        else:
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                value = value.item()
            if not isinstance(value, (int, float)):
                raise ValueError("公开用例的标量与 CUDA 接口不一致。")
            converted = c_type(value)
            if isinstance(value, int) and isinstance(converted.value, int) and value != converted.value:
                raise ValueError(f"标量 {value} 超出 CUDA 接口的范围。")
            arguments.append(converted)
    return arguments


def upstream_leetgpu(problem):
    # These recipes produce the same kinds of tensors as the public challenge files request.
    # The public repository omits the service's core.challenge_base module.
    core = types.ModuleType("core")
    core.__path__ = []
    base = types.ModuleType("core.challenge_base")
    sys.modules["core"] = core
    sys.modules["core.challenge_base"] = base

    class ChallengeBase:
        device = torch.device("cuda")
        rtol = 1e-5
        atol = 1e-6

    def alloc(shape, dtype):
        dtype = getattr(torch, dtype.removeprefix("torch.")) if isinstance(dtype, str) else dtype
        return torch.empty(shape, dtype=dtype, device=ChallengeBase.device)

    def out(shape, dtype="float32"):
        return alloc(shape, dtype).zero_()

    def uniform(shape, low=0.0, high=1.0, dtype="float32"):
        return alloc(shape, dtype).uniform_(low, high)

    def normal(shape, mean=0.0, std=1.0, dtype="float32"):
        return alloc(shape, dtype).normal_(mean, std)

    def integers(shape, low, high, dtype="int32"):
        return alloc(shape, dtype).random_(low, high)

    base.ChallengeBase = ChallengeBase
    base.FullTensor = lambda shape, value, dtype="float32": alloc(shape, dtype).fill_(value)
    base.OutTensor = out
    base.RandTensor = uniform
    base.RandnTensor = normal
    base.RandIntTensor = integers
    return load_module("public_leetgpu_challenge", problem.path.parent / "challenge.py").Challenge()


def leetgpu_plan(challenge):
    torch.manual_seed(BASE_SEED)
    examples = challenge.generate_functional_test()
    if not examples:
        raise ValueError("公开配置没有功能用例。")

    def reload(number, seed):
        torch.manual_seed(seed)
        return challenge.generate_functional_test()[number - 1]

    return [
        *(Case(number, "公开功能用例", lambda case=case: case)
          for number, case in enumerate(examples, start=1)),
        *published_replays(len(examples), reload, len(examples) + 1),
    ]


def selected_cases(plan, case):
    if case is None:
        return plan
    selected = [workload for workload in plan if workload.number == case]
    if not selected:
        raise ValueError(f"用例编号为 1 到 {len(plan)}。")
    return selected


def candidate_arguments(values, case):
    copies, guards = {}, {}
    for key, value in values.items():
        guarded = guarded_copy(value) if case.guarded else None
        if guarded is None:
            copies[key] = clone(value)
        else:
            copies[key], *guard = guarded
            guards[key] = guard
    return copies, guards


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone(memory_format=torch.preserve_format)
    return copy.deepcopy(value)


def leetgpu_abi(problem, signature):
    starter = leetgpu_template(problem)
    abi = read_abi(SOLVE.search(C_COMMENTS.sub("", starter.read_text()))[0])
    upstream_pointer = [issubclass(c_type, ctypes._Pointer) for c_type, _ in signature.values()]
    if upstream_pointer != [c_type is ctypes.c_void_p for c_type in abi]:
        raise ValueError("公开用例和 CUDA 模板的参数类型不一致。")
    return abi


def leetgpu_cases(challenge, case, invoke, abi=None):
    signature = challenge.get_solve_signature()
    outputs = [name for name, (_, role) in signature.items() if role in ("out", "inout")]
    if not outputs:
        raise ValueError("公开接口没有可比较的输出。")
    if case == 0:
        tests = [Case(0, "公开样例", challenge.generate_example_test)]
    else:
        tests = selected_cases(leetgpu_plan(challenge), case)
    judge = Judge(len(tests))
    for workload in tests:
        number, data = workload.number, workload.factory()
        initial = {name: data[name] for name in signature}
        actual, guards = candidate_arguments(initial, workload)
        reference = {name: clone(initial[name]) for name in signature}
        challenge.reference_impl(**reference)
        torch.cuda.synchronize()
        arguments = list(actual.values()) if abi is None else cuda_args(actual.values(), abi)
        judge.run(number, invoke, arguments, torch.cuda.synchronize)
        verify_guards(judge, number, guards)
        for name in outputs:
            try:
                torch.testing.assert_close(actual[name], reference[name], atol=challenge.atol,
                                           rtol=challenge.rtol, equal_nan=True)
            except AssertionError as error:
                judge.wrong(number, f"输出 {name} 不匹配\n{error}")
        if workload.guarded:
            for name, (_, role) in signature.items():
                if role == "in" and isinstance(initial[name], torch.Tensor):
                    try:
                        torch.testing.assert_close(actual[name], initial[name], atol=0, rtol=0,
                                                   equal_nan=True)
                    except AssertionError as error:
                        judge.wrong(number, f"只读输入 {name} 被修改\n{error}")
        judge.accept(number)
    judge.finish()


def xpu_case_sizes(config, number):
    previous_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO(f"{number}\n")
        sizes = config.getTestCaseSize()
    finally:
        sys.stdin = previous_stdin
    if isinstance(sizes, tuple) and len(sizes) == 2 and isinstance(sizes[0], list):
        sizes = sizes[0]
    return sizes


def xpu_public_plan(config):
    def make(number, seed):
        torch.manual_seed(seed)
        sizes = xpu_case_sizes(config, number)
        return sizes, config.genTestCase(sizes)

    total = config.getNumOfTestcases()
    return [
        *(Case(number, "官方公开题包", lambda n=number: make(n, BASE_SEED + n))
          for number in range(1, total + 1)),
        *published_replays(total, make, total + 1),
    ]


def xpu_cases(problem, case, path):
    config = load_module("public_xpuoj_config", problem.pack / "testcase_config.py")
    abi = read_abi(ENTRYPOINT.search(xpu_cuda_statement(problem))[0])
    evaluate = cuda_function(path, "run_kernel", abi)
    tests = selected_cases(xpu_public_plan(config), case)
    judge = Judge(len(tests))
    for workload in tests:
        number = workload.number
        sizes, original = workload.factory()
        actual, guards = candidate_arguments(dict(enumerate(original)), workload)
        reference = [clone(value) for value in original]
        result = config.baseline(*reference)
        arguments = cuda_args(actual.values(), abi)
        torch.cuda.synchronize()
        judge.run(number, evaluate, arguments, torch.cuda.synchronize)
        verify_guards(judge, number, guards)
        if result is not None:
            reference = result
        if not config.check(sizes, original, list(actual.values()), reference):
            judge.wrong(number, "XPUOJ 官方公开检查器判定不通过。")
        judge.accept(number)
    judge.finish()


def xpu_statement_cases(problem, case, path):
    from practice_xpuoj import baseline, case_plan

    python = path.endswith(".py")
    if python:
        evaluate = python_function(path, "run_kernel")
    else:
        abi = read_abi(ENTRYPOINT.search(xpu_cuda_statement(problem))[0])
        evaluate = cuda_function(path, "run_kernel", abi)
    reference_impl = baseline(problem)
    tests = selected_cases(case_plan(int(problem.key)), case)
    judge = Judge(len(tests))
    for workload in tests:
        number = workload.number
        original, outputs = workload.factory()
        copies, guards = candidate_arguments(dict(enumerate(original)), workload)
        actual = list(copies.values())
        reference = [clone(value) for value in original]
        result = reference_impl(*reference)
        arguments = actual if python else cuda_args(actual, abi)
        torch.cuda.synchronize()
        judge.run(number, evaluate, arguments, torch.cuda.synchronize)
        verify_guards(judge, number, guards)
        if result is not None:
            reference = result
        for index, (got, want) in enumerate(zip(actual, reference)):
            if not isinstance(got, torch.Tensor):
                continue
            if (problem.key, index) in (("37", 5), ("38", 2)):
                continue
            if got.dtype == torch.float8_e4m3fn:
                got, want = got.view(torch.uint8), want.view(torch.uint8)
            if index in outputs:
                if got.dtype in (torch.float16, torch.bfloat16):
                    tolerance = {"rtol": 1e-2, "atol": 1e-2}
                elif problem.key in ("68", "74", "86", "113", "114"):
                    tolerance = {"rtol": 1e-3, "atol": 1e-3}
                else:
                    tolerance = {}
            else:
                want = original[index]
                if want.dtype == torch.float8_e4m3fn:
                    want = want.view(torch.uint8)
                tolerance = {"rtol": 0, "atol": 0}
            try:
                torch.testing.assert_close(got, want, equal_nan=True, **tolerance)
            except AssertionError as error:
                judge.wrong(number, f"参数 {index + 1} 不匹配\n{error}")
        judge.accept(number)
    judge.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("platform", choices=("leetgpu", "xpuoj"))
    parser.add_argument("key")
    parser.add_argument("--case", type=int)
    parser.add_argument("--mode", choices=("test", "bench", "profile"), default="test")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("implementation")
    args = parser.parse_args()
    problem = find_problem(args.platform, args.key)
    torch.manual_seed(2026)
    with torch.no_grad():
        if args.repeats < 2 or args.warmup < 0:
            raise ValueError("至少采样 2 次，预热次数不能小于 0。")
        if args.platform == "xpuoj":
            if args.mode != "test":
                from practice_bench import compare_xpu

                compare_xpu(problem, args)
                return
            if local_test(problem) in ("xpu-statement", "xpu-python"):
                xpu_statement_cases(problem, args.case, args.implementation)
            else:
                xpu_cases(problem, args.case, args.implementation)
            return
        challenge = upstream_leetgpu(problem)
        abi = None
        if args.implementation.endswith(".py"):
            solve = python_function(args.implementation, "solve")
        else:
            abi = leetgpu_abi(problem, challenge.get_solve_signature())
            solve = cuda_function(args.implementation, "solve", abi)
        if args.mode == "test":
            leetgpu_cases(challenge, args.case, solve, abi)
        else:
            from practice_bench import compare

            compare(problem, challenge, solve, abi, args)


if __name__ == "__main__":
    try:
        main()
    except SubmissionError as error:
        sys.exit(fail(error))
    except (AssertionError, ValueError) as error:
        sys.exit(f"评测错误 · {error}")
