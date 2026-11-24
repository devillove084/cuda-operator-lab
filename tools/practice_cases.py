"""Inspect the exact correctness inputs used by the offline judge."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import torch

from practice_catalog import PLATFORMS, find_problem, local_test, problems, test_description
from practice_check import leetgpu_plan, load_module, upstream_leetgpu, xpu_case_sizes, xpu_public_plan
from practice_workloads import Case
from practice_leetcuda_cases import leetcuda_cases
from practice_leetcuda_suites import suite_cases
from practice_xpuoj import case_plan


def describe(value):
    if isinstance(value, torch.Tensor):
        return f"{tuple(value.shape)} {str(value.dtype).removeprefix('torch.')}"
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    return type(value).__name__


def xpu_config(problem):
    return load_module("public_xpuoj_config", problem.pack / "testcase_config.py")


def leetcuda_workloads(problem):
    mode = local_test(problem)
    return leetcuda_cases(mode.stem, problem.title) if isinstance(mode, Path) else ()


def list_cases(problem):
    mode = local_test(problem)
    print(f"{PLATFORMS[problem.platform]} #{problem.key} · {problem.title}", flush=True)
    if problem.platform == "leetgpu":
        challenge = upstream_leetgpu(problem)
        tests = leetgpu_plan(challenge)
        for workload in [Case(0, "公开样例", challenge.generate_example_test), *tests]:
            data = workload.factory()
            arguments = " · ".join(f"{name} {describe(value)}" for name, value in data.items())
            print(f"  #{workload.number} {workload.label} · {arguments}")
        print(f"{len(tests)} 个功能用例，另有 1 个样例")
        return
    if problem.platform == "xpuoj":
        if mode == "cuda":
            config = xpu_config(problem)
            tests = xpu_public_plan(config)
            total = config.getNumOfTestcases()
            print(f"官方公开题包 · {total} 个原始用例 + {len(tests) - total} 个边界复验")
            for workload in tests:
                number = workload.number
                position = number if number <= total else (1, total // 2 + 1, total)[number - total - 1]
                print(f"  #{number} {workload.label} · {xpu_case_sizes(config, position)}")
        else:
            tests = case_plan(int(problem.key))
            print(f"本地输入 + 题面参考实现 · {len(tests)} 个用例")
            for workload in tests:
                data, outputs = workload.factory()
                description = " · ".join(("输出 " if i in outputs else "") + describe(value)
                                         for i, value in enumerate(data))
                print(f"  #{workload.number} {workload.label} · {description}")
        return
    entries = leetcuda_workloads(problem)
    if entries:
        print(f"独立全输出校验 · {len(entries)} 个用例")
        for number, (label, shape) in enumerate(entries, start=1):
            print(f"  #{number} {label} {shape}")
    else:
        specs = suite_cases(problem.title, problem.key, mode)
        if specs:
            print(f"共享算子套件 · {len(specs)} 个工作负载")
            for number, spec in enumerate(specs, start=1):
                print(f"  #{number} {spec.label} · {spec.specification}")
        else:
            print(test_description(problem))


def summarize(platform):
    counts = {}
    for problem in problems():
        if platform not in ("all", problem.platform):
            continue
        torch.manual_seed(2026)
        if problem.platform == "leetgpu":
            count = len(leetgpu_plan(upstream_leetgpu(problem)))
            suite = "公开功能用例及边界复验"
        elif problem.platform == "xpuoj":
            if local_test(problem) == "cuda":
                count = len(xpu_public_plan(xpu_config(problem)))
                suite = "官方公开题包及边界复验"
            else:
                count = len(case_plan(int(problem.key)))
                suite = "本地生成用例"
        else:
            count = len(leetcuda_workloads(problem))
            suite = "独立全输出校验"
            if not count:
                count = len(suite_cases(problem.title, problem.key, local_test(problem)))
                suite = "共享驱动工作负载"
        if count:
            counts.setdefault((problem.platform, suite), []).append(count)
    for (platform, suite), values in counts.items():
        distribution = " · ".join(f"{size} 个/题×{frequency}" for size, frequency
                                 in sorted(Counter(values).items()))
        print(f"{PLATFORMS[platform]} · {suite} · {len(values)} 题 · {sum(values)} 个")
        print(f"  {distribution}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("platform", choices=("all", *PLATFORMS))
    parser.add_argument("key", nargs="?")
    args = parser.parse_args()
    torch.manual_seed(2026)
    with torch.no_grad():
        if args.key:
            list_cases(find_problem(args.platform, args.key))
        else:
            summarize(args.platform)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
