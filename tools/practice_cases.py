"""Inspect the exact correctness inputs used by the offline judge."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import torch

from practice_catalog import PLATFORMS, find_problem, local_test, problems, test_description
from practice_check import load_module, matching_tests, upstream_leetgpu, xpu_case_sizes
from practice_leetcuda_cases import leetcuda_cases
from practice_xpuoj import case_inputs, case_numbers


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
        print("公开样例 #0 · 公开功能用例 · 固定种子 2026")
        tests = list(matching_tests(challenge, None))
        for number, data in [*matching_tests(challenge, 0), *tests]:
            print(f"  #{number} " + " · ".join(f"{name} {describe(value)}" for name, value in data.items()))
        print(f"{len(tests)} 个功能用例，另有 1 个样例")
        return
    if problem.platform == "xpuoj":
        if mode == "cuda":
            config = xpu_config(problem)
            count = config.getNumOfTestcases()
            print(f"官方公开题包 · {count} 个用例")
            for number in range(1, count + 1):
                print(f"  #{number} {xpu_case_sizes(config, number)}")
        else:
            numbers = case_numbers(int(problem.key))
            print(f"本地输入 + 题面参考实现 · {len(numbers)} 个用例")
            for number in numbers:
                torch.manual_seed(2026 + number)
                data, outputs = case_inputs(int(problem.key), number)
                description = " · ".join(("输出 " if i in outputs else "") + describe(value)
                                         for i, value in enumerate(data))
                print(f"  #{number} {description}")
        return
    entries = leetcuda_workloads(problem)
    if entries:
        print(f"独立全输出校验 · {len(entries)} 个用例")
        for number, (label, shape) in enumerate(entries, start=1):
            print(f"  #{number} {label} {shape}")
    else:
        print(f"{test_description(problem)} · 该条目使用算子级入口")


def summarize(platform):
    counts = {}
    for problem in problems():
        if platform not in ("all", problem.platform):
            continue
        torch.manual_seed(2026)
        if problem.platform == "leetgpu":
            count = len(upstream_leetgpu(problem).generate_functional_test())
            suite = "公开功能用例"
        elif problem.platform == "xpuoj":
            if local_test(problem) == "cuda":
                count = xpu_config(problem).getNumOfTestcases()
                suite = "官方公开题包"
            else:
                count = len(case_numbers(int(problem.key)))
                suite = "本地生成用例"
        else:
            count = len(leetcuda_workloads(problem))
            suite = "独立全输出校验"
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
