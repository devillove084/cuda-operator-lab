"""Run the correctness exercises for one roadmap stage."""
import argparse
import shlex
import subprocess
import sys

from practice_catalog import ROOT

STAGES = {
    "1": ("CUDA 索引与访存", (("leetgpu", "1"), ("leetgpu", "21"), ("leetgpu", "3"))),
    "2": ("归约与同步", (("leetgpu", "17"), ("leetgpu", "4"), ("leetgpu", "13"))),
    "3": ("数值稳定与融合", (("leetgpu", "5"), ("leetgpu", "113"), ("leetgpu", "50"), ("leetgpu", "83"), ("quack", "1"))),
    "4": ("GEMM、CUTLASS 和 CuTe", (("leetgpu", "2"), ("leetcuda", "127"), ("leetcuda", "157"), ("cutlass", "1"), ("triton", "1"))),
    "5": ("Attention 与 KV Cache",
          (("leetgpu", "6"), ("flashattn", "1"), ("flashinfer", "1"), ("flashinfer", "2"),
           ("flashinfer", "3"), ("flashinfer", "4"), ("vllm", "1"))),
    "6": ("状态递推与反向传播", (("fla", "1"), ("fla", "2"), ("fla", "3"), ("fla", "4"))),
}


def main():
    parser = argparse.ArgumentParser(prog="./practice.sh selftest")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--smoke", action="store_true", help="每题只跑用例 1")
    parser.add_argument("--plan", action="store_true", help="打印命令")
    args = parser.parse_args()
    title, exercises = STAGES[args.stage]
    print(f"阶段 {args.stage} · {title}", flush=True)
    failures = []
    for platform, key in exercises:
        command = ["./practice.sh", "test", platform, key]
        if args.smoke:
            command.extend(["--case", "1"])
        print(shlex.join(command), flush=True)
        if not args.plan:
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode:
                failures.append(f"{platform} #{key}")
    if args.plan:
        return 0
    print(f"阶段自测 · {len(exercises) - len(failures)}/{len(exercises)} 题通过")
    if failures:
        print("需复查 · " + " · ".join(failures))
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
