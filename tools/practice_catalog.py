"""Catalog, editable files, and offline test dispatch for practice.sh."""
import argparse
import ast
from dataclasses import dataclass
from functools import lru_cache
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from time import time_ns
from urllib.parse import unquote

from practice_leetcuda_cases import leetcuda_cases

ROOT = Path(__file__).resolve().parent.parent
PLATFORMS = {"leetgpu": "LeetGPU", "leetcuda": "LeetCUDA", "xpuoj": "XPUOJ"}
ROW = re.compile(r"^- \[([ xX])\] (?:(#\d+) )?\[([^\]]+)\]\(([^)]+)\)(.*)$")
ENTRYPOINT = re.compile(r'extern\s+"C"\s+void\s+run_kernel\s*\([^;{}]*\)\s*;')
LEETCUDA_UPSTREAM = ROOT / "references" / "LeetCUDA"
LEETCUDA_WORKSPACE = ROOT / "practice" / "leetcuda"
FFPA_WORKSPACE = LEETCUDA_WORKSPACE / "ffpa-attn"
XPU_STATEMENT_CASES = {
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
    133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    10001, 10002, 10003,
}
XPU_FUNCTIONS = {"getNumOfTestcases", "getTestCaseSize", "genTestCase", "baseline", "check"}


@dataclass(frozen=True)
class Problem:
    platform: str
    key: str
    title: str
    path: Path | None
    link: str
    detail: str
    done: bool
    catalog: Path
    line: int

    @property
    def pack(self):
        return ROOT / "references" / "XPUOJ" / f"problem_{self.key}"


def problems():
    for platform in PLATFORMS:
        catalog = ROOT / "catalogs" / f"{platform}.md"
        count = 0
        for line_no, line in enumerate(catalog.read_text().splitlines()):
            match = ROW.match(line)
            if not match:
                continue
            state, number, title, link, detail = match.groups()
            count += 1
            path = None if link.startswith(("http://", "https://")) else (
                catalog.parent / unquote(link)
            ).resolve()
            yield Problem(platform, number[1:] if number else str(count), title, path, link,
                          detail.strip(), state.lower() == "x", catalog, line_no)


def find_problem(platform, key):
    for problem in problems():
        if problem.platform == platform and key in (problem.key, problem.title):
            return problem
    raise ValueError(f"清单没有这道题：{platform} {key}")


def leetgpu_template(problem):
    templates = problem.path.parent / "starter"
    for name in ("starter.cu", "starter.pytorch.py"):
        path = templates / name
        if path.is_file():
            return path
    raise ValueError("题目缺少 CUDA 或 PyTorch 模板。")


def xpu_cuda_statement(problem):
    sources = sorted(problem.pack.glob("zh_CN/*cuda.md"))
    if problem.path and problem.path.is_file():
        sources.append(problem.path)
    for path in sources:
        text = path.read_text()
        for block in re.findall(r"(?ms)^\x60{3}(?:cpp|cuda|c\+\+)\s*\n(.*?)^\x60{3}", text):
            if ENTRYPOINT.search(block):
                return block
    return ""


def write_new(path, contents):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        staged = Path(temp.name)
        try:
            temp.write(contents)
            temp.close()
            staged.chmod(0o644)
            try:
                os.link(staged, path)
            except FileExistsError:
                pass
        finally:
            staged.unlink()


def xpu_cuda_template(block, original=False):
    signature = ENTRYPOINT.search(block)[0][:-1].rstrip()
    headers = [line for line in block.splitlines() if line.lstrip().startswith("#include ")]
    if not original:
        required = ["stdint.h", "cuda_runtime.h"]
        for pattern, header in (
            (r"\b__nv_bfloat16\b", "cuda_bf16.h"),
            (r"\b(?:__half|half)\b", "cuda_fp16.h"),
            (r"\b__nv_fp8\w*\b", "cuda_fp8.h"),
            (r"\bcu(?:Double)?Complex\b", "cuComplex.h"),
        ):
            if re.search(pattern, signature):
                required.append(header)
        for name in required:
            include = f"#include <{name}>"
            if include not in headers:
                headers.append(include)
    return ("\n".join(headers) + "\n\n" + signature + " {\n"
            "    // TODO: 实现并启动 CUDA kernel。\n}\n").lstrip()

def xpu_tilelang_template(problem):
    if xpu_cuda_statement(problem):
        return None
    for path in problem.pack.glob("zh_CN/*tilelang.md"):
        for block in re.findall(r"(?ms)^\x60{3}python\s*\n(.*?)^\x60{3}", path.read_text()):
            if "def run_kernel(" in block:
                return block
    return None


def leetcuda_workspace(path):
    return LEETCUDA_WORKSPACE / path.relative_to(LEETCUDA_UPSTREAM.resolve())


def solution(problem, create=False):
    if problem.platform == "leetcuda":
        path = problem.path
        if path is None:
            if "persistent" in problem.title:
                return LEETCUDA_WORKSPACE / "kernels" / "interview" / "flash_attn.cuh"
            filename = ("ffpa_attn_fwd_split_kv.cuh" if "decode" in problem.title
                        else "ffpa_attn_fwd.cuh")
            return FFPA_WORKSPACE / "csrc" / "cuffpa" / filename
        if path.is_dir():
            scripts = sorted(path.glob("triton_*.py"))
            path = scripts[0] if scripts else path / "README.md"
        target = leetcuda_workspace(path)
        if not target.is_file():
            raise ValueError(f"练习目录缺少 {target.relative_to(ROOT)}")
        return target
    template = leetgpu_template(problem) if problem.platform == "leetgpu" else None
    tilelang = xpu_tilelang_template(problem) if problem.platform == "xpuoj" else None
    suffix = template.suffix if template else ".py" if tilelang else ".cu"
    path = ROOT / "practice" / problem.platform / problem.key / f"solution{suffix}"
    if create and not path.exists():
        if template:
            contents = template.read_bytes()
        elif tilelang:
            contents = tilelang.encode()
            previous = path.with_suffix(".cu")
            placeholder = f"// XPUOJ #{problem.key} {problem.title}\n// 按题面填写接口。\n"
            if previous.is_file() and previous.read_text() == placeholder:
                previous.unlink()
        else:
            block = xpu_cuda_statement(problem)
            signature = ENTRYPOINT.search(block)
            if signature:
                contents = xpu_cuda_template(block).encode()
            else:
                contents = f"// XPUOJ #{problem.key} {problem.title}\n// 按题面填写接口。\n".encode()
        write_new(path, contents)
    return path


@lru_cache(maxsize=None)
def scripts_for(folder):
    result = {}
    for script in sorted(folder.glob("*.py")):
        for node in ast.walk(ast.parse(script.read_text(encoding="utf-8-sig"))):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "load":
                continue
            for keyword in node.keywords:
                if keyword.arg != "sources" or not isinstance(keyword.value, (ast.List, ast.Tuple)):
                    continue
                for item in keyword.value.elts:
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        result.setdefault(item.value, script)
    return result


def leetcuda_script(problem):
    path = problem.path
    if path is None and "persistent" not in problem.title:
        return "ffpa"
    if path is None or "interview" in path.parts:
        script = LEETCUDA_WORKSPACE / "kernels" / "interview" / "build.sh"
        return "interview" if script.is_file() else None
    if "wgmma" in path.parts:
        return None
    if "hgemm" in path.parts:
        return LEETCUDA_UPSTREAM / "kernels" / "hgemm" / "hgemm.py"
    if "flash-attn" in path.parts:
        return LEETCUDA_UPSTREAM / "kernels" / "flash-attn" / "flash_attn_mma.py"
    if path.name == "mma_simple_swizzle.cu":
        return path.parent / "print_swizzle_layout.py"
    if path.is_dir() and "openai-triton" in path.parts:
        if path.name == "merge-attn-states":
            return path / "test_merge_attn_states.py"
        return next(iter(sorted(path.glob("triton_*.py"))), None)
    script = scripts_for(path.parent).get(path.name)
    if script:
        return script
    return None


def local_test(problem):
    if problem.platform == "leetgpu":
        if (problem.path.parent / "challenge.py").is_file():
            return "py" if leetgpu_template(problem).suffix == ".py" else "cuda"
    elif problem.platform == "xpuoj":
        config = problem.pack / "testcase_config.py"
        if config.is_file() and xpu_cuda_statement(problem):
            functions = {node.name for node in ast.walk(ast.parse(config.read_text(encoding="utf-8-sig")))
                         if isinstance(node, ast.FunctionDef)}
            if XPU_FUNCTIONS <= functions:
                return "cuda"
        if int(problem.key) in XPU_STATEMENT_CASES:
            return "xpu-python" if solution(problem).suffix == ".py" else "xpu-statement"
    else:
        return leetcuda_script(problem)
    return None


def test_description(problem):
    mode = local_test(problem)
    if isinstance(mode, Path):
        count = len(leetcuda_cases(mode.stem, problem.title))
        return f"独立判题 {count} 例" if count else "算子级脚本"
    if problem.platform == "leetcuda" and problem.path:
        if "wgmma" in problem.path.parts:
            return "需 Hopper SM90a"
        if problem.path.is_dir() and problem.path.name == "nvidia-nsight":
            return "本地阅读"
    if mode == "py":
        return "离线 PyTorch"
    if mode == "interview":
        return "实战套件"
    if mode == "ffpa":
        return "FFPA CUDA"
    if mode in ("xpu-statement", "xpu-python"):
        return "本地用例+公开参考"
    if mode:
        return "离线 CUDA"
    return "缺离线用例"


def list_problems(status, platform):
    for problem in problems():
        if platform != "all" and platform != problem.platform:
            continue
        if status == "done" and not problem.done or status == "todo" and problem.done:
            continue
        state = "完成" if problem.done else "待做"
        display = (
            f"[{state}] {PLATFORMS[problem.platform]:8} #{problem.key:<5} "
            f"{problem.title}  [{test_description(problem)}]"
        )
        print(problem.platform, problem.key, display, sep="\t")


def statement(problem):
    print(f"# {PLATFORMS[problem.platform]} #{problem.key} {problem.title}\n")
    print("实现文件", solution(problem).relative_to(ROOT))
    print("测试入口", test_description(problem), "\n")
    if problem.platform == "leetgpu":
        from render_leetgpu import render

        print(render(problem.path))
    elif problem.platform == "xpuoj":
        if problem.path.is_relative_to(problem.pack):
            language = ".tilelang.md" if solution(problem).suffix == ".py" else ".cuda.md"
            for path in sorted(problem.path.parent.glob("0[0-5]_*.md")):
                if not path.name.startswith("01_") or path.name.endswith(language):
                    print(path.read_text())
        else:
            print(problem.path.read_text())
    elif problem.path:
        print("源码", problem.path.relative_to(ROOT))
        folder = problem.path if problem.path.is_dir() else problem.path.parent
        guide = folder / "README.md"
        if guide.is_file():
            print("\n" + guide.read_text())
    else:
        if "persistent" in problem.title:
            print("练习文件使用 LeetCUDA 实战章节的 persist-D 版本。")
        print("上游来源", problem.link)
        guide = FFPA_WORKSPACE / "README.md"
        if guide.is_file():
            print(guide.read_text())


def toggle(problem):
    lock = ROOT / "build" / "practice" / "catalog.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        problem = find_problem(problem.platform, problem.key)
        lines = problem.catalog.read_text().splitlines(keepends=True)
        lines[problem.line] = re.sub(r"^- \[[ xX]\]", "- [ ]" if problem.done else "- [x]",
                                     lines[problem.line])
        with tempfile.NamedTemporaryFile(mode="w", dir=problem.catalog.parent,
                                         delete=False) as temp:
            staged = Path(temp.name)
            try:
                temp.writelines(lines)
                temp.close()
                staged.chmod(problem.catalog.stat().st_mode & 0o777)
                staged.replace(problem.catalog)
            finally:
                staged.unlink(missing_ok=True)
    print("待做" if problem.done else "已完成", problem.title)


def select_gpu(env):
    if "CUDA_VISIBLE_DEVICES" in env:
        return
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True
    )
    idle = [(int(memory), gpu.strip()) for gpu, memory, load in
            (line.split(",") for line in output.splitlines())
            if int(load) == 0 and int(memory) < 32]
    if not idle:
        raise ValueError("暂时没有空闲 GPU。可设置 CUDA_VISIBLE_DEVICES 指定运行设备。")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = min(idle)[1]


def run_leetcuda(command, folder, env, problem, action):
    if action != "bench":
        subprocess.run(command, cwd=folder, env=env, check=True)
        return
    reports = ROOT / "build" / "practice" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"leetcuda-{problem.key}-{time_ns()}.log"
    print(f"性能日志 {path}", flush=True)
    with path.open("w") as log, subprocess.Popen(command, cwd=folder, env=env, text=True,
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def run_judge(command, folder, env, timeout):
    try:
        if timeout is None:
            subprocess.run(command, cwd=folder, env=env, check=True)
            return
        with subprocess.Popen(command, cwd=folder, env=env, start_new_session=True) as process:
            try:
                returncode = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    except subprocess.TimeoutExpired:
        sys.exit(f"TLE · 评测进程超过 {timeout:g} 秒")
    except subprocess.CalledProcessError as error:
        if error.returncode < 0:
            print(f"RE · 评测进程被信号 {-error.returncode} 终止", file=sys.stderr)
        raise


def test(problem, args):
    action = args.action
    case = args.case
    if action == "profile" and problem.platform == "leetcuda":
        raise ValueError("LeetCUDA 的算子脚本可用 bench 运行。")
    mode = local_test(problem)
    if not mode:
        raise ValueError(f"{test_description(problem)}。可打开题面和实现文件。")
    if action == "test" and problem.platform == "leetcuda":
        numbered = isinstance(mode, Path) and leetcuda_cases(mode.stem, problem.title)
        if not numbered and (case is not None or args.timeout is not None):
            raise ValueError("这个算子级入口不支持 --case 或 --timeout。")
    env = os.environ.copy()
    select_gpu(env)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("MAX_JOBS", "2")

    print(f"GPU {env['CUDA_VISIBLE_DEVICES']} · {PLATFORMS[problem.platform]} #{problem.key}",
          flush=True)
    if mode == "ffpa":
        preferred_compiler = Path("/opt/rh/gcc-toolset-12/root/usr/bin/g++")
        if preferred_compiler.is_file():
            env.setdefault("NVCC_CCBIN", str(preferred_compiler))
        env.update({"FFPA_BUILD_ARCH": "120", "FFPA_BUILD_MAX_STAGES": "2",
                    "ENABLE_FFPA_CUDA_IMPL": "1", "FFPA_DEV_HEADDIMS": "320,512"})
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(FFPA_WORKSPACE / "src"), env.get("PYTHONPATH")))
        )
        lock = ROOT / "build" / "practice" / "ffpa.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"],
                           cwd=FFPA_WORKSPACE, env=env, check=True)
            run_leetcuda([sys.executable, "-u", str(ROOT / "tools" / "practice_ffpa.py"),
                          problem.key, action], FFPA_WORKSPACE, env, problem, action)
        return
    if mode == "interview":
        folder = LEETCUDA_WORKSPACE / "kernels" / "interview"
        binary = folder / "bin" / "leetcuda_bench_sm120a.bin"
        preferred_compiler = Path("/opt/rh/gcc-toolset-12/root/usr/bin/g++")
        if preferred_compiler.is_file():
            env.setdefault("NVCC_CCBIN", str(preferred_compiler))
        lock = ROOT / "build" / "practice" / "interview.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            inputs = [folder / "build.sh", *folder.glob("*.cuh"), *folder.glob("bench/*.cu")]
            if not binary.is_file() or any(
                source.stat().st_mtime_ns > binary.stat().st_mtime_ns for source in inputs
            ):
                subprocess.run(["bash", "build.sh", "--arch", "sm_120a", "--jobs", env["MAX_JOBS"]],
                               cwd=folder, env=env, check=True)
        command = [str(binary)]
        if action == "bench":
            command.extend(["--bench", "--mnk", "2048"])
        elif problem.key == "213":
            command.extend(["--fp4-gemm", "512", "512", "512"])
        elif problem.path is None:
            command.append("--pd-cute")
        run_leetcuda(command, folder, env, problem, action)
        return
    if isinstance(mode, Path):
        workspace_script = leetcuda_workspace(mode)
        if action == "test" and leetcuda_cases(mode.stem, problem.title):
            command = [sys.executable, "-u", str(ROOT / "tools" / "practice_leetcuda.py"),
                       workspace_script.name, problem.title]
            if case is not None:
                command.extend(["--case", str(case)])
            run_judge(command, workspace_script.parent, env, args.timeout)
            return
        if case is not None:
            raise ValueError("这个上游脚本按整个算子运行，无法指定 --case。")
        print(f"运行 {mode.name}。检查方式由该脚本决定，有些脚本只打印输出和耗时。",
              flush=True)
        command = [sys.executable, "-u", workspace_script.name]
        if mode.name == "hgemm.py":
            env.setdefault("NVCC_CCBIN", "/opt/rh/gcc-toolset-12/root/usr/bin/g++")
            command.extend(["--MNK", "1024", "--cuda-all", "--wmma-all", "--mma-all",
                            "--mma-tn", "--cute-tn", "--torch", "--show-a", "--force-build"])
        elif mode.name == "flash_attn_mma.py":
            env.setdefault("NVCC_CCBIN", "/opt/rh/gcc-toolset-12/root/usr/bin/g++")
            command.extend(["--B", "1", "--H", "8", "--N", "512", "--D", "64",
                            "--sdpa", "--check", "--f32", "--build-others", "--seed", "2026"])
        run_leetcuda(command, workspace_script.parent, env, problem, action)
        return
    source = solution(problem, create=True)
    command = [sys.executable, "-u", str(ROOT / "tools" / "practice_check.py"),
               problem.platform, problem.key]
    if case is not None or action == "profile":
        default_case = 1 if problem.platform == "xpuoj" else 0
        command.extend(["--case", str(case if case is not None else default_case)])
    if action in ("bench", "profile"):
        command.extend(["--mode", action, "--warmup", str(args.warmup)])
    if action == "bench":
        command.extend(["--repeats", str(args.repeats)])
    if action == "profile":
        folder = ROOT / "build" / "practice" / "reports"
        folder.mkdir(parents=True, exist_ok=True)
        report = folder / f"{problem.platform}-{problem.key}-{time_ns()}"
        if args.tool == "ncu":
            command = ["ncu", "--profile-from-start", "off", "--set", "full",
                       "--launch-count", "20", "--export", str(report), *command]
        else:
            command = ["/usr/local/bin/nsys" if Path("/usr/local/bin/nsys").is_file() else "nsys", "profile", "--trace=cuda,nvtx",
                       "--sample=none", "--cpuctxsw=none", "--capture-range=cudaProfilerApi",
                       "--capture-range-end=stop", "--output", str(report), *command]
        print(f"Profiler 报告 {report}", flush=True)
    if mode == "xpu-python":
        runtime = ROOT / "build" / "practice" / "tilelang-runtime"
        if runtime.is_dir():
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(runtime), env.get("PYTHONPATH"))))
    if mode in ("py", "xpu-python"):
        if action == "test":
            run_judge(command + [str(source)], source.parent, env, args.timeout)
        else:
            subprocess.run(command + [str(source)], env=env, cwd=source.parent, check=True)
        return
    build = ROOT / "build" / "practice" / problem.platform / problem.key
    build.mkdir(parents=True, exist_ok=True)
    with (build / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(["cmake", "--preset", "pro5000", "-S", str(ROOT), "-B", str(build),
                        f"-DCUDA_PRACTICE_SOURCE={source}"], cwd=ROOT, env=env, check=True)
        try:
            subprocess.run(["cmake", "--build", str(build), "--parallel", env["MAX_JOBS"],
                            "--target", "practice_solution"], env=env, check=True)
        except subprocess.CalledProcessError as error:
            print("CE · 编译失败", file=sys.stderr)
            raise
        command.append(str(build / "libpractice_solution.so"))
        if action == "test":
            run_judge(command, source.parent, env, args.timeout)
        else:
            subprocess.run(command, env=env, cwd=source.parent, check=True)


def audit():
    for platform in PLATFORMS:
        entries = [problem for problem in problems() if problem.platform == platform]
        ready = sum(bool(local_test(problem)) for problem in entries)
        files = sum(solution(problem).is_file() for problem in entries)
        print(f"{PLATFORMS[platform]}  条目 {len(entries)}  实现文件 {files}  测试入口 {ready}")


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    listing = actions.add_parser("list")
    listing.add_argument("status", nargs="?", default="all", choices=("all", "todo", "done"))
    listing.add_argument("platform", nargs="?", default="all", choices=("all", *PLATFORMS))
    actions.add_parser("audit")
    case_list = actions.add_parser("cases")
    case_list.add_argument("platform", nargs="?", default="all", choices=("all", *PLATFORMS))
    case_list.add_argument("key", nargs="?")
    init = actions.add_parser("init")
    init.add_argument("platform", choices=("leetgpu", "xpuoj"))
    for action in ("source", "show", "preview", "toggle", "test", "bench", "profile"):
        sub = actions.add_parser(action)
        sub.add_argument("platform", choices=PLATFORMS)
        sub.add_argument("key")
        if action in ("test", "bench", "profile"):
            sub.add_argument("--case", type=int)
        if action in ("bench", "profile"):
            sub.add_argument("--warmup", type=int, default=8)
        if action == "test":
            sub.add_argument("--timeout", type=float)
        if action == "bench":
            sub.add_argument("--repeats", type=int, default=30)
        if action == "profile":
            sub.add_argument("--tool", choices=("nsys", "ncu"), default="nsys")
    args = parser.parse_args()
    if args.action == "list":
        list_problems(args.status, args.platform)
        return
    if args.action == "audit":
        audit()
        return
    if args.action == "cases":
        if args.key and args.platform == "all":
            parser.error("查询单题用例需要指定题库。")
        env = os.environ.copy()
        if args.platform in ("all", "leetgpu") or args.platform == "xpuoj" and args.key:
            select_gpu(env)
        command = [sys.executable, "-u", str(ROOT / "tools" / "practice_cases.py"), args.platform]
        if args.key:
            command.append(args.key)
        subprocess.run(command, env=env, check=True)
        return
    if args.action == "test" and args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout 必须大于 0。")
    if args.action == "init":
        for problem in problems():
            if problem.platform == args.platform:
                solution(problem, create=True)
        return
    problem = find_problem(args.platform, args.key)
    if args.action == "source":
        print(solution(problem, create=True))
    elif args.action in ("show", "preview"):
        statement(problem)
    elif args.action == "toggle":
        toggle(problem)
    elif args.action in ("test", "bench", "profile"):
        test(problem, args)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")
    except (ValueError, OSError) as error:
        sys.exit(str(error))
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode if error.returncode > 0 else 128 - error.returncode)
