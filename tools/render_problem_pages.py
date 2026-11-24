"""Generate standalone Markdown statements from the checked-in problem snapshots."""
import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
from html.entities import html5
from io import StringIO
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote

import torch

from practice_cases import list_cases
from practice_catalog import PLATFORMS, ROOT, leetgpu_template, local_test, problems, solution
from render_leetgpu import render_html

OUTPUT = ROOT / "problems"
SVG = re.compile(r"(?is)<svg\b[^>]*>.*?</svg>")
TITLE = re.compile(r"(?m)^\s*#\s+[^\n]+\n")
C_SIGNATURE = re.compile(r'extern\s+"C"\s+void\s+solve\s*\([^)]*\)')
FAMILY_GOALS = {
    "elementwise": "计算两个同形张量的逐元素加法，按该变体的标量或向量宽度写回输出。",
    "histogram": "统计非负整数输入的出现次数。按最大输入索引确定直方图长度，并正确累加重复值。",
    "sigmoid": "逐元素计算 sigmoid，保持输入与输出的形状和变体规定的精度。",
    "relu": "逐元素计算 max(x, 0)，覆盖负输入、零以及不同向量宽度。",
    "elu": "逐元素计算 ELU。正值直接输出，非正值使用 exp(x) - 1。",
    "gelu": "逐元素计算 tanh 近似 GELU。关注低精度下中间指数溢出对有限输出的影响。",
    "swish": "逐元素计算 x × sigmoid(x)，保留精度与 pack 方式。",
    "hardswish": "逐元素计算 x × clamp(x + 3, 0, 6) / 6，覆盖 -3 与 3 的分段边界。",
    "hardshrink": "保留绝对值大于 0.5 的输入，其余位置写零。验证阈值两侧的结果。",
    "embedding": "按 token 索引从二维权重表收集行。保持输出行顺序，并覆盖重复 token。",
    "mat-transpose": "把 M×N 输入转置成 N×M，说明读写方向和共享内存布局。",
    "reduce": "计算输入所有元素的和。遵守该变体的输入与累加精度，处理多 warp 和多 block。",
    "dot-product": "对两个同形输入计算内积，验证低精度乘法和分块归约。",
    "softmax": "沿最后一维做归一化。safe 和 online 变体需要避免大 logits 的指数溢出。",
    "rope": "按 token 位置旋转相邻的通道对。位置零应保持恒等，长序列应控制相位误差。",
    "layer-norm": "沿最后一维计算均值和方差，再归一化并施加缩放和平移。",
    "rms-norm": "沿最后一维按均方根归一化，再乘以缩放。使用模块定义的 epsilon。",
    "nms": "按置信度从高到低保留候选框，抑制与已保留框的 IoU 超过阈值的候选。",
    "sgemv": "用 FP32 计算矩阵和单列向量的乘积。按选定变体组织 K 维载入和归约。",
    "hgemv": "用 FP16 输入计算矩阵和单列向量的乘积，核对输出形状与累加误差。",
    "sgemm": "计算 C = A × B。根据条目使用 FP32 CUDA Core 或 TF32 Tensor Core 路径。",
    "hgemm": "用 FP16 输入计算 C = A × B。根据条目实现相应的 tile、pipeline、布局或 cuBLAS 对照。",
    "ws-hgemm": "练习 warp-specialized FP16 GEMM，在保持正确性的前提下将搬运和计算重叠。",
    "flash-attn": "计算 scaled dot-product attention，用流水线和片上存储组织 QK、归一化和 PV。",
    "ffpa": "计算大 head 维度的注意力，按条目实现分维、split-KV 解码或 persistent CTA。",
    "swizzle": "比较共享内存的行列地址、bank 映射和 swizzle 后的访问顺序。",
    "nvidia-nsight": "为一个 CUDA 算子采集时间线和单 kernel 计数器，并定位 launch 与 kernel 执行区间。",
    "merge-attn-states": "用稳定的 log-sum-exp 权重合并多个 attention 分片的输出和归一化状态。",
    "vector-add": "用 Triton 计算两个向量的和，并用掩码覆盖无法填满一个 block 的尾部。",
    "fused-softmax": "用 Triton 在一个 kernel 内完成逐行 softmax，遮蔽超出行宽的位置。",
    "layer_norm": "用 Triton 实现 LayerNorm 前向和反向路径，对照 PyTorch 的输出及梯度。",
    "interview": "在实战套件使用的布局和精度下实现本条目算子，对照套件提供的参考误差。",
}
INTERVIEW_GOALS = {
    "SGEMV": FAMILY_GOALS["sgemv"], "SGEMM": FAMILY_GOALS["sgemm"], "HGEMM": FAMILY_GOALS["hgemm"],
    "Collective GEMM": "把分块搬运、同步和矩阵乘积组织成 collective，验证 TMA/warp specialization 路径。",
    "FP8 GEMM": "实现带 scale 的 FP8 矩阵乘法，覆盖不同 scale 布局和非整 tile。",
    "FP4 GEMM": "实现 NVFP4 矩阵乘法，比较单级和两级 scale，并报告相对于 CPU FP64 的误差。",
    "FlashAttention": FAMILY_GOALS["flash-attn"], "FFPA Attention": FAMILY_GOALS["ffpa"],
}


def relative_link(path, target):
    return quote(os.path.relpath(path, target.parent), safe="/._-")


def fenced(text, language=""):
    return f"~~~{language}\n{text.rstrip()}\n~~~"


def xml_entity(match):
    name = match[1]
    if name in ("amp", "lt", "gt", "quot", "apos"):
        return match[0]
    decoded = html5.get(name + ";")
    if decoded is None:
        return "&amp;" + name + ";"
    return "".join(f"&#x{ord(character):x};" for character in decoded)


def leetgpu_body(problem, target, emit):
    html = problem.path.read_text()
    next_diagram = 0

    def extract(match):
        nonlocal next_diagram
        next_diagram += 1
        name = f"{problem.key}-{next_diagram}.svg"
        svg = re.sub(r"&([a-zA-Z][a-zA-Z0-9]+);", xml_entity, match[0])
        if "xmlns=" not in svg.split(">", 1)[0]:
            svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
        emit(target.parent / "assets" / name, svg + "\n")
        return f'<img src="assets/{name}" alt="题面图示 {next_diagram}">'

    text = render_html(SVG.sub(extract, html))
    starter = leetgpu_template(problem)
    if starter.suffix == ".cu":
        source = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", starter.read_text())
        signature = C_SIGNATURE.search(source)[0]
        interface = fenced(signature + ";", "cpp")
    else:
        solve = next(item for item in ast.parse(starter.read_text()).body
                     if isinstance(item, ast.FunctionDef) and item.name == "solve")
        interface = fenced(ast.unparse(solve).split("\n", 1)[0], "python")
    return text + "\n\n## 本地接口\n\n" + interface


def xpu_body(problem):
    if not problem.path.is_relative_to(problem.pack):
        text = TITLE.sub("", problem.path.read_text(), count=1)
        return re.sub(r"\]\((/[^)]+)\)", r"](https://xpuoj.com\1)", text).strip()

    language = ".tilelang.md" if solution(problem).suffix == ".py" else ".cuda.md"
    sections = []
    for path in sorted(problem.path.parent.glob("0[0-5]_*.md")):
        if path.name.startswith("01_") and not path.name.endswith(language):
            continue
        parts = path.read_text().split("---", maxsplit=2)
        if len(parts) != 3:
            sections.append(path.read_text().strip())
        else:
            title = re.search(r'(?m)^sectionTitle:\s*["\']?([^"\'\n]+)', parts[1])
            heading = title[1].strip() if title else path.stem[3:]
            sections.append(f"## {heading}\n\n{parts[2].strip()}")
    return re.sub(r"\]\((/[^)]+)\)", r"](https://xpuoj.com\1)", "\n\n".join(sections))


def leetcuda_body(problem, target):
    path = problem.path
    if path is None:
        family = "ffpa"
    else:
        parts = path.relative_to(ROOT / "references" / "LeetCUDA").parts
        family = parts[2] if parts[1] == "openai-triton" else parts[1]
        if family == "layer-norm" and parts[1] == "openai-triton":
            family = "layer_norm"
    goal = INTERVIEW_GOALS.get(problem.title, FAMILY_GOALS[family])
    if "wgmma" in problem.title:
        goal += " 本变体使用 Hopper SM90a 的 WGMMA。验证时应先选与该 tile 尺寸相匹配的对齐矩阵，再比较不同 M、N 和 K。"
    if problem.key in ("194", "195"):
        goal += " 本工作区使用实战篇的 persist-D 实现，覆盖 dense、causal、GQA 和尾部 tile。"
    body = f"## 练习目标\n\n{goal}\n\n"
    if problem.detail:
        body += f"清单规格为 {problem.detail.removeprefix('·').strip()}。\n\n"
    if path:
        folder = path if path.is_dir() else path.parent
        # Some grouped variants keep their driver a few levels above the kernel file.
        for parent in (folder, *folder.parents):
            if parent.name == "LeetCUDA":
                break
            guide = parent / "README.md"
            if guide.is_file():
                body += f"[算子说明]({relative_link(guide, target)})\n\n"
                break
    return body.rstrip()


def case_section(problem):
    output = StringIO()
    torch.manual_seed(2026)
    with redirect_stdout(output):
        list_cases(problem)
    lines = output.getvalue().splitlines()[1:]
    details = fenced("\n".join(lines), "text")
    commands = [f"./practice.sh cases {problem.platform} {problem.key}"]
    if local_test(problem):
        commands.append(f"./practice.sh test {problem.platform} {problem.key}")
    command = fenced("\n".join(commands), "bash")
    return f"## 本地用例\n\n{command}\n\n<details>\n<summary>展开输入规格</summary>\n\n{details}\n\n</details>"


def page(problem, target, emit):
    source = relative_link(problem.path, target) if problem.path else problem.link
    editable = relative_link(solution(problem), target)
    header = (f"# {problem.title}\n\n{PLATFORMS[problem.platform]} · #{problem.key}\n\n"
              f"[打开实现文件]({editable}) · [原始材料]({source})")
    if problem.platform == "leetgpu":
        body = leetgpu_body(problem, target, emit)
    elif problem.platform == "xpuoj":
        body = xpu_body(problem)
    else:
        body = leetcuda_body(problem, target)
    return "\n\n".join((header, body, case_section(problem))) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    pending = []

    def emit(path, content):
        content = "\n".join(line.rstrip() for line in content.rstrip().splitlines()) + "\n"
        if args.check:
            if not path.is_file() or path.read_text() != content:
                pending.append(path.relative_to(ROOT))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    counts = Counter()
    with torch.no_grad():
        for problem in problems():
            target = OUTPUT / problem.platform / f"{problem.key}.md"
            emit(target, page(problem, target, emit))
            counts[problem.platform] += 1
    index = ["# 题面目录", "", "| 题库 | 条目 |", "| --- | ---: |"]
    for platform, title in PLATFORMS.items():
        index.append(f"| [{title}](../catalogs/{platform}.md) | {counts[platform]} |")
    emit(OUTPUT / "README.md", "\n".join(index) + "\n")
    if pending:
        sys.exit(f"{len(pending)} 个 Markdown 或图示需要重新生成，首个为 {pending[0]}")
    print(" · ".join(f"{PLATFORMS[platform]} {count}" for platform, count in counts.items()))


if __name__ == "__main__":
    main()
