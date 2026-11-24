"""Numbered workloads for LeetCUDA entries that share an upstream driver."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuiteCase:
    label: str
    specification: str
    args: tuple[str, ...] = ()


def suite_cases(title, key, mode):
    if mode == "ffpa":
        if int(key) in (192, 193):
            return (SuiteCase("单 token 解码", "D=320, Kv=512, Q=1"),
                    SuiteCase("宽 head 解码", "D=512, Kv=1024, Q=1"),
                    SuiteCase("更长 KV cache", "D=320, Kv=2048, Q=1"))
        return (SuiteCase("短 prefill", "D=320, N=512"), SuiteCase("宽 head", "D=512, N=1024"),
                SuiteCase("更长 prefill", "D=320, N=2048"))
    if mode == "interview":
        if int(key) in (194, 195):
            return (SuiteCase("持久 CTA", "实战套件覆盖 dense、causal、GQA、尾 tile",
                              ("--pd-cute",)),)
        if key == "212":
            return (SuiteCase("FP8 量化及尾 tile", "512×512×512；额外非整 tile 由套件生成",
                              ("--fp8-gemm",)),)
        if key == "213":
            return (SuiteCase("FP4 量化及尾 tile", "512×512×512；各 scale 布局、WS 与尾 tile",
                              ("--fp4-gemm",)),)
        if key == "214":
            return (SuiteCase("CuTe FA2", "cp.async 流水线、参考误差",
                              ("--fa2-cute-cpasync",)),
                    SuiteCase("CuTe FA2 TMA", "TMA / warp specialization、参考误差",
                              ("--fa2-cute",)))
        if key == "215":
            return (SuiteCase("Split-D", "CPU FP64 参考与尾 tile", ("--sdnw-cute",)),
                    SuiteCase("持久 CTA", "dense / causal / GQA", ("--pd-cute",)))
        if key == "211":
            return (SuiteCase("TMA 与 MMA", "M=256 N=256 K=256", ("--tma-mma-ws", "256", "256", "256")),
                    SuiteCase("大 K", "M=512 N=512 K=1024", ("--tma-mma-ws", "512", "512", "1024")))
        return (SuiteCase("综合正确性套件", "块归约、SGEMV、SGEMM、HGEMM 和 Attention 的公开参考误差"),)
    if not isinstance(mode, Path):
        return ()
    if mode.stem == "hgemm":
        return tuple(SuiteCase(label, f"M={m} N={n} K={k}",
                               ("--M", str(m), "--N", str(n), "--K", str(k)))
                     for label, m, n, k in (
                         ("方阵", 1024, 1024, 1024), ("更多输出列", 512, 1024, 512),
                         ("长归约", 512, 512, 2048), ("大矩阵", 2048, 2048, 1024)))
    if mode.stem == "flash_attn_mma":
        return tuple(SuiteCase(label, f"B={b} H={h} N={n} D={d}",
                               ("--B", str(b), "--H", str(h), "--N", str(n), "--D", str(d),
                                "--sdpa", "--check", "--f32", "--build-others"))
                     for label, b, h, n, d in (
                         ("标准 head", 1, 8, 512, 64), ("长序列", 1, 4, 2048, 64),
                         ("宽 head", 2, 4, 512, 128), ("宽 head 多 batch", 2, 4, 1024, 128)))
    if mode.stem == "print_swizzle_layout":
        return (SuiteCase("共享内存布局", "打印 swizzle 映射和 bank 分布"),)
    drivers = {
        "nms": "重叠框、固定回归例和生成的框集合",
        "cuda_merge_attn_states": "合并 CUDA attention 状态",
        "test_merge_attn_states": "合并状态与 PyTorch 参考",
        "triton_vector_add": "向量加法与 PyTorch 参考",
        "triton_fused_softmax": "沿行归一化与 PyTorch 参考",
        "triton_layer_norm": "LayerNorm 前向及梯度参考",
        "ws_hgemm": "warp-specialized HGEMM 算子驱动",
        "embedding": "共用的 embedding 驱动，目录中的 f16x2 未导出独立绑定",
    }
    label = drivers.get(mode.stem, f"{title} 的上游驱动")
    return (SuiteCase("原始驱动", label), SuiteCase("固定随机重跑", f"{label} · 种子 2041"))
