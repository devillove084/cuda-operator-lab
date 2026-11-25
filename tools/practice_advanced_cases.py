"""Offline workload specifications for the source-based exercise sets."""
from dataclasses import dataclass


ADVANCED = {
    "flashinfer": "FlashInfer",
    "fla": "FLA",
    "flashattn": "FlashAttention",
    "cutlass": "CUTLASS",
    "triton": "Triton",
    "quack": "Quack",
    "vllm": "vLLM",
}


@dataclass(frozen=True)
class Workload:
    label: str
    shape: tuple
    variant: str = ""

    def description(self):
        tail = f" · {self.variant}" if self.variant else ""
        return f"{self.label} · {self.shape}{tail}"


# Paged cache: (sequence lengths, page size, KV heads, head dimension[, query heads]).
# Merge: (batch, heads, split count, head dimension).
# Dense recurrence: (batch, tokens, heads, K, V).
# Packed recurrence: (sequence lengths, heads, K, V).
PLANS = {
    ("flashinfer", "1"): (
        Workload("单 token", ((1,), 4, 1, 4)),
        Workload("刚好一页", ((8,), 8, 2, 16)),
        Workload("跨页，尾页一项", ((9, 16, 17), 8, 3, 33)),
        Workload("空序列夹在中间", ((7, 0, 2, 21), 8, 2, 15)),
        Workload("共享首个满页", ((16, 17, 33), 16, 2, 64), "shared"),
        Workload("全空", ((0, 0), 16, 1, 8)),
        Workload("长序列与乱序物理页", ((8193, 3671, 31, 513), 16, 4, 64)),
    ),
    ("flashinfer", "2"): (
        Workload("单 token", ((1,), 4, 1, 4)),
        Workload("每条序列只更新后缀", ((7, 8, 9), 8, 2, 16), "suffix"),
        Workload("跨页的稀疏乱序更新", ((35, 1, 17), 8, 3, 33), "sparse"),
        Workload("含空序列", ((0, 7, 0, 17), 16, 2, 15), "suffix"),
        Workload("整页与尾页同批", ((16, 31, 32, 33), 16, 1, 128)),
        Workload("无更新，缓存原样", ((7, 19), 8, 1, 8), "no_updates"),
        Workload("长序列更新", ((8193, 3671, 31, 513), 16, 4, 64), "sparse"),
    ),
    ("flashinfer", "3"): (
        Workload("单分片", (1, 1, 1, 7)),
        Workload("权重相等", (2, 3, 2, 16), "equal"),
        Workload("部分空分片为 NaN 毒值", (3, 2, 7, 17), "missing"),
        Workload("全空分片", (2, 3, 3, 33), "all_missing"),
        Workload("指数极值", (2, 2, 5, 64), "extreme"),
        Workload("分片数 33 与头维尾部", (1, 5, 33, 65), "missing"),
        Workload("大批量分片合并", (32, 16, 32, 128), "missing"),
    ),
    ("flashinfer", "4"): (
        Workload("单 token 恒等与 GQA", ((1,), 8, 1, 8, 4)),
        Workload("多头 MHA 与整页", ((8, 16), 8, 2, 16, 2)),
        Workload("共享首个满页与 GQA", ((16, 17, 33), 16, 2, 64, 8), "shared"),
        Workload("空序列与非整头维", ((0, 7, 1, 65), 8, 2, 33, 6)),
        Workload("大 logits 与 FP32 LSE", ((31, 17), 8, 2, 64, 8), "extreme"),
        Workload("全空", ((0, 0), 16, 2, 32, 8)),
        Workload("长上下文与不均匀批次", ((8193, 4096, 1025, 17), 16, 4, 128, 16)),
        Workload("较大 batch 与多组 GQA", ((513,) * 16, 32, 2, 128, 8)),
    ),
    ("fla", "1"): (
        Workload("单步", (1, 1, 1, 4, 3)),
        Workload("零初始态，多 head", (2, 31, 3, 8, 7), "zero_state"),
        Workload("无衰减", (1, 33, 2, 9, 11), "no_decay"),
        Workload("一次完全遗忘", (2, 65, 2, 13, 17), "forget"),
        Workload("多 batch 非方状态", (3, 17, 4, 64, 33)),
        Workload("长序列尾块", (4, 257, 4, 32, 32)),
    ),
    ("fla", "2"): (
        Workload("单步", (1, 1, 1, 4, 3)),
        Workload("beta=0，只衰减", (1, 7, 3, 11, 8), "beta_zero"),
        Workload("beta=1，无衰减", (2, 33, 2, 17, 15), "overwrite"),
        Workload("一次完全遗忘", (2, 65, 2, 13, 17), "forget"),
        Workload("零初始态", (3, 17, 4, 64, 33), "zero_state"),
        Workload("长序列尾块", (4, 257, 4, 32, 32)),
    ),
    ("fla", "3"): (
        Workload("单序列和单状态", ((1,), 1, 4, 3)),
        Workload("首尾空序列", ((0, 7, 0), 3, 11, 8)),
        Workload("标称 32 的块边界", ((31, 32, 33), 2, 17, 15), "overwrite"),
        Workload("同一调用中所有序列为空", ((0, 0), 2, 13, 17)),
        Workload("非均匀序列与初始状态隔离", ((129, 1, 0, 2, 65), 3, 19, 33)),
        Workload("长序列与小序列混合", ((257, 127, 63, 33, 1, 0), 4, 32, 32)),
    ),
    ("fla", "4"): (
        Workload("单步核对所有梯度", (1, 1, 1, 4, 3)),
        Workload("只有输出梯度", (2, 7, 2, 11, 5), "output_only"),
        Workload("只有最终状态梯度", (1, 17, 3, 9, 13), "state_only"),
        Workload("无衰减，门梯度仍有效", (1, 33, 2, 9, 11), "no_decay"),
        Workload("零初始态，多 batch", (3, 19, 4, 32, 17), "zero_state"),
        Workload("长序列反向与尾块", (2, 129, 4, 32, 32)),
    ),
    ("flashattn", "1"): (
        Workload("Q 比 KV 短的右下对齐", (1, 2, 5, 2, 7, -1)),
        Workload("Q 比 KV 长与全掩码行", (1, 5, 2, 2, 16, -1)),
        Workload("滑窗只含当前位置", (2, 7, 17, 1, 33, 0)),
        Workload("不等长的滑窗", (2, 33, 17, 3, 15, 8)),
        Workload("极端 logits", (2, 31, 65, 2, 64, 15), "extreme"),
        Workload("序列与头维尾部", (2, 129, 513, 4, 65, 97)),
        Workload("长序列 prefill", (1, 512, 2048, 8, 64, -1)),
    ),
    ("cutlass", "1"): (
        Workload("单组单元素", ((1, 1, 1),)),
        Workload("长短矩形组", ((1, 7, 9), (17, 5, 3), (3, 19, 11))),
        Workload("M、N 或 K 为空", ((0, 9, 11), (5, 0, 7), (3, 9, 0), (2, 3, 4))),
        Workload("全部组的 M 为零", ((0, 4, 3), (0, 9, 11))),
        Workload("非整 tile 和分隔区", ((31, 17, 65), (49, 33, 15), (1, 7, 129))),
        Workload("多组小 M 和大组混合", ((129, 513, 255),) + ((1, 128, 255),) * 8),
        Workload("矩形组压力", ((257, 129, 513), (129, 511, 255), (31, 513, 1023))),
    ),
    ("triton", "1"): (
        Workload("一个量化块", (1, 1, 16, 16)),
        Workload("非整量化块", (7, 9, 35, 16)),
        Workload("非对称布局与零 scale", (17, 5, 33, 8), "zeros"),
        Workload("K 比量化块小", (3, 9, 7, 32)),
        Workload("量化极值与跨块尺度", (9, 11, 257, 32), "extreme"),
        Workload("多块长 K 与奇数 M/N", (129, 65, 1025, 32)),
    ),
    ("quack", "1"): (
        Workload("词表只有一项", (2, 1, 0.0)),
        Workload("默认交叉熵与多行尺度", (7, 33, 0.0)),
        Workload("标签平滑与忽略行", (5, 32, 0.15), "ignore"),
        Workload("全忽略", (3, 19, 0.2), "all_ignore"),
        Workload("大 logits 与接近并列", (3, 65, 0.1), "extreme"),
        Workload("全平滑与非整词表", (17, 997, 1.0)),
        Workload("大词表归约", (32, 50257, 0.05), "ignore"),
    ),
    ("vllm", "1"): (
        Workload("单 token 和空 expert", (1, 1, 4, 4)),
        Workload("均匀路由与整块边界", (32, 2, 8, 8), "round_robin"),
        Workload("单 expert 热点", (33, 2, 16, 16), "hot"),
        Workload("无效 route 与非整 tile", (19, 3, 11, 11), "invalid"),
        Workload("全部 route 无效", (17, 2, 8, 8), "all_invalid"),
        Workload("零 token", (0, 4, 11, 16)),
        Workload("多 expert 大批次", (4097, 4, 64, 32), "invalid"),
    ),
}


def advanced_plan(platform, key):
    return PLANS[(platform, key)]
