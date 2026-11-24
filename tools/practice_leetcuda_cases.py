"""Workloads for LeetCUDA bindings with a dedicated correctness checker."""
POINTWISE = {
    "elementwise", "sigmoid", "relu", "elu", "gelu", "swish", "hardswish", "hardshrink",
}

POINTWISE_CASES = (
    ("分段阈值与零", (1, 16)),
    ("非整 block 的向量对齐长度", (31, 72)),
    ("非整行数与较宽行", (257, 1024)),
    ("宽行", (1024, 4096)),
    ("大规模", (8192, 1024)),
)
ROW_CASES = (
    ("全零行", (1, 256)),
    ("不同尺度的相邻行", (7, 256)),
    ("数值边界", (33, 512)),
    ("非整行数与宽行", (257, 1024)),
    ("大量行与宽向量", (2049, 1024)),
)
HISTOGRAM_CASES = (
    ("全落到同一个桶", (16,)),
    ("重复索引与空桶", (256,)),
    ("非整 block", (1028,)),
    ("桶数量大于数据量", (32,)),
    ("大输入与热点", (1048576,)),
)
EMBEDDING_CASES = (
    ("首尾索引与重复 token", (17, 4, 16)),
    ("非整齐词表与奇数序列", (257, 33, 72)),
    ("窄向量", (1024, 127, 64)),
    ("真实形状", (4096, 2048, 512)),
    ("宽向量与大序列", (4097, 4097, 1024)),
)


def leetcuda_cases(family, title):
    if family in POINTWISE:
        return POINTWISE_CASES
    if family in ("softmax", "layer_norm", "rms_norm"):
        return ROW_CASES
    if family == "histogram":
        return HISTOGRAM_CASES
    if family == "embedding" and title != "embedding_f16x2":
        return EMBEDDING_CASES
    if family == "dot_product":
        pack = 8 if "x8" in title else 4 if "x4" in title else 2 if "x2" in title else 1
        return (
            ("最小 warp", (1, 32 * pack)),
            ("完全抵消", (2, 64 * pack)),
            ("非整行数", (37, 128 * pack)),
            ("多 block 归约", (513, 256 * pack)),
            ("大规模与宽向量", (2049, 1024 * pack)),
        )
    return ()
