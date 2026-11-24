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
    ("零输入与非整行数", (7, 248)),
    ("幅值交替", (65, 264)),
    ("另一组宽向量", (2049, 2048)),
)
ROW_CASES = (
    ("全零行", (1, 256)),
    ("不同尺度的相邻行", (7, 256)),
    ("数值边界", (33, 512)),
    ("非整行数与宽行", (257, 1024)),
    ("大量行与宽向量", (2049, 1024)),
    ("相同元素的一整行", (3, 512)),
    ("相邻列和值域变化", (65, 256)),
    ("另一组大矩阵", (4097, 512)),
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
    if family == "mat_transpose":
        shapes = ((32, 32), (128, 128), (256, 256), (1024, 1024), (2048, 2048))
        if "diagonal" not in title:
            shapes = ((64, 128), (128, 64), (256, 128), (512, 1024), (1024, 2048))
        return tuple(("矩阵转置", shape) for shape in shapes)
    if family in ("hgemv", "sgemv"):
        width = 16 if "_k16_" in title else 128
        return tuple(("矩阵乘向量", (rows, width)) for rows in (128, 256, 512, 1024, 4096))
    if family == "rope":
        return (("零位置恒等", (1, 128)), ("短序列", (31, 256)),
                ("长序列", (513, 512)), ("宽 head", (257, 1024)),
                ("工作负载", (4096, 512)))
    if family == "block_all_reduce":
        return (("全零", (1, 512)), ("正负抵消", (2, 1024)),
                ("非整行数", (33, 1024)), ("重复值", (257, 2048)),
                ("大规模", (1025, 4096)))
    if family == "sgemm":
        return (("小矩阵", (256, 256, 256)), ("N 大于 M", (256, 512, 256)),
                ("K 大于 N", (512, 256, 512)), ("零乘积", (512, 512, 256)),
                ("多 tile", (1024, 1024, 1024)))

    if family in POINTWISE:
        return POINTWISE_CASES
    if family in ("softmax", "layer_norm", "rms_norm"):
        return ROW_CASES
    if family == "histogram":
        return HISTOGRAM_CASES
    if family == "embedding":
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
