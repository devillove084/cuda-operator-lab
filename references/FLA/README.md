# Flash Linear Attention 上游源码快照

选自 FLA `v0.4.0`，提交 `2e0f6c31f9db50f010cb157c712c048b8a3a89d2`。目录保持上游路径，保留 [MIT 许可](LICENSE) 和 [引用信息](CITATION.cff)。

- [Simple GLA 数值参考](fla/ops/simple_gla/naive.py) · [递推](fla/ops/simple_gla/fused_recurrent.py) · [Chunk](fla/ops/simple_gla/chunk.py) · [前反向对拍](tests/ops/test_simple_gla.py)
- [Gated Delta Rule 递推](fla/ops/gated_delta_rule/fused_recurrent.py) · [Chunk](fla/ops/gated_delta_rule/chunk.py) · [WY 实现](fla/ops/gated_delta_rule/wy_fast.py) · [推导对应的测试](tests/ops/test_gated_delta.py)

这些文件供离线源码阅读。题库内的 CUDA C++ 实验使用本地 PyTorch 对拍，完整项目入口见上游。[固定提交](https://github.com/fla-org/flash-linear-attention/tree/2e0f6c31f9db50f010cb157c712c048b8a3a89d2)
