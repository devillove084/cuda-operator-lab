# Flash Linear Attention 源码练习清单

这些题把固定上游版本中的测试和算法改写成 CUDA C++ 离线练习，包含独立题面、练习入口和本地用例。

[上游源码快照](../references/FLA/README.md) · [学习路线](../docs/roadmap.md)

- [ ] #1 [Simple GLA 递推前向](../problems/fla/1.md) · 进阶 · 状态递推 · [原始材料](../references/FLA/fla/ops/simple_gla/fused_recurrent.py)
- [ ] #2 [Gated Delta Rule 递推前向](../problems/fla/2.md) · 进阶 · 状态更新 · [原始材料](../references/FLA/fla/ops/gated_delta_rule/fused_recurrent.py)
- [ ] #3 [Packed 变长 Gated Delta Rule](../problems/fla/3.md) · 综合 · 批处理与状态隔离 · [原始材料](../references/FLA/fla/ops/gated_delta_rule/chunk.py)
- [ ] #4 [Simple GLA 反向传播](../problems/fla/4.md) · 综合 · 训练算子 · [原始材料](../references/FLA/tests/ops/test_simple_gla.py)
