# FlashInfer 源码练习清单

这些题把固定上游版本中的测试和算法改写成 CUDA C++ 离线练习，包含独立题面、练习入口和本地用例。

[上游源码快照](../references/FlashInfer/README.md) · [学习路线](../docs/roadmap.md)

- [ ] #1 [分页 KV Cache Gather](../problems/flashinfer/1.md) · 入门 · 地址计算 · [原始材料](../references/FlashInfer/include/flashinfer/page.cuh)
- [ ] #2 [分页 KV Cache Append](../problems/flashinfer/2.md) · 进阶 · 稀疏写入 · [原始材料](../references/FlashInfer/tests/attention/test_page.py)
- [ ] #3 [Split-KV Attention 状态合并](../problems/flashinfer/3.md) · 进阶 · 数值稳定 · [原始材料](../references/FlashInfer/include/flashinfer/attention/cascade.cuh)
- [ ] #4 [Paged GQA Decode](../problems/flashinfer/4.md) · 综合 · LLM 推理 · [原始材料](../references/FlashInfer/include/flashinfer/attention/decode.cuh)
