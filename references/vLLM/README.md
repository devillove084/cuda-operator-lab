# vLLM 上游源码快照

选自 `v0.11.0`，提交 `b8b302cde434df8c9289a2b465406b47ebab1c2d`。文件保留上游路径和原有 [许可](LICENSE)。CUDA 练习使用仓库内的数值参考与固定工作负载。完整项目见上游。[固定提交](https://github.com/vllm-project/vllm/tree/b8b302cde434df8c9289a2b465406b47ebab1c2d)

- [csrc/moe/moe_align_sum_kernels.cu](csrc/moe/moe_align_sum_kernels.cu)
- [tests/kernels/moe/test_moe_align_block_size.py](tests/kernels/moe/test_moe_align_block_size.py)
- [vllm/model_executor/layers/fused_moe/moe_align_block_size.py](vllm/model_executor/layers/fused_moe/moe_align_block_size.py)
