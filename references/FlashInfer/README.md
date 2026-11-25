# FlashInfer 上游源码快照

选自 FlashInfer `v0.6.0`，提交 `4230a48aedd21fc8823249147c4b8b363bb0dc36`。目录保持上游路径，保留 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。

- [分页与 ragged KV 布局](docs/tutorials/kv_layout.rst) · [页表 CUDA 实现](include/flashinfer/page.cuh) · [追加测试](tests/attention/test_page.py)
- [Cascade 合并](include/flashinfer/attention/cascade.cuh) · [Log₂ attention 状态](include/flashinfer/attention/state.cuh) · [Cascade 对拍](tests/utils/test_triton_cascade.py)
- [Decode CUDA](include/flashinfer/attention/decode.cuh) · [调度](include/flashinfer/attention/scheduler.cuh) · [Batch decode 测试](tests/attention/test_batch_decode_kernels.py) · [Decode LSE 测试](tests/attention/test_decode_prefill_lse.py)

这些文件供离线源码阅读。练习从固定的 PyTorch 数值参考和 CUDA C++ 模板开始，完整项目入口见上游。[固定提交](https://github.com/flashinfer-ai/flashinfer/tree/4230a48aedd21fc8823249147c4b8b363bb0dc36)
