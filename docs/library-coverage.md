# 算子库覆盖表

新增题目从上游的算法、测试或编程教程选材，再把语义写成可以独立练习的 CUDA C++ 接口。题面、实现模板、固定种子的用例和源码快照随仓库提供。原样读取上游源码的入口见下表。

| 上游 | 本轮已入库 | 本地功能用例 | 源码与测试 |
| --- | --- | ---: | --- |
| FlashInfer | [分页 gather、append、Split-KV 合并、Paged GQA Decode](../catalogs/flashinfer.md) | 29 | [分页、Cascade 和 Decode 实现及上游测试](../references/FlashInfer/README.md) |
| FlashAttention | [不等长、右下对齐因果 mask、滑窗](../catalogs/flashattn.md) | 7 | [mask、forward 和测试](../references/FlashAttention/README.md) |
| FLA | [Simple GLA、Gated Delta、packed 变长和 GLA 反向](../catalogs/fla.md) | 24 | [递推、分块、WY、前反向测试](../references/FLA/README.md) |
| NVIDIA CUTLASS | [异形 Grouped GEMM 与 Bias/ReLU](../catalogs/cutlass.md) | 7 | [分组 GEMM 示例](../references/cutlass/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu) · [许可](../references/cutlass/LICENSE.txt) |
| Triton | [K 分块缩放 FP8 GEMM](../catalogs/triton.md) | 6 | [Block-scaled matmul 教程](../references/Triton/README.md) |
| Quack | [带 label smoothing 的 Cross Entropy 前反向](../catalogs/quack.md) | 7 | [CuTe DSL 实现、测试与基准](../references/Quack/README.md) |
| vLLM | [MoE 路由分桶与 GEMM tile 对齐](../catalogs/vllm.md) | 7 | [路由 API、CUDA 实现与测试](../references/vLLM/README.md) |

以下现有题库也覆盖常用算子，适合搭配上面的进阶练习。

- **基础 CUDA 与 Triton** 向量操作、转置、归约、归一化、常规 attention、前反向。 [LeetGPU](../catalogs/leetgpu.md) · [LeetCUDA](../catalogs/leetcuda.md)
- **混合精度与推理** FP8 GEMM、分组 masked GEMM、FP4 量化、RoPE/MLA 与采样。 [XPUOJ](../catalogs/xpuoj.md) · [LeetCUDA](../catalogs/leetcuda.md)
- **下一批覆盖缺口** 从 DeepGEMM 或 TensorRT-LLM 选 MoE 分组量化与 scale packing，从 FlashMLA 选压缩 KV 与稀疏索引，从 Transformer Engine 选 FP8 缩放和梯度路径。移植时先定义可在 SM120 跑的数值语义，再单独核对目标库的原生优化架构与版本。[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) · [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) · [FlashMLA](https://github.com/deepseek-ai/FlashMLA) · [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

## 本地覆盖如何验证

~~~bash
./practice.sh list all flashattn
./practice.sh cases cutlass 1
./practice.sh test vllm 1
./practice.sh bench flashinfer 4 --case 7
./practice.sh profile flashattn 1 --case 6 --tool ncu
~~~

用例验证每个输出张量和不应写入的区域，以源码任务的数值定义为准。上游源码快照供阅读和迁移测试，当前性能对照为 PyTorch eager FP32。比较优化库性能时，要对齐形状、布局、精度、架构、输出语义与计时区间。[路线中的实验方式](roadmap.md#性能实验如何验收)
