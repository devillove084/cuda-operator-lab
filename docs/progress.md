# 我的练习进度

[首页](../README.md) · [学习路线](roadmap.md) · [全部题目](../problems/README.md)

~~~bash
./practice.sh progress --edit
./practice.sh progress
~~~

第一条命令创建并打开你在这份仓库中的 `docs/progress.local.md`。做完一项，把 `[ ]` 改成 `[x]` 并保存。以后用第二条命令查看自己的进度。个人文件已被 Git 忽略。

每个阶段的最后一项包含完整自测和一项可复查的实验。单题进度可在终端题库用 Ctrl-X 标记。

## 01 · 索引与访存

[路线与阶段验收](roadmap.md#stage-1)

- [ ] [Vector Addition](../problems/leetgpu/1.md) 全测通过，能手算长度 33 的尾部。
- [ ] [ReLU](../problems/leetgpu/21.md) 全测通过。
- [ ] [Matrix Transpose](../problems/leetgpu/3.md) 实现朴素版和共享内存版，验证非方阵。
- [ ] `./practice.sh selftest 1` 通过。记录一个小向量和一个大向量的延迟与有效带宽。

## 02 · 归约与同步

[路线与阶段验收](roadmap.md#stage-2)

- [ ] [Dot Product](../problems/leetgpu/17.md) 通过，覆盖不满一个 warp 的输入。
- [ ] [Reduction](../problems/leetgpu/4.md) 写出 shared-memory tree 与 warp shuffle 两版。
- [ ] [Histogramming](../problems/leetgpu/13.md) 通过，对比均匀输入和热点冲突。
- [ ] `./practice.sh selftest 2` 通过。保存尾例的 memcheck 和 racecheck 结果。

## 03 · 数值稳定与融合

[路线与阶段验收](roadmap.md#stage-3)

- [ ] [Softmax](../problems/leetgpu/5.md) 通过，能手造让朴素指数溢出的反例。
- [ ] [LayerNorm](../problems/leetgpu/113.md) 与 [RMSNorm](../problems/leetgpu/50.md) 通过，验证近常数和尾部输入。
- [ ] [Fused Residual Add and RMSNorm](../problems/leetgpu/83.md) 通过，用时间线比较拆分和融合的 launch。
- [ ] [Quack Cross Entropy](../problems/quack/1.md) 前反向通过，覆盖忽略行和标签平滑。
- [ ] `./practice.sh selftest 3` 通过。留存一个数值反例和一份融合对比。

## 04 · GEMM、Tensor Core 与 CuTe

[路线与阶段验收](roadmap.md#stage-4)

- [ ] [Matrix Multiplication](../problems/leetgpu/2.md) 和 [SGEMM](../problems/leetcuda/127.md) 通过，比较朴素、共享内存分块、寄存器分块。
- [ ] 跑通 [CuTe HGEMM](../problems/leetcuda/157.md)，能画出一个线程块的 tile、layout 和访存映射。
- [ ] [CUTLASS Grouped GEMM](../problems/cutlass/1.md) 通过，覆盖非齐整分组和零大小维度。
- [ ] [Triton FP8 分块缩放](../problems/triton/1.md) 通过，覆盖最后一个短量化块。
- [ ] `./practice.sh selftest 4` 通过。给小 M、方阵和非对齐矩形各留一组相同精度的基准结果。

## 05 · 推理 Attention 与分页 KV

[路线与阶段验收](roadmap.md#stage-5)

- [ ] [Softmax Attention](../problems/leetgpu/6.md) 和 [不等长滑窗](../problems/flashattn/1.md) 通过，能手算右下对齐掩码。
- [ ] [Paged Gather](../problems/flashinfer/1.md) 和 [Paged Append](../problems/flashinfer/2.md) 通过，验证共享满页和尾页。
- [ ] [Split-KV Merge](../problems/flashinfer/3.md) 通过，覆盖全空分片和极端 LSE。
- [ ] [Paged GQA Decode](../problems/flashinfer/4.md) 通过，测单步、长上下文和较大 batch。
- [ ] [vLLM MoE 路由分桶](../problems/vllm/1.md) 通过，覆盖热点和无效路由。
- [ ] `./practice.sh selftest 5` 通过。保存两种不同形状的 decode 性能及解释。

## 06 · 状态递推、变长和反向

[路线与阶段验收](roadmap.md#stage-6)

- [ ] [Simple GLA](../problems/fla/1.md) 通过，拼接两段递推的输出和末状态。
- [ ] [Gated Delta](../problems/fla/2.md) 通过，检查零 beta 和完全遗忘。
- [ ] [Packed 变长](../problems/fla/3.md) 通过，覆盖空序列和跨块状态。
- [ ] [GLA 反向](../problems/fla/4.md) 通过，用中心差分和 autograd 验证三类梯度。
- [ ] `./practice.sh selftest 6` 通过。保存一份含吞吐和中间状态显存占用的实验结果。

## 综合项目

- [ ] 从分页 Attention 或状态递推选一项，复现一处开源实现的优化。记录回归用例、可复跑命令、性能前后对比和出现退化的形状。 [实验验收](roadmap.md#性能实验如何验收)
