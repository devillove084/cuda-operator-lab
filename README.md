# CUDA Operator Lab

**离线练 CUDA kernel、CUTLASS 和 CuTe。**

从一道题的输入输出出发，写实现、做对拍，再用同一张 GPU 比较性能。题库、可编辑代码、参考源码和用例随仓库提供。

| 题库 | 条目 | 练习路径 |
| --- | ---: | --- |
| [LeetGPU](catalogs/leetgpu.md) | 147 | `practice/leetgpu/<题号>/solution.cu` |
| [XPUOJ](catalogs/xpuoj.md) | 144 | `practice/xpuoj/<题号>/solution.cu` |
| [LeetCUDA](catalogs/leetcuda.md) | 215 | `practice/leetcuda/kernels/<算子>/` |

少数 Python / TileLang 题使用 `solution.py`。LeetCUDA 的多个变体可能共用一份源码文件。

## 快速开始

~~~bash
git clone https://github.com/devillove084/cuda-operator-lab.git
cd cuda-operator-lab
./practice.sh
~~~

克隆完成后，列题、编辑、编译、判题和性能测试都使用本地文件。运行需要 Bash、Python 3.10+、带 CUDA 的 PyTorch、CUDA Toolkit、CMake 和 Ninja。交互菜单使用 `fzf` 和 `less`。可以直接运行命令行子命令。

| 菜单按键 | 操作 |
| --- | --- |
| 输入题名 · ↑ / ↓ | 搜索、选题 |
| Enter · q | 打开题面、退出题面 |
| Ctrl-E | 编辑实现 |
| Ctrl-T · Ctrl-B | 编译并判题、跑性能 |
| Ctrl-S · Ctrl-P | 切换进度筛选、切换题库 |
| Ctrl-X · Esc | 标记完成、退出 |

编辑器按 `PRACTICE_EDITOR`、`VISUAL`、`EDITOR` 的顺序选择，默认使用 `vi`。也可以直接用任意编辑器打开实现文件。用 `./practice.sh source <题库> <题号>` 查询路径。

~~~bash
EDITOR=vim ./practice.sh edit leetgpu 1
./practice.sh show xpuoj 67
./practice.sh list todo leetcuda
~~~

## 判题与性能

~~~bash
./practice.sh cases                         # 统计逐题用例
./practice.sh cases leetgpu 1               # 查看输入形状、类型和用例编号
./practice.sh test leetgpu 1                # 编译一次，逐例判定
./practice.sh test xpuoj 67 --case 3        # 单例复跑
./practice.sh test leetcuda elementwise_f32
./practice.sh bench leetgpu 1
./practice.sh profile leetgpu 1 --case 6 --tool ncu
~~~

- **判题** 用固定输入或随机种子生成张量，运行参考实现，再比对完整的输出。独立判题输出 `AC`、`WA`、`CE` 或 `RE`，也可以用 `--timeout 秒` 设置整个评测进程时限。LeetGPU 的 `--case 0` 单跑题面样例。
- **性能** 先验算结果，再在同一 GPU、形状和精度下采样。统一性能报告记录 p10、p50、p90 和原始样本。LeetCUDA 也提供上游算子驱动和实战套件。
- **分析** `profile` 用 Nsight Systems 记录时间线，或用 `--tool ncu` 采集硬件计数器。报告写入 `build/practice/reports/`。
- **进度** `Ctrl-X` 更新 `catalogs/` 内的复选框。测试和性能测量可以重复运行。

LeetGPU 使用公开功能用例。XPUOJ 同时使用公开题包的校验器和依据题面编写的本地用例。LeetCUDA 清单区分逐变体判题、算子级入口和阅读条目，Hopper WGMMA 示例需要相应硬件编译和运行。用 `cases` 查看当前覆盖。

## 学习路线

| 阶段 | 练习 | 验收 |
| --- | --- | --- |
| 1 | Vector Addition → ReLU → Transpose | 处理尾部线程和非方阵，检查合并访存 |
| 2 | Dot Product → Reduction → Histogram | 对比共享内存和 warp shuffle，覆盖部分 warp |
| 3 | Softmax → LayerNorm → RMSNorm | 处理大值和极小方差，验证累加精度 |
| 4 | SGEMM → HGEMM → CUTLASS GEMM | 覆盖非整 tile 尺寸，和 cuBLAS 比较 |
| 5 | CuTe layout → Attention → FP8 / FP4 | 验证布局、mask、长序列和量化误差 |

## 环境与源码

在 `pro5000` 上验证的是 CUDA 13.2、RTX PRO 5000 Blackwell 和 PyTorch CUDA 版。设置 `PRACTICE_PYTHON` 可指定 Python，`CUDA_HOME` 可指定 Toolkit。`CUDA_VISIBLE_DEVICES` 可选择 GPU。CMake 默认编译 SM120a，用 `PRACTICE_CUDA_ARCH` 可覆盖，例如 `86-real`。LeetCUDA 的专项驱动按各自的 GPU 要求构建。界面可以在任何支持的编辑器旁边使用。

`references/` 保存冻结的题面、上游参考实现和编译需要的头文件。`practice/` 保存可编辑的实现。上游保留原有著作权和许可文本，题目作者和来源见各清单。

- LeetGPU 题目与模板来自 [AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges) 的 `3b2ec31`。
- LeetCUDA 来自 [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA) 的 `v4.2.2`。
- XPUOJ 来自 [公开题库](https://xpuoj.com/p) 和 [XPUOJ-ProblemSet](https://github.com/XPUOJ/XPUOJ-ProblemSet) 的 `1a1dd54`。
- CUTLASS / CuTe、cuDNN Frontend 和 FFPA 的上游许可与源码一并保存在对应目录。
