# GPU 版物理筛选（自穿透 469 对）—— 实现与验证

日期：2026-11-22　服务器：`gpufree-container`（L40 46 GB，torch 2.10.0+cu128，CUDA 12.8）
基准数据：0009 归档 `smp_gsi_paired_0009/formal/0009_location_seed2027_..._vwjhPg/batch_*.pt`

## 改了什么

| 文件 | 作用 |
|---|---|
| `gpu_filter.py` | `GpuPhysicalFilter(PhysicalFilter)`：只把 `process` 里 469 对线段距离循环搬到 GPU（torch float64），其余（关节/穿地/FK/诊断）完全沿用参考实现 |
| `gpu_scorer.py` | `GpuScorer`：替代 `PhysicalScorer.evaluate`，一次 FK + 一次 469 对计算，同时给出 `passed` 与连续 `overlap`，消除参考实现里重复的第二遍 |
| `gpu_validate.py` / `gpu_score_validate.py` | 等价性验证脚本 |
| `bs_sweep.py` / `gpu_search_bench2.py` | 基准脚本 |

## 等价性验证（逐条，不是近似）

**筛选器**（`passed`、输出数组、全部诊断计数）：

| 数据集 | 条数 | 结果 |
|---|---|---|
| 标定 300 + 300 | 600 | 全部一致 |
| 随机 | 4096 | 全部一致 |
| 修正模式（repair=True） | 256 | 全部一致 |

**评分器**（`passed`/`numeric`/`joint`/`ground`/`overlap`/`fitness` 六个数组）：

| 数据集 | 条数 | 结果 |
|---|---|---|
| 标定 300 + 300 | 600 | `ALL_EQUAL` |
| 随机 | 4096 | `ALL_EQUAL` |

48M 个“对×帧”判定中符号翻转 0 次，`max|Δgap| = 4.4e-16` m（float64 机器精度）。

**注意**：`overlap` 的连续值最初差 `1.4e-14 mm`（torch 与 numpy 的 `sqrt` 差 1 ulp），
现已改为“用 numpy 精确重算所有实际穿透的对”，因此也做到逐位一致。

## 性能

单次 `process()`（含 FK、关节、穿地），每窗口毫秒：

| 批量 | numpy | GPU | 加速 |
|---|---|---|---|
| 32 | 3.014 | 0.320 | 9.4× |
| 128 | 1.971 | 0.214 | 9.2× |
| 512 | 1.752 | 0.225 | 7.8× |
| 2048 | 1.732 | 0.238 | 7.3× |
| 8192 | 1.868 | 0.240 | 7.8× |

端到端 `search_batch`（1024 行、随机初始噪声、30 代上限）：

| 实现 | 墙钟 | 每候选 | 代数 | 成功 | 耗尽 |
|---|---|---|---|---|---|
| CPU 参考 | 473.4 s | 2.750 ms | 5380 | 890 | 134 |
| GPU 版 | **71.5 s** | **0.415 ms** | 5380 | 890 | 134 |

**6.6× 端到端，且通过的条目集合逐条相同（`verdict identical: True`）。**

## 折算到正式实验

| 场景 | 现状 | GPU 后 |
|---|---|---|
| 00083 搜索 17,610 条 | 8.56 h | ≈ 1.3 h |
| 00083 整跑（含训练 1.5 h） | 10.06 h | ≈ 2.8 h |

## 使用方法

```python
import sys; sys.path.insert(0, "/root/gpufree-data/smp-g1/tools/gpu_filter")
from gpu_filter import GpuPhysicalFilter
from gpu_scorer import GpuScorer

flt = GpuPhysicalFilter(env.sim.mj_model, FilterConfig(repair=False, control_dt=0.02), device="cuda:0")
scorer = GpuScorer(flt)          # 直接传给 search_batch
```

## 限制与风险

1. **只支持 fp64**（GPU）：fp32 会有判定翻转，未采用。
2. **FK 仍在 CPU**（mujoco 逐窗口），占 `process` 时间约 8%；如需再快需搬 FK，风险更高。
3. **需要 CUDA**：无 GPU 时无法使用。
4. 等价性是在**当前模型**（G1，33 碰撞体/469 对，manifest `9b039f63…`）上验证的；换模型或改碰撞体配置后必须重跑验证脚本。
5. 已完成的实验（0002/0003/00031/00033/0005/00051/00052/00053/0008/00083/0009/00091）**未使用**本实现，其结果不受影响。
