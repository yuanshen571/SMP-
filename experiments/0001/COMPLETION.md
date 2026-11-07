# 实验 0001 完成摘要（G1 Location baseline）

运行标识：`0001_location_seed2027_20261106_225159`
生成时间：`2026-11-07T00:21:33.067813+08:00`（Asia/Shanghai）

## 结果

| 项目 | 数值 |
| --- | --- |
| Python 退出码 | `0` |
| 完成迭代 | `10000` / 10000 |
| 训练样本 | `245760000` / 245760000 |
| 训练耗时 | 1.47 h（`Time/training_elapsed_s`） |
| 环境数 / 每轮步数 / 种子 | 1024 / 24 / 2027 |
| TensorBoard 固定标量 | 21/21 |
| TensorBoard 条件标量 | 7/7 |
| 最终检查点 | `model_9999.pt` |
| 最终检查点 SHA256 | `1c4644d83bc8b28b8979712594d29d10e87cd8b4bc20d17d2754b725dd949adf` |
| final=model_9999.pt actor_params=218977 max|d|=3.7981e-01 | |

## 训练趋势（首 100 轮 vs 末 100 轮，TensorBoard 原值窗口均值）

| 指标 | 首 100 轮 | 末 100 轮 |
| --- | --- | --- |
| 任务奖励（原始 r_task） | 0.3089 | 0.4916 |
| SMP 奖励（原始 r_smp） | 0.1721 | 0.2204 |
| 乘积奖励 r_product | 0.0531 | 0.1197 |
| 环境返回奖励（0.02×r_product） | 0.0011 | 0.0024 |
| 目标距离均值 m | 5.5233 | 4.3488 |
| 完整回合长度（步） | 21.7759 | 776.6642 |
| 完整回合失败率 | 1.0000 | 0.2757 |

## 产出位置

- 日志 / TensorBoard：`/root/gpufree-data/smp-g1/repo/logs/0001_location_seed2027_20261106_225159`
- 检查点 / 索引：`/root/gpufree-data/smp-g1/repo/ckpt/0001_location_seed2027_20261106_225159`
- 先验：`datasets/pretrain_ckpt/pretrained_lafan_run.pt`（sha256 `74787104ff124191136eeed2268f79448da3ac2e30e710d42aebfc5a60e58cb3`，未修改）
- 代码提交：`382f0f03f73c66ae225e487b2ac53e4f121a4a1a`（分支 `exp0001-location-baseline`）；启动时工作区包含本实验新增的 `experiments/0001/` 支撑文件（见 `run_info.json` 的 `git_dirty`）

## 验收明细

| 检查项 | 结果 | 说明 |
| --- | --- | --- |
| python exit code == 0 | PASS | exit_code=0 |
| run reported 10000 iterations / 245760000 samples | PASS | console tail matched |
| run_info completed_iterations | PASS | 10000 |
| run_info total_samples | PASS | 245760000 |
| run_info fixed parameters (1024 envs / 24 steps / seed 2027) | PASS | 1024/24/2027 |
| config deviations recorded | PASS | dec=4 horizon=20.0 dt=0.005 scaled=True |
| metrics layer enabled | PASS | True |
| prior file unchanged (sha256) | PASS | 74787104ff124191 |
| checkpoint index present | PASS | 22 entries |
| interval + final + initial checkpoints present | PASS | 22 files |
| every indexed checkpoint matches its sha256 | PASS | 22 verified |
| native iter -> completed iterations mapping (i+1) | PASS | 22 entries |
| final checkpoint loads with actor/critic/iter | PASS | keys=['actor_state_dict', 'critic_state_dict', 'infos', 'iter', 'optimizer_state_dict'] |
| policy actually learned (actor MLP weights changed vs untrained) | PASS | max|d|=3.7981e-01 actor_params=218977 |
| obs normalizer count == training samples | PASS | 245760000 |
| action std stays fixed at 0.30 (learn_std=False) | PASS | 0.3000 |
| all 21 fixed TensorBoard scalars present | PASS | 21/21 |
| each fixed scalar has 10000 unique steps | PASS | 21 tags |
| progress axis is 1..10000 with step == value | PASS | 1..10000 |
| training samples final == 245760000 | PASS | 245760000 |
| count identities hold on every iteration | PASS | violations=0 iters=10000 |
| failure duration mean recomputable from sum/count | PASS | checked=10000 bad=0 |
| episode length mean recomputable from sum/count | PASS | checked=10000 bad=0 |
| env reward == 0.02 x r_product (dt scaling) | PASS | mean ratio=0.020000 |
| key reward/time scalars finite | PASS | non-finite=0 |
| no forbidden per-iteration/video/GSI artefacts | PASS | clean |

## 结论

全部检查通过（26 项）。

## 复现命令

```bash
cd /root/gpufree-data/smp-g1 && source env.sh && cd repo
experiments/0001/launch_0001.sh            # 正式训练（阶段 C）
experiments/0001/accept_0001.py --num-envs 64 --steps 2400 --out logs/_accept/b_stage  # 阶段 B
experiments/0001/verify_completion.py --log-dir logs/<run> --ckpt-dir ckpt/<run> --out COMPLETION.md
```

TensorBoard：`127.0.0.1:6009`（`experiments/0001/launch_tensorboard.sh 6009`）。
