# 实验 00031：在 0003 基础上加入小幅关节限位修正

完整共同设置、物理条件、四项修正要求、产物规则和两组串行流程见 [总任务单](../0003/EXECUTION.md)。如果总队列已经运行，不要另行启动本文件中的命令。

本文件供 **0003 已正式完成核验 PASS 后，单独继续 00031** 时使用。00031 是独立训练，不加载 0003 最终策略；加载与 0003 相同的 0002 `initial_untrained.pt`，新建优化器。

## 唯一实验差异

`experiments/00031/EXPERIMENT.json` 中 `physical_filter.repair_small_joints=true`。共享的物理筛选/修正实现与 0003 相同，其余训练配置、奖励、缓存容量/配额及刷新规则均不变。

- 整条十帧最大关节超限 ≤1° 才允许修正，任意关节/帧 >1° 整条淘汰。已有 `1e-5 rad` 容差内的片段原样保留。
- 修正完整十帧的越界角至原始模型边界；根状态与范围内关节不变，受影响末端特征按 FK 位移差同步更新。初始关节速度用修正后角度重新差分，不沿用旧速度、不统一清零。
- 修正后重新检查十帧关节限位、十帧 469 对自穿透（0 mm）及十帧地面穿透（≤19.72 mm）。
- 记录差分速度/加速度变化和末端位移，验收中专门初始化已修正条目并观察首秒响应。角度合规不直接代表动态可行或性能改善；不额外加入未约定的速度/存活阈值。

## 环境和准备

仍使用唯一的 `/root/gpufree-data/smp-g1/repo/` 和 `venv/`。不创建新 repo/venv，不执行旧 0002 激活脚本，不安装依赖或更新锁。

先核对 0003 完成报告、真实退出码与共同初始模型路径；确认没有训练在运行、工作区干净，再拉取和切换：

```bash
cd /root/gpufree-data/smp-g1/repo
git fetch origin refs/heads/00031:refs/remotes/origin/00031
git switch 00031  # 本地没有分支时：git switch -c 00031 --track origin/00031
git merge --ff-only origin/00031
source experiments/00031/activate_00031.sh
export SMP_INITIAL_CHECKPOINT=<0003运行记录中同一份0002初始模型的绝对路径>
```

## 验收、冒烟、正式训练

先运行：

```bash
python experiments/00031/test_reward_formula.py
python experiments/00031/test_passive_metrics.py
python experiments/00031/test_physics.py
python experiments/00031/test_runtime.py
python experiments/00031/test_initialization.py
python experiments/00031/accept_00031.py --num-envs 64 --steps 2400 --out /root/gpufree-data/smp-g1/logs/00031_accept_<唯一时间戳>
```

然后用 `run_00031.py --num-envs 1024 --max-iterations 2 --seed 2027 --initial-checkpoint "$SMP_INITIAL_CHECKPOINT"` 做独立冒烟，明确指定统一根目录下独立的 `--log-root`、`--ckpt-root` 和 `--run 00031_smoke_<时间戳>`；保存 Python/tee 退出码。使用 `verify_completion.py --iterations 2` 核验对应冒烟目录，不使用 `--allow-incomplete`。

全部通过后正式启动：

```bash
bash experiments/00031/launch_00031.sh
```

正式为 1024 环境、10000 轮、seed 2027；从共同初始模型重新开始，不接着冒烟或 0003 模型训练。启动后禁止切共享 repo 分支。保留 tmux、TensorBoard、日志和模型路径。

完成后：

```bash
python experiments/00031/verify_completion.py --log-dir /root/gpufree-data/smp-g1/logs/<正式运行名> --ckpt-dir /root/gpufree-data/smp-g1/ckpt/<正式运行名>
```

要求 10000 轮/245760000 样本、101 次足额 GSI 入池/106496 条、确有修正片段入池、所有初始模型 tensor 与 0003 的共同来源一致、新优化器、完整事件/检查点/先验锁核对通过。失败时保留实际错误和退出码，停止后续动作，不覆盖旧运行。
