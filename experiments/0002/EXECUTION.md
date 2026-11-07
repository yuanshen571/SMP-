# 实验 0002：G1 Location，原版 SMP 加法及权重

将本文件交给服务器上的执行 AI。用户授权完成本实验的环境准备、验收和正式训练；按 A→B→C→D 执行，前一阶段通过才进入下一阶段。本机准备阶段没有连接训练服务器，没有启动训练。执行时保留已有 0001 的代码、日志和检查点，不覆盖同名目录，不在 0001 的工作目录切换分支。

## 1. 本实验唯一的方法改动

基线为用户仓库 `https://github.com/yuanshen571/SMP-.git` 的 `exp0001-location-baseline` 分支，基准提交 **`ff8b7f3a90d9b87fa230b5b9777b75ee51cb2d5d`**。0002 分支为 **`exp0002-location-additive`**，本机目录为 `/home/yang/Desktop/项目/SMP-G1+/0002`。

| 设置 | 0001 | 0002 |
| --- | --- | --- |
| 奖励组合 | `r_task * r_smp` | **`0.5*r_task + 0.5*r_smp`** |
| Location 内部位置权重 | 1 | **1** |
| Location 内部速度权重 | 0 | **0** |
| Location 内部朝向权重 | 0 | **0** |
| 位置奖励实际公式 | `exp(-0.25 * 水平目标距离)` | 相同 |
| SMP 的 `ws` | 4 | 相同 |
| RewardTerm 权重 / dt 缩放 | 1 / `scale_rewards_by_dt=True` | 相同 |

因此 0002 环境实际返回 `0.02 * (0.5*r_task + 0.5*r_smp)`。任务内部仍只有位置项；速度与朝向为零权重，不额外计算它们。这里采用原版 SMP 的**组合形式和权重**；距离奖励核、SMP 指数尺度、分位数输入缩放继续使用 G1 基线。不得顺带将距离公式改成原 Humanoid 的平方形式、把 `ws` 改成 6、或去掉框架 dt 缩放。

已实现的核心差异仅有：

1. `src/smp/rl/rewards.py` 新增 `task_smp_additive`；既有 guidance 和其他任务的乘法函数原样保留。
2. `src/smp/rl/tasks/location/location_env_cfg.py` 使用该函数，显式配置 `.5/.5` 外层权重。
3. `experiments/0002/` 包含本实验入口、被动指标、启动和验收文件。0001 支持文件留在第一分支，不复制其完成报告冒充 0002 结果。

两条奖励路径（开启记录／`--no-metrics`）均为加法。单步始终只调用一次 `smp_guidance_reward`，不为指标重复更新历史、误差归一化或抽噪声。

## 2. 固定训练条件

- G1 仓库的 `Smp-Location-G1`，**29 DOF G1**，mjlab / MuJoCo-Warp 后端；单张 **L40 48 GB**。
- **1024 环境，10000 PPO 轮，seed=2027，每轮每环境 24 步**，正式总样本 **245760000**。
- 从未训练初始化开始，独立优化器；不加载 0001 最终策略继续训练。
- 物理步长 `.005 s`、decimation=4，控制 **50 Hz**；回合上限 20 s / 1000 控制步。
- 目标距离 1～10 m，每 5～10 s 重新生成；没有目标速度指令。
- 使用相同 LaFAN 跑步先验 `datasets/pretrain_ckpt/pretrained_lafan_run.pt`，冻结先验，不重新训练先验。
- SMP 完整十帧窗口、分位数缩放及误差归一化、噪声层级 `(8,15,22)`、50 步 DDPM 与默认编译/warm-up 保持不变。
- GSI 启动 4096 条，每 2400 同步控制步新生成 1024 条 FIFO 替换，均匀有放回抽样；按每轮 24 步换算为每 **100 轮**刷新。
- Actor/Critic `(512,256,128)`、ELU，Actor 当前观测和默认噪声，Critic 十帧历史无观测噪声；固定动作标准差 .30。
- PPO 保留 Adam、学习率 .001、adaptive KL、目标 KL .01、5 epochs、4 mini-batches、clip .2、gamma .99、lambda .95、entropy 0、max_grad_norm 1。
- 推扰、摩擦、编码器偏差、质心随机化，原自碰撞/根高度终止、正常超时、起始回合长度随机偏移均不改。
- 使用 `play=False`；不添加周期独立测试、全环境重置、物理筛选、混合缓存、分簇或其他改进。

## 3. 支持文件

| 文件（均在 `experiments/0002/`） | 用途 |
| --- | --- |
| `EXPERIMENT.json` | 固定参数、基线提交和 SHA256 |
| `activate_0002.sh` | 读取原 `env.sh`，切到独立 uv 环境并设置 UTF-8 |
| `run_0002.py` | 正式训练、初始/中途/最终模型保存、配置与版本元数据 |
| `passive_metrics.py` | 被动记录 `0002/` TensorBoard 指标 |
| `launch_0002.sh` | tmux 启动，记录 Python 及 tee 退出码，拒绝覆盖旧运行 |
| `launch_tensorboard.sh` | 本机回环地址上的 TensorBoard，默认端口 6010 |
| `test_reward_formula.py` | 无额外依赖的真实函数公式、配置和不变项检查；标量 stub 不替代 GPU 验收 |
| `test_passive_metrics.py` | 使用 torch 的计数、加法均值、无 RNG 消耗检查 |
| `accept_0002.py` | 独立进程仿真验收，零策略更新，检查完整/部分重置及 GSI 刷新 |
| `inspect_metrics.py` | 读取 TensorBoard，不导出另一份逐轮数据 |
| `verify_completion.py` | 完成轮数、奖励恒等式、事件覆盖及检查点核验，生成完成报告 |

## 4. A：拉取与环境准备

服务器沿用 `/root/gpufree-data/smp-g1/`。在该根目录创建独立代码副本 **`repo-0002/`** 和 uv 环境 **`venv-0002/`**。原因是 Python 的 editable 安装会指向源码目录，复用/重装原 venv 可能让 0001 指向另一份代码。独立环境仍使用同一个 `uv.lock`，不会改变实验依赖版本。

1. 先只检查本项目的已有目录、0001 是否运行、GPU/驱动、磁盘余量及 `tools/` 内已安装的 uv 0.13.7；不操作其他实验。
2. 拉取明确分支：

   ```bash
   cd /root/gpufree-data/smp-g1
   git clone --branch exp0002-location-additive --single-branch https://github.com/yuanshen571/SMP-.git repo-0002
   cd repo-0002
   source experiments/0002/activate_0002.sh
   ```

   若 `repo-0002/` 已存在，先核对分支、HEAD 和是否干净，不覆盖、不强制重置；以交接时推送的明确提交为准，记录完整 SHA。

3. 激活脚本读取已有根目录 `env.sh`，随后将 `UV_PROJECT_ENVIRONMENT` 指向 `venv-0002/`。不改 0001 的 env.sh / venv / repo。原缓存变量继续复用，尤其不要删除 `cache/python/` 中被环境依赖的 Python 本体。每个后续进程均重新 source 激活脚本。
4. 使用已验证的 uv 路径（通常 `tools/uv`，先确认），执行 frozen 同步；Python 按 `.python-version` 使用 **3.13**。若 `uv` 已在 PATH，可执行 `uv sync --frozen --python 3.13`。不得重写锁、改成 Conda 或装进系统 Python。
5. 核对 Python 指向 `venv-0002`，实际导入的 `smp.__file__` 在 `repo-0002/src/` 内。记录实际版本：mjlab 1.6.0、MuJoCo 3.11.0、mujoco-warp 3.11.0、rsl-rl-lib 5.4.2、torch 2.10.0、TensorBoard 2.20.0；检查 CUDA 可用与 L40 名称。
6. 核对以下文件未变：

   | 文件 | SHA256 |
   | --- | --- |
   | `uv.lock` | `ce041c7774b1d4abf42e868fa3b4af2973954cec61d35c3b4e58a7e27464d9ce` |
   | `datasets/pretrain_ckpt/pretrained_lafan_run.pt` | `74787104ff124191136eeed2268f79448da3ac2e30e710d42aebfc5a60e58cb3` |

7. 将环境验收小摘要写到 `logs/0002_setup_<时间>/`，所有中文文件显式 UTF-8。若磁盘/驱动/网络存在实际问题，给出具体错误并修复准备层；不得静默减少环境数、训练轮数或改核心依赖。

## 5. B：正式训练前的验收

全部在 `repo-0002/` 内、激活 0002 环境后运行；输出只属于 0002。下面示例路径加时间戳，防止重复执行混入事件文件。

```bash
python experiments/0002/test_reward_formula.py
python experiments/0002/test_passive_metrics.py
python experiments/0002/accept_0002.py --num-envs 64 --steps 2400 --out /root/gpufree-data/smp-g1/logs/0002_accept_$(date +%Y%m%d_%H%M%S)
```

要求：

- 单元检查全部通过，无日志路径额外消耗随机数。
- 仿真进程零策略更新；观测、奖励有限，50 Hz / 1000 步时限等与默认配置一致。
- 完整与部分 GSI 重置正确；2400 控制步触发 1024 条刷新，只替换对应池行。
- **逐步核对** `r_additive=.5*r_task+.5*r_smp` 和 `r_env=.02*r_additive`。
- 独立计数器与 TensorBoard 的终止分类、完整回合长度、纯任务回报及训练回报一致。
- 失败与超时同一步出现只计失败；无失败时不伪造失败时长为零；首回合随机时间偏移单独归类。

随后做一次 **1024 环境、2 轮**的独立 PPO 冒烟检查，验证正式规模显存、更新和保存路径。调用 `run_0002.py --max-iterations 2 --num-envs 1024`，使用 `0002_smoke_<时间>` 独立运行名及根目录日志/模型目录，保存真实退出码。检查两轮自定义事件、样本数 49152、Actor/Critic 参数更新、初始及最终检查点可读取、加法恒等式。10000 轮完成核验脚本不用来判断这个两轮短检查。

验收及冒烟均不计入正式训练。正式进程重新按 seed 2027 从未训练初始化开始；不从冒烟模型继续，不重复启动正式进程。必要实现修复先提交、记录差异，不改变本文固定方法与预算。

## 6. C：正式训练

A、B 通过后，启动一次：

```bash
cd /root/gpufree-data/smp-g1/repo-0002
source experiments/0002/activate_0002.sh
bash experiments/0002/launch_0002.sh
```

默认值已经固定为 1024 / 10000 / 2027 / cuda:0。检查启动输出及 `run_info.json`，确认没有残留 `NUM_ENVS`、`MAX_ITERATIONS`、`SEED` 等环境变量把参数覆盖掉。若有残留，先清除或显式设置后启动；不能在正式启动后才发现参数不符。

启动时向用户报告：完整提交、运行名、tmux 会话、日志/模型路径及是否已经产生第一轮有效记录。训练进程异常则停止并保留证据，不自动换参数、追加轮数或重跑正式训练。会话名为 `smp_g1_<运行名>`。

输出：

- `/root/gpufree-data/smp-g1/logs/0002_location_seed2027_<时间>/`
- `/root/gpufree-data/smp-g1/ckpt/0002_location_seed2027_<时间>/`

检查点沿用 0001 的框架保存编号：训练前 `initial_untrained.pt`；原生编号 `model_0.pt`、`model_500.pt` 等，每 500 轮保存间隔与 0001 相同；最后为 `model_9999.pt`。框架原生编号从 0 开始，索引中的完成轮次为原生编号+1；最终完成轮数必须是 10000，不能把文件名 9999 误认为少训练一轮。

可用下面的只读服务看曲线：

```bash
bash experiments/0002/launch_tensorboard.sh 6010
```

监听 `127.0.0.1:6010`，从本地终端以自己的服务器 SSH 配置做端口转发。不要占用或停止 0001 已有的 6009 服务。

## 7. TensorBoard 与保留规则

正式逐轮原始指标**只保存在 TensorBoard**。自定义命名空间 `0002/`，step 为已完成 PPO 轮次 1～10000；`Progress/training_samples` 给出样本横轴。默认框架日志同时保留。

21 个固定标量包含：

- `Reward/task_raw_mean`：未乘 .5、未乘 dt 的纯位置任务奖励。
- `Reward/smp_raw_mean`：未乘 .5、未乘 dt 的 SMP 奖励；仍使用各自运行的误差归一化。
- `Reward/additive_raw_mean`：`.5*task+.5*smp` 的均值；0001 对应是 product，标签不同且公式不同。
- `Reward/env_reward_mean`：最终传给 PPO 的 `.02*additive` 均值。
- 目标距离、实际水平速度（不是速度误差）；终止/失败/纯超时/首回合/完整回合计数，完整回合的长度、失败时长、纯任务回报与训练回报汇总；轮数、样本、时间。

有相应分母才记录 7 个条件均值，包括完整回合任务回报、训练回报、长度、存活时长、失败率、**失败回合平均时长**及所有已结束回合失败率。完整回合统计排除启动时被随机时间偏移的首回合；跨 PPO 轮保留未结束回合累加器，不每轮清零、不周期强制重置环境。

统计 100 轮等较长窗口时，失败率和失败时长优先用事件中的**失败数/回合数与时长总和重新汇总**，而不是把每轮不同分母的比率直接等权平均。不能把训练回报的上涨直接解释成任务能力改进：0001、0002 的奖励定义不同。纯任务回报、目标距离、存活与失败指标沿用相同定义。

保留：事件文件、console.log、真实退出码、环境/算法 YAML、指标说明、轻量运行元数据/依赖版本/检查点索引、初始/间隔/最终模型、验收/完成小报告。

不生成/归档：逐轮 CSV/JSONL 或其他重复原始指标表、逐步完整轨迹、逐回合明细、GSI 十帧片段/缓存快照、视频/截图、W&B 运行文件、模型上传或 ONNX/JIT 导出、独立策略评估及结果。内存中的训练 rollout 与 GSI 继续正常使用。配置和索引 JSON 不作为逐轮指标数据库。

## 8. D：完成核验

正式进程退出后，使用实际运行名：

```bash
python experiments/0002/verify_completion.py \
  --log-dir /root/gpufree-data/smp-g1/logs/<运行名> \
  --ckpt-dir /root/gpufree-data/smp-g1/ckpt/<运行名> \
  --out /root/gpufree-data/smp-g1/logs/<运行名>/COMPLETION.md
```

正式验收禁止使用 `--allow-incomplete`。如发现日志 tee 出错，即使 Python 退出为 0 也应报告失败，核查日志完整性。

要求退出码 0、10000 轮/245760000 样本；21 固定标量各有完整的 1～10000 点；条件指标与分母对应；计数恒等式与逐轮加法/dt 恒等式成立；Actor/Critic 检查点可读取且结构、有限值检查通过，参数实际更新；先验/依赖锁保持不变，未生成用户取消的数据。无独立策略评估，模型加载检查不推进仿真。

最后提供：实际分支/提交、启动与完成时间、真实退出码、完成轮数/样本、训练耗时、日志/模型/TensorBoard 路径、最终模型校验值、验收结论及实际警告。尚未执行的检查不能写成 PASS。若需要修正支持文件，保持在 0002 分支提交；不要修改或推送到第一分支。

## 9. 可核查来源

- 原版 SMP 外层权重：<https://github.com/xbpeng/MimicKit/blob/fbe1a36e878f036d6bf5521defb8eb3b8ab4170d/data/agents/smp_task_humanoid_agent.yaml>（task=.5 / smp=.5）。
- 原版 Location 内部权重：<https://github.com/xbpeng/MimicKit/blob/fbe1a36e878f036d6bf5521defb8eb3b8ab4170d/data/envs/smp_location_humanoid_env.yaml>（position=1 / velocity=0 / facing=0）。
- 0001 实际代码：<https://github.com/yuanshen571/SMP-/tree/ff8b7f3a90d9b87fa230b5b9777b75ee51cb2d5d>。
- G1 上游基准：<https://github.com/SUZ-tsinghua/smp/tree/28d1dfd1b7b4447977fbdb2062167d9f87a2faaa>。

本地准备只完成静态/标量公式检查；完整 torch 指标与 GPU/仿真/训练检查由执行 AI 按本文在 L40 上完成。
