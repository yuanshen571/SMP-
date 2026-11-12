# 00053完成后自动接00033：服务器执行任务单

本次可立即部署**等待任务**。00053仍在运行时，只在共享repo之外下载衔接文件并挂tmux，不切换repo分支，不更改00053进程。用户授权的顺序为：00053完成且验收通过 → 独立执行00033。

## 实验00033

00033复制00053提交 `9cecb1244c3d83eb1e9c159682ea2cc591b3fd55` 的方法，**仅训练种子2028改为2027**。源码新增于 `experiments/00033/`，不改既有实验文件、src、依赖或先验。

- G1 Location，1024环境×24步×10000轮＝245760000正式交互样本。
- 奖励、PPO、目标生成、推扰、随机化和记录口径沿用00053；总奖励0.5任务＋0.5 SMP。
- GSI容量4096，启动生成4096；每100个PPO轮次刷新1024，FIFO替换、均匀有放回采样。
- 原片段物理通过则原样入池。全部失败片段逐条执行初始噪声Sep-CMA-ES补救：sigma0.05、每代32候选、最多30代、RMS≤0.15、固定49次后验噪声。
- 首次出现物理通过候选的代结束后停止，取固定顺序第一个通过候选；30代仍失败，取搜索中的历史最佳候选入池，原失败片段不参与择优或回退。全部原始条目均保留一个入池代表，不再拒绝重采样。
- 完整十帧物理标准：29个单轴关节硬限位容差1e-5 rad、全部碰撞体地面穿透≤19.72mm、469对自穿透容差0mm、数值有效。直接关节裁剪关闭，混合缓存和分簇关闭。
- 每条独立优化器种子 `(2027+10000+全程原始条目编号) % 2**32`。批量布局和早停、历史最佳逻辑与00053相同。

## 2027公共未训练初始化

使用0002的真实初始模型，和原0003、00031相同；不承接00053已训练策略、优化器、缓存或归一化统计。路径与内容已由此前2026-11-11只读核验档案确认，本次部署前再校验：

- 模型：`/root/gpufree-data/smp-g1/ckpt/0002_location_seed2027_20261107_114103/initial_untrained.pt`
- 来源：`/root/gpufree-data/smp-g1/logs/0002_location_seed2027_20261107_114103/run_info.json`
- 文件SHA256：`7109c7215ed8691c72b151601d6145c5b7e4f6d7fae479dc6db30a6d40d9ffdb`
- Actor/Critic内容SHA256：`774d69745290cdcce1522fe5875d48022cc4789f1ab47abb386bb45a19fec4eb`

0002的旧格式没有initialization_seed/experiment两个检查点字段，不能套用2028版的字段要求。00033通过已归档的双指纹、来源run_info、检查点index、stage、run_id、common_step_counter=0、空优化器和normalizer count=0建立身份，严格加载Actor/Critic并保留新优化器。不要改写旧0002材料。

## 现在部署等待任务

1. 读取适用AGENTS.md。沿用 `/root/gpufree-data/smp-g1/repo/` 和唯一 `venv/`，不安装或升级软件。
2. 在本次00053的tmux、运行日志和 `logs/00053_pipeline_*/runs.txt` 中确定**当前正在执行的确切pipeline目录**。对应 `revision.txt` 必须为branch=00053、commit=`9cecb1244c3d83eb1e9c159682ea2cc591b3fd55`。若有多个候选，逐一对照正在跑的run，不能只凭时间戳挑最新目录。
3. 将该绝对目录记为 `PIPELINE`。核对当前共享repo仍为00053上述提交且干净，现有训练仍正常。
4. 取得本次交付的00033完整提交SHA，记为 `REVISION`；优先使用用户提供的任务单链接里的固定提交。如只收到分支链接，用 `git ls-remote https://github.com/yuanshen571/SMP-.git refs/heads/00033` 读取并固定40位SHA；此操作不改变共享repo。
5. 在共享repo之外准备工具目录，下载**固定提交**下的 `arm_after_00053.sh`，再执行下方命令。该脚本会从同一提交下载独立watcher和配置，并先用CPU核验2027初始模型文件身份。不要先fetch/switch共享repo来取得它。

```bash
TOOLS="/root/gpufree-data/smp-g1/tools/handoff_00053_to_00033_${REVISION:0:12}"
mkdir -p "$TOOLS"
curl --fail --location "https://raw.githubusercontent.com/yuanshen571/SMP-/${REVISION}/experiments/00033/arm_after_00053.sh" -o "$TOOLS/arm_after_00053.sh"
bash "$TOOLS/arm_after_00053.sh" "$PIPELINE" "$REVISION"
```

`PIPELINE`和`REVISION`由前面核对结果设置，不能原样传占位文字。若工具文件已存在，先核对内容一致，保留用户已有修改。

等待任务在独立tmux内运行，无需保持SSH终端在线。它在tools目录运行，等待期间只用标准库检查共享锁，不导入GPU框架、不生成GSI、不创建仿真，也不fetch或switch共享repo。

部署后立即向用户返回脚本输出的 `HANDOFF_SESSION`、`HANDOFF_LOG`、`HANDOFF_STATE` 和实际phase=waiting；再次核对00053原进程、分支、提交未变。若脚本当场退出，报告原因，不能声称已排队。

## 自动接续条件和顺序

`chain_after_00053.py`每30秒检查一次。它必须获得原工作流的 `.0003_00031_queue.lock`，之后仍持有同一个锁，直到00033整个流程结束：

1. 00053的pipeline退出码=0、last_stage=complete、progress=complete、COMPLETED=PASS；formal训练及formal_verify阶段的Python/tee退出码均为0。
2. 确认run_info是该pipeline的同一正式00053，seed2028、1024环境、10000轮、245760000样本，且源提交、30代和RMS0.15均吻合。
3. 在原00053分支用原 `verify_completion.py` 再次严格核验，报告写到handoff目录。失败即停，不补跑00053。
4. 在日志中持久记录启动意图，再fetch固定00033分支。远程提交必须与部署时固定SHA相同；工作区必须干净。仅创建分支或快进合并，不reset/clean/stash、不强制覆盖。
5. 切换00033，核验外置watcher和该提交内文件一致，将同一个共享锁传给 `execute_00033.sh`。
6. 独立完成00033源文件/共享环境/公共初始化核验、单元测试、真实采样器验收、64环境2400步零训练验收、1024环境2轮独立冒烟，最后从同一2027未训练参数重新开始10000轮正式训练。
7. 完成后严格核验00033，写COMPLETION；handoff状态改为complete。

00033使用现有共享L40/venv；不复制服务器环境，不重跑00053。完整训练流程及额外验收与00053一致。若只看到GPU空闲，但缺少成功完成记录，也不会启动00033。

## 退出、重复启动和保存

任何阶段失败停止后续流程，保存真实退出码；不自动重试训练、不承诺断点续训。相同00053 pipeline的等待任务不能同时注册两次；一旦记录启动意图/运行/完成/异常停止，再次运行不会自动重开00033。等待阶段可在核对身份后恢复watcher，不能借此重跑实验。

保存位置：

- watcher：`tools/handoff_00053_to_00033_<提交前12位>/watcher_console.log`
- 状态与接续核验：`logs/handoff_00053_to_00033_<原pipeline目录名>/`
- 新流程：`logs/00033_pipeline_<时间>_<随机后缀>/`
- 正式训练：统一logs/ckpt中的 `00033_location_seed2027_*`

TensorBoard沿用统一logs目录，不另起服务。训练保留物与00053相同：TensorBoard、控制台、初始/每500轮/最终模型、配置、轻量验收报告；不保存视频、轨迹、原GSI片段、逐轮CSV/JSONL、W&B或独立策略评估。

最终检查：10000轮/245760000样本；101次GSI启动/刷新事务、原始生成及入池各106496；`direct_inserted+attempted=inserted`、`succeeded+exhausted=attempted`；30代上限、RMS≤0.15、直接关节修正/原片段回退/丢弃计数为0。补救耗尽入池数可以非零。

本地CPU协议检查不代表服务器GPU验收或正式训练已完成。实际完成后向用户汇报00053核验、00033真实退出码和训练样本数、初始化身份、补救统计，以及两个实验的TensorBoard/完成报告路径。
