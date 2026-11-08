# 00041：00031＋320维冻结分簇模块＋上限2的GSI温和加权

共同实验设置、筛选修正、采样公式、指标／产物规则、完整串行队列见 [0004／00041总执行单](../0004/EXECUTION.md)。优先把总任务单交给服务器执行AI，一次按0004→00041自动完成。

**若总队列已在运行，不要另行启动本文件中的独立流程。**

本分支基于0004交付提交，`src/smp`与0004相同。算法设置仅将`EXPERIMENT.json`中的模块目录由`/root/gpufree-data/dim64`换为`/root/gpufree-data/dim320`，编码维度由64换为320；实验编号、输出目录和TensorBoard namespace相应改为00041。

- 基线仍为00031；物理筛选与小幅关节修正全部保留。
- 当前GSI缓存按每条状态赋权：`min(2,sqrt(nmax/nk))`，全池归一化、有放回抽样，空簇跳过。
- 固定模块代码`/root/gpufree-data/ts2vec_g1_code/`，不重新训练模块、normalizer或K-means。
- 1024环境、24步／轮、10000轮、种子2027，共245760000正式样本。
- 使用与00031／0004同一公共未训练初始模型和新的优化器，不接着0004最终策略训练。
- 复用唯一`/root/gpufree-data/smp-g1/repo`及`venv`，不另建环境。

## 单独继续00041

只在0004已完成核验、共享repo没有其他任务运行时使用。先读取总任务单，核对0004报告及两版模块指纹。干净共享repo切到`00041`，拉取本次交付提交；不得强制覆盖用户改动。

在独立tmux会话内，设置总单找到的两个实际路径，并依次执行：

```bash
set -euo pipefail
cd /root/gpufree-data/smp-g1/repo
source experiments/00041/activate_00041.sh
export SMP_INITIAL_CHECKPOINT=/实际公共初始模型目录/initial_untrained.pt
export SMP_00031_RUN_INFO=/实际00031日志目录/run_info.json
exec 9>"${SMP_00041_ROOT}/.0003_00031_queue.lock"
flock -n 9
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
STAGE_ROOT="$(mktemp -d "${SMP_00041_ROOT}/logs/00041_stages_${STAMP}_XXXXXX")"
PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
"${PYTHON}" experiments/00041/preflight.py --baseline-info "${SMP_00031_RUN_INFO}" --initial-checkpoint "${SMP_INITIAL_CHECKPOINT}" --out "${STAGE_ROOT}/preflight.json"
for test in test_reward_formula test_passive_metrics test_physics test_runtime test_initialization test_clusters; do
  "${PYTHON}" "experiments/00041/${test}.py"
done
"${PYTHON}" experiments/00041/accept_00041.py --num-envs 64 --steps 2400 --baseline-info "${SMP_00031_RUN_INFO}" --out "${STAGE_ROOT}/accept"
```

然后按总单执行一次1024环境×2轮的`run_00041.py`短训练并用`verify_completion.py --iterations 2`严格核验。参数包含相同的`--initial-checkpoint`、`--baseline-info`，日志和模型均用独立smoke运行标识，不覆盖已有数据。

以上全部通过后，仍在这个tmux会话内启动正式训练并核验：

```bash
RUN_ID="00041_location_seed2027_${STAMP}"
mkdir "${SMP_00041_ROOT}/logs/${RUN_ID}"
bash experiments/00041/launch_00041.sh --worker "${RUN_ID}"
"${PYTHON}" experiments/00041/verify_completion.py --log-dir "${SMP_00041_ROOT}/logs/${RUN_ID}" --ckpt-dir "${SMP_00041_ROOT}/ckpt/${RUN_ID}" --out "${SMP_00041_ROOT}/logs/${RUN_ID}/COMPLETION.md"
```

执行AI需将独立流程的真实退出码、阶段控制台输出、实际目录和tmux会话记录到`STAGE_ROOT`；管道须保留程序和tee各自退出码。异常停止，不自动重试、不覆盖、不用`--allow-incomplete`。不另外启动PPO评估或延长训练。

完整完成条件沿用总单：10000轮／245760000样本、101次分类／106496条累计分类、当前有效池4096、加权重置实际执行、模块内存及文件冻结、TensorBoard与最终模型核验通过。
