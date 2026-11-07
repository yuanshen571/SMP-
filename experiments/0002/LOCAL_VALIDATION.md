# 0002 本地准备检查

检查时间：2026-11-07T11:30:39.284446+08:00

基线：`ff8b7f3a90d9b87fa230b5b9777b75ee51cb2d5d`。

| 检查 | 结果 |
| --- | --- |
| 核心/记录/无记录三条路径：`.5*task+.5*smp`，每步各调用一次任务函数和 guidance | PASS，执行实际函数 AST，guidance 使用标量替身 |
| 外层 `.5/.5`，Location 位置/速度/朝向 `1/0/0`，位置核与 ws=4 不变 | PASS |
| 既有 guidance 与其他奖励函数语义不变，其余 Location 配置不变 | PASS |
| src 仅两处文件变化，PPO/GSI/归一化源码不变 | PASS |
| Location 先验 / uv.lock SHA256 | PASS |
| 9 个 Python 文件语法，3 个 Shell 文件语法 | PASS |
| 启动器：包含空格的路径、真实 Python 退出码 0/23、tee 退出码 | PASS，临时目录/替身程序，未调用 tmux |
| 固定正式预算、运行名边界检查 | PASS |
| 第一分支和原 0001 副本 | 仍为基线提交，工作区干净 |

本机 Python 没有 torch；没有运行 torch 指标单元检查，没有创建 GPU 仿真或启动 PPO。完整运行验收由服务器执行 AI 按 `EXECUTION.md` 完成，不将上述替身检查当作 GPU/模型质量结论。
