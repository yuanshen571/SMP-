# 本地准备与验证范围

本地准备机使用独立CPU检查环境验证，未连接服务器，未执行GPU采样、仿真或正式训练，也未修改服务器环境。

已完成的本地检查：

- `contract.py`：00053身份、参数、38项原始源码/先验/依赖SHA256和vendor来源。
- `test_variant.py`：相对00063的训练参数仅改变种子和搜索预算；搜索器仅改变命名空间和30代上限，物理标准、恢复/FIFO逻辑和奖励源码保持一致。
- `test_runtime.py`：12项协议测试，含真实DDPM调度器的噪声记录/重放及原生随机流一致性、真实Sep-CMA-ES独立RNG、原合格条目保持、所有失败条目处理、首次通过停止、历史最佳保留、失败候选即使劣于原片段也入池、半径拒绝、并行任务续接、FIFO环绕及非刷新步、仅替换指定GSI回调。
- `test_scoring.py`：完整十帧关节/地面/球体重叠的独立闭式几何检查，以及大TensorBoard计数舍入检查。
- `test_initialization.py`：seed2028公共未训练策略身份、参数加载、独立空优化器和RNG中立。
- `test_passive_metrics.py`：计数恒等式、空分母、终止/超时分类和不消耗随机数；CPU机没有mjlab时，仅将未使用的guidance入口隔离为显式报错占位，真实guidance不在此项声称验收。
- `test_reward_formula.py`：实际奖励AST、原权重和原源码一致性。
- Python语法、自有代码Ruff检查/格式和Shell语法。固定SHA256的第三方vendor源码保持原样，不纳入格式检查。

CPU检查环境：Python3.11.17、Torch2.6.0+cpu、NumPy2.2.6、MuJoCo3.3.7、TensorBoard2.20.0。它不是服务器共享环境，不用于规定服务器版本。

仍必须由服务器执行AI完成的检查：

- 既有L40/共享venv/实际MuJoCo3.11 SDK与G1资产核验。
- `test_physics.py`在真实SDK角色上的碰撞体和硬限位测试。
- `accept_sampler.py`的真实冻结先验、torch.compile和固定1024布局检查。
- 64环境2400步验收、独立1024环境2轮冒烟及正式10000轮训练和收尾核验。

不能把本文件视为00053训练或GPU验收已通过的报告。
