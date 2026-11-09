# 实验00042执行入口

本分支只运行00042：基于0004，64维TS2Vec＋K-means保持不变，逐条温和加权采样上限由2改为1.5。

完整服务器任务单：[experiments/00042/EXECUTION.md](experiments/00042/EXECUTION.md)。

执行文件：`experiments/00042/execute_00042.sh`。复用共享repo／venv，1024环境、10000轮；它会先预检、验收和短训练，再从公共未训练模型重新开始正式训练。不要运行旧0004／00041任务单。
