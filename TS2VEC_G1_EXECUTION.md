# 本分支入口

请将 [G1 TS2Vec＋K-means执行单](experiments/ts2vec_g1/EXECUTION.md) 交给服务器执行AI。

本地目录：`/home/yang/Desktop/项目/TS2Vec+K-means`。

分支：`TS2Vec+K-means`，基于00031，包含一份十万条去重GSI数据的生成流程，以及64维和320维两版独立模块训练流程。

服务器复用已有repo和venv，结果放在`/root/gpufree-share/TS2Vec+K-means/`；不运行PPO，两版均不生成或保留TensorBoard。

实际L40数据生成和正式训练由服务器执行AI完成；本机没有连接或启动服务器实验。
