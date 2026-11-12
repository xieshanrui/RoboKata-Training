# robokata-ig-fi · RoboKata Training

RoboKata A+B1+Foresight Indexing 的独立 IG C15 适配，初始化 Zero-WAM pretrain。原运行曾验证并训练到 step958 后按用户要求暂停；本仓提供从头训练，不自动恢复原作业。

从主分支克隆完整集合并按 `docs/TRAINING.md` 操作：

```bash
git clone https://github.com/xieshanrui/RoboKata-Training.git
cd RoboKata-Training
git submodule update --init
```

当前分支是可公开的独立源码快照。来源提交 `bb5ae196287f9899dad5ea70e7bf4562c5d80ad6`，逐文件来源在 [SOURCE.json](SOURCE.json)。原仓库和原有 Git 历史未改写。训练数据、权重、凭据、服务器运行目录和未公开论文不在本仓。

- [baseline/README.md](baseline/README.md)

[原始说明](SOURCE_README.md) 保留为来源记录；当前交接范围和可用命令以 main 的文档为准。
