# zero-wam · RoboKata Training

Zero-WAM 上游模型与本项目推理适配。IG C15 训练入口在 environments/imitator_game；RoboTwin 保留官方入口/已发布参考，不声称新增了一套本项目43任务训练。

从主分支克隆完整集合并按 `docs/TRAINING.md` 操作：

```bash
git clone https://github.com/xieshanrui/RoboKata-Training.git
cd RoboKata-Training
git submodule update --init
```

当前分支是可公开的独立源码快照。来源提交 `d32c5a54d450323ec084357440c425e0b061a97e`，逐文件来源在 [SOURCE.json](SOURCE.json)。原仓库和原有 Git 历史未改写。训练数据、权重、凭据、服务器运行目录和未公开论文不在本仓。

- [adapters/README.md](adapters/README.md)

[原始说明](SOURCE_README.md) 保留为来源记录；当前交接范围和可用命令以 main 的文档为准。
