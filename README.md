# robokata · RoboKata Training

RoboKata 既有基线：IG Version A（posttrain-robotwin 基座）及 HumanGen RoboTwin A+B1（pretrain 基座）。不要与 main 中 models/robokata_ig_fi 混用。

从主分支克隆完整集合并按 `docs/TRAINING.md` 操作：

```bash
git clone https://github.com/xieshanrui/RoboKata-Training.git
cd RoboKata-Training
git submodule update --init
```

当前分支是可公开的独立源码快照。来源提交 `045d5806f91518fdaa7d1c8944efe050028b66d8`，逐文件来源在 [SOURCE.json](SOURCE.json)。原仓库和原有 Git 历史未改写。训练数据、权重、凭据、服务器运行目录和未公开论文不在本仓。

- [baseline/README.md](baseline/README.md)
- [ig_evaluation/README.md](ig_evaluation/README.md)
- [robotwin_b/README.md](robotwin_b/README.md)

[原始说明](SOURCE_README.md) 保留为来源记录；当前交接范围和可用命令以 main 的文档为准。
