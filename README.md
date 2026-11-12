# lingbot-va · RoboKata Training

LingBot-VA 原生 text IG C15 全参数训练，包括混长 batch 掩码和可恢复优化器状态。不是旧 RoboICL Demo LoRA。尚无本项目 RoboTwin43 text 训练适配。

从主分支克隆完整集合并按 `docs/TRAINING.md` 操作：

```bash
git clone https://github.com/xieshanrui/RoboKata-Training.git
cd RoboKata-Training
git submodule update --init
```

当前分支是可公开的独立源码快照。来源提交 `045d5806f91518fdaa7d1c8944efe050028b66d8`，逐文件来源在 [SOURCE.json](SOURCE.json)。原仓库和原有 Git 历史未改写。训练数据、权重、凭据、服务器运行目录和未公开论文不在本仓。

- [lingbot_ig_text/README.md](lingbot_ig_text/README.md)
