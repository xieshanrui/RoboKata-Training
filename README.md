# osvi-wm · RoboKata Training

OSVI-WM 的 RoboTwin EEF16 与 IG C15 joint16 适配。共同训练入口 osvi_robotwin/train.py。依赖同一 main checkout 的 models/act 与 third_party/osvi_wm。

从主分支克隆完整集合并按 `docs/TRAINING.md` 操作：

```bash
git clone https://github.com/xieshanrui/RoboKata-Training.git
cd RoboKata-Training
git submodule update --init
```

当前分支是可公开的独立源码快照。来源提交 `045d5806f91518fdaa7d1c8944efe050028b66d8`，逐文件来源在 [SOURCE.json](SOURCE.json)。原仓库和原有 Git 历史未改写。训练数据、权重、凭据、服务器运行目录和未公开论文不在本仓。

- [osvi_ig/README.md](osvi_ig/README.md)
- [osvi_robotwin/README.md](osvi_robotwin/README.md)
