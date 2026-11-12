# RoboKata-Training

交给协作者的 **RoboTwin 43→6 / IG-10K C15 训练代码集合**。`main` 管理固定子模块、数据版本、训练命令和预算；每个模型在本库同名分支维护。原工作仓库和服务器作业不受本仓影响。

先读 [运行与模型一致性审查](docs/METHOD_AUDIT.md)：11/12配方通过既有Linux环境的真实参数解析；官方ACT IG独立训练环境待补齐。本次未做整套GPU前后向验证。

```bash
git clone --branch main --single-branch https://github.com/xieshanrui/RoboKata-Training.git
cd RoboKata-Training
git submodule update --init
python tools/preflight.py
cp configs/paths.env.example local.env
# 编辑 local.env 中的绝对路径，激活对应模型的独立 Linux 环境
source local.env
TRAIN_DRY_RUN=1 bash recipes/act_robotwin.sh
# 已准备数据与权重、分配 GPU 后，去掉 TRAIN_DRY_RUN=1 正式运行
```

`local.env` 已被忽略，不能提交凭据。代码支持路径自选，无需维护者 VPN/SSH/outputs。下载代码不会启动训练、购买 GPU、上传权重或改动其他进程。

| 模型 / 分支 | RoboTwin43→6 | IG C15 | 训练入口与性质 |
|---|---|---|---|
| ACT/DINOv2 · `act` | 本项目适配，实际训练10epoch | 官方入口；独立训练环境待验证，本项目未重训 | `act_robotwin.train` / IG官方ACT |
| OSVI-WM · `osvi-wm` | 原生EEF16适配，72k预算 | 原生joint16适配，72k预算 | `osvi_robotwin.train --dataset ...` |
| LingBot-VA text · `lingbot-va` | 本项目未接入 | 原生全参，60epoch | `lingbot_ig_text/train.py`；human description |
| 分段Demo π0 · `openpi` | 已接入并正式训练 | 已接入并正式训练 | `examples/segmented_demo/train.py` |
| 原生text π0.5 · `openpi` | 已接入并正式训练 | 使用官方参考，未实现本项目训练适配 | `examples/robotwin_text/train.py` |
| Zero-WAM · `zero-wam` | 官方代码/已发布参考 | 官方完整入口 | 本项目不重复宣称新基线训练 |
| RoboKata A / A+B1 · `robokata` | A+B1，200epoch历史配方 | A，60epoch历史配方 | 不同基座/配置，明确区分 |
| RoboKata A+B1+FI · `robokata-ig-fi` | 本分支不提供 | 新pretrain基座，60epoch计划 | 原正式运行验证过，后于958步暂停；交接从头训练 |

先读 [训练步骤](docs/TRAINING.md)、[数据与权重](docs/DATA.md)、[环境](docs/ENVIRONMENTS.md)、[实验边界](docs/PROTOCOLS.md)。模型/环境路径及来源固定在 `sources.lock.json`。

本仓没有把未经实现的消融开关标成可运行。现有模块的适用范围见 [模块与消融边界](docs/ABLATION_STATUS.md)。两份未公开PDF不随公开代码发布。

验证边界：本次做源码、脚本、配置和公开克隆验证；不新增GPU训练。既有运行证据只能支持对应源码/数据/硬件，不能保证新机器已通过运行验证。
