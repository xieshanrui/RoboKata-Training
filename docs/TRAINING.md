# 训练交接

以下命令从 `main` 根目录执行。完成数据/权重下载，激活对应独立环境，`source local.env`，为每个实验设置新的 `$RUN`、`$CONFIG`、缓存路径与 `CUDA_VISIBLE_DEVICES`。先执行dry-run再去掉 `TRAIN_DRY_RUN=1`。本库不自动恢复旧作业。

实际启动检查与已知边界见 [模型审查](METHOD_AUDIT.md)。新ACT/OSVI训练的checkpoint包含适配源码身份；旧checkpoint使用原冻结源码恢复。官方ACT IG当前仅提供原入口，完整训练环境尚未验收。

## 预算索引

| 配方脚本（recipes/） | GPU × 实际batch | 有限训练边界 | 来源性质 |
|---|---|---|---|
| act_robotwin.sh | 1×128 | 10epoch /35760更新 | 实际正式配方；2步检查峰值41.81GiB，不是全程上限 |
| act_ig_c15.sh | 1×128 | 10epoch | 官方ACT/DINO入口，本项目未重训 |
| osvi_robotwin.sh / osvi_ig10k.sh | 1×64 | 72000更新 | 既有原生动作适配；禁用旧10k配置 |
| lingbot_ig_c15.sh | 8×8 | 60epoch /2820更新 | 原生全参数，FSDP与重算，尾批保留 |
| segmented_pi0_robotwin.sh / segmented_pi0_ig10k.sh | 2×32 | 20000更新 | 保留原RT30k/IG60k学习率曲线 |
| pi05_robotwin.sh | 4×16 | 20000更新 | 原生text；保留30k学习率曲线 |
| robokata_robotwin_b1.sh | 4×16 | 200epoch /6600更新 | 历史B1配方，需按实际显存验证；未声称A800可容纳该batch |
| robokata_ig_a.sh / robokata_ig_fi.sh | 8×8 | 60epoch /2820更新 | 混长batch/掩码/激活重算；A和FI基座不同 |
| zerowam_ig_c15.sh | 4×1 | 60epoch /45000更新 | 官方采样，开启重算适配显存 |

所有配方无梯度累计。RoboKata global64和官方Zero-WAM global4同为60遍数据，但优化器更新次数不同。RT B1脚本显式关闭历史121GiB显存预留，仅取消占位分配，不降低模型真实显存需求。预算和GPU数未经重新冻结不能随意更改。

## 1. RoboTwin基础准备与ACT

```bash
source recipes/common.sh
python -m act_robotwin.prepare --data "$DATA_RT" --output "$PREPARED_RT_ORIGINAL" --download
python -m act_robotwin.prepare --data "$DATA_RT" --output "$PREPARED_RT" --download --exclude-pairs configs/robotwin_rgb_exclusions.json
python -m act_robotwin.cache_demo --data "$DATA_RT" --prepared "$PREPARED_RT" --ig-root "$IG_ROOT" --cache "$FEATURE_CACHE" --hf-cache "$WORK/hf"
TRAIN_DRY_RUN=1 bash recipes/act_robotwin.sh
```
原版prepare的解码长度检查不代表全视频严格解码；固定六条排除来自此前完整RGB扫描。保留原manifest供head-only OSVI，clean manifest供ACT/π系列。`FEATURE_CACHE`是ACT raw CLS特征，不可与RoboKata DINO patch缓存共用。

## 2. IG C15 VAE与动作归一化

```bash
source recipes/common.sh
export ZERO_WAM_MODEL="$MODEL" ZERO_WAM_EMPTY_TEXT="$EMPTY_TEXT" ZERO_WAM_CACHE="$IG_CACHE"
cd "$IG_ROOT"
python -m examples.baselines.zero_wam.precompute_vae --protocol c15 --manifest "$MANIFEST_IG" --model-path "$MODEL" --human-root "$HUMAN_ROOT" --sim-root "$SIM_ROOT" --cache-dir "$IG_CACHE" --action-per-frame 16 --robot-frame-stride 4 --image-size 224 --robot-width 288 --human-height 320 --human-width 480
cd "$ROOT"
python tools/prepare_ig_metadata.py --manifest "$MANIFEST_IG" --statistics "$STATS_IG" --selection "$WORK/ig_text_selection.json"
```
采用对应基座VAE。准备输出带源码/数据合同，不能把不同基座缓存未经身份核验复用。OSVI/π/LingBot共享IG原始动作与时钟合同，不把EE标签替代joint标签。官方ACT IG直接读LeRobot数据，使用自己的完整官方环境，不需上述VAE。

## 3. OSVI-WM

RT直接使用2149对的 `$PREPARED_RT_ORIGINAL`。IG需准备真实RGB/原生joint轨迹缓存：
```bash
source recipes/common.sh
python -m osvi_ig.prepare --manifest "$MANIFEST_IG" --output "$(dirname "$PREPARED_IG")"
TRAIN_DRY_RUN=1 bash recipes/osvi_ig10k.sh
# RT: TRAIN_DRY_RUN=1 bash recipes/osvi_robotwin.sh
```
完整算法/坐标适配/SoftDTW约定见 `models/osvi/osvi_robotwin/README.md`；里面的早期smoke状态保留作历史说明，正式预算以上表为准。

## 4. 分段Demo π0与原生text π0.5

```bash
source recipes/common.sh
export PYTHONPATH="$ROOT/models/openpi/src:$ROOT/models/openpi/examples/segmented_demo:$ROOT/models/openpi/packages/openpi-client/src:$PYTHONPATH"
python models/openpi/examples/segmented_demo/prepare_transfer.py --dataset robotwin --input "$PREPARED_RT" --root "$DATA_RT" --output "$PI_DATA" --workers 4
# IG改为 --dataset ig10k --input "$MANIFEST_IG"，不传--root，另设PI_DATA。
python models/openpi/examples/segmented_demo/cache_fixed.py --data "$PI_DATA" --base "$PI_BASE" --output "$PI_CACHE" --world 1 --rank 0
python models/openpi/examples/segmented_demo/cache_fixed.py --data "$PI_DATA" --base "$PI_BASE" --output "$PI_CACHE" --world 1 --merge
python tools/render_config.py configs/segmented_pi0_robotwin.json "$CONFIG"
TRAIN_DRY_RUN=1 bash recipes/segmented_pi0_robotwin.sh
```
IG选 `configs/segmented_pi0_ig10k.json` 和对应脚本。SigLIP只缓存冻结raw patches，投影/池化仍训练。多GPU缓存时显式分rank且不重复同输出；完成全部shard再merge。

π0.5 RT复用clean `$PI_DATA`，更换为π0.5 `$PI_BASE`，无需Demo feature cache：
```bash
python models/openpi/examples/robotwin_text/train.py --prepare-data "$PI_DATA" --assets "$PI_ASSETS"
python tools/render_config.py configs/pi05_robotwin.json "$CONFIG"
TRAIN_DRY_RUN=1 bash recipes/pi05_robotwin.sh
```
config模板不覆盖已有文件。学习率原始曲线与stop20k见协议说明。

## 5. LingBot-VA text IG

```bash
source recipes/common.sh
test ! -e "$LINGBOT_SOURCE"
mkdir -p "$(dirname "$LINGBOT_SOURCE")"
cp -a third_party/lingbot_va "$LINGBOT_SOURCE"
python models/lingbot/lingbot_ig_text/prepare_upstream.py "$LINGBOT_SOURCE" --mixed-batch
python models/lingbot/lingbot_ig_text/encode_text.py --source "$LINGBOT_SOURCE" --model "$LINGBOT_MODEL" --descriptions "$IG_ROOT/examples/baselines/lerobot_dataset/task_desc/human_desc.json" --selection "$WORK/ig_text_selection.json" --output "$TEXT_CACHE"
python tools/render_config.py configs/lingbot_ig_c15.json "$CONFIG"
TRAIN_DRY_RUN=1 bash recipes/lingbot_ig_c15.sh
```
仅human task description第一条，不读human视频。代码强制8卡×8及2820更新。`STORAGE`使用容量足够的节点本地存储；全参Adam checkpoint显著大于LoRA。已保留lazy Adam slot恢复修复，不能替换成早期源码。

## 6. RoboKata与Zero-WAM

RT B1，基座用pretrain。先按 DATA.md 下载含VAE latents/action_stats/receipt的完整发布，再构建DINO缓存：
```bash
export PYTHONPATH="$PWD/models/robokata/backend:$PWD/models/robokata/roboicl:$PWD/models/zerowam"
python models/robokata/robotwin_b/feature_cache.py --data "$DATA_RT" --encoder "$DINO_BASE" --cache "$FEATURE_CACHE" --init
python models/robokata/robotwin_b/feature_cache.py --data "$DATA_RT" --encoder "$DINO_BASE" --cache "$FEATURE_CACHE" --shards 1 --shard 0
python models/robokata/robotwin_b/feature_cache.py --data "$DATA_RT" --encoder "$DINO_BASE" --cache "$FEATURE_CACHE" --verify
TRAIN_DRY_RUN=1 bash recipes/robokata_robotwin_b1.sh
```
IG A用posttrain-robotwin；IG FI用pretrain。在下面选择匹配目录，并为不同schema指定不同特征文件：
```bash
source recipes/common.sh
COMPONENT="$ROOT/models/robokata"  # FI改为models/robokata_ig_fi
export PYTHONPATH="$COMPONENT/backend:$IG_ROOT"
python "$COMPONENT/roboicl/ig10k_features.py" --manifest "$MANIFEST_IG" --encoder "$DINO_BASE" --output "$DEMO_FEATURES" --device cuda:0
# 仅FI额外执行：
# python "$COMPONENT/roboicl/ig10k_semantic_features.py" --manifest "$MANIFEST_IG" --encoder "$DINO_BASE" --output "$SEMANTIC_FEATURES" --device cuda:0
TRAIN_DRY_RUN=1 bash recipes/robokata_ig_a.sh
# FI改用recipes/robokata_ig_fi.sh；官方Zero-WAM用recipes/zerowam_ig_c15.sh。
```
上述缓存输入都是RGB，Query未来语义仅作训练目标，不能作为推理输入。原958步暂停的FI作业不由本库恢复。

## 7. 评测与消融交接边界

本次提供训练入口及已有部署代码，未启动新的训练/评测。正式评测还须冻结six-task seeds、Demo映射、专家可行性、执行器和检查点；IG区分ZS与目标任务微调。不要直接用不同既有小样本成功率填同一论文列。现有代码的适用边界见 `ABLATION_STATUS.md`，历史A/B/FI不自动构成单变量消融。

RT B1特征提取器保留历史单进程GPU显存比例0.02上限（当时按H200设置）。在其他卡上若触发该缓存阶段的显存限制，可显式 `--device cpu` 完成特征准备；不要把它误判为正式训练batch的显存上限。正式训练关闭占位预留，但仍须验证真实batch。
