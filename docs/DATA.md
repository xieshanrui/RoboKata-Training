# 数据、基础权重与缓存

## 固定数据版本

| 数据集 | revision | 训练范围 |
|---|---|---|
| [HumanGen-RoboTwin-43Train-7Test](https://huggingface.co/datasets/xieshanrui/HumanGen-RoboTwin-43Train-7Test) | `e2a347bb78dd48fd869bd7642a51430a60676876` | 43任务，2149原始human→robot对 |
| [IG-10K-Dataset](https://huggingface.co/datasets/imitator-game/IG-10K-Dataset) | `9a57b27edcb54823d8da7d2b77efa61c092ffa95` | C15：3000 robot +750 human |

RT原训练manifest SHA256：`1fa416a57cd93983ceb31393947a0db2f7eaa3f82dfbebb5ae4de983638c0802`。
RT用 `act_robotwin.prepare --download` 下载训练清单引用的文件；可独立准备原2149和过滤后2143两份manifest。六条固定排除记录见 `configs/robotwin_rgb_exclusions.json`，只用于原发布版本，不能根据评测成败改动。

IG按 `configs/ig_c15_download_manifest.json` 只下载C15所需文件（7,235,724,007字节，不含VAE特征/权重）：
```bash
python tools/download_ig.py --output /your/data/IG-10K
```
下载工具固定revision、核验文件大小及LFS SHA256/Git blob OID；重跑可续传。目标目录之后设置 `HUMAN_ROOT=.../imitator_human_v1`、`SIM_ROOT=.../imitator_sim_v1_zed2i`。

## 初始化权重

| 用途 | HF仓库 / 固定revision |
|---|---|
| RoboKata IG A / 官方Zero-WAM IG | `robbyant-research/zero-wam-posttrain-robotwin` / `07ee865f175d9474a5653e9147698390e47b6143` |
| RoboKata RT B1 / IG A+B1+FI | `robbyant-research/zero-wam-pretrain` / `7040c4195df216c900334ef62d5fdcf05c0601aa` |
| LingBot IG text | `robbyant/lingbot-va-base` / `68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c` |
| ACT Demo DINO | `facebook/dinov2-large` / `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` |
| RoboKata Demo/语义 DINO | DINOv2-base；用实际下载文件SHA256固定，记录在缓存与checkpoint |
| 分段π0 | 官方π0 base的PyTorch权重；训练config和缓存记录SHA256 |
| π0.5 text | 官方π0.5 base转换的PyTorch权重；使用本库OpenPI转换器 |

示例（依具体配方选择基座，不能统一填同一个MODEL）：
```bash
hf download robbyant-research/zero-wam-posttrain-robotwin --revision 07ee865f175d9474a5653e9147698390e47b6143 --local-dir "$MODEL"
```
`EMPTY_TEXT` 指向固定Zero-WAM源码中的 `empty_text_emb.pt`（用 `find models/zerowam -name empty_text_emb.pt` 定位）。ACT `cache_demo` 默认自动下载固定DINO-Large并写入 `revision.json`；手动指定DINO路径时也必须有这份逐文件SHA记录。

π权重不在本库。分段π0需要官方π0 PyTorch `model.safetensors` 与 `paligemma_tokenizer.model`。π0.5可使用 `models/openpi/examples/convert_jax_model_to_pytorch.py --checkpoint-dir ... --config-name pi05_aloha --output-path ... --precision bfloat16` 转换官方JAX权重；先查看该固定版本 `--help`。`PI_BASE` 指向转换后的文件，不能把π0和π0.5混用。

## 路径与缓存

复制并填写 `configs/paths.env.example`；示例路径是占位符。不要提交token、SSH密钥、数据或权重。模型下载遵守其原许可证。
VAE manifest包含绝对路径；移动原始数据/缓存后重新准备并验证，不能只改字符串后跳过来源检查。DINO缓存绑定模型SHA和manifest；IG A、IG FI、RT B1各用自己的builder与独立输出路径。训练只读取已有缓存；准备VAE、DINO、SigLIP、LingBot text cache需要GPU预算，不能计作纯CPU准备。

RoboKata RT B1额外读取发布的三视角VAE latents、human latents、action_stats和download_receipt。ACT的RGB下载不包含这些；B1前先执行：
```bash
python tools/download_robotwin.py --output "$DATA_RT"
```
该命令下载完整固定发布（含原test文件），训练入口仍只使用原train.jsonl。
