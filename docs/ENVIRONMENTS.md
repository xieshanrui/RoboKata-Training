# 环境准备

训练面向 Linux NVIDIA CUDA，Python 3.12。每组模型单独建 venv；仿真器另设环境。先 `git submodule update --init`，不要递归初始化历史子依赖，也不要在共享环境执行安装脚本。

本仓 `environments/runtime_versions.json` 是本次从既有成功运行环境读取的包版本，仅含包名/版本/Python版本，无服务器路径。以下是训练所需的直接依赖清单；它们来自实际环境，但本次未在全新GPU机器重新安装验证。

```bash
python3.12 -m venv /your/env/act
source /your/env/act/bin/activate
python -m pip install -r environments/act-py312.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

| 配方 | 依赖 | 说明 |
|---|---|---|
| ACT RT | `act-py312.txt` | Torch 2.9 / CUDA12.8、TorchCodec0.8.1；需FFmpeg共享库 |
| OSVI RT/IG | `osvi-py312.txt` | 同Torch；SoftDTW必须使用记录的numba-cuda/CUDA bindings组合 |
| RoboKata、Zero-WAM、LingBot IG | `wam-py312.txt` | 本次记录的共用环境为CUDA12.8；保留各自PYTHONPATH |
| 分段π0 / π0.5 | `models/openpi/examples/segmented_demo/requirements-linux-py312.txt` | 独立环境，Torch2.7.1；不与WAM依赖混装 |
| 官方ACT IG | IG官方ACT文档/依赖 | 完整官方训练入口还会导入ManiSkill；不能将RT训练环境直接视为完整IG仿真环境 |

OSVI需系统已有兼容CUDA toolkit（原检查为12.6），设置 `CUDA_HOME`、`CUDA_PATH`。避免回退到已发现会segfault的旧Numba CUDA target。TorchCodec需要系统 `ffmpeg` 及可加载的 `libavcodec/libavformat/libavutil`；`ffmpeg -version` 和 `python -c "from torchcodec.decoders import VideoDecoder"` 均应通过。

π模型安装后，在其专用环境运行：
```bash
export PYTHONPATH="$PWD/models/openpi/src:$PWD/models/openpi/examples/segmented_demo:$PWD/models/openpi/packages/openpi-client/src"
python models/openpi/examples/segmented_demo/prepare_transformers.py --apply
```
该补丁针对记录的 Transformers4.53.2。不要改动其他模型环境。

LingBot必须将固定上游源码复制到 `$LINGBOT_SOURCE`，在副本上执行一次 `prepare_upstream.py --mixed-batch`，见训练步骤。该脚本验证原文再修改，不能反复打补丁。它使FlashAttention按需导入，并加入混长序列掩码；不应因上游requirements包含flash_attn就无条件编译安装。

训练前先做 `python tools/preflight.py` 和对应 `TRAIN_DRY_RUN=1 bash recipes/....sh`。再在分配的GPU上验证真实batch、损失、显存、检查点恢复；遇到OOM保留失败证据，不能静默改batch/梯度累计并宣称同一配方。

激活对应环境后，可用 `python tools/preflight.py --profile act` 核对安装版本；另支持 `osvi`、`wam`、`openpi`。该命令不使用GPU。ACT清单已补齐IG backbone直接依赖的IPython。官方ACT IG需要单独的完整训练环境，当前审查在已有IG仿真环境发现缺少TensorBoard，完整依赖链仍未验证；不要把这套仿真环境或RT ACT清单当成已验证的IG ACT训练环境。
