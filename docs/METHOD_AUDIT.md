# 训练入口与模型一致性审查 · 2026-11-12

**当前结论：不能把整个仓库标为“新机器开箱即训、全套GPU已验证”。** 已检查公开代码的依赖、实际启动参数、动作表示、主要损失、采样、掩码和恢复路径。使用既有Linux隔离环境，隐藏全部GPU，执行导入和真实参数解析；没有加载大模型训练，也没有安装或修改共享环境。

## 已修复的问题

| 问题 | 影响 | 修复与验证 |
|---|---|---|
| ACT最小依赖漏了IPython | IG的DETR backbone直接导入IPython；干净环境会失败，原训练环境中已安装因此未暴露 | 补入记录的 `ipython==8.37.0`；核对实际backbone导入 |
| LingBot把SystemExit当成训练故障 | `--help`本应退出0，却因找不到`--config`再次抛错；配置读取失败也可能被失败日志覆盖 | 只捕获普通Exception；日志失败保留原异常；修复后实际完整命令解析通过 |
| ACT恢复未核验适配/上游源码 | 参数结构相同但预处理/训练实现变化时，可能静默续训 | checkpoint写入并校验两类源码SHA；相同身份接受、缺失/改变拒绝 |
| OSVI恢复只核验上游源码 | 本地RT/IG数据适配及共享ACT数据工具变化未被覆盖 | 保存并验证相关适配源码SHA；校验后才写运行配置；同样通过身份检查 |

这些修复不改变前向、损失、优化器、学习率或采样公式。新ACT/OSVI checkpoint按新合同恢复；旧checkpoint应使用原冻结源码恢复。旧权重的推理加载格式未改，既有训练结果没有重新计分或宣称失效。

## 启动验证范围

| 配方 | 真实参数解析/导入 | 本轮GPU前后向 |
|---|---|---|
| ACT RoboTwin | 通过；另检查IG backbone导入 | 未运行 |
| OSVI RoboTwin / IG | 两项通过；另检查原生模型、SoftDTW导入 | 未运行 |
| LingBot text IG | 修复后通过 | 未运行 |
| 分段π0 RoboTwin / IG、text π0.5 RoboTwin | 三项通过 | 未运行 |
| RoboKata RT B1、IG A、IG FI | 三项通过 | 未运行 |
| 官方Zero-WAM IG | 通过tyro真实参数解析 | 未运行 |
| 官方ACT IG | 当前可用IG环境在 `tensorboard` 导入处失败；完整LeRobot/训练环境尚未核验 | 未运行 |

共11/12配方通过实际命令解析。此检查在argparse/tyro完成解析后主动退出，不执行数据加载、分布式初始化或优化器更新。官方ACT IG为保留的官方入口，本项目此前未重训；不能将RT ACT的依赖清单当成完整IG ACT训练环境。仅补一个tensorboard也不足以宣称整个依赖链通过。本次不使用上游包含全部基线的大型安装清单覆盖现有环境。

## 与原模型的关系

| 模型 | 保留的核心 | 跨基准适配/必须说明的区别 |
|---|---|---|
| ACT/DINOv2 | IG DETRVAE、冻结DINO-Large、可训练adapter、prior z=0推理、L1+10×KL | RT为三视角、16D双臂EEF、训练集mean/std、24步chunk；RGB仅/255、尾部重复末动作参与损失，沿用IG版本；未加IG图像增强 |
| OSVI-WM | 原生ResNet18、动作注意力与自回归latent模型、1–5 waypoint头、SoftDTW、mixup、stop-gradient latent L1 | 单臂XYZ+gripper扩为双臂16D；RT为XYZ/WXYZ/gripper、IG为原生joint；按双臂平均SoftDTW；原相机投影改原生坐标，去掉几何增强；72k预算及损失系数衰减尺度是适配配方 |
| LingBot-VA text | 原生全参Trainer、噪声/时钟、chunk/window采样、视频和动作flow loss、Adam/clip/LR | IG zed2i、joint16映射至30通道、human description第一条；混长batch补掩码并逐episode等权；8×8/60epoch为项目预算 |
| 分段Demo π0 | OpenPI PyTorch训练循环与flow matching | Demo分段/池化、冻结视觉缓存是项目模型改动，不能当作原生text π0；RT与IG使用各自原生动作 |
| text π0.5 RT | 原生π0.5架构、text/state编码与flow matching、OpenPI训练循环 | robot_instruction文本、双臂EEF16填充至32维、屏蔽无效动作维/尾部；最终停止20k，保留原30k LR曲线 |
| Zero-WAM IG | IG的原生episode数据合同、LoRA106、MCP、动作表示/时钟 | 开启激活重算；global4/60epoch=45000更新 |
| RoboKata A/B1/FI | 共享Zero-WAM训练主干，A增加Demo交叉注意力，B1使用DINO语义目标，FI增加预测key检索 | 不同分支有不同基座及数据配方；IG global64/60epoch=2820更新；RT B1有裁窗、分桶和丢尾，不能套用IG完整序列协议 |

原始ACT在 [官方policy.py](https://github.com/tonyzhaozh/act/blob/main/policy.py) 中使用ImageNet图像归一化和padding损失掩码。当前ACT实现对齐的是 [IG版本](../environments/imitator_game/examples/baselines/act/train_act_imitator.py)，不是原始ACT的逐行复现。为保持既有模型身份，本次没有偷偷更换图像预处理或重新定义尾部损失。论文建议写“ACT/DINOv2, adapted to RoboTwin”。

OSVI上游当前实现的latent loss使用 `enc_features` 的停止梯度结果；其 `target_encoder` 虽执行EMA更新，但返回的target特征未接入该损失。这是固定上游代码的现有行为，本次保留；不要额外声称该损失已使用EMA teacher targets。

## 仍需处理的验证边界

1. **RoboKata RT B1归一化来源**：沿用发布的 `robotwin_data/meta/action_stats.json`。数据卡明确说明未独立证明它只由43训练任务计算。当前不能声称这份统计完全来自train split，也不能据此直接判定已经泄漏。严格train-only实验需另行拟合、记录并从头训练，不能给旧权重换统计后混报结果。
2. **新机器的GPU短训**：真实数据、指定实际batch、一次保存和恢复后的更新尚需验证。历史运行证据不能替代新环境检查；RT B1每卡16尤其不能直接承诺适配A80080GB。
3. **官方ACT IG独立环境**：补齐并验证TensorBoard、LeRobot及其数据/仿真依赖后，才标为可启动。当前可继续引用匹配协议的官方C15结果，不必为交接而自动重训。
4. **公平比较**：不同基座、global batch、更新次数、2143/2149数据范围与输入模态已不同；保持各自身份并在实验设置中说明。单变量消融需另行控制这些变量。

本轮未发现需要立即改写其他模型训练目标的确定证据；这份审查覆盖上述入口和合同，不是对全部上游代码的无缺陷保证。机器可读摘要见 `audit_20261112.json`。
