# 冻结协议

## RoboTwin 43→6
使用 HumanGen 发布的43任务训练划分，原7个测试任务保留。仅评测时排除 `open_microwave`，不把它放回训练。六任务为：place_object_scale、stamp_seal、move_stapler_pad、place_bread_basket、place_empty_cup、stack_blocks_three。

原训练清单2149对。三视角ACT与π0/π0.5排除六条已发现的损坏腕部视频，保留2143对/457680帧，见 configs/robotwin_rgb_exclusions.json。OSVI历史只读head RGB，使用原2149对；不得静默声称所有模型训练样本完全相同。RoboKata历史B1保留原2149及其既有采样配方。

Query 原生50Hz、16D绝对双臂EEF（XYZ/WXYZ/gripper），不混成joint标签。Demo条件模型只读human RGB；π0.5原生text用发布的robot_instruction。环境资产/任务修复/专家可行初态/执行器/推理种子需在评测前另行冻结。stock RoboTwin eval_policy.py 的100条/专家搜索不等于本项目六任务各50条评测协议。

## IG C15
3000 robot +750 human，15训练任务，官方zed2i，原生joint16动作和官方时钟。5未见任务×L0–L3×每格10=200条ZS；目标任务不更新参数。官方FT/Scratch/Seen与ZS分开，不能混填。

RGB模型从同任务human库配对；LingBot text读取对应human_desc的第一条说明，不能用目标任务标签替代描述。归一化只拟合训练集。VAE缓存绑定基座/VAE、相机尺寸、抽帧时钟、动作表示和来源SHA；不同encoder或DINO schema不能混用。

## 检查点与预算
固定最终epoch/update，不能用测试成功率选权重。π模型历史计划RT30k/IG60k后被要求在20k结束，但保留原LR曲线：本仓配置保留原schedule，启动命令通过 --steps 20000 停止。换batch、epoch或从头20k余弦曲线属于新实验。

所有训练脚本均为有限epoch/update；无梯度累计。恢复必须同模型、同数据、同训练预算与归一化并明确传resume参数。现有历史作业不会由本仓自动恢复。
