# 现有模块与消融边界

本仓交接已实现的公开基准训练代码。未公开的论文和详细消融计划不在本仓发布。

| 已有代码 | 可用于新实验的基础 | 使用边界 |
|---|---|---|
| 官方 Zero-WAM / IG C15 | 无新增Demo交叉注意力的基线入口 | 与global64的RoboKata预算不同 |
| RoboKata IG A | DINO Demo交叉注意力 + 原生MCP | posttrain-robotwin初始化 |
| RoboKata RT A+B1 | Demo交叉注意力 + Query语义预测 | pretrain初始化；RoboTwin专用采样/动作表示 |
| RoboKata IG A+B1+FI | 语义预测与foresight indexing | 独立冻结分支，pretrain初始化 |
| ACT / OSVI / LingBot text / OpenPI | 各基准对照训练入口 | 覆盖范围见主README，不能把未接入项视为可运行 |

现有不同基准、初始化、数据量和训练预算的权重不能直接组成单变量消融。新消融须先固定训练数据、初始化、预算、实现开关与独立验证集，再训练或评测。没有实现的开关不能通过杜撰CLI调用。测试集不用于选择checkpoint。

本次不自动重训、恢复旧作业或运行消融；历史模型与基准训练入口均保留独立身份。
