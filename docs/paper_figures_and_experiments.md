# MAPR 论文图表与补充实验清单

本文档梳理原论文 *Towards Reasoning in Large Language Models via Multi-Agent
Peer Review Collaboration* 中的全部图表，并记录使用 Qwen3-8B 重新实验时的
完成状态、数据来源与补充实验需求。

论文：<https://arxiv.org/abs/2311.08152>

## 实验配置口径

原论文主实验设置：

- 基座模型：`gpt-3.5-turbo-0613`
- Agent 数量：3
- Review cycle 数量：1
- 每个数据集最多随机抽取500个样本
- 最终答案由三个 Agent 多数投票得到

当前复现实验使用：

- 基座模型：`qwen3-8b`
- Agent 数量：3
- 主表包含 Ours、全部 baseline 和两种消融

### `rounds` 参数的特别说明

论文中的一个 review cycle 表示一次完整的 `Review -> Revision`。但是当前
`peer_review.py` 将 Creation、Review、Revision 写成三次循环：

```text
round_num = 0  Creation
round_num = 1  Review
round_num = 2  Revision
```

因此当前代码的正确参数是：

| 方法 | 命令行参数 |
|---|---:|
| Ours | `--rounds 3` |
| Ours (w/o confidence) | `--rounds 3` |
| Ours (w/o solution) | `--rounds 3` |
| Multi-agent Debate | `--rounds 2` |

如果 Ours 使用 `--rounds 2`，程序只执行到 Review，没有执行最终 Revision，
不能作为论文中的 Ours 结果。

## 主表

### Table 1：数据集信息

论文列出10个数据集的任务类型、测试集规模及实验采样数量：

- 数学推理：GSM8K、SVAMP、AQuA、MultiArith、AddSub、SingleEq
- 常识推理：ARC-c、StrategyQA
- 符号推理：Colored Objects、Penguins

不需要新增模型调用，可由 `params.py`、`datasets/` 和 `processed_data/` 整理。

### Table 2：数学推理主结果

包含6个数学推理数据集以及以下方法：

- Zero-shot CoT
- Self-correct
- Multi-agent Majority
- Multi-agent Debate
- Ours
- Ours (w/o confidence)
- Ours (w/o solution)

状态：已完成。

### Table 3：常识推理主结果

包含 ARC-c 和 StrategyQA，方法与 Table 2 相同。

状态：已完成。

### Table 4：符号推理主结果

包含 Colored Objects 和 Penguins，方法与 Table 2 相同。

状态：已完成。

## 正文图表

### Figure 1：Self-correction 与 MAPR 对比

性质：方法动机示意图，不是定量实验。

内容：

- Self-correct 依赖模型自身检查，可能无法发现原有错误。
- MAPR 通过其他 Agent 的解答和反馈引入外部信息。

处理方式：重新设计示意图即可，不需要调用模型。

### Figure 2：Creation、Review、Revision 流程

性质：方法流程与 GSM8K 案例示意图。

内容：

1. 三个 Agent 独立生成初始解答。
2. 每个 Agent 依次评审其他 Agent，并给出 confidence。
3. 每个 Agent 根据其他解答和收到的 feedback 修订答案。
4. 最终通过多数投票得到预测。

处理方式：可以从已有 GSM8K `peer_review` 结果中筛选案例，无需补跑。

### Figure 3：GSM8K 答案变化

比较 Self-correct 与 Ours 修订前后的答案变化：

- No Change
- Correct -> Incorrect
- Incorrect -> Correct
- Incorrect -> Incorrect

统计单位为 Agent 回答，因此当前 GSM8K 分母为：

```text
500 questions * 3 agents = 1500 answers
```

状态：已完成，并额外加入两种消融。

相关文件：

- `visualize/plot_answer_transitions.py`
- `pics/answer_transitions_GSM8K_0713.png`
- `pics/answer_transitions_GSM8K_0713.pdf`

### Figure 4：Agent 数量和 review round

数据集：GSM8K、SVAMP。

#### Figure 4a：Agent number

横轴设置：

```text
Agent number = 2, 3, 4, 5
```

保持模型、数据、prompt 和 review cycle 不变，只改变 Agent 数量。

状态：需要补跑。

建议为保证曲线内部一致性，统一重新运行 Agent 2–5，而不是只复用主表的
Agent 3 点。

#### Figure 4b：Review round

横轴设置：

```text
Review round = 1, 2, 3, 4
```

每一轮应当执行完整的：

```text
Review -> Revision
```

修订后的答案继续进入下一轮 Review。论文连续执行4轮，并在每轮 Revision
结束后计算准确率。

状态：需要补跑并新增实现。

当前 `peer_review.py` 不支持真正的多轮 review cycle。简单传入
`--rounds 4` 无效，因为 `round_num == 3` 没有任何处理逻辑。相关代码应放在
`supplementary/`，不能通过修改主表实现来完成。

### Figure 5：Confidence 与 feedback correctness

数据集：GSM8K、Penguins。

论文从每个数据集中人工标注600条 feedback，共1200条。每条 feedback 标注：

```text
correct feedback / wrong feedback
```

图包含：

1. verbalized confidence 分布；
2. reliability diagram；
3. Feedback accuracy；
4. AUROC；
5. ECE。

现有结果已经保存 peer solution、feedback 和 confidence，不需要重新调用模型。
但是 feedback correctness 不能仅由目标 Agent 的最终答案自动推导，必须进行
人工语义标注，才能严格复现论文。

状态：需要新增数据抽取、人工标注模板、指标计算和绘图程序。

### Table 5：异构 LLM 协作

论文使用两种不同 LLM 组成两 Agent peer review，比较：

- Initial accuracy
- Updated accuracy
- Capability gap
- Diversity（INCON）
- Accuracy delta

原论文模型组合：

- GPT-3.5-0301 + GPT-3.5-0613
- GPT-3.5-0613 + Claude Instant 1.2
- Claude Instant 1.2 + Claude 2.1

这些旧接口不适合直接复跑。Qwen 实验可以改为例如：

- Qwen3-4B + Qwen3-8B
- Qwen3-8B + Qwen3-14B
- Qwen3-8B + 一个能力接近但来源不同的模型

状态：需要补跑，但属于异构模型扩展，优先级低于 Figure 4、Figure 5 和
Table 7。

## 附录图表

### Table 6：Confidence 消融案例

展示同一个错误初始答案在以下两种条件下的修订过程：

- 不包含 confidence：被错误 feedback 误导，最终仍错误；
- 包含 confidence：根据不同 confidence 选择信息，最终修订正确。

处理方式：可以从已有 `feedback` 和 `peer_review` 结果中筛选，不必重新调用
模型。由于两种方法是独立运行，初始答案可能不同，因此最终案例需要人工确认
可比性。

### Figure 6：Penguins 答案变化

统计定义与 Figure 3 相同，数据集改为 Penguins。

状态：不需要补跑，可以直接用现有结果生成。

生成命令：

```bash
MPLCONFIGDIR=/tmp/mapr-matplotlib \
conda run --no-capture-output -n MAPR \
python visualize/plot_answer_transitions.py \
  --result-dir result_qwen3_8b \
  --task Penguins \
  --time-flag 0713
```

### Table 7：多样化角色提示词

数据集：GSM8K、SVAMP。

实验组：

- Single Agent
- Multi Agent (single role)
- Multi Agent (diverse role)

论文使用5种角色：

1. AI Assistant
2. Math Teacher
3. Mathematical Scientist
4. Engineer
5. Computer Scientist

Single-role 条件下所有 Agent 都使用 AI Assistant；diverse-role 条件下，每道
题从5个角色中随机选择角色，通过 system message 初始化 Agent。

状态：需要新增代码并补跑。当前项目没有角色 prompt 实现；应在
`supplementary/` 中创建独立实验，固定随机种子，并保持其他 MAPR 设置不变。

### Table 8–11：完整 Case Study

- Table 8：Stage 1 Creation
- Table 9：Agent A、B 的 Stage 2 Review
- Table 10：Agent C 的 Stage 2 Review
- Table 11：Stage 3 Revision

可以从已有 GSM8K `peer_review` 结果中自动筛选：

- 至少两个 Agent 初始答案错误；
- 最终多数投票正确；
- 至少一个 Agent 从错误修订为正确；
- feedback 中包含清晰的纠错过程。

状态：不需要补跑，需要案例筛选和排版程序。

## 补充实验优先级

### P0：配置核查

- 确认主表 Ours 和两种消融实际使用 `--rounds 3`。
- 如果使用了 `--rounds 2`，必须重跑对应结果。

### P1：直接利用现有结果

1. 生成 Figure 6 Penguins answer changes。
2. 筛选 Table 6 confidence ablation 案例。
3. 筛选 Table 8–11 完整 MAPR case study。

### P2：核心补充实验

1. Figure 4a：Agent number = 2、3、4、5。
2. Figure 4b：Review round = 1、2、3、4。
3. Figure 5：抽取并人工标注1200条 feedback，计算校准指标。
4. Table 7：single-role 与 diverse-role 实验。

### P3：可选扩展

- Table 5：基于 Qwen 系列或跨模型组合重新设计异构 LLM 协作实验。

## 工作区目录规范

```text
visualize/      所有绘图程序
pics/           所有生成的 PNG、PDF 图表
supplementary/  主表以外的补充实验代码
docs/           实验设计、图表梳理和复现说明
```

补充实验的运行结果仍应写入独立结果目录，避免与主表结果混合或覆盖。
