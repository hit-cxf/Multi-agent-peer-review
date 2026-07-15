# MAPR 实验备忘

- Answer-change 图中的 `No Change` 是答案未改变的比例，不是准确率；其中同时包含保持正确和保持错误。

- Answer-change 使用 Agent-level 统计。GSM8K 为 `500 × 3 = 1500` 条回答轨迹；主表中的多 Agent 方法使用500道题的多数投票准确率，两者不能直接逐项对应。

- Self-correct、Ours 与 Zero-shot CoT 分别独立调用模型，不能把主表 Zero-shot CoT 的准确率当作其他方法的初始准确率。

- DashScope 会在 StrategyQA 第385条（`Does the Dalai Lama believe in the divine barzakh?`）的 Creation 输入阶段返回 `data_inspection_failed`。

- Figure 4 的所有 StrategyQA 配置统一将第385条标为 `excluded=true`，按共同的499条有效样本评估。排除规则只适用于 `result_agent_num_review_rounds` 中的补充实验及其副本，不得修改 `result_qwen3_8b` 主表数据。

- 被排除样本保留在结果 JSON 的原始位置，使用空 `agent_contexts` 和明确的 `exclusion_reason` 占位，使文件仍有500条并维持数据索引对齐。
