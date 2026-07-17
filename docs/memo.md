# MAPR 实验备忘

- Answer-change 图中的 `No Change` 是答案未改变的比例，不是准确率；其中同时包含保持正确和保持错误。

- Answer-change 使用 Agent-level 统计。GSM8K 为 `500 × 3 = 1500` 条回答轨迹；主表中的多 Agent 方法使用500道题的多数投票准确率，两者不能直接逐项对应。

- Self-correct、Ours 与 Zero-shot CoT 分别独立调用模型，不能把主表 Zero-shot CoT 的准确率当作其他方法的初始准确率。

- DashScope 会在 StrategyQA 第385条（`Does the Dalai Lama believe in the divine barzakh?`）的 Creation 输入阶段返回 `data_inspection_failed`。

- 样本过滤统一使用 `--exclude-sample <数据集>:<1-based样本编号>`，例如 `--exclude-sample StrategyQA:385`；多个样本可重复传入该参数。生成端保留 `excluded=true` 占位记录以维持原始索引，评估端自动排除占位记录。
- StrategyQA 第385条在后续主表重跑和 Figure 4 补充实验中统一过滤，按共同的499条有效样本评估。历史 `result_qwen3_8b` 文件不自动改写；只有使用新参数续跑或重跑的结果才会写入占位记录。

- 被排除样本保留在结果 JSON 的原始位置，使用空 `agent_contexts` 和明确的 `exclusion_reason` 占位，使文件仍有500条并维持数据索引对齐。
