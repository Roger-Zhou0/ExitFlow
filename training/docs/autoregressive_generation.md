## 自回归逐 token 早退：工程实现思路

这个文档把你们后续要做的“逐 token 自回归 + 真实基准验证”进一步拆解为明确的代码模块与步骤。重点围绕两个核心目标：

> 1. 如何在现有 ModelBundle/exit head 结构上实现“token-by-token”的生成；
> 2. 如何把这个流程在 MMLU、IFEval 等真实 benchmark 上跑通，同时记录延迟、退出层与准确率等 KPI。

### 一、可复用的现有构件

| 组件 | 作用 |
|---|---|
| `training/exporter.py` | 已把 backbone（HF save_pretrained）和 exit head TorchScript 保存为 `artifacts/`，runtime 可以直接加载。 |
| `runtime/forwarder.py` | 定义 gate 三元组（logit/margin/entropy），可沿用在自回归判深/回退。 |
| `training/evaluate_exit_heads.py` | 演示了从 manifest+metadata+thresholds 装载 exit head、backbone 的全流程，可重用建立 runtime 的 loader。 |
| `evaluation/benchmarks.py` 等 | 提供 benchmark loader 与 MetricTracker，便于测试环节复用。 |

### 二、自回归生成模块设计

#### 1. 新增模块与概念

1. **`runtime/hf_truncated.py`（建议）**
   - 目标：构造一个“只计算前 L 层且可增量更新”的 HF wrapper。
   - 主要方法：
     - `prime(input_ids)`：Run 一次前向，返回前 L 层 hidden states 与 `past_key_values`（长度为 L）。
     - `step(next_token_id, state)`：每生成一个 token，仅运行前 L 层，更新时间步缓存，输出该层最后 token hidden state。
   - 实现细节：直接复用 HF 模型的 `model.model.layers[:L]` 顺序迭代；`use_cache=True`；`attention_mask`/`position_ids` 兼容。

2. **`runtime/autoregressive.py`**
   - 类 `AutoRegressiveGenerator`
     - 初始化：加载 manifest -> artifacts -> tokenizer；根据 manifest.precision 推断 dtype；按 exit_spec 生成 `ExitHeadRunner`（封装 TorchScript head + thresholds）。
     - `select_exit_depth(input_ids)`：先用 `AutoModelForCausalLM`（完全版本）前向一次（或逐层跑 truncated probe）来拿到各层 hidden_state，通过 `ExitHeadRunner` 计算 gate 分数，返回第一个通过 threshold 的层 ID（对应的 L_exit）。
     - `generate(prompt, max_new_tokens, eos_token_id=None, decode_temp=1.0)`：
       1. tokenizer 编码 prompt，调用 `select_exit_depth` 得到 `L_exit` 与 `forwarder_id`。
       2. 用 `TruncatedCausalLM(max_layer=L_exit)` `prime` prompt，得到 initial past/hidden。
       3. 进入 token loop：每步 `step(next_token)`，取 hidden -> exit head logits -> 采样（greedy/top-k） -> 记录 latency & gate scores -> append token -> 直到 `eos` 或 max steps。
       4. 最后用 tokenizer 解码 tokens 输出文本，并返回包含 `exit_depth`、`per_step_latency`、`exit_scores`（top_logit/margin/entropy）、`fallback` 等的结构。
     - 可选功能：在生成早期加入“低置信度 fallback”，当连续 `r` 次 entropy 高于阈值时重跑 `select_exit_depth` 以升深并重新 prime；也可以提供“单调不降深度”的扩展（在每步检查 gate）。
   - `ExitHeadRunner`：封装 TorchScript head + thresholds，提供 `score(hidden_state)` 返回 logits+scores，与 `forwarder.Forwarder.evaluate` 内部一致。
   - 统计：在生成过程中记录
     - 探针耗时、prime 时间（可独立计数）；
     - 每步 latency（二次 step 执行）；
     - 每步 gate scores；
     - 最终 `exit_depth`、`fallback_count`、`tokens_generated`。

#### 2. 关键逻辑和理由

| 问题 | 解决方案 | 理由 |
|---|---|---|
| 自回归时只算浅层，如何保持 KV 连续 | 每次生成 token 只运行至 L_exit，使用 `TruncatedCausalLM` 维护 `past`；一旦升深则重新 prime（可限制仅在前几 token），避免多次回填。 | 避免每 token 都要从头跑全部层，保留早退收益。 |
| 早退判定何时执行 | 在生成前用 `select_exit_depth` 探针（一次完整前向）决定 L_exit；也可以在每步判断 gate，再决定是否升深。 | 提前决策能稳定控制每 token 的算力；后续再考虑“单调递增”策略。 |
| 生成策略 | 先用贪心/温度1的 argmax；后期可加 top-k/top-p、beam；MMLU 类任务用候选 token 集直接打分。 | 简化初版，实现可控；MMLU 只需要“下一个 token”对应选项即可。 |

### 三、真实数据上的测试方案

#### 1. 数据准备

1. 使用 `training/scripts/download_benchmarks.py` 下载：
   - `mmlu_en.jsonl`（validation split）；
   - `ifeval`, `gsm8k`, `alpacaeval2` 等（可选）。
2. 若已有 JSONL，确保格式 `{"prompt": "...","response": "A"}` ；可以按需添加 `input_ids` 字段（已 tokenized），方便快速运行。

#### 2. 评测策略（不生成完整文本版先行）

1. **“一步分类”评测（字符型选项）**
   - 步骤：
     - 用 tokenizer 编码 prompt，取最后一个 token 的 hidden_state（可直接调用 `training/evaluate_exit_heads.py` 的方式）。
     - 每个 exit head 输出 logits，按 options 中 A/B/C/D 构建 candidate token 集（包含 `_`、空格等常见分词变体），以 `sum`/`max` 聚合概率。
     - 选出最高分的选项，对比 `response`（A/B 等），统计准确率。
     - 记录每层 exit head 的准确率和 gate statistics（top_logit/margin/entropy）。
   - 指标：exit_rate（按 thresholds 早退的比例）、`avg_exit_depth`、`p95_latency`（single forward time）、`accuracy`。
2. **门限级联版本**
   - 使用 `select_exit_depth` 的结果（带 threshold 决策），仅对最终选中层做一步分类，记录实际 exit depth。
   - 对比每个 threshold 组合下的 accuracy/latency/exit_rate。 

#### 3. 生成评测（自回归）

1. 运行 `AutoRegressiveGenerator.generate`，目标任务可选 MMLU prompt（一次生成多 token 但重点是首选项）、或开一个简单问答（如 AlpacaEval 提示）。
2. 测量：
   - 每步 latency（生成前/后），计算 tokens/s；
   - 每层对应 exit depth 的统计；
   - fallback/升级发生次数；
   - 生成结果是否包含正确选项（可用 scorer 进行模糊匹配）；
3. 记录 `GenerationResult` JSONL，便于后续分析。

#### 4. KPI 和调参反馈

1. **主要 KPI**
   - `Latency`：端到端（包括探针）或 decode-only；
   - `Throughput`：tokens/s；
   - `Accuracy`：MMLU/IFEval 结果；
   - `Exit rate`：按 layer 统计实际提前退出比例；
2. **调参建议**
   - 在 MMLU 的 `accuracy` 允许 1–2% 下降时，尝试降低 thresholds（缩放 α < 1），提高 exit rate；
   - 如果 accuracy 下降过快，先收集 gate scores 的分布（top_logit/margin/entropy 的分位）来判断 threshold 是否偏向稀疏/稠密；
   - 记录 fallback 触发及其对 latency 的影响，避免“多次回退”导致延迟反而更高。

### 四、实现进度建议

| 阶段 | 内容 | 产出 |
|---|---|---|
| Stage1: 准备 | 抽象 TK truncated wrapper + exit head scoring 工具 | `runtime/hf_truncated.py`, `runtime/autoregressive.py` 的 scaffolding |
| Stage2: 非生成测试 | 实现 step1 分类，用标准 benchmark 统计 accuracy/exit depth/latency | 结果 CSV/JSON + gate scores |
| Stage3: 自回归生成 | 完整 `AutoRegressiveGenerator.generate`，包括 fallback 逻辑与 metrics | JSONL 结果 + KPI 报告 |
| Stage4: 阈值与深度搜索 | 结合 gate score histogram 调整 thresholds，验证在 real dataset 上的 exit rate 与 accuracy trade-off | 复用 `calibrate_thresholds_from_artifacts.py` 的 calibrator + new scan script |

### 五、小结

这套方案充分利用现有的 manifest、runtime、exit head 与门限计算逻辑，新增 `TruncatedCausalLM`+`AutoRegressiveGenerator` 组合，将“早退”机制与自回归生成解耦。真实数据测试分成“只评分类”与“生成解码”两条线，先验证浅层 exit head 的 accuracy 曲线，再在保证误差允许的情况下做全速自回归。你可以先依据这个计划实施新模块，然后用 benchmark 数据逐步收敛 threshold/ fallback 策略。
