# 单节点四卡 A100 早退推理实现指南

> 目标：在**单节点、4×A100** 环境下，基于本仓库现有代码，实现一个可用的早退推理路径和对照（无早退）路径，用于课程项目实验和展示。  
> 约束：不做跨机器编排，不引入复杂的外部 orchestrator，仅在单节点内部做好 GPU 资源划分、简单负载均衡和推理协议。

---

## 1. 场景与设计目标

### 1.1 场景假设

- 单节点，4 张 A100（或类似性能 GPU）。
- 运行环境为 Perlmutter 上的 conda 环境（如 `ddl`），可直接执行本仓库中的脚本。
- 主要需求：
  - 对 **早退模型** 做推理：使用训练好的 exit heads + 阈值，在不同深度提前退出；
  - 对 **对照模型（无早退）** 做推理：使用 backbone 最后一层 logits，以便比较准确率与延迟；
  - 在 MMLU 等基准上按批次评估；
  - 有一套清晰的“推理协议”和“负载策略”，可在 PPT/报告中说明。

### 1.2 设计目标

1. **保持现有逻辑不变**：不破坏当前 `runtime/autoregressive.py`、`runtime/pipeline.py` 等核心逻辑，仅通过：
   - 新增可选参数（例如 `--force-final-exit`）；
   - 新增辅助脚本/说明。
2. **简单但可扩展**：
   - 单节点内部，以“多进程 + 每进程绑定 1 张 GPU”的方式扩展吞吐量；
   - 负载均衡尽量简化（例如 round-robin 把请求分配到不同进程）。
3. **易于解释**：
   - 有清晰的请求/响应协议（哪怕只是在 Python 脚本层）；
   - Forwarder 的职责和“forwarder→forwarder”的通信方式清楚可讲。

---

## 2. 资源规划与进程模型（含多 Segment 多副本）

### 2.1 最小原型：单进程单卡

最简单的配置是单进程 + 单 GPU：

- 使用 `runtime/autoregressive.py` / `runtime/autoregressive_batch.py` 直接在一张卡上跑完整的 backbone + exit heads；
- 优点：实现最简单，调试方便；
- 缺点：吞吐量有限，但对课程项目的实验与展示通常足够。

建议首先确保**单卡单进程**的以下路径完全跑通：

1. 早退推理（默认行为，使用 exit heads）；
2. 对照推理（使用 `--force-final-exit`）。

在此基础上再考虑多卡扩展。

### 2.2 多卡扩展：4 进程 × 4 GPU（整模型）

这是最朴素的多卡方案：每张 GPU 上都加载完整 backbone + exit heads，用于 baseline 或快速对比。

- 启动 4 个独立的 Python 进程，每个进程：
  - 设置 `CUDA_VISIBLE_DEVICES=<gpu_id>`；
  - 在本进程内加载 **完整** 的 backbone + exit heads；
  - 暴露一个“接收 prompt、返回结果”的接口（可以是 CLI、也可以是简单的 Python 函数调用）。
- 上层（可以是一个简单的“调度脚本”或 Slurm job array）按 round-robin 把推理任务分配到 4 个进程。

进程划分示例（伪代码）：

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python runtime/autoregressive_batch.py ...

# GPU 1
CUDA_VISIBLE_DEVICES=1 python runtime/autoregressive_batch.py ...

# GPU 2
CUDA_VISIBLE_DEVICES=2 python runtime/autoregressive_batch.py ...

# GPU 3
CUDA_VISIBLE_DEVICES=3 python runtime/autoregressive_batch.py ...
```

对于课程项目，通常不需要在代码里实现复杂的集中式负载均衡，简单地用 shell 或 Slurm 把不同数据分片分配到不同 GPU 即可（例如把 MMLU 按索引切成四段）。

### 2.3 多 Segment 多副本：按层段拆分模型

如果希望验证“浅层 Segments 有更多副本，深层 Segments 副本更少”的设计，可以进一步把模型按层段拆分，每个 Segment 只加载自己负责的几层：

- Segment 1：embed + 层 1–4 + forwarder@4  
- Segment 2：层 5–8 + forwarder@8  
- Segment 3：层 9–12 + forwarder@12  
- ……  
- Segment 8：层 29–32 + final head（无 forwarder）

这些划分在 `config/manifest.json` 里已经写清楚，并且 `weights/segment_*.pt`、`weights/exit_head_*.pt` 已经与之对应。

### 2.4 多副本配置建议（示例）

根据 exit 统计（例如 `training/evaluate_exit_heads.py` 在 Dolly15k/Ultrachat 上的分布），可以给一个示例性的副本分布：

- Segment 1（1–4 层）：8 个副本  
- Segment 2（5–8 层）：4 个副本  
- Segment 3（9–12 层）：2 个副本  
- Segment 4–7：每段 1 个副本  
- Segment 8：1 个副本（最终 head）

在 4×A100 上的一种简单映射方式（示例）：

- GPU0：
  - 2 × Segment‑1 副本（S1_0, S1_1）
  - 1 × Segment‑2 副本（S2_0）
- GPU1：
  - 2 × Segment‑1 副本（S1_2, S1_3）
  - 1 × Segment‑2 副本（S2_1）
- GPU2：
  - 2 × Segment‑1 副本（S1_4, S1_5）
  - 1 × Segment‑3 副本（S3_0）
- GPU3：
  - 2 × Segment‑1 副本（S1_6, S1_7）
  - 1 × Segment‑3 副本（S3_1）
  - 1 × “后段串联进程”（S4–S8 一起）

注意：上面只是例子，实际可以根据 exit 分布、显存情况调整副本数量。

每个“Segment‑k 副本”进程只加载：

- 对应的 `segment_k.pt`；
- 对应的 `exit_head_{layer}.pt`（若该段有 forwarder）；  
因此单进程 VRAM 占用会小于整模。

### 2.5 Segment 间通信（单节点简化版）

在单节点上，不引入外部 orchestrator 的前提下，可以使用 Python 多进程 + 进程间队列来连接各段：

1. 定义“中间消息”的统一结构（伪代码）：

   ```python
   msg = {
       "request_id": "<全局唯一 ID>",
       "segment_id": "segment-1",
       "hidden_state": <tensor 或 numpy 数组>,
       "generated_ids": [...],  # 已生成的 token（若做自回归）
       "metadata": {...},       # prompt 长度、时间戳等
   }
   ```

2. 对每类 Segment 维护一个输入队列（`multiprocessing.Queue`）：

   - `queue_s1_in`：Segment‑1 的输入队列；
   - `queue_s2_in`：Segment‑2 的输入队列；
   - ……；
   - `queue_results`：最终结果队列。

3. 每个 Segment‑k 进程循环执行：

   ```python
   while True:
       msg = queue_sk_in.get()
       # 1）取出 hidden_state，送入 segment_k.pt
       h_out = segment_k(msg["hidden_state"])
       # 2）forwarder 决策
       decision = forwarder_k.evaluate(h_out)  # 若该段有 forwarder

       if decision.exit_now:
           # 早退，写入结果队列
           queue_results.put({
               "request_id": msg["request_id"],
               "generated_ids": msg["generated_ids"] + [sampled_token],
               "exit_segment": f"segment-{k}",
               "metadata": msg["metadata"],
           })
       else:
           # 继续向下，把 forwarded_state 写入下一段队列
           msg_next = {
               **msg,
               "segment_id": decision.next_segment_id,
               "hidden_state": decision.forwarded_state,
           }
           queue_s(k+1)_in.put(msg_next)
   ```

4. Segment‑1 的“入口”由主进程或简单 router 负责：

   - 接收外部 prompt；
   - 编码为初始 hidden_state（或 input_ids）；
   - 创建 `request_id` 和初始 `msg`；
   - 放入 `queue_s1_in`。

这样实现的优点：

- 完全在单节点内部，不需要外部 pub/sub；  
- Forwarder→Forwarder 的通信体现在“哪个 Segment 队列收到消息”；  
- 可以在 S1/S2 段配置更多进程副本，以吸收大部分 easy 请求。

### 2.6 初始 batch + 低延迟通道的具体建议

在上述多进程 + 队列框架下，可以进一步结合“初始 batch 输入”和“低延迟传输”：

1. **Segment‑1 端显式 batch**  
   - 主进程从请求源（例如 MMLU jsonl）中一次取出 B 条样本（如 32 或 64 条），打包成一个 batch：  
     ```python
     prompts = [sample["prompt"] for sample in batch]
     ```  
   - 将这 B 条样本统一编码为 `[B, T]` 的 `input_ids`，作为一个批输入送入某个 S1 worker 进程；  
   - S1 内部：
     - 在 GPU 上一次性跑 embed + 层 1–4；  
     - 对 batch 内每条样本分别跑 forwarder@4，得到每条样本的 `exit_now` / `forwarded_state`；  
     - 对于 early exit 样本：直接在 S1 内部写结果 `msg` 到 `queue_results`；  
     - 对于需继续的样本：从 batch 输出中取出对应的 hidden_state（例如 `hidden[B_i, T_i, d]`），打包到单条 `msg` 并写入 `queue_s2_in`。

2. **后续 Segment 的 micro‑batch 策略**  
   - S2/S3 等 worker 从各自队列中按条获取 `msg`，可以积攒到一定数目（例如 8–16 条）再组装成一个 batch：  
     ```python
     msgs = [queue_s2_in.get() for _ in range(N)]
     hidden_batch = torch.stack([m["hidden_state"] for m in msgs], dim=0)
     ```  
   - 然后在 GPU 上一次性跑 segment_k + forwarder_k；  
   - 对 batch 中每条消息独立决定 exit / forward，并分别写回结果或下游段队列。  
   - 由于样本经过早退筛选，越往后段 batch size 通常越小，这符合早退设计的预期。

3. **低延迟传输的实现层面**  
   - **基础实现（推荐起步方式）**：  
     - 在 S1/S2 等进程间传输时，先把 hidden_state 从 GPU 拷贝到 CPU：  
       ```python
       cpu_state = hidden_state.detach().cpu()  # 可选: .to(torch.bfloat16)
       ```  
     - 在下一段进程中再搬回 GPU：  
       ```python
       hidden_state = torch.tensor(cpu_state, device="cuda", dtype=...)
       ```  
     - 配合 pinned memory（`pin_memory=True`）和 `non_blocking=True` 可以减少同步开销；  
     - 将 hidden_state 存为 `bf16`/`fp16` 以减小数据量。  
     - 这种方式实现简单，非常适合课程项目的原型阶段。
   - **利用 NVLink 的 GPU P2P（可选优化）**：  
     - 若 S1 和 S2 位于同一 GPU 或同一进程，可以直接在 GPU 上做 P2P：  
       ```python
       h_out = h_out.to("cuda:next_id", non_blocking=True)
       ```  
     - 底层通过 NVLink 进行 peer‑to‑peer 传输，无需显式管理 RDMA；  
     - 更复杂的跨进程 GPU P2P（CUDA IPC / torch.distributed send/recv）实现成本较高，不建议在课程项目中作为必须项，可以在文档中提为“潜在优化方向”。

总体策略是：**先用 GPU→CPU→GPU 的队列传输跑通整个 pipeline 和统计，再根据时间和需求考虑是否在某些 Segment 对之间引入 P2P 优化**。

---

## 3. 推理协议（逻辑层面）

本部分定义的是“逻辑协议”，不要求你一定做 HTTP/gRPC 服务，只要在代码结构上清晰即可。

### 3.1 请求格式

一个推理请求（无论是 CLI 参数还是 Python 调用）建议包含：

- `prompt`: 字符串，用户输入或 benchmark 题目（如 MMLU 的一条题干+选项）；
- `max_new_tokens`: 生成的最大 token 数（如 64、128）；
- `temperature`: 抽样温度（如 0.8 或 1.0）；
- `threshold_profile`: 阈值配置名（例如 `"dolly_0.6"` 或 `"oasst_0.8"`）；
- `force_final_exit`: 布尔值，是否使用最终 LM head 作为对照（不做 early exit）。

实例（Python dict）：

```python
request = {
    "prompt": "Explain early-exit inference.",
    "max_new_tokens": 64,
    "temperature": 1.0,
    "threshold_profile": "dolly_0.6",
    "force_final_exit": False,
}
```

### 3.2 响应格式

建议至少包含：

- `generated_text`: 最终生成的文本；
- `generated_ids`: 生成的 token ID 列表；
- `exit_decision`: 最终 exit 头的元信息（id、layer 等）；
- `total_latency_ms` / `probe_latency_ms`：一次完整生成的耗时；
- `final_exit_forwarder_id` / `final_exit_layer_index`；
- 对于 MCQ 任务（如 MMLU）：
  - `choice_logits`: [logit_A, logit_B, logit_C, logit_D]；
  - `predicted_choice`: 'A'/'B'/'C'/'D'。

`runtime/autoregressive.py` 的 `AutoRegressiveGenerator.generate()` 已经基本输出了上面这些字段，只需要在你的“上层服务”里按需使用/丢弃即可。

---

## 4. 受控模型（无早退）与早退模型的使用方法

### 4.1 受控模型：`--force-final-exit`

我们已经在 `runtime/autoregressive.py` 中扩展了一个对照模式：

- 在 CLI 层：`parse_args()` 增加了 `--force-final-exit`；
- 在 `AutoRegressiveGenerator.__init__` 中有 `force_final_exit` 参数；
- 内部逻辑：
  - 若 `force_final_exit=True`：
    - `select_exit_depth()` 不再调用 `_evaluate_exit_from_hidden()`，而是使用 `_final_head_decision()`，直接基于 backbone 最后一层 logits 返回一个决策；
    - `generate()` 每一步也使用 `_final_head_decision()`，完全绕过所有 exit heads。

运行示例（单题 smoke）：

```bash
python runtime/autoregressive.py \
  --manifest config/manifest.json \
  --metadata artifacts/trained-exit-heads_kd_round_final/metadata.json \
  --thresholds artifacts/trained-exit-heads_kd_round_final/dolly_15k/tr_0.6/thresholds.json \
  --prompt "Explain early-exit inference." \
  --max-new-tokens 64 \
  --temperature 1.0 \
  --device cuda \
  --force-final-exit \
  --output logs/smoke_single_prompt_baseline.json
```

注意：在对照模式下，`thresholds.json` 内容实际上不影响退出行为，只保留了 `temperature_scale` 等信息；退出逻辑完全由最终 LM head 决定。

### 4.2 早退模型：默认行为（不加 `--force-final-exit`）

不传 `--force-final-exit` 时，行为仍然是原始设计：

1. 探测阶段 (`select_exit_depth`)：
   - 运行 backbone，取所有层 hidden states；
   - 调用 `_evaluate_exit_from_hidden()`，对 forwarder@4, @8, …, @28 依次评估；
   - 找到第一个通过阈值的 exit，记为 initial exit。
2. 自回归阶段 (`generate`)：
   - 每一步都重新跑 backbone；
   - 同样按照各层 exit head 顺序评估，找到第一个通过 gate 的层作为本步的退出点；
   - 用该 exit 的 logits 采样下一个 token。

在早退模式下，`thresholds.json` 的内容（logit_threshold, margin_gate, entropy_gate, temperature_scale）会直接影响每一层 forwarder 的 gate。

---

## 5. MMLU 等 benchmark 的单节点运行流程

这里给出一个完整的“单节点基准评测”配方，供实验与报告使用。

### 5.1 使用现有脚本跑早退 MMLU

脚本：`scripts/run_mmlu_benchmark.sh` + `scripts/run_mmlu_full_eval.sh`。

1. 在 `scripts/run_mmlu_benchmark.sh` 中设置阈值路径，例如使用 OASST1 tr_0.8：

   ```bash
   --thresholds "$BASE_DIR/artifacts/trained-exit-heads_kd_round_final/oasst1/tr_0.8/thresholds.json"
   ```

2. 执行：

   ```bash
   cd /pscratch/sd/$USER:0:1/$USER/extracted/exit-model
   bash scripts/run_mmlu_full_eval.sh
   ```

3. 输出：
   - `logs/benchmark/mmlu/question_*.json`：每题的详细日志；
   - `logs/benchmark/mmlu/mmlu_summary.csv`：按题的汇总；
   - 终端会打印类似：

     ```text
     Processed 1531 samples, accuracy 0.2391 (366/1531)

     Latency (total_latency_ms per sample):
       mean   = ...
       median = ...
       p95    = ...

     Final-exit distribution (per-sample, by layer):
       layer  4: ...
       ...
     ```

### 5.2 换成 Dolly15k 阈值重新测 MMLU

1. 修改 `scripts/run_mmlu_benchmark.sh` 中的 `--thresholds` 为 Dolly 版本，例如：

   ```bash
   --thresholds "$BASE_DIR/artifacts/trained-exit-heads_kd_round_final/dolly_15k/tr_0.6/thresholds.json"
   ```

2. 执行同样的命令：

   ```bash
   bash scripts/run_mmlu_full_eval.sh
   ```

3. 可以将结果保存到不同目录（手动移动 `logs/benchmark/mmlu` 下的文件），然后比较：
   - Accuracy；
   - 各层 exit 率；
   - 平均/95 分位 latency。

### 5.3 在对照模式下测 MMLU（可选）

对照模式推荐单独跑一个 batch 脚本（或拷贝 `run_mmlu_benchmark.sh`）：

```bash
PYTHON_BIN=python3
DATA_FILE="$BASE_DIR/data/raw_benchmarks/mmlu_en.jsonl"
OUTPUT_DIR="$BASE_DIR/logs/benchmark/mmlu_baseline"

"$PYTHON_BIN" "$BASE_DIR/runtime/autoregressive_batch.py" \
  --manifest "$BASE_DIR/config/manifest.json" \
  --metadata "$BASE_DIR/artifacts/trained-exit-heads_kd_round_final/metadata.json" \
  --thresholds "$BASE_DIR/artifacts/trained-exit-heads_kd_round_final/dolly_15k/tr_0.6/thresholds.json" \
  --questions "$DATA_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --device cuda \
  --no-per-step \
  --force-final-exit

python3 "$BASE_DIR/scripts/aggregate_mmlu_results.py" \
  --logs "$OUTPUT_DIR" \
  --dataset "$BASE_DIR/data/raw_benchmarks/mmlu_en.jsonl" \
  --output "$OUTPUT_DIR/mmlu_summary.csv"
```

这样就得到一个“无早退”的 baseline MMLU accuracy，可与早退配置对比。

---

## 6. 单节点推理时的“负载均衡”说明

在不引入复杂 orchestrator 的前提下，可以采用以下简单策略：

1. **多进程 + 多 GPU（整模型模式）**：
   - 每个 GPU 上跑一个独立进程，各自加载完整模型；
   - 用 shell/Slurm 手动切分数据集到不同进程：
     - 例如，把 MMLU 拆成 4 份，每个进程跑 1/4 的问题；
     - 或者在 CLI 增加 `--start-index` / `--end-index`，不同进程跑不同索引区间。
2. **多 Segment 多副本模式**：
   - 每类 Segment 拥有自己的进程集合和输入队列（例如 `queue_s1_in`, `queue_s2_in` 等）；
   - 主进程只负责把请求送入 Segment‑1 队列，并从 `queue_results` 中收集结果；
   - Segment‑1 拥有最多副本，使用队列实现“谁空谁取”的自然负载均衡；
   - Segment‑2/3 副本较少，但输入量也自然更小，因为经过早退筛选；
   - 深层 Segment 只有极少数副本，服务那部分真正“难”的请求。
3. **单进程内部的 router（纯函数调用版）**：
   - 如果不想做多进程，也可以在单进程内用 `SegmentedPipeline` + `RequestRouter` 做逻辑验证；
   - 这时“多副本”只是多个 `SegmentWorker` 实例，`RequestRouter` 用 round-robin 分配到不同 worker；
   - 早退行为和多进程版本一致，只是所有计算集中在一张 GPU 上。
4. **协议层面说明（汇报用）**：
   - 对于课程汇报，重点说明：
     - 我们把模型按层段切成多个 Segment，每个 Segment n 副本，越浅的 Segment 副本越多；
     - 请求进入 Segment‑1 池，经 forwarder 判定后，要么 early exit，要么将 hidden state 传给下一段的池；
     - 整个过程在单节点内通过进程队列或函数调用完成，不依赖外部 orchestrator；
     - 对照模式通过 `--force-final-exit` 强制使用 final LM head，以衡量早退在准确率/延迟/资源占用上的 trade-off。

---

## 7. Forwarder→Forwarder 通信在单节点内的实现方式

在单节点、不做跨进程 orchestrator 的前提下，`forwarder` 之间的通信可以完全通过函数调用和返回值完成（无需 pub/sub 或消息队列）：

1. 每个 `SegmentWorker.run(hidden_state)`：
   - 先用段内 transformer 层处理 hidden_state；
   - 再把段输出传给本段的 `Forwarder.evaluate()`；
   - 返回 `(segment_output, ForwarderDecision)`。
2. `SegmentedPipeline.run(prompt)`：
   - 把 prompt 编码成初始 hidden_state；
   - 从 `entry_segment_id`（segment-1）开始，在一个 `while` 循环中依次调用下游 `SegmentWorker`：
     - 如果 `decision.exit_now=True`，则直接结束循环，返回结果；
     - 否则把 `decision.forwarded_state` 传给下一个 segment；
   - 整个过程完全在一个进程/一个 GPU 内完成。
3. 若要把不同 segment 放到不同 GPU 上（进阶，不是必须）：
   - 可以在 Python 代码中显式调用 `.to(device="cuda:1")` 之类的方法，把 hidden_state 从一个 GPU 显式搬到另一个 GPU；
   - `SegmentedPipeline` 中的 workers 表示的是逻辑段，真正的“通信”只是 tensor 在不同 GPU 之间的 copy；
   - 对课程项目来说，这一步是“可选加分项”，不是必须实现。

---

## 8. 总结：可以直接实施的最小方案

综合上面的设计，建议按以下顺序实施：

1. **单卡单进程验证**：
   - 用早退模式跑 MMLU（已有 0.2391 的结果）；
   - 用 `--force-final-exit` 跑一次 baseline MMLU；
   - 记录两者的 accuracy / exit 分布 / latency。
2. **阈值对比实验**：
   - 使用 Dolly15k 0.6 / OASST1 0.8 等不同 thresholds.json，重复 MMLU 实验；
   - 用 `scripts/aggregate_mmlu_results.py` 的输出比较不同配置的准确率与 exit 行为。
3. **单节点多卡扩展**（可选）：
   - 在 4 张 A100 上启动 4 个进程，每个进程跑不同数据分片；
   - 在报告中说明“这是我们在单节点上扩展吞吐量的方式”。
4. **如有时间，再尝试 SegmentedPipeline**：
   - 使用 `runtime/builder.py` + `runtime/pipeline.py` 构建分段 runtime；
   - 在单 GPU 内跑一次短 context 的早退推理，展示 forwarder 分层 exit 的行为。

这样，你既能在课程报告中给出清晰的“推理协议 + 负载策略”，又不会陷入复杂的多机调度或 pub/sub 实现。  
所有步骤都基于现有代码的小幅扩展，可以直接按本指南一步步落地。 
