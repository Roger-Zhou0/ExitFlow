# ExitFlow Quad‑A100 早退推理金字塔（单节点多段多副本）实施说明

目标：在单节点 4×A100 上把模型分成 8 个 segment，浅段（Segment‑1）多副本分布在 GPU0/1，Segment‑2 副本在 GPU2，Segment‑3+后续段在 GPU3，通过 `SegmentedQueueExecutor` 做竞争式拉取，实现早退推理的吞吐扩展。

> 当前环境没有 4 张 GPU，未做本地实机跑通；此文档给出配置和命令，待拿到生产节点直接执行。

## 1. 关键改动与配置

- `config/manifest.json` 的 `runtime` 段已配置好金字塔设备映射：  
  - Segment‑1：6 副本，分布在 `cuda:0/1`（各 3 个）；  
  - Segment‑2：3 副本在 `cuda:2`；  
  - Segment‑3：3 副本在 `cuda:3`；  
  - Segment‑4~8：各 1 副本在 `cuda:3`；  
  - `queue_capacity: 128`；`router` 字段为 build_segmented_pipeline 的占位（不影响 queue executor）。
- 依赖的权重：`weights/segment_*.pt`、`weights/exit_head_*.pt`、`weights/forwarder_*.pt`、`weights/final_head.pt` 已在 manifest 中引用；`thresholds.json` 需指向对应的阈值文件（例如 Dolly15k 0.6 或 OASST1 0.8）。
- 运行入口使用 `runtime/autoregressive_queue.py`，它会读取 manifest 的 `segment_pools` 自动把各段分配到指定 GPU，不需要额外手工分配。

## 2. 运行前准备

```bash
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ddl
cd /pscratch/sd/x/xz446/extracted/exit-model

# 可选：确保看到 4 张 A100
nvidia-smi
```

检查输入文件（任选其一）：
- 简单多行文本：`experiments/queue_demo_questions.txt`（每行一个 prompt）。  
- 或 `.jsonl` 含 `prompt` / `question` + `choices/options` 字段的文件（同 `runtime/autoregressive_batch.py` 的解析规则）。

## 3. 单节点 4×A100 早退推理命令（queue executor）

将 `<THRESHOLDS>` 替换为实际阈值路径，例如 `artifacts/trained-exit-heads_kd_round_final/dolly_15k/tr_0.6/thresholds.json`。

```bash
BASE_DIR=/pscratch/sd/x/xz446/extracted/exit-model
cd "$BASE_DIR"
PY=python  # 或指定绝对路径

PROMPTS=experiments/queue_demo_questions.txt  # 或任意自定义问题文件

$PY runtime/autoregressive_queue.py \
  --manifest config/manifest.json \
  --thresholds "$BASE_DIR/<THRESHOLDS>" \
  --questions "$PROMPTS" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --queue-capacity 128 \
  --show-steps
```

执行逻辑说明：
- `build_queue_executor()` 读取 manifest 的 `segment_pools`，按 GPU 列表创建各段的 `SegmentWorker` 副本；`SegmentedQueueExecutor` 为每段开线程和队列，采用“谁空谁取”的轮询。  
- 每个 token step 会提交一个 `WorkItem` 到 Segment‑1 队列；浅段副本（GPU0/1）先处理，未早退的样本被 forwarder 送往下一段的队列（Segment‑2 在 GPU2，Segment‑3+ 在 GPU3）。  
- `ForwarderDecision` 同时携带 logits 和 `forwarded_state`，满足 gate 则 `exit_now` 并返回 logits；否则把隐藏状态推给下游段。  
- `AutoRegressiveTracker` 在 CPU 侧记录每步的 token 和 exit 段序列，达到 EOS 或 `max_new_tokens` 后结束。

## 4. 架构与参数择取理由

- **浅段多副本**（6× Segment‑1）吸收大量易样本，充分利用 GPU0/1；  
- **中段适中**（3× Segment‑2 @ GPU2）承接被筛选后的请求；  
- **深段收敛**（3× Segment‑3 + 后续各 1 副本 @ GPU3）处理少量难样本；  
- `queue_capacity=128` 足够覆盖多副本竞争；需要更大 batch 可调高但会占用更多内存；  
- `router.segment_1_pool_size=2` 仅用于 `build_segmented_pipeline` 的 fallback，queue executor 不依赖该字段。

## 5. 可能的后续扩展（非本次必须）

- 在 `runtime/autoregressive_queue.py` 中增加初始大 batch + 后续 micro-batch 聚合（文档 2.6 的优化）；  
- 为 `runtime/autoregressive_batch.py` 添加 `--num-shards/--shard-index`，便于无需手工 `split` 文件时做多进程数据并行；  
- 将 `SegmentedPipeline._prepare_hidden_state` 改为真实 embedding/hidden 输入，便于单进程调试和测试。

## 6. 测试说明（以及当前“占位权重”限制）

- 当前环境缺少 4×A100，未做实机验证；生产节点可直接使用上述命令。  
- 现有仓库的 `weights/segment_*.pt` 是占位的 2.6KB stub，不是真正导出的 4 层分段模型；因此用 queue executor 跑出来的文本不可用，只能验证流程。  
- **不需要重训 exit heads**：`artifacts/trained-exit-heads_kd_round_final/exit_heads/forwarder@*.pt` 已经是训练好的头。  
- **需要做的是真实分段导出**（用现有 backbone 导出 8 段或 3 段 TorchScript）：  
  1. 拆 backbone 为若干层段（例如 1–8、9–16、17–32 或每 4 层一段），导出为 `segment_X.pt`。  
  2. 在 manifest 的 `model.segments[*].artifacts.segment_weights` 中填入导出路径；保持 forwarder 的 `exit_head_weights` 指向已训练好的 exit 头。  
  3. 保持 `runtime.segment_pools` 的 GPU 映射，再跑 `runtime/autoregressive_queue.py` 即可获得真实早退输出。  
- 若要做最小烟囱测试，可在单 GPU 上把 `segment_pools` 临时改成全走 `cuda:0`，并用导出的 segment 权重验证逻辑流；确认后改回四卡映射。 

## 7.（更新）使用原生 HF 模型分段，无需 TorchScript 导出

由于 TorchScript 导出在 RoPE/SDPA 上不断踩坑，我们改用“原生分段”方案：runtime 直接加载 HF 模型并按层切分，不依赖 `weights/segment_*.pt`。这仍然是单节点多段多副本的早退流水线，只是绕过了导出。

- 配置：`config/manifest.json` 已增加  
  ```json
  "runtime": {
    "native_backbone_dir": "artifacts/trained-exit-heads_kd_round_final/backbone",
    "native_dtype": "float16",
    ...
  }
  ```  
  只要目录存在，runtime 会忽略 `segment_weights`，直接把 backbone 切成段并分配到 `segment_pools`。
- 不需要再跑导出脚本即可测试金字塔早退；`scripts/export_segments_3way.sh` 仅在想要独立导出 TorchScript 时备用。

### 7.1 运行命令（四卡）

```bash
python runtime/autoregressive_queue.py \
  --manifest config/manifest.json \
  --thresholds artifacts/trained-exit-heads_kd_round_final/dolly_15k/tr_0.6/thresholds.json \
  --questions experiments/queue_demo_questions.txt \
  --max-new-tokens 64 \
  --temperature 0.8 \
    --queue-capacity 128 \
    --show-steps
```

注意：
- `runtime.segment_pools` 依然决定段与 GPU 的映射（浅段多副本在 GPU0/1，中段 GPU2，深段 GPU3）。  
- `forwarder.exit_head_weights` 继续使用训练好的 `artifacts/trained-exit-heads_kd_round_final/exit_heads/forwarder@*.pt`。  
- 原生分段用的是 PyTorch 模型切片，输出可直接用于早退评估；不再受 TorchScript 导出报错影响。 

### 7.2 调试日志

- 若想在运行时看到每个段的设备/数据类型/序列长度，可在命令前设置：`SEGMENT_DEBUG=1`。  
  例如：  
  ```bash
  SEGMENT_DEBUG=1 python runtime/autoregressive_queue.py ...
  ```
  只会在每个 segment 第一次运行时打印一次，帮助确认“金字塔”映射是否生效。 
- 运行结束会尝试采集 GPU 利用率（依赖 `pynvml`，缺失则跳过），打印每张卡的 util/mem_used，便于观察负载分布。
- 若要高频记录 GPU 利用率以便后续画图：设置环境变量 `GPU_LOG_PATH=/path/to/gpu_log.csv`（可选 `GPU_SAMPLE_INTERVAL=1.0` 秒），运行结束会写入 CSV，字段包括 ts/gpu/gpu_util/mem_util/mem_used_mb/mem_total_mb。 
- 日志目录现在按时间戳分段：`logs/HIT_inference/<timestamp>/`，会记录本次命令（commands.md）、运行信息（run.log）、GPU 采样（gpu_log.csv）。
- 提供绘图脚本：`python scripts/plot_hit_inference.py --run-dir logs/HIT_inference/<timestamp>`（默认读取最新 run），生成 `gpu_util.png` / `gpu_mem.png`。加 `--show` 可弹出窗口。
