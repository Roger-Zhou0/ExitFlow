# Historical Proposal: ExitFlow on Llama 3.1 8B

> This document records the original project plan. Numerical goals below are
> targets, not measured outcomes. See the README's **Results and claim status**
> section for the recoverable experimental evidence.

## 1. Project Title

**ExitFlow: Segmented Inference with Forwarder-Controlled Early Exits in Llama 3.1 8B Instruct**

---

## 2. Background and Motivation

Large Language Models (LLMs) such as Llama-3 8B Instruct deliver strong performance but suffer from high inference latency and compute costs. A promising optimization is **early-exit inference**, where intermediate exit heads allow the model to terminate generation earlier if confidence thresholds are met.

Instead of deploying a centralized gateway, this project introduces **in-segment forwarders**: lightweight components positioned at the tail of each segment. Each forwarder decides whether to exit immediately (returning a token to the user) or to forward activations downstream for further computation. This design eliminates the bandwidth bottlenecks of central gateways and avoids unnecessary microservice complexity, while still enabling measurable performance gains in short-context inference scenarios.

---

## 3. Scope (Course Project Focus)

This project is scoped for a **semester-length course project**, focusing on research and implementation feasibility, not full productionization. Specifically:

- **Base Model**: Llama-3 8B Instruct (frozen backbone).
- **Exit Heads**: Added at every **4-layer boundary (layers 4, 8, 12, 16, 20, 24, 28)**; trained with Knowledge Distillation (KD).
- **Segments**:
  - Segment 1 = embed + layers 1–4 + forwarder@4
  - Segment 2 = layers 5–8 + forwarder@8
  - Segment 3 = layers 9–12 + forwarder@12
  - Segment 4 = layers 13–16 + forwarder@16
  - Segment 5 = layers 17–20 + forwarder@20
  - Segment 6 = layers 21–24 + forwarder@24
  - Segment 7 = layers 25–28 + forwarder@28
  - Segment 8 = layers 29–32 + final head (always exit)
- **Forwarders**: In-process decision units, no containerization, no microservices.
- **Router**: A lightweight balancer only for assigning requests into Segment-1 worker pool; not involved in exit logic.
- **KV Cache**: Disabled, so each step recomputes prefix (limiting acceleration for long contexts, but acceptable for course scope).
- **Evaluation**: Accuracy vs latency tradeoffs on benchmarks (IFEval, AlpacaEval2, GSM8K, MMLU subset).

---

## 4. Objectives

1. **Attach and train exit heads** at layers 4, 8, 12, 16, 20, 24, and 28 using KD against the final logits.
2. **Implement forwarders** at the end of each segment to make exit/forward decisions based on calibrated thresholds (temperature scaling + margin + entropy dual-gate).
3. **Build a segmented inference runtime**:
   - Router → Segment1 pool → forwarder@4 → … → forwarder@28 → Segment8 final head.
   - All forwarders operate in-process with their segment worker.
4. **Measure benefits and tradeoffs**:
   - Compare accuracy drop (<1%), compliance drop (<0.5%), and latency reduction (>35% at ≤256 tokens with finer exit granularity).
   - Document diminishing returns for longer contexts (since KV is disabled).

---

## 5. Methodology

- **Training**:
  - Freeze backbone, train only forwarder RMSNorms.
  - KD: temperature τ=2.0, weighted KL loss with exit-specific weights increasing from α₄=0.4 up to α₂₈=1.0.
  - 1.5B training tokens, bf16 precision, 8×A100 GPUs.
- **Threshold Calibration**:
  - Static thresholds per forwarder: `t_4=1.65`, `t_8=1.60`, `t_12=1.55`, `t_16=1.50`, `t_20=1.45`, `t_24=1.40`, `t_28=1.35`; margin/entropy dual-gates with fixed values.
- **System Design**:
  - Segment workers pass full activations in-process (threads) or via CUDA IPC/SHM (multi-process, same host).
  - Forwarders execute locally; only exit decisions or sampled tokens go back to Router.
- **Evaluation**:
  - Benchmarks: IFEval, AlpacaEval2, GSM8K, MMLU (English subset).
  - Metrics: accuracy, compliance, average exit depth, P95 latency.

---

## 6. Expected Outcomes

- **Deliverables**:
  - Exit head weights (`forwarder@4` … `forwarder@28`), segmented model weights, `manifest.json`, `thresholds.json`, launch scripts.
  - Report documenting methodology, evaluation, and analysis of latency/accuracy tradeoffs.
- **Expected Results**:
  - Short contexts: ≥40% reduction in average depth, ≥30% latency improvement at P95 thanks to denser exit coverage.
  - Accuracy/compliance degradation within defined tolerance.
  - Insight into limitations of early-exit without KV cache.

---

## 7. Timeline (10–12 weeks)

- **Week 1–2**: Data preparation, attach exit heads, KD training setup.
- **Week 3–5**: Train exit heads, run calibration, export segment weights.
- **Week 6–7**: Implement segmented runtime with in-process forwarders.
- **Week 8–9**: Integrate router + worker pools, test short-context inference.
- **Week 10–11**: Benchmark on evaluation datasets, collect metrics.
- **Week 12**: Write final report and present findings.

---

## 8. Course Value

This project is well-scoped for a course:

- It combines **ML training (KD, calibration)** with **systems design (segmentation, runtime orchestration)**.
- It exposes tradeoffs between **algorithmic optimization (early exit)** and **systems constraints (no KV cache)**.
- It yields concrete artifacts (weights, manifest, benchmarks, report) for demonstration and grading.

---

**In summary:** This proposal defines a practical, challenging, but tractable **course project** that blends ML and systems perspectives by implementing segmented inference with in-segment forwarders for early exits in Llama-3 8B.

---

## 9. Benchmark Evaluation Recipes

### 9.1 MMLU (English subset)

We run the exported early-exit runtime over the 1,531 prompts in `data/raw_benchmarks/mmlu_en.jsonl`, log per-question metrics, and aggregate accuracy/latency summaries.

1. **Full evaluation (inference + aggregation)**

   ```bash
   cd /pscratch/sd/x/xz446/extracted/exit-model
   bash scripts/run_mmlu_full_eval.sh
   ```

   - `scripts/run_mmlu_benchmark.sh` iterates through the dataset, calling `runtime/autoregressive_batch.py` to generate JSON logs under `logs/benchmark/mmlu/question_XXXX.json`.
   - After inference completes, the script automatically executes `scripts/aggregate_mmlu_results.py` to summarize accuracy, per-choice logits, exit layer, and latency into `logs/benchmark/mmlu/mmlu_summary.csv`.

2. **Aggregating existing logs only**

   If inference has already been run, aggregate the stored logs directly:

   ```bash
   cd /pscratch/sd/x/xz446/extracted/exit-model
   bash scripts/aggregate_existing_mmlu_logs.sh \
     [LOG_DIR (default: logs/benchmark/mmlu)] \
     [DATASET (default: data/raw_benchmarks/mmlu_en.jsonl)] \
     [OUTPUT (default: logs/benchmark/mmlu/mmlu_summary.csv)]
   ```

   This script is a thin wrapper over `aggregate_mmlu_results.py` and prints overall accuracy, e.g.:

   ```
   Processed 1531 samples, accuracy 0.2391 (366/1531)
   ```

3. **How accuracy is computed**

   - During generation, each step stores a `choice_logits` dictionary over {A,B,C,D}. After decoding, `runtime/autoregressive.py` selects the step whose exit metadata matches the final exit head (or fallbacks to the earliest step) and writes a top-level `choice_logits` list ordered `[A, B, C, D]` plus `predicted_choice`.
   - `aggregate_mmlu_results.py` reads each `question_*.json`, extracts the top-level `choice_logits` if present (otherwise step-0), converts it to a dict, chooses the label with the highest logit, and compares it to the gold answer from `data/raw_benchmarks/mmlu_en.jsonl`.
   - Accuracy is simply the fraction of questions where `pred == gold`, so the reported number is directly comparable with mainstream MMLU evaluations.

4. **Artifacts to inspect**

   - Raw logs: `logs/benchmark/mmlu/question_XXXX.json` (contains per-step tokens, exit gates, latency, choice logits).
   - Aggregated CSV: `logs/benchmark/mmlu/mmlu_summary.csv` (one row per question with prompt, prediction, logits, exit layer, latency, and correctness).
   - Console summary: accuracy and counts printed by the aggregation script for quick verification.
