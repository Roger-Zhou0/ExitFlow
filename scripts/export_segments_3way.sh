#!/usr/bin/env bash
set -euo pipefail

# 3 段导出脚本（示例）：从已有 backbone 拆出 1–8、9–16、17–32 三段的 TorchScript。
# 生成的权重会放在 artifacts/segments_3way/ 下，manifest 中的 segment_weights
# 可指向这些路径。
#
# 运行环境要求：
# - 已下载好的 backbone（例如 artifacts/trained-exit-heads_kd_round_final/backbone）
# - GPU 可用（建议在 1 张 A100 上跑导出）
# - torch + transformers 已安装
#
# 用法：
#   bash scripts/export_segments_3way.sh \
#     --backbone artifacts/trained-exit-heads_kd_round_final/backbone \
#     --outdir artifacts/segments_3way
#
# 导出结果：
#   artifacts/segments_3way/segment_1_8.pt
#   artifacts/segments_3way/segment_9_16.pt
#   artifacts/segments_3way/segment_17_32.pt

BACKBONE=""
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone)
      BACKBONE="$2"; shift 2;;
    --outdir)
      OUTDIR="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$BACKBONE" || -z "$OUTDIR" ]]; then
  echo "Usage: $0 --backbone <backbone_dir> --outdir <output_dir>" >&2
  exit 1
fi

python - <<PY
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

# 避免 SDPA / Flash Attention 在半精度下产生 dtype 冲突
# 可通过环境变量控制导出设备/精度（默认 CPU + fp32，最稳）。
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
EXPORT_DTYPE="${EXPORT_DTYPE:-float32}"

if torch.cuda.is_available():
    torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)


def build_embed_segment(model, start_idx, end_idx, rotary_emb):
    layers = model.model.layers
    subset = torch.nn.ModuleList(layers[start_idx:end_idx])
    embed = model.model.embed_tokens
    norm = model.model.norm if end_idx == len(layers) else None

    class EmbedSegment(torch.nn.Module):
        def __init__(self, embed, subset, norm, rotary_emb):
            super().__init__()
            self.embed = embed
            self.subset = subset
            self.norm = norm
            self.rotary_emb = rotary_emb

        def forward(self, input_ids):
            hidden_state = self.embed(input_ids)
            seq_len = hidden_state.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_state.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_state, position_ids)
            attn_mask = torch.zeros((1, 1, seq_len, seq_len), device=hidden_state.device, dtype=hidden_state.dtype)
            for block in self.subset:
                hidden_state = block(
                    hidden_states=hidden_state,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )[0]
            if self.norm is not None:
                hidden_state = self.norm(hidden_state)
            return hidden_state

    return EmbedSegment(embed, subset, norm, rotary_emb)


def build_hidden_segment(model, start_idx, end_idx, rotary_emb):
    layers = model.model.layers
    subset = torch.nn.ModuleList(layers[start_idx:end_idx])
    norm = model.model.norm if end_idx == len(layers) else None

    class HiddenSegment(torch.nn.Module):
        def __init__(self, subset, norm, rotary_emb):
            super().__init__()
            self.subset = subset
            self.norm = norm
            self.rotary_emb = rotary_emb

        def forward(self, hidden_state):
            seq_len = hidden_state.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_state.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_state, position_ids)
            attn_mask = torch.zeros((1, 1, seq_len, seq_len), device=hidden_state.device, dtype=hidden_state.dtype)
            for block in self.subset:
                hidden_state = block(
                    hidden_states=hidden_state,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )[0]
            if self.norm is not None:
                hidden_state = self.norm(hidden_state)
            return hidden_state

    return HiddenSegment(subset, norm, rotary_emb)


def main(backbone_dir, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(EXPORT_DEVICE)
    dtype = torch.float16 if EXPORT_DTYPE in {"fp16", "float16", "half"} else torch.float32
    print(f"[export] loading backbone from {backbone_dir} to {device} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        backbone_dir,
        dtype=dtype,
        device_map=None,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    # 定义三段：1–8, 9–16, 17–32（索引从 0 开始）
    segments = [
        ("segment_1_8.pt", 0, 8, "embed"),
        ("segment_9_16.pt", 8, 16, "hidden"),
        ("segment_17_32.pt", 16, 32, "hidden"),
    ]

    # dummy 输入
    seq_len = 16
    dummy_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)

    for filename, start, end, mode in segments:
        if mode == "embed":
            seg = build_embed_segment(model, start, end, model.model.rotary_emb).to(device)
            example_inputs = (dummy_ids,)
        else:
            dummy_hidden = torch.zeros((1, seq_len, model.config.hidden_size), dtype=model.dtype, device=device)
            seg = build_hidden_segment(model, start, end, model.model.rotary_emb).to(device)
            example_inputs = (dummy_hidden,)
        seg.eval()
        print(f"[export] tracing {filename} layers {start+1}–{end}")
        traced = torch.jit.trace(seg, example_inputs, strict=False)
        traced.save(str(outdir / filename))
        print(f"[export] saved -> {outdir/filename}")

    print("[export] done.")


if __name__ == "__main__":
    main("${BACKBONE}", "${OUTDIR}")
PY
