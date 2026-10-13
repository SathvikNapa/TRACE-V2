# TRACE-V2

Successor to [TRACE](https://github.com/SathvikNapa/NaturalnessPrediction) (v1, [arXiv:2606.30543](https://arxiv.org/abs/2606.30543)) — a classifier for emotional-entrainment detection in dyadic speech. v1 achieved 93.47% accuracy / 0.999 ROC-AUC (TRACE-CE, speech+relationship) on the DyadEE dataset.

**Status: early, in progress.** See [`docs/TRACE_v2_proposal.md`](docs/TRACE_v2_proposal.md) for the full research proposal (experiment matrix E0-E9, dual-stream causal Mamba-2, cross-speaker JEPA, TRILLsson fusion, distillation). This repo currently has the first concrete step from that plan: a drop-in Mamba backbone for the existing v1 architecture, as a stepping stone toward the proposal's dual-stream causal design.

## What's here

- `train.py` / `extract.py` / `infer.py` — the v1 pipeline, inherited unchanged from [NaturalnessPrediction](https://github.com/SathvikNapa/NaturalnessPrediction), **plus** a new `--audio-backbone {llama,mamba}` flag on `train.py` (orthogonal to `--model-type`, which picks the fusion strategy). Diff against v1 is just the backbone swap — nothing else changed, so v1 results stay reproducible with `--audio-backbone llama` (the default).
- `mamba_ssm_lite.py` — a pure-PyTorch, bidirectional, chunk-parallel selective-scan encoder (`NaturalnessMambaAudioEncoder`), used when `--audio-backbone mamba`. No `mamba_ssm`/`nvcc` dependency (not available on this cluster). Chunked (not a naive per-step loop) because TRACE sequences run up to ~512 tokens, where a naive scan is a real bottleneck — measured ~5x faster than the naive loop at that length on an A100.
- `docs/TRACE_v2_proposal.md` — the research proposal this repo is working toward.

## Known gap vs. the proposal

The proposal's architecture (§4) is a **dual-stream causal** Mamba-2 — separate recurrent state per speaker, explicit synchronous cross-speaker exchange at each timestep, used as the backbone for **cross-speaker JEPA** pretraining (§5: predict the other speaker's future latent, not your own). `mamba_ssm_lite.py` today is a **bidirectional single-stream** encoder (forward+backward scan over the already-interleaved A/B sequence) — a reasonable stand-in for a first backbone comparison (proposal's E2), but not yet the dual-stream design the JEPA objective in §5 needs. That's the next piece.

## Setup

Same as v1 — see [NaturalnessPrediction's README](https://github.com/SathvikNapa/NaturalnessPrediction) for the `vox-profile-release` dependency and data/embedding layout. `pip install -r requirements.txt` plus `torch`/`transformers`/`sentence-transformers` as in v1.

## Training with the Mamba backbone

```bash
python train.py \
    --train-csv data/train.csv --test-csv data/test.csv \
    --embed-root /path/to/embeds_vec \
    --out-dir runs/mamba_crossattn \
    --model-type crossattn --audio-backbone mamba \
    --mamba-d-state 16 --mamba-chunk-size 32 \
    --batch-size 256 --num-epochs 25 --crossattn-layers 2
```

`--audio-backbone llama` (default) reproduces v1 exactly.
