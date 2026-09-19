# Research LLM

A production-quality, end-to-end GPT-style language model pipeline for training
on large research paper corpora stored in Parquet files.

---

## Features

| Feature | Details |
|---|---|
| **Data** | Streaming Parquet ingestion (PyArrow / Polars), memory-mapped pre-tokenised binaries |
| **Preprocessing** | Unicode norm, LaTeX stripping, URL/email removal, length filtering, paragraph chunking |
| **Tokenizer** | SentencePiece BPE / Unigram *or* HuggingFace BPE — vocab trained from your corpus |
| **Model** | Decoder-only Transformer: RoPE, GQA, SwiGLU FFN, RMSNorm, weight tying |
| **Training** | Gradient accumulation, AMP (BF16 / FP16), grad clipping, cosine LR + warmup |
| **Checkpointing** | Save / load / resume; keeps last-N + best-by-val-loss |
| **Evaluation** | Perplexity on val set; sample generation during training |
| **Inference** | Temperature / top-k / nucleus / beam search; streaming output; interactive REPL |
| **Config** | YAML config, CLI overrides, model size presets (nano → xl) |
| **Logging** | Rich console, file logs, TensorBoard, optional WandB |

---

## Project Structure

```
research_llm/
├── config/
│   └── default.yaml          ← All hyperparameters
├── data/
│   ├── dataset.py            ← Parquet streaming & IterableDataset
│   └── preprocessing.py      ← Text cleaning pipeline
├── tokenizer/
│   └── tokenizer.py          ← SentencePiece / HF wrapper
├── model/
│   └── transformer.py        ← GPT architecture (RoPE, GQA, SwiGLU)
├── training/
│   ├── trainer.py            ← Training loop + evaluation
│   └── scheduler.py          ← LR schedules (cosine, linear, constant)
├── inference/
│   └── generator.py          ← TextGenerator, beam search, streaming
├── utils/
│   ├── config.py             ← Config object, YAML loading, CLI overrides
│   ├── checkpoint.py         ← Checkpoint save / load / pruning
│   ├── logging_utils.py      ← Rich logging, MetricsLogger (WandB/TB)
│   └── seed.py               ← Reproducibility + device helpers
├── scripts/
│   ├── train_tokenizer.py    ← Standalone tokenizer training script
│   ├── tokenize_corpus.py    ← Pre-tokenise corpus to binary
│   └── evaluate.py           ← Compute perplexity on a text file
└── main.py                   ← Unified entry point
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Optional: Flash Attention (requires CUDA 11.8+)
pip install flash-attn --no-build-isolation
```

### 2. Prepare your data

Your Parquet files must have at least one text column (default: `"text"`).
Put them all under a single directory:

```
data/
└── parquet/
    ├── papers_0000.parquet
    ├── papers_0001.parquet
    └── ...
```

### 3. Edit the config

```bash
cp config/default.yaml config/my_run.yaml
# Edit: data.parquet_dir, tokenizer.vocab_size, model size, training params
```

### 4. Train the tokenizer

The tokenizer is trained automatically at the start of `train` if the
`tokenizer.model_path` does not exist. Or train it separately:

```bash
python scripts/train_tokenizer.py --config config/my_run.yaml
```

### 5. (Optional) Pre-tokenise the corpus

For large corpora, pre-tokenising once and writing binary files speeds up
all subsequent training runs significantly:

```bash
python scripts/tokenize_corpus.py --config config/my_run.yaml
```

### 6. Train

```bash
python main.py train --config config/my_run.yaml

# Override settings on the command line:
python main.py train --config config/my_run.yaml \
    training.batch_size=16 \
    training.grad_accumulation_steps=4 \
    model.n_layers=24 \
    model.d_model=1024 \
    training.max_steps=200000
```

Training automatically:
- Resumes from the latest checkpoint if one exists
- Logs to `logs/` and optionally WandB / TensorBoard
- Saves checkpoints to `checkpoints/`
- Keeps the best checkpoint by validation loss at `checkpoints/best.pt`

### 7. Generate text

```bash
# Single prompt
python main.py generate \
    --config config/my_run.yaml \
    --checkpoint checkpoints/best.pt \
    --prompt "Abstract: In this paper we introduce"

# Streaming output
python main.py generate --config config/my_run.yaml \
    --checkpoint checkpoints/best.pt \
    --prompt "Introduction:" --stream

# Interactive REPL
python main.py generate --config config/my_run.yaml \
    --checkpoint checkpoints/best.pt --interactive
```

### 8. Evaluate perplexity

```bash
python scripts/evaluate.py \
    --config config/my_run.yaml \
    --checkpoint checkpoints/best.pt \
    --text data/test.txt
```

---

## Model Size Presets

Set `model.preset` in your config (explicit values override the preset):

| Preset | Layers | Heads | d_model | d_ff | Params |
|--------|--------|-------|---------|------|--------|
| nano   | 4      | 4     | 256     | 1024 | ~10M   |
| small  | 12     | 12    | 768     | 3072 | ~117M  |
| medium | 24     | 16    | 1024    | 4096 | ~345M  |
| large  | 36     | 20    | 1280    | 5120 | ~762M  |
| xl     | 48     | 25    | 1600    | 6400 | ~1.5B  |

---

## Configuration Reference

All settings live in `config/default.yaml`. Key sections:

| Section | Key setting | Default | Notes |
|---------|-------------|---------|-------|
| `data` | `parquet_dir` | `./data/parquet` | Root for Parquet files |
| `data` | `text_column` | `text` | Column name |
| `tokenizer` | `vocab_size` | 32000 | Larger = better coverage, more memory |
| `model` | `preset` | `small` | Or set n_layers/n_heads/d_model directly |
| `model` | `context_length` | 1024 | Max tokens per training example |
| `training` | `batch_size` | 8 | Per-GPU micro-batch |
| `training` | `grad_accumulation_steps` | 8 | Effective batch = batch × accum |
| `training` | `dtype` | `bfloat16` | Use `float32` for non-Ampere GPUs |
| `training` | `max_steps` | 100000 | Total optimiser steps |
| `training` | `learning_rate` | 3e-4 | Peak LR |
| `system` | `compile` | false | Set true for PyTorch 2.0+ speed boost |

---

## Architecture Details

The model is a **decoder-only Transformer** inspired by GPT-2 and LLaMA:

- **No absolute position embeddings** — position is encoded via RoPE applied inside each attention head
- **Pre-norm** with RMSNorm for training stability
- **SwiGLU** feed-forward with 3 projection matrices (gate × up → down)
- **Grouped-Query Attention** (GQA): set `n_kv_heads < n_heads` to reduce KV-cache memory
- **Weight tying**: input embedding and output projection share the same matrix
- **Flash Attention**: automatically used via `F.scaled_dot_product_attention` (PyTorch 2.0+)

---

## Multi-GPU Training

For distributed training across multiple GPUs, launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 main.py train \
    --config config/my_run.yaml \
    system.distributed=true \
    training.batch_size=4
```

> **Note**: The `system.distributed=true` flag wraps the model in
> `DistributedDataParallel`. The effective batch size is automatically
> `batch_size × grad_accumulation_steps × world_size`.

---

## License

MIT
# paper_pulse
