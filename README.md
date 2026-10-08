# LLMForge

**A decoder-only Transformer language model implemented and trained from scratch using PyTorch, with a focus on training systems, data pipeline efficiency, and reproducibility on Apple Silicon.**

LLMForge is an experimental language-model training framework designed to explore the engineering decisions behind modern autoregressive Transformers.

Rather than relying on a pretrained language model, the project implements the model architecture, tokenization, data loading, training loop, checkpoint management, and inference pipeline.

## Features

* **Transformer architecture:** Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), RMSNorm, SwiGLU, and causal self-attention.
* **Training engine:** AdamW, gradient accumulation, learning-rate scheduling, validation, and resumable checkpoints.
* **Data processing:** Streaming Parquet ingestion, SentencePiece tokenization, and optional memory-mapped token datasets.
* **Parallel tokenization:** Sequential and multiprocessing implementations with deterministic output comparison.
* **Evaluation:** Validation loss, perplexity utilities, throughput measurements, and experimental profiling.
* **Hardware:** PyTorch MPS support for Apple Silicon, with CPU execution for smaller experiments.

## Research Objectives

1. Understand Transformer architecture and autoregressive language modeling at the implementation level.
2. Compare streaming versus pre-tokenized dataset pipelines.
3. Measure tokenization and training throughput under different batching configurations.
4. Investigate memory usage and training performance on Apple Silicon.
5. Develop reproducible experiments with recoverable checkpoints.

## Dataset

The project uses a subset of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) for continued experimentation with language-model pretraining.

Dataset files, checkpoints, and trained model weights are not distributed with the repository.

## Project Status

**Research implementation / experimental.**

The repository includes model, training, data-processing, and evaluation components. Performance figures and model-quality claims should be based on documented reproducible experiments rather than inferred from implementation alone.
