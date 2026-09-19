# mini-gpt on Apple Silicon

Train a decoder-only GPT model with PyTorch on the Mac GPU through MPS. The default corpus is the 14 FineWeb EDU Parquet shards in `data/fineweb-edu/sample/10BT`.

## Setup

From the repository root, use Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.backends.mps.is_available())"
```

The last command must print `True`. Training defaults to `system.device=mps` and float32. The `nano` model and batch size 2 are conservative starting settings for Mac memory.

## Train

The first run automatically trains a SentencePiece tokenizer if `tokenizer/spm.model` does not exist. To cap that preparation work at 100,000 documents:

```bash
python main.py train --config config/default.yaml tokenizer.train_on_n_docs=100000
```

Training reads the Parquet shards directly. It writes full checkpoints to `checkpoints/checkpoint-step-*` and the best model weights to `checkpoints/best.pt`. Rerunning the same command resumes from the latest full checkpoint. You can explicitly resume a specific one with `--checkpoint checkpoints/checkpoint-step-XXXXXXXX`.

For a short validation run in a separate output directory:

```bash
python main.py train --config config/default.yaml \
  training.max_steps=10 training.warmup_steps=0 \
  training.eval_every_n_steps=10 training.eval_steps=2 \
  training.save_every_n_steps=10 training.output_dir=./checkpoints/smoke
```

## Generate text

```bash
python main.py generate --config config/default.yaml \
  --checkpoint checkpoints/best.pt \
  --prompt "Once upon a time" --max_new_tokens 100
```

## Data and performance

The Parquet loader splits files deterministically into 13 training shards and one validation shard. It starts with `data.num_workers=0`; try `data.num_workers=2` if data loading limits training speed. It assigns different files to each worker.

Pre-tokenization is optional. It writes `train.bin` and `val.bin` alongside the Parquet shards and may need roughly 20 GB for 10 billion uint16 tokens:

```bash
python main.py tokenize --config config/default.yaml
```

Check GPU activity in macOS Activity Monitor under **Window → GPU History**. The training log also prints `Device: mps` at startup. To adjust the model, batch size, learning rate, or checkpoint interval, edit `config/default.yaml` or pass `KEY=VALUE` overrides to `main.py`.

Source dataset: [HuggingFaceFW/fineweb-edu sample/10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT).
