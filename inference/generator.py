"""
Inference / text generation for Research LLM.

Provides:
  - ``TextGenerator``: high-level interface for loading a model and generating text
  - ``generate_text``: convenience function for one-shot generation

Sampling strategies:
  - Greedy (temperature=0)
  - Temperature sampling
  - Top-k filtering
  - Nucleus (top-p) sampling
  - Repetition penalty
  - Beam search (``beam_size > 1``)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from model.transformer import GPTModel, ModelConfig
from tokenizer.tokenizer import ResearchTokenizer
from utils.logging_utils import get_logger
from utils.device import get_device, get_dtype

logger = get_logger(__name__)


class TextGenerator:
    """
    High-level text generation interface.

    Loads a trained model and tokenizer from disk, then provides a
    ``generate()`` method for interactive or batch inference.

    Args:
        checkpoint_path:  Path to a ``model.pt`` or checkpoint directory.
        tokenizer_path:   Path to tokenizer model file.
        model_config:     ``ModelConfig`` (required if not stored in checkpoint).
        device_str:       ``"mps"`` or ``"cpu"``.
        dtype_str:        ``"float32"`` / ``"float16"`` / ``"bfloat16"``.
        tokenizer_backend: ``"sentencepiece"`` or ``"hf"``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        model_config: Optional[ModelConfig] = None,
        device_str: str = "auto",
        dtype_str: str = "bfloat16",
        tokenizer_backend: str = "sentencepiece",
    ):
        self.device = get_device(device_str)
        self.dtype = get_dtype(dtype_str)
        self.use_amp = self.dtype in (torch.float16, torch.bfloat16)

        # ---- Load tokenizer ----
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = ResearchTokenizer.load(
            tokenizer_path, backend=tokenizer_backend
        )
        logger.info(f"Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")

        # ---- Load model ----
        if model_config is None:
            model_config = self._load_config_from_checkpoint(checkpoint_path)
        model_config.vocab_size = self.tokenizer.vocab_size

        logger.info(
            f"Building model: {model_config.n_layers}L / d={model_config.d_model}"
        )
        self.model = GPTModel(model_config)
        self._load_weights(checkpoint_path)

        self.model.eval()
        self.model.to(self.device)

        logger.info(
            f"TextGenerator ready: "
            f"{self.model.num_parameters():,} params on {self.device} ({self.dtype})"
        )

    # ------------------------------------------------------------------ #
    #  Main generation API                                                 #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        num_return_sequences: int = 1,
        beam_size: int = 1,
        stream: bool = False,
    ) -> List[str]:
        """
        Generate text continuations for a given prompt.

        Args:
            prompt:               Input text string.
            max_new_tokens:       Maximum tokens to add.
            temperature:          Sampling temperature (0 = greedy).
            top_k:                Top-k filter (0 = disabled).
            top_p:                Nucleus sampling threshold (1.0 = disabled).
            repetition_penalty:   > 1.0 penalises repeated tokens.
            num_return_sequences: How many independent samples to generate.
            beam_size:            Beam search width (1 = sampling).
            stream:               If True, print tokens as they are generated.

        Returns:
            List of generated text strings (prompt included).
        """
        # Tokenise prompt
        ids = self.tokenizer.encode(prompt, add_bos=True)
        if not ids:
            logger.warning("Prompt encoded to empty token list.")
            ids = [self.tokenizer.bos_token_id]

        input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)

        if beam_size > 1:
            outputs = self._beam_search(
                input_tensor,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            return [
                self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs
            ]

        # Independent sampling
        results = []
        for _ in range(num_return_sequences):
            if stream:
                output_ids = self._generate_streaming(
                    input_tensor.clone(),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            else:
                output_ids = self.model.generate(
                    input_tensor.clone(),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p < 1.0 else None,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(
                output_ids[0].tolist(), skip_special_tokens=True
            )
            results.append(text)

        return results

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Generate continuations for a list of prompts (one per prompt, no beam search).
        Prompts are processed sequentially to avoid variable-length padding issues.
        """
        return [
            self.generate(
                p,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )[0]
            for p in prompts
        ]

    # ------------------------------------------------------------------ #
    #  Perplexity evaluation                                               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        """
        Compute perplexity of ``text`` under the model.

        Returns:
            Perplexity scalar (lower is better).
        """
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        max_ctx = self.model.cfg.context_length
        total_loss = 0.0
        n_chunks = 0

        # Slide over the text in overlapping windows
        stride = max_ctx // 2
        for start in range(0, len(ids) - 1, stride):
            chunk = ids[start : start + max_ctx + 1]
            if len(chunk) < 2:
                break
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=self.device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=self.device)

            with torch.autocast(
                device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp
            ):
                _, loss = self.model(x, labels=y)

            total_loss += loss.item()
            n_chunks += 1

        avg_loss = total_loss / max(n_chunks, 1)
        return math.exp(min(avg_loss, 20))

    # ------------------------------------------------------------------ #
    #  Streaming generation                                                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _generate_streaming(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Generate tokens one at a time, printing each to stdout."""
        import sys

        model = self.model
        eos = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            ctx = input_ids[:, -model.cfg.context_length :]
            logits, _ = model(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if repetition_penalty != 1.0:
                for tok in set(input_ids[0].tolist()):
                    logits[0, tok] /= repetition_penalty

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if 0 < top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                cum_p = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                sorted_l[cum_p - F.softmax(sorted_l, dim=-1) > top_p] = float("-inf")
                logits = torch.scatter(logits, 1, sorted_i, sorted_l)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

            # Stream the decoded token
            token_str = self.tokenizer.decode(
                [next_tok.item()], skip_special_tokens=False
            )
            sys.stdout.write(token_str)
            sys.stdout.flush()

            if next_tok.item() == eos:
                break

        sys.stdout.write("\n")
        return input_ids

    # ------------------------------------------------------------------ #
    #  Beam search                                                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        beam_size: int,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
    ) -> List[List[int]]:
        """
        Simple beam search.  Returns top ``beam_size`` sequences.
        """
        model = self.model
        eos = self.tokenizer.eos_token_id
        vocab_size = model.cfg.vocab_size

        # (beam_size, seq_len)
        beams = [input_ids[0].tolist()]
        beam_scores = [0.0]
        completed: List[Tuple[float, List[int]]] = []

        for _ in range(max_new_tokens):
            all_candidates: List[Tuple[float, List[int]]] = []

            for score, seq in zip(beam_scores, beams):
                if seq[-1] == eos:
                    completed.append((score, seq))
                    continue

                ctx = torch.tensor(
                    [seq[-model.cfg.context_length :]], device=self.device
                )
                logits, _ = model(ctx)
                logits = logits[0, -1, :] / max(temperature, 1e-8)

                if repetition_penalty != 1.0:
                    for tok in set(seq):
                        logits[tok] /= repetition_penalty

                log_probs = F.log_softmax(logits, dim=-1)
                top_log_probs, top_ids = torch.topk(log_probs, beam_size)

                for log_p, tok_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                    new_score = score + log_p
                    all_candidates.append((new_score, seq + [tok_id]))

            if not all_candidates:
                break

            # Keep top beam_size candidates
            all_candidates.sort(
                key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True
            )
            all_candidates = all_candidates[:beam_size]
            beam_scores = [c[0] for c in all_candidates]
            beams = [c[1] for c in all_candidates]

        # Merge active beams into completed
        for score, seq in zip(beam_scores, beams):
            completed.append((score, seq))

        # Sort by normalised log prob
        completed.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        return [seq for _, seq in completed[:beam_size]]

    # ------------------------------------------------------------------ #
    #  Loading helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_weights(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if path.is_dir():
            model_file = path / "model.pt"
        else:
            model_file = path
        state = torch.load(model_file, map_location=self.device, weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}")
        logger.info(f"Loaded weights from {model_file}")

    def _load_config_from_checkpoint(self, checkpoint_path: str) -> ModelConfig:
        """Try to read meta.json from a checkpoint directory."""
        import json

        path = Path(checkpoint_path)
        meta_candidates = [
            path / "meta.json",
            path.parent / "meta.json",
            path.parent / "best_meta.json",
        ]
        for meta_path in meta_candidates:
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                if "model_config" in meta:
                    return ModelConfig(**meta["model_config"])
        raise ValueError(
            "Could not find model config in checkpoint. "
            "Pass model_config=ModelConfig(...) explicitly."
        )


# ------------------------------------------------------------------ #
#  Convenience wrapper                                                #
# ------------------------------------------------------------------ #


def generate_text(
    prompt: str,
    checkpoint_path: str,
    tokenizer_path: str,
    model_config: Optional[ModelConfig] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    device_str: str = "auto",
    dtype_str: str = "bfloat16",
    stream: bool = False,
) -> str:
    """
    One-shot convenience function: load model and generate text.

    Args:
        prompt:           Input text.
        checkpoint_path:  Path to trained model weights.
        tokenizer_path:   Path to tokenizer model file.
        model_config:     ModelConfig (required if not in checkpoint meta).
        **kwargs:         Sampling parameters.

    Returns:
        Generated text string.
    """
    gen = TextGenerator(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        model_config=model_config,
        device_str=device_str,
        dtype_str=dtype_str,
    )
    results = gen.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stream=stream,
    )
    return results[0] if results else ""
