"""
Tokenizer wrapper for Research LLM.

Supports:
  - SentencePiece (BPE or Unigram) - default for research corpora
  - HuggingFace Tokenizers (BPE) - optional, requires ``tokenizers`` package

Usage::

    # Train a new tokenizer
    tok = ResearchTokenizer.train(
        texts=iter_texts(),
        vocab_size=32000,
        model_path="./tokenizer/spm.model",
    )

    # Load existing
    tok = ResearchTokenizer.load("./tokenizer/spm.model")

    # Encode / decode
    ids = tok.encode("Hello world")
    text = tok.decode(ids)
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Union

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<mask>"


class ResearchTokenizer:
    """
    Unified tokenizer interface wrapping SentencePiece or HuggingFace Tokenizers.

    Provides:
      - ``encode(text) → List[int]``
      - ``decode(ids)  → str``
      - ``vocab_size`` property
      - Standard special token IDs

    Args:
        backend:      ``"sentencepiece"`` or ``"hf"``
        model_path:   Path to the trained model file.
    """

    def __init__(self, backend: str = "sentencepiece", model_path: Optional[str] = None):
        self.backend = backend
        self.model_path = model_path
        self._sp = None     # SentencePiece processor
        self._hf = None     # HuggingFace tokenizer

        if model_path:
            self._load(model_path)

    # ------------------------------------------------------------------ #
    #  Factory / training                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32_000,
        model_path: str = "./tokenizer/spm.model",
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        max_sentence_length: int = 16_384,
        num_threads: int = 16,
        max_texts: int = 2_000_000,
        backend: str = "sentencepiece",
    ) -> "ResearchTokenizer":
        """
        Train a new tokenizer on the provided text stream and save to ``model_path``.

        Args:
            texts:              Iterable of raw text strings.
            vocab_size:         Target vocabulary size.
            model_path:         Where to save the trained model.
            model_type:         ``"bpe"`` or ``"unigram"`` (SentencePiece).
            character_coverage: Fraction of characters to cover (0.9995 suits Latin scripts).
            max_sentence_length: Max chars per sentence for SentencePiece training.
            num_threads:        SentencePiece training threads.
            max_texts:          Cap on the number of documents used for training.
            backend:            ``"sentencepiece"`` or ``"hf"``.

        Returns:
            A loaded ``ResearchTokenizer`` instance.
        """
        model_path = str(model_path)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        if backend == "sentencepiece":
            cls._train_sentencepiece(
                texts=texts,
                vocab_size=vocab_size,
                model_path=model_path,
                model_type=model_type,
                character_coverage=character_coverage,
                max_sentence_length=max_sentence_length,
                num_threads=num_threads,
                max_texts=max_texts,
            )
        elif backend == "hf":
            cls._train_hf(
                texts=texts,
                vocab_size=vocab_size,
                model_path=model_path,
                max_texts=max_texts,
            )
        else:
            raise ValueError(f"Unknown backend: '{backend}'")

        tokenizer = cls(backend=backend, model_path=model_path)
        logger.info(f"Tokenizer trained and saved to {model_path}  (vocab_size={tokenizer.vocab_size})")
        return tokenizer

    @classmethod
    def load(cls, model_path: str, backend: str = "sentencepiece") -> "ResearchTokenizer":
        """Load an existing tokenizer model."""
        return cls(backend=backend, model_path=model_path)

    # ------------------------------------------------------------------ #
    #  Encode / decode                                                     #
    # ------------------------------------------------------------------ #

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Tokenise ``text`` into a list of integer token IDs.

        Args:
            text:       Input string.
            add_bos:    Prepend BOS token.
            add_eos:    Append EOS token.
            max_length: Truncate to this many tokens (after BOS/EOS).
        """
        if self.backend == "sentencepiece":
            ids = self._sp.encode(text, out_type=int)
        else:
            enc = self._hf.encode(text)
            ids = enc.ids

        if max_length is not None:
            ids = ids[:max_length]

        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to a string."""
        if self.backend == "sentencepiece":
            # SentencePiece handles special token filtering automatically
            return self._sp.decode(ids)
        else:
            return self._hf.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(t, **kwargs) for t in texts]

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        if self.backend == "sentencepiece":
            return self._sp.get_piece_size()
        return self._hf.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        if self.backend == "sentencepiece":
            return self._sp.pad_id() if self._sp.pad_id() >= 0 else 0
        return self._hf.token_to_id(PAD_TOKEN) or 0

    @property
    def bos_token_id(self) -> int:
        if self.backend == "sentencepiece":
            return self._sp.bos_id()
        return self._hf.token_to_id(BOS_TOKEN) or 1

    @property
    def eos_token_id(self) -> int:
        if self.backend == "sentencepiece":
            return self._sp.eos_id()
        return self._hf.token_to_id(EOS_TOKEN) or 2

    @property
    def unk_token_id(self) -> int:
        if self.backend == "sentencepiece":
            return self._sp.unk_id()
        return self._hf.token_to_id(UNK_TOKEN) or 3

    def __len__(self) -> int:
        return self.vocab_size

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _load(self, model_path: str) -> None:
        if self.backend == "sentencepiece":
            try:
                import sentencepiece as spm
            except ImportError:
                raise ImportError("sentencepiece not installed: pip install sentencepiece")
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(model_path)
        elif self.backend == "hf":
            try:
                from tokenizers import Tokenizer as HFTokenizer
            except ImportError:
                raise ImportError("tokenizers not installed: pip install tokenizers")
            self._hf = HFTokenizer.from_file(model_path)
        else:
            raise ValueError(f"Unknown backend: '{self.backend}'")

    # ------------------------------------------------------------------ #
    #  Training backends                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _train_sentencepiece(
        texts: Iterable[str],
        vocab_size: int,
        model_path: str,
        model_type: str,
        character_coverage: float,
        max_sentence_length: int,
        num_threads: int,
        max_texts: int,
    ) -> None:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("sentencepiece not installed: pip install sentencepiece")

        logger.info(f"Training SentencePiece tokenizer (vocab_size={vocab_size}, type={model_type}) …")

        # Write texts to a temp file (SentencePiece needs a file path)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            tmp_path = f.name
            count = 0
            for text in texts:
                if count >= max_texts:
                    break
                # Replace newlines so each line ≈ one sentence
                f.write(text.replace("\n", " ") + "\n")
                count += 1
            logger.info(f"Wrote {count:,} documents to temp file for tokenizer training.")

        # Model prefix = path without .model extension
        prefix = model_path[:-6] if model_path.endswith(".model") else model_path

        try:
            spm.SentencePieceTrainer.train(
                input=tmp_path,
                model_prefix=prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                max_sentence_length=max_sentence_length,
                num_threads=num_threads,
                pad_id=0,
                bos_id=1,
                eos_id=2,
                unk_id=3,
                # Extra special tokens
                user_defined_symbols=[MASK_TOKEN],
                # Byte fallback ensures full Unicode coverage
                byte_fallback=True,
                # Remove very rare tokens
                vocab_size_threshold=0.9999,
                input_sentence_size=max_texts,
                shuffle_input_sentence=True,
            )
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _train_hf(
        texts: Iterable[str],
        vocab_size: int,
        model_path: str,
        max_texts: int,
    ) -> None:
        try:
            from tokenizers import Tokenizer as HFTokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.processors import TemplateProcessing
        except ImportError:
            raise ImportError("tokenizers not installed: pip install tokenizers")

        logger.info(f"Training HuggingFace BPE tokenizer (vocab_size={vocab_size}) …")

        tokenizer = HFTokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = ByteLevel()

        special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN]
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=2,
        )

        def text_iterator() -> Iterator[str]:
            for i, text in enumerate(texts):
                if i >= max_texts:
                    break
                yield text

        tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

        # Add BOS/EOS post-processor
        tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[
                (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
                (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
            ],
        )

        # HF tokenizer saves as JSON
        save_path = model_path if model_path.endswith(".json") else model_path + ".json"
        tokenizer.save(save_path)
