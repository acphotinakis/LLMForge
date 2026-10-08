"""
Text preprocessing, normalization, and filtering for research paper corpora.

Handles:
  - Unicode normalization
  - LaTeX artefact removal
  - Reference/boilerplate stripping
  - Deduplication-ready fingerprinting
  - Length filtering
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterator, List, Optional


class TextPreprocessor:
    """
    Configurable text preprocessing pipeline.

    Args:
        min_length:         Minimum character length to accept a document.
        max_length:         Maximum character length.  Longer documents are
                            split at paragraph boundaries (see ``chunk``).
        normalize_unicode:  Apply NFKC Unicode normalization.
        fix_whitespace:     Collapse multiple whitespace / fix line endings.
        remove_latex:       Strip common LaTeX commands (\\cite, \\ref, …).
        remove_urls:        Replace URLs with a placeholder token.
        remove_emails:      Replace email addresses.
        lowercase:          Lowercase all text (not recommended for LLMs).
    """

    # Compiled patterns (class-level, shared across instances)
    _RE_MULTI_SPACE = re.compile(r" {2,}")
    _RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
    _RE_TABS = re.compile(r"\t")
    _RE_URL = re.compile(
        r"https?://[^\s\)\]\}\"\'<>]+" r"|www\.[^\s\)\]\}\"\'<>]+",
        re.IGNORECASE,
    )
    _RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
    _RE_LATEX_CITE = re.compile(r"\\cite\w*\{[^}]*\}")
    _RE_LATEX_REF = re.compile(r"\\(?:ref|label|eqref)\{[^}]*\}")
    _RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?\{[^}]*\}")
    _RE_LATEX_MATH = re.compile(r"\$+[^$]*\$+")
    _RE_FIGURE_REF = re.compile(
        r"\b(?:Fig(?:ure)?|Table|Eq\.?)\s*\.?\s*\d+", re.IGNORECASE
    )
    _RE_LEADING_PUNCT = re.compile(r"^[^\w(\"\']+")
    _RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 100_000,
        normalize_unicode: bool = True,
        fix_whitespace: bool = True,
        remove_latex: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        lowercase: bool = False,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.normalize_unicode = normalize_unicode
        self.fix_whitespace = fix_whitespace
        self.remove_latex = remove_latex
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.lowercase = lowercase

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process(self, text: str) -> Optional[str]:
        """
        Process and filter a single document.

        Returns:
            Cleaned text, or ``None`` if the document should be discarded.
        """
        if not isinstance(text, str) or len(text) < self.min_length:
            return None

        text = self._clean(text)

        if len(text) < self.min_length:
            return None

        # Truncate very long documents to max_length at a paragraph boundary
        if len(text) > self.max_length:
            text = self._truncate_at_paragraph(text, self.max_length)

        return text if text else None

    def chunk(self, text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """
        Split a document into non-overlapping (or overlapping) chunks of
        approximately ``chunk_size`` characters, splitting at paragraph
        boundaries where possible.

        Args:
            text:       Input text (should already be cleaned).
            chunk_size: Target characters per chunk.
            overlap:    Number of characters to overlap between chunks.

        Returns:
            List of text chunks.
        """
        if len(text) <= chunk_size:
            return [text]

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > chunk_size and current:
                chunks.append("\n\n".join(current))
                if overlap > 0:
                    # Carry over last few paragraphs for context
                    keep = []
                    keep_len = 0
                    for p in reversed(current):
                        if keep_len + len(p) < overlap:
                            keep.insert(0, p)
                            keep_len += len(p)
                        else:
                            break
                    current = keep
                    current_len = keep_len
                else:
                    current = []
                    current_len = 0
            current.append(para)
            current_len += len(para)

        if current:
            chunks.append("\n\n".join(current))

        return [c for c in chunks if len(c) >= self.min_length]

    def iter_process(self, texts: Iterator[str]) -> Iterator[str]:
        """Generator that processes an iterable of raw texts."""
        for text in texts:
            result = self.process(text)
            if result is not None:
                yield result

    # ------------------------------------------------------------------ #
    #  Internal cleaning steps                                             #
    # ------------------------------------------------------------------ #

    def _clean(self, text: str) -> str:
        # Strip control characters
        text = self._RE_CONTROL.sub("", text)

        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # LaTeX artefacts
        if self.remove_latex:
            text = self._RE_LATEX_CITE.sub("[REF]", text)
            text = self._RE_LATEX_REF.sub("", text)
            text = self._RE_LATEX_MATH.sub(" [MATH] ", text)
            text = self._RE_LATEX_CMD.sub("", text)

        # URLs and emails
        if self.remove_urls:
            text = self._RE_URL.sub("[URL]", text)
        if self.remove_emails:
            text = self._RE_EMAIL.sub("[EMAIL]", text)

        # Whitespace normalization
        if self.fix_whitespace:
            text = self._RE_TABS.sub(" ", text)
            # Normalize Windows / Mac line endings
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = self._RE_MULTI_NEWLINE.sub("\n\n", text)
            text = self._RE_MULTI_SPACE.sub(" ", text)
            # Strip trailing whitespace from each line
            text = "\n".join(line.rstrip() for line in text.split("\n"))
            text = text.strip()

        if self.lowercase:
            text = text.lower()

        return text

    @staticmethod
    def _truncate_at_paragraph(text: str, max_len: int) -> str:
        """Truncate text to max_len at the nearest preceding paragraph break."""
        truncated = text[:max_len]
        last_break = truncated.rfind("\n\n")
        if last_break > max_len // 2:
            return truncated[:last_break].strip()
        return truncated.strip()
