"""Disk-backed, resumable without-replacement sampling of token blocks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


def _write_json_atomic(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=".order-meta-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(value, file, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class CoverageSampler(Sampler[int]):
    """Yield every block once per epoch, continuing into the next epoch.

    ``issued_position`` may be ahead of ``committed_position`` while a batch is
    being fetched or trained. Only the latter is serialized in checkpoints.
    This sampler requires a single-process DataLoader (no worker prefetch).
    """

    STATE_VERSION = 1

    def __init__(
        self,
        bin_path: str | Path,
        manifest_path: str | Path,
        n_blocks: int,
        context_length: int,
        order_dir: str | Path,
        seed: int,
    ) -> None:
        if n_blocks <= 0 or n_blocks >= 2**32:
            raise ValueError("CoverageSampler needs 1 to 2^32-1 blocks")
        if seed < 0:
            raise ValueError("shuffle seed must be nonnegative")
        self.n_blocks = int(n_blocks)
        self.context_length = int(context_length)
        self.seed = int(seed)
        self.order_dir = Path(order_dir)
        self.order_dir.mkdir(parents=True, exist_ok=True)

        binary = Path(bin_path)
        stat = binary.stat()
        manifest_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        identity = {
            "path": str(binary.resolve()),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "manifest_sha256": manifest_hash,
            "n_blocks": self.n_blocks,
            "context_length": self.context_length,
        }
        self.binary_identity = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self.committed_position = 0
        self.issued_position = 0
        self._cached_epoch: int | None = None
        self._cached_order: np.memmap | None = None
        self._cached_sha256: str | None = None

    def __len__(self) -> int:
        return self.n_blocks

    def __iter__(self) -> Iterator[int]:
        position = self.issued_position
        while True:
            epoch, offset = divmod(position, self.n_blocks)
            order = self._order_for_epoch(epoch)
            index = int(order[offset])
            position += 1
            self.issued_position = position
            yield index

    def commit(self, expected_blocks: int) -> None:
        pending = self.issued_position - self.committed_position
        if pending != expected_blocks:
            raise RuntimeError(
                f"Expected {expected_blocks} trained blocks, but sampler issued {pending}; "
                "refusing to checkpoint an uncertain position"
            )
        self.committed_position = self.issued_position

    def progress(self) -> dict:
        epoch, offset = divmod(self.committed_position, self.n_blocks)
        return {
            "epoch": epoch,
            "blocks_seen_this_epoch": offset,
            "blocks_per_epoch": self.n_blocks,
            "epoch_coverage_pct": 100.0 * offset / self.n_blocks,
        }

    def state_dict(self) -> dict:
        if self.issued_position != self.committed_position:
            raise RuntimeError(
                "Cannot checkpoint while a training update is in progress"
            )
        epoch, offset = divmod(self.committed_position, self.n_blocks)
        self._order_for_epoch(epoch)
        return {
            "version": self.STATE_VERSION,
            "committed_position": self.committed_position,
            "epoch": epoch,
            "block_offset": offset,
            "shuffle_sha256": self._cached_sha256,
            "binary_identity": self.binary_identity,
            "seed": self.seed,
            "n_blocks": self.n_blocks,
            "context_length": self.context_length,
        }

    def load_state_dict(self, state: dict) -> None:
        position = state.get("committed_position")
        if (
            state.get("version") != self.STATE_VERSION
            or state.get("binary_identity") != self.binary_identity
            or state.get("seed") != self.seed
            or state.get("n_blocks") != self.n_blocks
            or state.get("context_length") != self.context_length
            or not isinstance(position, int)
            or position < 0
        ):
            raise ValueError(
                "Checkpoint sampler state does not match this training binary/configuration"
            )
        epoch, offset = divmod(position, self.n_blocks)
        if state.get("epoch") != epoch or state.get("block_offset") != offset:
            raise ValueError(
                "Checkpoint sampler epoch and block offset are inconsistent"
            )
        self._order_for_epoch(epoch)
        if state.get("shuffle_sha256") != self._cached_sha256:
            raise ValueError(
                "Checkpoint shuffle order identity does not match the saved order"
            )
        self.committed_position = position
        self.issued_position = position

    def _order_for_epoch(self, epoch: int) -> np.memmap:
        if self._cached_epoch == epoch and self._cached_order is not None:
            return self._cached_order
        base = f"{self.binary_identity[:16]}-seed{self.seed}-epoch{epoch:08d}"
        path = self.order_dir / f"{base}.u32"
        meta_path = self.order_dir / f"{base}.json"
        if not path.exists() or not meta_path.exists():
            values = np.arange(self.n_blocks, dtype=np.uint32)
            np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).shuffle(
                values
            )
            checksum = hashlib.sha256(memoryview(values)).hexdigest()
            fd, name = tempfile.mkstemp(prefix=".order-stage-", dir=self.order_dir)
            try:
                with os.fdopen(fd, "wb") as file:
                    values.tofile(file)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(name, path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
            _write_json_atomic(
                meta_path,
                {
                    "binary_identity": self.binary_identity,
                    "seed": self.seed,
                    "epoch": epoch,
                    "n_blocks": self.n_blocks,
                    "sha256": checksum,
                },
            )
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("binary_identity") != self.binary_identity
            or meta.get("seed") != self.seed
            or meta.get("epoch") != epoch
            or meta.get("n_blocks") != self.n_blocks
            or path.stat().st_size != self.n_blocks * 4
        ):
            raise ValueError(f"Invalid shuffle order metadata: {path}")
        order = np.memmap(path, mode="r", dtype=np.uint32, shape=(self.n_blocks,))
        if hashlib.sha256(memoryview(order)).hexdigest() != meta.get("sha256"):
            raise ValueError(f"Corrupt shuffle order: {path}")
        self._cached_epoch = epoch
        self._cached_order = order
        self._cached_sha256 = meta["sha256"]
        return order
