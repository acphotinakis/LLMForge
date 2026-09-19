#!/usr/bin/env python3
"""Audit the Parquet corpus, active split, text limits, and tokenizer provenance.

Reads Parquet metadata exactly and samples text from randomly selected row groups.
The tokenizer provenance scan follows the current training code's file order and
document preprocessing; it is a reconstruction, not a saved training manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.compute as pc
import pyarrow.parquet as pq

from data.preprocessing import TextPreprocessor
from utils.config import load_config

SAMPLE_COLUMNS = (
    "text",
    "id",
    "url",
    "language",
    "language_score",
    "token_count",
    "score",
    "int_score",
)
LENGTH_BINS = (
    ("<100", 0, 100),
    ("100–499", 100, 500),
    ("500–999", 500, 1000),
    ("1k–4.9k", 1000, 5000),
    ("5k–9.9k", 5000, 10000),
    ("10k–49.9k", 10000, 50000),
    ("50k–99.9k", 50000, 100000),
    ("100k+", 100000, math.inf),
)


def split_files(
    files: list[Path], val_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Match build_dataloaders' Parquet split exactly."""
    if not 0 <= val_fraction < 1:
        raise ValueError("data.val_split must be in [0, 1)")
    ordered = sorted(files)
    random.Random(seed).shuffle(ordered)
    n_val = max(1, int(len(ordered) * val_fraction))
    if n_val >= len(ordered):
        raise ValueError("The split leaves no training Parquet files")
    return ordered[n_val:], ordered[:n_val]


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sample_rows(
    pf: pq.ParquetFile, wanted: int, groups: int, seed: int
) -> list[tuple[int, int, dict]]:
    """Cluster sample rows across the file without decoding the whole text column."""
    if pf.num_row_groups == 0:
        return []
    rng = random.Random(seed)
    selected = sorted(
        rng.sample(range(pf.num_row_groups), min(groups, pf.num_row_groups))
    )
    columns = [c for c in SAMPLE_COLUMNS if c in pf.schema_arrow.names]
    base, extra = divmod(wanted, len(selected))
    result = []
    for position, group in enumerate(selected):
        count = base + (position < extra)
        if count == 0:
            continue
        table = pf.read_row_group(group, columns=columns)
        for row in rng.sample(range(table.num_rows), min(count, table.num_rows)):
            result.append((group, row, {c: table[c][row].as_py() for c in columns}))
    return result


def column_sum(pf: pq.ParquetFile, name: str) -> int | None:
    if name not in pf.schema_arrow.names:
        return None
    total = 0
    for batch in pf.iter_batches(batch_size=65536, columns=[name]):
        value = pc.sum(batch.column(0)).as_py()
        total += int(value or 0)
    return total


def null_count_from_metadata(pf: pq.ParquetFile, name: str) -> int | None:
    if name not in pf.schema_arrow.names:
        return None
    index = pf.schema_arrow.get_field_index(name)
    total = 0
    for group in range(pf.num_row_groups):
        stats = pf.metadata.row_group(group).column(index).statistics
        if stats is None or stats.null_count is None:
            return None
        total += stats.null_count
    return total


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def percentile(values: list[int | float], fraction: float) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def tokenizer_limit(model_path: Path, configured: int) -> tuple[int, dict]:
    """Read the saved SentencePiece training limit when possible."""
    info = {
        "exists": model_path.exists(),
        "path": str(model_path),
        "configured_docs": configured,
        "source": "current configuration",
    }
    if not model_path.exists():
        return configured, info
    try:
        from sentencepiece import sentencepiece_model_pb2

        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(model_path.read_bytes())
        spec = proto.trainer_spec
        info.update(
            {
                "source": "saved SentencePiece model",
                "vocab_size": spec.vocab_size,
                "model_type": spec.model_type,
                "character_coverage": spec.character_coverage,
                "max_sentence_length_bytes": spec.max_sentence_length,
                "byte_fallback": spec.byte_fallback,
                "saved_input_sentence_size": spec.input_sentence_size,
            }
        )
        return int(spec.input_sentence_size or configured), info
    except (ImportError, ValueError) as exc:
        info["warning"] = f"Could not inspect model metadata: {exc}"
        return configured, info


def scan_tokenizer_provenance(
    files: list[Path], cfg, doc_limit: int
) -> tuple[list[dict], int]:
    """Recreate the stream fed to ResearchTokenizer.train before SentencePiece filters it."""
    ordered = sorted(files)
    random.Random(cfg.data.shuffle_seed).shuffle(ordered)
    pre = TextPreprocessor(
        min_length=cfg.data.min_text_length, max_length=cfg.data.max_text_length
    )
    counts = Counter()
    long_lines = Counter()
    accepted = 0
    sentence_limit = cfg.tokenizer.max_sentence_length
    for path in ordered:
        if accepted >= doc_limit:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024, columns=[cfg.data.text_column]):
            for value in batch.column(0).to_pylist():
                if not value:
                    continue
                clean = pre.process(value)
                if clean is None:
                    continue
                counts[path.name] += 1
                accepted += 1
                # The trainer writes each document as one line.
                if len(clean.replace("\n", " ").encode("utf-8")) > sentence_limit:
                    long_lines[path.name] += 1
                if accepted >= doc_limit:
                    break
            if accepted >= doc_limit:
                break
    return (
        [
            {
                "shard": path.name,
                "candidate_documents": counts[path.name],
                "candidate_lines_over_byte_limit": long_lines[path.name],
            }
            for path in ordered
        ],
        accepted,
    )


def bars(
    title: str, pairs: list[tuple[str, int | float]], color: str = "#5277bb"
) -> str:
    """Small self-contained SVG bar chart with an accessible title."""
    width, row_height, label_width = 840, 25, 230
    height = max(65, 45 + row_height * len(pairs))
    peak = max((float(v) for _, v in pairs), default=0) or 1
    out = [
        f'<svg role="img" aria-label="{html.escape(title)}" viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(title)}</title>",
    ]
    for i, (label, value) in enumerate(pairs):
        y = 22 + i * row_height
        bar_width = int((width - label_width - 125) * float(value) / peak)
        out.append(
            f'<text x="4" y="{y + 13}" class="axis">{html.escape(str(label))}</text>'
        )
        out.append(
            f'<rect x="{label_width}" y="{y}" width="{bar_width}" height="16" fill="{color}"/>'
        )
        out.append(
            f'<text x="{label_width + bar_width + 7}" y="{y + 13}" class="axis">{value:,.0f}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def html_table(rows: list[dict], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def render_report(
    summary: dict,
    inventory: list[dict],
    samples: list[dict],
    split: list[dict],
    provenance: list[dict],
    duplicates: list[dict],
) -> str:
    cfg = summary["settings"]

    def histogram(key: str) -> list[tuple[str, int]]:
        values = [r[key] for r in samples if r[key] is not None]
        return [
            (label, sum(lo <= n < hi for n in values)) for label, lo, hi in LENGTH_BINS
        ]

    split_chart = [(r["split"], r["rows_exact"]) for r in split]
    prov_chart = [(r["shard"], r["candidate_documents"]) for r in provenance]
    inventory_chart = [(r["shard"], r["rows_exact"]) for r in inventory]
    token_chart = [(r["shard"], r["token_count_sum_exact"] or 0) for r in inventory]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Mini GPT data and tokenizer audit</title><style>
body{{font:16px/1.5 system-ui,sans-serif;color:#1c2637;max-width:1100px;margin:2rem auto;padding:0 1rem}}
h1,h2{{line-height:1.2}}section{{margin:2.5rem 0}}.note{{background:#edf3fb;border-left:4px solid #5277bb;padding:.8rem 1rem}}
table{{border-collapse:collapse;width:100%;font-size:.85rem;overflow-x:auto;display:block}}
td,th{{border-bottom:1px solid #d8dfe8;text-align:left;padding:.45rem .65rem;white-space:nowrap}}
th{{background:#edf3fb}}svg{{max-width:100%;height:auto}}.axis{{font:12px system-ui,sans-serif;fill:#25344a}}
code{{background:#eef1f5;padding:.1rem .3rem}}small{{color:#536173}}
</style></head><body><h1>Mini GPT data and tokenizer audit</h1>
<p class="note">Parquet row counts and sums of the dataset's <code>token_count</code> field are exact;
the latter are not counts from mini-gpt's SentencePiece tokenizer. Text distributions and duplicate checks use a
cluster sample of {summary['sample_count']:,} rows across {len(inventory)} shards. The sample selects
{cfg['groups_per_shard']} random row groups per shard, then random rows within each group.
Zero sampled duplicates does not establish that the full corpus has none.</p>
<section><h2>1. Shard inventory</h2>
<p>{summary['total_rows_exact']:,} rows; {summary['total_bytes_exact'] / 2**30:,.1f} GiB on disk.
Columns and missing-text counts come from Parquet metadata where available.</p>
{bars('Rows per shard', inventory_chart)}
{bars('Dataset token_count per shard', token_chart, '#609b7b')}
{html_table(inventory, ['shard','split','rows_exact','token_count_sum_exact','disk_bytes','null_text_exact','sampled_rows','sampled_language','sampled_score_p10','sampled_score_median','sampled_score_p90'])}</section>
<section><h2>2. Document lengths and filters</h2>
<p>All lengths below are from sampled rows. The active preprocessor drops texts shorter than
<code>{cfg['min_text_length']}</code> characters and truncates cleaned text above
<code>{cfg['max_text_length']}</code> characters. SentencePiece's
<code>{cfg['max_sentence_length']}</code> limit is in UTF-8 bytes and applies only while learning a tokenizer.</p>
{bars('Sampled raw document length in characters', histogram('raw_chars'), '#609b7b')}
{bars('Sampled raw document length in UTF-8 bytes', histogram('raw_bytes'), '#b77c58')}
{bars('Sampled dataset token_count per document', histogram('token_count'), '#8165ad')}
{html_table([summary['length_sample']], ['sampled_nonnull','raw_median_chars','raw_p95_chars','below_min_raw','filtered_after_cleaning','above_max_before_truncation','above_sentence_byte_limit'])}</section>
<section><h2>3. Train and validation split</h2>
<p>The loader selects whole files: <code>max(1, int(file_count × val_split))</code>.
<code>train_split</code> is currently unused. The same seed fixes split membership and file order.</p>
{'<p class="note">Pre-tokenized train.bin and val.bin exist. Training loads those files, so the current Parquet split may differ from the active training split.</p>' if summary['pretokenized_bins_present'] else ''}
{bars('Rows by split', split_chart, '#b77c58')}
{html_table(split, ['split','files','rows_exact','token_count_sum_exact','sampled_rows','sampled_score_median','sampled_language_score_median'])}
<p>Cross-split normalized exact-text duplicate groups in sample: <strong>{summary['cross_split_text_groups']}</strong>.
Cross-split matching IDs: <strong>{summary['cross_split_id_groups']}</strong>;
matching URLs: <strong>{summary['cross_split_url_groups']}</strong>.</p>
{html_table(duplicates[:30], ['kind','fingerprint','train_shards','val_shards'])}</section>
<section><h2>4. Tokenizer-training provenance</h2>
<p>The saved model records an input limit of <strong>{summary['tokenizer']['saved_input_sentence_size']:,}</strong>
candidate documents; current config specifies <strong>{summary['tokenizer']['configured_docs']:,}</strong>.
The chart reconstructs the files supplying those documents under the current code, data, and seed.
The project does not save an original tokenizer input manifest, and SentencePiece may exclude
lines over its byte limit.</p>
{bars('Candidate tokenizer-training documents by shard', prov_chart, '#8165ad')}
{html_table(provenance, ['shard','candidate_documents','candidate_lines_over_byte_limit'])}
<p>Candidate documents supplied: {summary['tokenizer_candidates_scanned']:,}.</p></section>
<p><small>Companion files: shard_inventory.csv, document_sample.csv, split_summary.csv,
cross_split_duplicates.csv, tokenizer_provenance.csv, summary.json.</small></p></body></html>"""


def audit(cfg, output_dir: Path, sample_per_shard: int, groups_per_shard: int) -> dict:
    from data.dataset import ParquetStreamIterator

    files = ParquetStreamIterator.discover_files(cfg.data.parquet_dir)
    train_files, val_files = split_files(
        files, cfg.data.val_split, cfg.data.shuffle_seed
    )
    val_names = {p.name for p in val_files}
    pre = TextPreprocessor(
        min_length=cfg.data.min_text_length, max_length=cfg.data.max_text_length
    )
    inventory, samples = [], []
    for path in files:
        pf = pq.ParquetFile(path)
        if cfg.data.text_column not in pf.schema_arrow.names:
            raise ValueError(f"Missing text column {cfg.data.text_column!r} in {path}")
        split = "validation" if path.name in val_names else "train"
        sampled = sample_rows(
            pf,
            sample_per_shard,
            groups_per_shard,
            cfg.data.shuffle_seed
            + int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:4], "big"),
        )
        shard_scores, languages = [], Counter()
        for group, row, fields in sampled:
            raw = fields.get(cfg.data.text_column)
            if isinstance(raw, str):
                cleaned_before_limit = (
                    pre._clean(raw) if len(raw) >= pre.min_length else None
                )
                processed = pre.process(raw)
                raw_chars, raw_bytes = len(raw), len(raw.encode("utf-8"))
            else:
                cleaned_before_limit, processed, raw_chars, raw_bytes = (
                    None,
                    None,
                    None,
                    None,
                )
            if fields.get("score") is not None:
                shard_scores.append(fields["score"])
            if fields.get("language"):
                languages[fields["language"]] += 1
            samples.append(
                {
                    "shard": path.name,
                    "split": split,
                    "row_group": group,
                    "row_in_group": row,
                    "id": fields.get("id"),
                    "url": fields.get("url"),
                    "language": fields.get("language"),
                    "language_score": fields.get("language_score"),
                    "score": fields.get("score"),
                    "int_score": fields.get("int_score"),
                    "token_count": fields.get("token_count"),
                    "raw_chars": raw_chars,
                    "raw_bytes": raw_bytes,
                    "cleaned_before_limit_chars": (
                        len(cleaned_before_limit)
                        if cleaned_before_limit is not None
                        else None
                    ),
                    "processed_chars": (
                        len(processed) if processed is not None else None
                    ),
                    "filtered": processed is None,
                    "would_truncate": cleaned_before_limit is not None
                    and len(cleaned_before_limit) > pre.max_length,
                    "text_hash": (
                        hashlib.sha256(normalize_text(processed).encode()).hexdigest()
                        if processed
                        else None
                    ),
                }
            )
        inventory.append(
            {
                "shard": path.name,
                "split": split,
                "rows_exact": pf.metadata.num_rows,
                "row_groups": pf.num_row_groups,
                "columns": ", ".join(pf.schema_arrow.names),
                "token_count_sum_exact": column_sum(pf, "token_count"),
                "disk_bytes": path.stat().st_size,
                "null_text_exact": null_count_from_metadata(pf, cfg.data.text_column),
                "sampled_rows": len(sampled),
                "sampled_language": languages.most_common(1)[0][0] if languages else "",
                "sampled_language_counts": json.dumps(languages, sort_keys=True),
                "sampled_score_p10": percentile(shard_scores, 0.10),
                "sampled_score_median": median(shard_scores) if shard_scores else None,
                "sampled_score_p90": percentile(shard_scores, 0.90),
            }
        )
        print(
            f"Audited {path.name}: {pf.metadata.num_rows:,} rows, {len(sampled)} sampled",
            flush=True,
        )

    split_rows = []
    for split_name in ("train", "validation"):
        chosen = [x for x in inventory if x["split"] == split_name]
        chosen_samples = [x for x in samples if x["split"] == split_name]
        scores = [x["score"] for x in chosen_samples if x["score"] is not None]
        lang_scores = [
            x["language_score"]
            for x in chosen_samples
            if x["language_score"] is not None
        ]
        split_rows.append(
            {
                "split": split_name,
                "files": len(chosen),
                "rows_exact": sum(x["rows_exact"] for x in chosen),
                "token_count_sum_exact": sum(
                    x["token_count_sum_exact"] or 0 for x in chosen
                ),
                "sampled_rows": len(chosen_samples),
                "sampled_score_median": median(scores) if scores else None,
                "sampled_language_score_median": (
                    median(lang_scores) if lang_scores else None
                ),
            }
        )

    duplicate_rows = []
    group_counts = {}
    for kind, key in (("text", "text_hash"), ("id", "id"), ("url", "url")):
        seen = defaultdict(lambda: {"train": set(), "validation": set()})
        for row in samples:
            value = row[key]
            if value:
                seen[str(value)][row["split"]].add(row["shard"])
        matches = [
            (fingerprint, sides)
            for fingerprint, sides in seen.items()
            if sides["train"] and sides["validation"]
        ]
        group_counts[kind] = len(matches)
        for fingerprint, sides in matches:
            duplicate_rows.append(
                {
                    "kind": kind,
                    "fingerprint": fingerprint,
                    "train_shards": ", ".join(sorted(sides["train"])),
                    "val_shards": ", ".join(sorted(sides["validation"])),
                }
            )

    nonnull = [x for x in samples if x["raw_chars"] is not None]
    lengths = sorted(x["raw_chars"] for x in nonnull)
    byte_limit = cfg.tokenizer.max_sentence_length
    length_stats = {
        "sampled_nonnull": len(nonnull),
        "raw_median_chars": median(lengths) if lengths else None,
        "raw_p95_chars": (
            lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))]
            if lengths
            else None
        ),
        "below_min_raw": sum(
            x["raw_chars"] < cfg.data.min_text_length for x in nonnull
        ),
        "filtered_after_cleaning": sum(x["filtered"] for x in samples),
        "above_max_before_truncation": sum(x["would_truncate"] for x in samples),
        "above_sentence_byte_limit": sum(x["raw_bytes"] > byte_limit for x in nonnull),
    }

    model_path = Path(cfg.tokenizer.model_path)
    limit, tokenizer_info = tokenizer_limit(model_path, cfg.tokenizer.train_on_n_docs)
    if cfg.tokenizer.type != "sentencepiece":
        raise ValueError(
            "Tokenizer provenance reconstruction currently requires sentencepiece"
        )
    provenance, scanned = scan_tokenizer_provenance(files, cfg, limit)
    tokenizer_info["effective_candidate_limit"] = limit
    tokenizer_info["saved_input_sentence_size"] = tokenizer_info.get(
        "saved_input_sentence_size", limit
    )

    summary = {
        "total_rows_exact": sum(x["rows_exact"] for x in inventory),
        "total_bytes_exact": sum(x["disk_bytes"] for x in inventory),
        "sample_count": len(samples),
        "length_sample": length_stats,
        "cross_split_text_groups": group_counts["text"],
        "cross_split_id_groups": group_counts["id"],
        "cross_split_url_groups": group_counts["url"],
        "tokenizer_candidates_scanned": scanned,
        "tokenizer": tokenizer_info,
        "validation_shards": sorted(val_names),
        "pretokenized_bins_present": all(
            (Path(cfg.data.parquet_dir) / name).exists()
            for name in ("train.bin", "val.bin")
        ),
        "settings": {
            "val_split": cfg.data.val_split,
            "train_split_unused": cfg.data.train_split,
            "shuffle_seed": cfg.data.shuffle_seed,
            "min_text_length": cfg.data.min_text_length,
            "max_text_length": cfg.data.max_text_length,
            "max_sentence_length": byte_limit,
            "sample_per_shard": sample_per_shard,
            "groups_per_shard": groups_per_shard,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_write(output_dir / "shard_inventory.csv", inventory, list(inventory[0]))
    csv_write(
        output_dir / "document_sample.csv",
        samples,
        [
            "shard",
            "split",
            "row_group",
            "row_in_group",
            "id",
            "url",
            "language",
            "language_score",
            "score",
            "int_score",
            "token_count",
            "raw_chars",
            "raw_bytes",
            "cleaned_before_limit_chars",
            "processed_chars",
            "filtered",
            "would_truncate",
            "text_hash",
        ],
    )
    csv_write(output_dir / "split_summary.csv", split_rows, list(split_rows[0]))
    csv_write(
        output_dir / "cross_split_duplicates.csv",
        duplicate_rows,
        ["kind", "fingerprint", "train_shards", "val_shards"],
    )
    csv_write(output_dir / "tokenizer_provenance.csv", provenance, list(provenance[0]))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.html").write_text(
        render_report(
            summary, inventory, samples, split_rows, provenance, duplicate_rows
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default="outputs/data_audit")
    parser.add_argument("--sample-per-shard", type=int, default=512)
    parser.add_argument("--groups-per-shard", type=int, default=16)
    args = parser.parse_args()
    if args.sample_per_shard < 1 or args.groups_per_shard < 1:
        parser.error("sample-per-shard and groups-per-shard must be positive")
    cfg = load_config(args.config)
    summary = audit(
        cfg, Path(args.output_dir), args.sample_per_shard, args.groups_per_shard
    )
    print(f"Wrote {Path(args.output_dir) / 'report.html'}")
    print(
        f"Sampled {summary['sample_count']:,} rows; reconstructed "
        f"{summary['tokenizer_candidates_scanned']:,} tokenizer candidate documents"
    )


if __name__ == "__main__":
    main()
