"""Fine-tune the embedding model on implicit click-through feedback.

Reads (query_text, clicked_chunk_text) pairs from retrieval_log and trains
the base nomic-embed model with MultipleNegativesRankingLoss. In-batch
negatives provide the contrastive signal; the --hard-negatives flag adds
non-clicked-but-shown chunks as explicit per-query negatives.

Usage:
    # Basic (in-batch negatives only):
    python scripts/finetune.py

    # With hard negatives from non-clicked impressions:
    python scripts/finetune.py --hard-negatives

    # Custom output / device:
    python scripts/finetune.py --output models/bjj-nomic-ft --device cuda --epochs 5

After training, re-index with the fine-tuned model:
    python scripts/reembed.py --model models/bjj-nomic-ft --collection text_ft
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "0"

import argparse
import json
import random
import sqlite3
import time
from pathlib import Path


_DEFAULT_OUTPUT = "models/bjj-nomic-ft"
_DEFAULT_EPOCHS = 3
_DEFAULT_BATCH  = 16
_MIN_PAIRS      = 10


def _load_click_pairs(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return (query_text, chunk_text) for every clicked impression."""
    rows = conn.execute(
        """
        SELECT rl.query_text, c.transcript_text
        FROM retrieval_log rl
        JOIN chunks c ON c.id = rl.chunk_id
        WHERE rl.clicked = 1
          AND rl.query_text != ''
        ORDER BY rl.logged_at
        """
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _load_triplets_with_hard_negatives(
    conn: sqlite3.Connection,
) -> list[tuple[str, str, str]]:
    """Return (query_text, clicked_chunk, non_clicked_chunk) triplets.

    For each session, pairs each clicked chunk with a randomly sampled
    non-clicked-but-shown chunk from the same query as a hard negative.
    Only sessions with at least one click AND one non-click are included.
    """
    # Fetch all impressions grouped by session + query
    rows = conn.execute(
        """
        SELECT rl.session_id, rl.query_text, rl.chunk_id,
               rl.clicked, c.transcript_text
        FROM retrieval_log rl
        JOIN chunks c ON c.id = rl.chunk_id
        WHERE rl.query_text != ''
        ORDER BY rl.session_id, rl.query_text
        """
    ).fetchall()

    groups: dict[tuple[str, str], dict[str, list[str]]] = {}
    for row in rows:
        key = (row["session_id"], row["query_text"])
        if key not in groups:
            groups[key] = {"clicked": [], "shown": []}
        if row["clicked"]:
            groups[key]["clicked"].append(row["transcript_text"])
        else:
            groups[key]["shown"].append(row["transcript_text"])

    triplets: list[tuple[str, str, str]] = []
    for (_, query_text), g in groups.items():
        if not g["clicked"] or not g["shown"]:
            continue
        for pos in g["clicked"]:
            neg = random.choice(g["shown"])
            triplets.append((query_text, pos, neg))

    return triplets


def _train(
    examples,  # list[InputExample]
    model,
    epochs: int,
    batch_size: int,
    output_path: str,
) -> float:
    """Run model.fit() and return elapsed seconds."""
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss_fn = MultipleNegativesRankingLoss(model)
    warmup = max(1, int(0.1 * len(loader) * epochs))

    start = time.time()
    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup,
        show_progress_bar=True,
        output_path=output_path,
    )
    return time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model on click-through data"
    )
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=_MIN_PAIRS,
        help="Minimum click pairs required to proceed (default: 10)",
    )
    parser.add_argument(
        "--hard-negatives",
        action="store_true",
        help="Include non-clicked impressions as explicit hard negatives",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    from jitsudb.config import get_config
    from jitsudb.db import get_connection
    from jitsudb.pipeline.embedder import load_embedding_model

    cfg = get_config()
    conn = get_connection()
    device = args.device or cfg.embedding.device

    # Build training examples
    if args.hard_negatives:
        print("Loading click triplets with hard negatives…")
        triplets = _load_triplets_with_hard_negatives(conn)
        n_data = len(triplets)
        print(f"Found {n_data} (query, clicked, non-clicked) triplets")
    else:
        print("Loading click pairs…")
        pairs = _load_click_pairs(conn)
        n_data = len(pairs)
        print(f"Found {n_data} (query, chunk) click pairs")

    if n_data < args.min_pairs:
        print(
            f"Not enough data (need {args.min_pairs}, got {n_data}). "
            "Run more searches and click results to collect training signal."
        )
        return

    model_src = cfg.embedding.encoder_path or cfg.embedding.model_name
    print(f"Loading base model: {model_src}")
    model = load_embedding_model(
        model_name=cfg.embedding.model_name,
        device=device,
        model_path=cfg.embedding.encoder_path,
        trust_remote_code=cfg.embedding.trust_remote_code,
    )

    # Build InputExamples with nomic prefixes
    from sentence_transformers import InputExample

    q_pfx = cfg.embedding.query_prefix
    d_pfx = cfg.embedding.document_prefix

    if args.hard_negatives:
        examples = [
            InputExample(texts=[q_pfx + q, d_pfx + pos, d_pfx + neg])
            for q, pos, neg in triplets
        ]
    else:
        examples = [
            InputExample(texts=[q_pfx + q, d_pfx + chunk])
            for q, chunk in pairs
        ]

    output_path = str(Path(args.output).resolve())
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print(
        f"\nTraining: {len(examples)} examples, "
        f"{args.epochs} epoch(s), batch size {args.batch_size}, device={device}"
    )
    elapsed = _train(examples, model, args.epochs, args.batch_size, output_path)

    log = {
        "base_model": model_src,
        "n_examples": len(examples),
        "mode": "triplets_hard_neg" if args.hard_negatives else "pairs_inbatch_neg",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": device,
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": output_path,
    }
    log_path = Path(output_path) / "training_log.json"
    with log_path.open("w") as f:
        json.dump(log, f, indent=2)

    print(f"\nFine-tuned model saved → {output_path}")
    print(f"Training log         → {log_path}")
    print(
        f"\nNext: re-index with the fine-tuned model:\n"
        f"  python scripts/reembed.py "
        f"--model {args.output} "
        f"--collection {cfg.embedding.ft_collection}"
    )


if __name__ == "__main__":
    main()
