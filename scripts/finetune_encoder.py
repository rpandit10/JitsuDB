"""Fine-tune a sentence-transformer encoder on BJJ hard-negative triplets.

Usage:
    python -m scripts.finetune_encoder \\
        [--input data/bjj_triplets.jsonl] \\
        [--output models/bjj-minilm-ft] \\
        [--base-model all-MiniLM-L6-v2] \\
        [--epochs 3] \\
        [--batch-size 32] \\
        [--device cpu]

Requirements:
    sentence-transformers >= 2.2 (tested on 5.x — uses FitMixin API)
    torch

The script uses MultipleNegativesRankingLoss, which treats every positive in the
batch as a negative for the other anchors. Explicit hard negatives from the JSONL
are passed as the third InputExample text, giving the loss access to both in-batch
and mined negatives simultaneously.

80 / 20 train / val split. An EmbeddingSimilarityEvaluator on the val set drives
``save_best_model=True`` so the checkpoint with the lowest val loss is kept.
Training metrics are written to ``<output>/training_log.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def load_triplets(path: Path) -> list[dict[str, str]]:
    """Load newline-delimited JSON triplets from disk."""
    triplets: list[dict[str, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    return triplets


def finetune(
    input_path: Path,
    output_dir: Path,
    base_model: str = "all-MiniLM-L6-v2",
    epochs: int = 3,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    device: str = "cpu",
) -> dict:
    """Run contrastive fine-tuning with MultipleNegativesRankingLoss."""
    # Import here so the module can be imported without torch installed.
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader

    triplets = load_triplets(input_path)
    if not triplets:
        raise ValueError(f"No triplets found in {input_path}")

    print(f"Loaded {len(triplets)} triplets from {input_path}")

    random.shuffle(triplets)
    split = int(len(triplets) * (1.0 - val_fraction))
    train_data = triplets[:split]
    val_data = triplets[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Training set: pass all three texts so MNRL sees explicit hard negatives.
    train_examples = [
        InputExample(texts=[t["anchor"], t["positive"], t["negative"]])
        for t in train_data
    ]
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # Validation evaluator: measure cosine similarity of (anchor, positive) pairs.
    val_examples = [
        InputExample(texts=[t["anchor"], t["positive"]], label=1.0)
        for t in val_data
    ]
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples, name="bjj_val"
    )

    model = SentenceTransformer(base_model, device=device)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        evaluator=evaluator,
        evaluation_steps=len(train_loader),
        warmup_steps=max(1, int(len(train_loader) * epochs * 0.1)),
        output_path=str(output_dir),
        show_progress_bar=True,
        save_best_model=True,
    )
    elapsed = time.time() - start

    log: dict = {
        "base_model": base_model,
        "triplets_total": len(triplets),
        "triplets_train": len(train_data),
        "triplets_val": len(val_data),
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": str(output_dir),
    }
    log_path = output_dir / "training_log.json"
    with log_path.open("w") as f:
        json.dump(log, f, indent=2)

    print(f"Fine-tuning complete in {elapsed:.1f}s")
    print(f"Model saved → {output_dir}")
    print(f"Training log → {log_path}")
    return log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune sentence-transformer on BJJ triplets"
    )
    parser.add_argument(
        "--input",
        default="data/bjj_triplets.jsonl",
        help="Input JSONL path (default: data/bjj_triplets.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="models/bjj-minilm-ft",
        help="Output model directory (default: models/bjj-minilm-ft)",
    )
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="HuggingFace base model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction held out for validation (default: 0.2)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    finetune(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        device=args.device,
    )


if __name__ == "__main__":
    main()
