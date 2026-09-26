"""RAGAS evaluation harness for JitsuDB.

Computes five metrics against eval/golden_qa.jsonl:
    faithfulness        — LLM-based (requires ragas + Ollama)
    answer_relevance    — LLM-based (requires ragas + Ollama)
    context_precision   — chunk-ID-based (always available)
    context_recall      — chunk-ID-based (always available)
    ndcg_at_10          — rank-based   (always available)

Usage:
    python eval/run_ragas.py
    python eval/run_ragas.py --config config.yaml --qa eval/golden_qa.jsonl
    python eval/run_ragas.py --skip-llm   # skips faithfulness + answer_relevance
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when run directly
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain at k."""
    def dcg(ids: list[str]) -> float:
        return sum(
            (1.0 if ids[i] in relevant_ids else 0.0) / math.log2(i + 2)
            for i in range(min(len(ids), k))
        )

    ideal = sorted(retrieved_ids, key=lambda cid: int(cid in relevant_ids), reverse=True)
    idcg = dcg(ideal)
    return dcg(retrieved_ids) / idcg if idcg > 0 else 0.0


def _context_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Fraction of retrieved chunks that are in the relevant set."""
    if not retrieved_ids:
        return 0.0
    return sum(1 for cid in retrieved_ids if cid in relevant_ids) / len(retrieved_ids)


def _context_recall(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Fraction of relevant chunks that appear anywhere in retrieved results."""
    if not relevant_ids:
        return 1.0
    return sum(1 for cid in relevant_ids if cid in retrieved_ids) / len(relevant_ids)


def _try_ragas_evaluate(
    samples: list[dict],
    ollama_url: str,
    model: str,
) -> dict[str, float | None]:
    """Run ragas Faithfulness + AnswerRelevancy; returns None values on failure."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        print(
            "[eval] ragas or datasets not installed — skipping LLM metrics.\n"
            "       Install with: pip install ragas datasets"
        )
        return {"faithfulness": None, "answer_relevance": None}

    try:
        from langchain_community.llms import Ollama
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        llm = LangchainLLMWrapper(Ollama(model=model, base_url=ollama_url))
        emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
        faithfulness.llm = llm
        answer_relevancy.llm = llm
        answer_relevancy.embeddings = emb
    except Exception as exc:
        print(f"[eval] LLM setup failed ({exc}) — skipping LLM metrics.")
        return {"faithfulness": None, "answer_relevance": None}

    try:
        ds = Dataset.from_list([
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "ground_truth": s["ground_truth"],
            }
            for s in samples
        ])
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
        return {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevance": float(result["answer_relevancy"]),
        }
    except Exception as exc:
        print(f"[eval] ragas evaluation failed ({exc}) — LLM metrics set to None.")
        return {"faithfulness": None, "answer_relevance": None}


def _build_search_engine(cfg):
    from jitsudb.pipeline.embedder import get_chroma_collection, load_embedding_model
    from jitsudb.search.engine import SearchEngine

    chroma_dir = cfg.paths.data_dir / "chromadb"

    if cfg.embedding.encoder_path:
        from jitsudb.pipeline.embedder import get_ft_chroma_collection
        model = load_embedding_model(
            model_name=cfg.embedding.model_name,
            device=cfg.embedding.device,
            model_path=cfg.embedding.encoder_path,
        )
        collection = get_ft_chroma_collection(chroma_dir, cfg.embedding.ft_collection)
    else:
        model = load_embedding_model(cfg.embedding.model_name, cfg.embedding.device)
        collection = get_chroma_collection(chroma_dir)

    reranker = None
    if cfg.reranker.enabled:
        from jitsudb.search.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(
            model_name=cfg.reranker.model_name,
            device=cfg.reranker.device,
        )

    return SearchEngine(
        embedding_model=model,
        collection=collection,
        top_k=cfg.search.default_top_k,
        reranker=reranker,
        beam_width=cfg.search.beam_width,
        max_expand=cfg.search.max_expand,
        reranker_top_k=cfg.search.reranker_top_k,
        candidate_k=cfg.search.candidate_k,
        hyde_enabled=False,  # disable HyDE for eval reproducibility
    )


def _generate_answer(synthesizer, question: str, results: list) -> str:
    """Stream a full answer string from the RAG synthesizer."""
    try:
        tokens = synthesizer.synthesize_stream(question, results)
        return "".join(tokens)
    except Exception:
        return ""


def run_evaluation(
    config_path: str = "config.yaml",
    qa_path: str = "eval/golden_qa.jsonl",
    skip_llm: bool = False,
) -> dict:
    """Run the full evaluation and return a metrics dict.

    Always computes: context_precision, context_recall, ndcg_at_10.
    Computes if ragas + Ollama available: faithfulness, answer_relevance.
    """
    from jitsudb.config import load_config, reset_config
    reset_config()
    cfg = load_config(Path(config_path))

    qa_entries = [
        json.loads(line)
        for line in Path(qa_path).read_text().splitlines()
        if line.strip()
    ]
    print(f"[eval] Loaded {len(qa_entries)} QA pairs from {qa_path}")

    print("[eval] Initialising search engine …")
    engine = _build_search_engine(cfg)

    synthesizer = None
    if not skip_llm and cfg.rag.enabled:
        from jitsudb.rag.synthesizer import RagSynthesizer
        synthesizer = RagSynthesizer(
            ollama_url=cfg.rag.ollama_url,
            model=cfg.rag.model,
        )
        if not synthesizer.is_available():
            print("[eval] Ollama not available — answers will be empty strings.")
            synthesizer = None

    per_question: list[dict] = []
    ctx_precisions, ctx_recalls, ndcgs = [], [], []
    ragas_samples: list[dict] = []

    for i, entry in enumerate(qa_entries, 1):
        question = entry["question"]
        relevant_ids: set[str] = set(entry["relevant_chunk_ids"])
        ground_truth: str = entry["ground_truth"]

        print(f"[eval] ({i}/{len(qa_entries)}) {question[:70]}")

        results = engine.search(question)
        retrieved_ids = [r.chunk.id for r in results]

        cp = _context_precision(retrieved_ids, relevant_ids)
        cr = _context_recall(retrieved_ids, relevant_ids)
        ndcg = _ndcg_at_k(retrieved_ids, relevant_ids)

        ctx_precisions.append(cp)
        ctx_recalls.append(cr)
        ndcgs.append(ndcg)

        answer = ""
        if synthesizer:
            answer = _generate_answer(synthesizer, question, results)

        contexts = [r.chunk.transcript_text for r in results[:5]]

        per_question.append({
            "question": question,
            "category": entry.get("category", ""),
            "retrieved_ids": retrieved_ids[:10],
            "relevant_ids": list(relevant_ids),
            "context_precision": cp,
            "context_recall": cr,
            "ndcg_at_10": ndcg,
            "answer_preview": answer[:200] if answer else "",
        })

        if not skip_llm:
            ragas_samples.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            })

    # Aggregate retrieval-only metrics
    metrics: dict[str, float | None] = {
        "context_precision": sum(ctx_precisions) / len(ctx_precisions),
        "context_recall": sum(ctx_recalls) / len(ctx_recalls),
        "ndcg_at_10": sum(ndcgs) / len(ndcgs),
        "faithfulness": None,
        "answer_relevance": None,
    }

    # Attempt LLM-based metrics
    if not skip_llm and ragas_samples and synthesizer:
        llm_metrics = _try_ragas_evaluate(ragas_samples, cfg.rag.ollama_url, cfg.rag.model)
        metrics.update(llm_metrics)

    # Write per-run JSON
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "config": config_path,
        "qa_path": qa_path,
        "n_questions": len(qa_entries),
        "metrics": metrics,
        "per_question": per_question,
    }
    result_path.write_text(json.dumps(payload, indent=2))
    print(f"[eval] Results written → {result_path}")

    # Update RESULTS.md
    _update_results_md(metrics, timestamp, result_path)

    return metrics


def _update_results_md(metrics: dict, timestamp: str, result_path: Path) -> None:
    results_md = Path("eval/RESULTS.md")

    def _fmt(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    row = (
        f"| {timestamp} "
        f"| {_fmt(metrics['ndcg_at_10'])} "
        f"| {_fmt(metrics['context_precision'])} "
        f"| {_fmt(metrics['context_recall'])} "
        f"| {_fmt(metrics['faithfulness'])} "
        f"| {_fmt(metrics['answer_relevance'])} "
        f"| {result_path.name} |"
    )

    if not results_md.exists():
        header = textwrap.dedent("""\
            # JitsuDB Evaluation Results

            Metrics computed by `eval/run_ragas.py` against `eval/golden_qa.jsonl` (20 questions).

            **Retrieval metrics** (context_precision, context_recall, nDCG@10) are computed
            from `relevant_chunk_ids` in the golden set — no LLM required.

            **LLM metrics** (faithfulness, answer_relevance) require `ragas`, `datasets`,
            `langchain-community`, and a running Ollama instance. They are marked N/A otherwise.

            ---

            ## Baseline

            | Run | nDCG@10 | Ctx Precision | Ctx Recall | Faithfulness | Ans Relevance | File |
            |-----|---------|---------------|------------|--------------|---------------|------|
        """)
        results_md.write_text(header + row + "\n")
    else:
        existing = results_md.read_text()
        results_md.write_text(existing + row + "\n")

    print(f"[eval] RESULTS.md updated → {results_md}")


def _print_summary(metrics: dict) -> None:
    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A "

    print("\n" + "=" * 52)
    print("  JitsuDB RAGAS Evaluation Summary")
    print("=" * 52)
    print(f"  nDCG@10            {_fmt(metrics['ndcg_at_10'])}")
    print(f"  Context Precision  {_fmt(metrics['context_precision'])}")
    print(f"  Context Recall     {_fmt(metrics['context_recall'])}")
    print(f"  Faithfulness       {_fmt(metrics['faithfulness'])}")
    print(f"  Answer Relevance   {_fmt(metrics['answer_relevance'])}")
    print("=" * 52)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JitsuDB RAGAS evaluation harness")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--qa", default="eval/golden_qa.jsonl", help="Path to golden QA file")
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip faithfulness and answer_relevance (faster, no Ollama required)"
    )
    args = parser.parse_args()

    metrics = run_evaluation(
        config_path=args.config,
        qa_path=args.qa,
        skip_llm=args.skip_llm,
    )
    _print_summary(metrics)
