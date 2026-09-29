# JitsuDB

JitsuDB is a local-first search engine for Brazilian jiu-jitsu videos. Search "kimura from closed guard" and it finds the exact timestamps in your DVD, YouTube, or digital instructionals where that technique is taught, then writes a short answer citing those clips.

> **Proof of concept.** I built this to learn how the parts of a retrieval system work together: speech-to-text, dense and multimodal retrieval, reranking, late interaction, RAG, and the feedback loops around them. It runs on my own library. It isn't packaged or hardened for anyone else's, but I am working to add YouTube scraping functionalites so anyone can build their own library. 

**Stack:** Python 3.11, FastAPI, SQLite, ChromaDB, faster-whisper, sentence-transformers, CLIP, Ollama, React, Vite

## What's implemented

| Area | Details |
|---|---|
| Ingest | Local files or YouTube (yt-dlp). ffmpeg → faster-whisper primed with a BJJ glossary → correction pass for terms Whisper mishears → 45 s chunks with 15 s overlap, snapped to sentence boundaries → keyword taxonomy labels → nomic-embed-text (768-d) and CLIP (512-d) embeddings → temporal chunk graph. Each stage checkpoints, so an interrupted run resumes where it stopped. |
| Retrieval | Abbreviation expansion and filter detection → HyDE, weighted more for short queries → text ANN and CLIP text-to-frame ANN, fused 0.7 / 0.3 → cross-encoder rerank → beam walk to neighboring chunks → ColBERT-style MaxSim rescoring → boost from past clicks. |
| Calibration | Conformal threshold on cross-encoder scores for a target recall (default 90%), calibrated from a labeled query set or click logs. |
| Answers | llama3.2:3b via Ollama, streamed over SSE with inline citations. An optional NLI check marks each sentence as entailed, neutral, or contradicted by the sources. |
| Feedback loop | Entropy scoring flags chunks with ambiguous labels. HDBSCAN clusters propose labels by majority vote for human review. Drift detection flags videos far from their category centroid. Click logs and taxonomy-mined hard negatives drive contrastive (MNRL) fine-tuning of the embedder. |
| Evaluation | Precision@k, Recall@k, nDCG@k, MRR. RAGAS faithfulness, answer relevance, context precision and recall. P50/P95/P99 latency benchmark. |

## Architecture

```
index   video → audio → transcribe → correct → chunk → classify → embed → thumbnail → CLIP → graph

query   preprocess ─┬─ HyDE → text ANN ─┬─ fuse → rerank → beam walk → ColBERT → click boost → answer
                    └─ CLIP visual ANN ─┘
```

SQLite (WAL mode) stores videos, chunks, labels, the processing queue, click logs, and graph edges. ChromaDB stores the text and visual vectors. ColBERT token embeddings sit in a numpy memmap under `data/colbert/`.

## Running it

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 18+, ffmpeg on `PATH`, and [Ollama](https://ollama.com) for HyDE and answers. An NVIDIA GPU is recommended. Whisper falls back to CPU on its own; set the other `device` fields in `config.yaml` to `cpu` if you don't have one.

```bash
uv sync --extra dev
cp config.example.yaml config.yaml          # set paths.video_library
ollama pull llama3.2:3b

uv run jitsudb-server                       # API + processing worker on 127.0.0.1:8000
cd frontend && npm install && npm run dev   # UI on localhost:5173, in a second terminal
```

Videos are expected in this layout:

```
<library>/<NN-Category>/<Instructor - Series Title>/<part>.mp4
```

Add a series from the UI or with `uv run python scripts/add_instructional.py --path <folder>`. An optional `Index.md` at the library root supplies canonical instructor names, series titles, and gi/no-gi; see [Index.example.md](Index.example.md).

Other scripts in `scripts/`: `build_colbert.py` builds the token index, `finetune.py` and `reembed.py` fine-tune on clicks and re-index, `reclassify.py` relabels after taxonomy edits, `benchmark.py` measures latency.

## Evaluation

```bash
uv run pytest
uv run python eval/run_eval.py                 # P@k, R@k, nDCG@k, MRR
uv run --extra eval python eval/run_ragas.py   # RAGAS, needs Ollama
```

Without a golden set, `run_eval.py` generates synthetic queries from the labels in your index. For meaningful numbers, write `eval/golden_qa.jsonl` against your own library, using [the example](eval/golden_qa.example.jsonl) as a template. I don't publish results because they only mean something for a specific library.

## Limitations

- The taxonomy is hand-built for BJJ, and the classifier is keyword matching, not a trained model.
- No videos are included. You are responsible for providing them to the software. 
- The API binds to loopback and has no auth.
- Videos process one at a time to stay within a single GPU's VRAM.

## License

MIT
