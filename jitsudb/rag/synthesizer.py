"""Local LLM RAG synthesizer — streams answers from Ollama over retrieved BJJ chunks."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Generator

import httpx
from loguru import logger

if TYPE_CHECKING:
    from jitsudb.models import SearchResult

_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_nli_model_cache: dict = {}

# Matches terminal punctuation followed by whitespace — sentence boundary signal
_SENTENCE_END_RE = re.compile(r'(?<=[.?!])\s+')

SYSTEM_PROMPT = (
    "You are a Brazilian Jiu-Jitsu coach reviewing instructional footage to answer a student's question. "
    "Use ONLY the numbered excerpts provided. Your answer must:\n"
    "  1. Open with the key principle or controlling concept — not a position definition.\n"
    "  2. Focus on how to control your opponent's reactions and limit their counters, "
    "not on situational micro-adjustments.\n"
    "  3. Cite each claim inline as [1], [2], etc.\n"
    "  4. Use active voice. Be direct. 3–5 sentences total.\n"
    "  5. Only include what the excerpts support. Do not add disclaimers, hedges, or "
    "requests for more information."
)


def build_rag_prompt(query: str, results: list["SearchResult"], max_chunks: int = 5) -> str:
    """Format retrieved chunks as numbered citations for the LLM."""
    lines = ["Transcript excerpts from BJJ instructional videos:\n"]
    for i, r in enumerate(results[:max_chunks], 1):
        mins = int(r.chunk.start_time // 60)
        secs = int(r.chunk.start_time % 60)
        lines.append(f'[{i}] "{r.video.title}" at {mins}:{secs:02d}:')
        text = r.chunk.transcript_text[:600]
        if len(r.chunk.transcript_text) > 600:
            text += "…"
        lines.append(f'"{text}"\n')
    lines.append(
        f"Question: {query}\n\n"
        "Answer with the controlling principle and how it limits what your opponent can do. "
        "Cite each claim. No disclaimers."
    )
    return "\n".join(lines)


class RagSynthesizer:
    """Streams answers synthesized from retrieved BJJ chunks via a local Ollama model."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        max_context_chunks: int = 5,
        max_tokens: int = 400,
        temperature: float = 0.1,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.max_context_chunks = max_context_chunks
        self.max_tokens = max_tokens
        self.temperature = temperature

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and the configured model is pulled."""
        try:
            resp = httpx.get(f"{self.ollama_url}/api/tags", timeout=2.0)
            if resp.status_code != 200:
                return False
            model_names = [m["name"] for m in resp.json().get("models", [])]
            # Match both "llama3.2:3b" and "llama3.2:3b-instruct-q4_0" etc.
            base = self.model.split(":")[0]
            return any(base in name for name in model_names)
        except Exception:
            return False

    def synthesize_stream(
        self,
        query: str,
        results: list["SearchResult"],
    ) -> Generator[str, None, None]:
        """Yield answer tokens one by one from Ollama's streaming chat endpoint."""
        prompt = build_rag_prompt(query, results, self.max_context_chunks)
        logger.debug(f"RAG: {len(results)} chunks, prompt {len(prompt)} chars, query={query!r}")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        with httpx.Client(timeout=httpx.Timeout(5.0, read=120.0)) as client:
            with client.stream(
                "POST",
                f"{self.ollama_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = event.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if event.get("done"):
                        logger.debug(
                            f"RAG complete: {event.get('eval_count', '?')} tokens generated"
                        )
                        break

    def synthesize_stream_with_faithfulness(
        self,
        query: str,
        results: list["SearchResult"],
    ) -> Generator[dict, None, None]:
        """Stream tokens with per-sentence NLI entailment verdicts.

        Yields dicts of two shapes:
        - ``{"type": "token", "content": str}`` — one per generated token
        - ``{"type": "faithfulness_update", "sentence_idx": int, "verdict": str}``
          emitted after each sentence completes; verdict is one of
          ``ENTAILED``, ``NEUTRAL``, or ``CONTRADICTED``.

        Sentence boundaries are detected by terminal punctuation (.?!)
        followed by whitespace. NLI runs synchronously after each sentence;
        generation continues while the verdict is computed.
        """
        nli = _load_nli_model()
        contexts = [r.chunk.transcript_text[:600] for r in results[: self.max_context_chunks]]

        sentence_idx = 0
        buffer = ""

        for token in self.synthesize_stream(query, results):
            yield {"type": "token", "content": token}
            buffer += token

            # Check for one or more complete sentences in the buffer
            parts = _SENTENCE_END_RE.split(buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence and nli is not None:
                        verdict = _nli_verdict(nli, sentence, contexts)
                        yield {
                            "type": "faithfulness_update",
                            "sentence_idx": sentence_idx,
                            "verdict": verdict,
                        }
                    sentence_idx += 1
                buffer = parts[-1]  # remainder starts the next sentence

        # Flush any remaining text as a final sentence
        remainder = buffer.strip()
        if remainder and nli is not None:
            verdict = _nli_verdict(nli, remainder, contexts)
            yield {
                "type": "faithfulness_update",
                "sentence_idx": sentence_idx,
                "verdict": verdict,
            }

    def _prompt_hyde(self, query: str) -> Generator[str, None, None]:
        """Generate a hypothetical BJJ transcript passage that would answer query.

        Used by HyDE (Hypothetical Document Embeddings): the generated passage
        is embedded and averaged with the original query embedding so retrieval
        searches for content that *looks like* a good answer rather than the
        question itself.
        """
        prompt = (
            f"Write a short passage (3-4 sentences) from a BJJ instructional video transcript "
            f"that directly answers this question: {query}\n\n"
            "Write as a BJJ instructor speaking during a technique demonstration. "
            "Use natural spoken language. No preamble."
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"num_predict": 80, "temperature": 0.3},
        }
        with httpx.Client(timeout=httpx.Timeout(5.0, read=30.0)) as client:
            with client.stream(
                "POST",
                f"{self.ollama_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = event.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if event.get("done"):
                        break


def _load_nli_model():
    """Load and cache the NLI cross-encoder; returns None if unavailable."""
    if _NLI_MODEL_NAME in _nli_model_cache:
        return _nli_model_cache[_NLI_MODEL_NAME]
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(_NLI_MODEL_NAME)
        _nli_model_cache[_NLI_MODEL_NAME] = model
        logger.info(f"NLI model loaded: {_NLI_MODEL_NAME}")
        return model
    except Exception as exc:
        logger.warning(f"NLI model unavailable ({exc}) — faithfulness check disabled")
        _nli_model_cache[_NLI_MODEL_NAME] = None
        return None


def _nli_verdict(nli_model, sentence: str, contexts: list[str]) -> str:
    """Return ENTAILED / NEUTRAL / CONTRADICTED for a sentence vs. source chunks.

    Checks the sentence against each context chunk and takes the verdict
    from the chunk with the highest entailment score.

    Label order for nli-deberta-v3-small: contradiction=0, entailment=1, neutral=2
    """
    if not contexts or not sentence:
        return "NEUTRAL"

    try:
        pairs = [(ctx, sentence) for ctx in contexts]
        scores = nli_model.predict(pairs, show_progress_bar=False)
        # scores shape: (n_contexts, 3) — contradiction, entailment, neutral
        import numpy as np
        scores = np.array(scores)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        # Best chunk = highest entailment score (index 1)
        best_idx = int(np.argmax(scores[:, 1]))
        label_idx = int(np.argmax(scores[best_idx]))
        labels = {0: "CONTRADICTED", 1: "ENTAILED", 2: "NEUTRAL"}
        return labels.get(label_idx, "NEUTRAL")
    except Exception as exc:
        logger.debug(f"NLI check failed: {exc}")
        return "NEUTRAL"
