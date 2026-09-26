"""ConformalReranker and NLI faithfulness check."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jitsudb.retrieval.conformal import ConformalReranker


def _make_result(chunk_id: str, text: str, score: float = 0.5):
    from jitsudb.models import Chunk, SearchResult, Video
    from pathlib import Path

    chunk = Chunk(
        id=chunk_id,
        video_id="vid-1",
        start_time=0.0,
        end_time=45.0,
        transcript_text=text,
    )
    video = Video(
        id="vid-1",
        title="Test Video",
        instructor="Test Instructor",
        file_path=Path("/tmp/test.mp4"),
        duration_seconds=300.0,
        gi_nogi="nogi",
        primary_category="submissions",
    )
    return SearchResult(chunk=chunk, video=video, relevance_score=score, snippet=text[:50])


def _make_mock_reranker(scores: list[float]):
    """CrossEncoderReranker mock whose model.predict returns given scores."""
    mock = MagicMock()
    mock.model.predict.return_value = np.array(scores, dtype=np.float32)
    return mock


class TestCalibrate:
    def test_calibrate_returns_q_hat(self):
        reranker = _make_mock_reranker([8.0, 6.0, 9.0, 7.0])
        conformal = ConformalReranker(reranker, target_coverage=0.90)
        pairs = [
            ("q1", "chunk1", True),
            ("q2", "chunk2", True),
            ("q3", "chunk3", True),
            ("q4", "chunk4", True),
        ]
        q_hat = conformal.calibrate(pairs)
        assert isinstance(q_hat, float)

    def test_calibrate_only_uses_positive_pairs(self):
        reranker = MagicMock()
        reranker.model.predict.return_value = np.array([8.0, 6.0], dtype=np.float32)
        conformal = ConformalReranker(reranker, target_coverage=0.90)
        pairs = [
            ("q1", "positive_chunk", True),
            ("q2", "positive_chunk2", True),
            ("q3", "negative_chunk", False),
            ("q4", "negative_chunk2", False),
        ]
        conformal.calibrate(pairs)
        # Only 2 positive pairs should have been scored
        call_args = reranker.model.predict.call_args[0][0]
        assert len(call_args) == 2
        assert all(text in ("positive_chunk", "positive_chunk2") for _, text in call_args)

    def test_calibrate_empty_positives_sets_neg_inf(self):
        reranker = MagicMock()
        conformal = ConformalReranker(reranker, target_coverage=0.90)
        q_hat = conformal.calibrate([("q", "text", False)])
        assert q_hat == -float("inf")
        reranker.model.predict.assert_not_called()

    def test_q_hat_is_low_quantile_of_positive_scores(self):
        # With target_coverage=0.90, q_hat = 10th percentile of positives
        reranker = MagicMock()
        scores = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        reranker.model.predict.return_value = np.array(scores, dtype=np.float32)
        conformal = ConformalReranker(reranker, target_coverage=0.90)
        pairs = [("q", f"c{i}", True) for i in range(10)]
        q_hat = conformal.calibrate(pairs)
        expected = float(np.quantile(scores, 0.10))
        assert abs(q_hat - expected) < 1e-5


class TestEmpiricalCoverage:
    def test_coverage_above_target_after_calibration(self):
        # 10 positive pairs; calibrate at 90% coverage → coverage should be >= 90%
        scores = list(range(10, 0, -1))  # 10 down to 1
        reranker = MagicMock()
        reranker.model.predict.return_value = np.array(scores, dtype=np.float32)

        conformal = ConformalReranker(reranker, target_coverage=0.90)
        pairs = [("q", f"c{i}", True) for i in range(10)]
        conformal.calibrate(pairs)

        # Reset mock so the same scores are returned on coverage check
        reranker.model.predict.return_value = np.array(scores, dtype=np.float32)
        coverage = conformal.empirical_coverage(pairs)
        assert coverage >= 0.90

    def test_coverage_raises_before_calibration(self):
        reranker = MagicMock()
        conformal = ConformalReranker(reranker, target_coverage=0.90)
        with pytest.raises(RuntimeError, match="calibrate"):
            conformal.empirical_coverage([("q", "text", True)])

    def test_coverage_is_one_with_no_positives(self):
        reranker = MagicMock()
        reranker.model.predict.return_value = np.array([5.0], dtype=np.float32)
        conformal = ConformalReranker(reranker, target_coverage=0.90)
        conformal.calibrate([("q", "text", True)])
        coverage = conformal.empirical_coverage([("q", "neg", False)])
        assert coverage == 1.0


# ConformalReranker.predict_set / rerank
class TestPredictSet:
    def test_predict_set_filters_below_q_hat(self):
        # q_hat calibrated at 5.0; candidates with score < 5.0 should be excluded
        reranker = MagicMock()
        # Calibration: one positive with score 5.0 → q_hat ≈ 5.0 (100th percentile of [5.0])
        reranker.model.predict.side_effect = [
            np.array([5.0], dtype=np.float32),   # calibration call
            np.array([8.0, 3.0, 6.0], dtype=np.float32),  # predict call
        ]
        conformal = ConformalReranker(reranker, target_coverage=1.0)
        conformal.calibrate([("q", "text", True)])

        candidates = [
            _make_result("c1", "text1"),
            _make_result("c2", "text2"),
            _make_result("c3", "text3"),
        ]
        result = conformal.predict_set("query", candidates)
        ids = [r.chunk.id for r in result]
        assert "c2" not in ids  # score 3.0 < q_hat 5.0
        assert "c1" in ids
        assert "c3" in ids

    def test_predict_set_falls_back_to_top1_when_all_below(self):
        reranker = MagicMock()
        reranker.model.predict.side_effect = [
            np.array([100.0], dtype=np.float32),   # calibration: high q_hat
            np.array([1.0, 2.0, 0.5], dtype=np.float32),  # all below q_hat
        ]
        conformal = ConformalReranker(reranker, target_coverage=1.0)
        conformal.calibrate([("q", "text", True)])

        candidates = [_make_result(f"c{i}", f"text{i}") for i in range(3)]
        result = conformal.predict_set("query", candidates)
        # Fallback: exactly 1 result (the highest-scoring one)
        assert len(result) == 1

    def test_rerank_is_alias_for_predict_set(self):
        reranker = MagicMock()
        reranker.model.predict.side_effect = [
            np.array([5.0], dtype=np.float32),
            np.array([7.0, 3.0], dtype=np.float32),
        ]
        conformal = ConformalReranker(reranker, target_coverage=0.90)
        conformal.calibrate([("q", "text", True)])
        candidates = [_make_result("c1", "text1"), _make_result("c2", "text2")]

        rerank_result = conformal.rerank("q", candidates, top_k=10)
        assert isinstance(rerank_result, list)

    def test_scores_normalised_after_filter(self):
        reranker = MagicMock()
        reranker.model.predict.side_effect = [
            np.array([0.0], dtype=np.float32),   # calibration: q_hat = 0
            np.array([4.0, 8.0], dtype=np.float32),  # both above q_hat
        ]
        conformal = ConformalReranker(reranker, target_coverage=1.0)
        conformal.calibrate([("q", "text", True)])

        candidates = [_make_result("c1", "text1"), _make_result("c2", "text2")]
        result = conformal.predict_set("q", candidates)
        scores = [r.relevance_score for r in result]
        # After normalisation, max score should be 1.0
        assert max(scores) == pytest.approx(1.0, abs=1e-5)
        assert min(scores) == pytest.approx(0.0, abs=1e-5)

    def test_uncalibrated_falls_back_to_reranker(self):
        reranker = MagicMock()
        reranker.rerank.return_value = [_make_result("c1", "text")]
        conformal = ConformalReranker(reranker, target_coverage=0.90)
        candidates = [_make_result("c1", "text")]
        conformal.predict_set("query", candidates)
        reranker.rerank.assert_called_once()


# NLI faithfulness check (synthesizer helpers)
class TestNliFaithfulness:
    def test_nli_verdict_returns_string(self):
        from jitsudb.rag.synthesizer import _nli_verdict

        mock_model = MagicMock()
        # scores: (n_contexts, 3) — contradiction=0, entailment=1, neutral=2
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)

        verdict = _nli_verdict(mock_model, "The kimura locks the shoulder.", ["source text"])
        assert verdict in ("ENTAILED", "NEUTRAL", "CONTRADICTED")

    def test_nli_entailed_when_entailment_highest(self):
        from jitsudb.rag.synthesizer import _nli_verdict

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.05, 0.90, 0.05]], dtype=np.float32)
        verdict = _nli_verdict(mock_model, "sentence", ["context"])
        assert verdict == "ENTAILED"

    def test_nli_contradicted_when_contradiction_highest(self):
        from jitsudb.rag.synthesizer import _nli_verdict

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.90, 0.05, 0.05]], dtype=np.float32)
        verdict = _nli_verdict(mock_model, "sentence", ["context"])
        assert verdict == "CONTRADICTED"

    def test_nli_neutral_on_empty_contexts(self):
        from jitsudb.rag.synthesizer import _nli_verdict

        mock_model = MagicMock()
        verdict = _nli_verdict(mock_model, "sentence", [])
        assert verdict == "NEUTRAL"
        mock_model.predict.assert_not_called()

    def test_nli_neutral_on_exception(self):
        from jitsudb.rag.synthesizer import _nli_verdict

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model error")
        verdict = _nli_verdict(mock_model, "sentence", ["context"])
        assert verdict == "NEUTRAL"

    def test_synthesize_with_faithfulness_emits_updates(self):
        from jitsudb.rag.synthesizer import RagSynthesizer

        synth = RagSynthesizer()
        mock_result = MagicMock()
        mock_result.chunk.transcript_text = "source context"
        mock_result.video.title = "Test"
        mock_result.chunk.start_time = 0.0

        fake_tokens = ["The kimura locks the shoulder. ", "Control the wrist first."]

        with patch.object(synth, "synthesize_stream", return_value=iter(fake_tokens)):
            nli_mock = MagicMock()
            nli_mock.predict.return_value = np.array([[0.05, 0.85, 0.10]], dtype=np.float32)

            with patch("jitsudb.rag.synthesizer._load_nli_model", return_value=nli_mock):
                events = list(synth.synthesize_stream_with_faithfulness("heel hook", [mock_result]))

        token_events = [e for e in events if e["type"] == "token"]
        faith_events = [e for e in events if e["type"] == "faithfulness_update"]

        assert len(token_events) == len(fake_tokens)
        assert len(faith_events) >= 1
        assert all(e["verdict"] in ("ENTAILED", "NEUTRAL", "CONTRADICTED") for e in faith_events)
        # sentence indices must be sequential from 0
        indices = [e["sentence_idx"] for e in faith_events]
        assert indices == list(range(len(indices)))
