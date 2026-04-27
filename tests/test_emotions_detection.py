"""Tests for the emotions detection engine."""

import pytest
from unittest.mock import MagicMock, patch

from emotions_detection import (
    detect_emotions,
    format_results,
    analyze_text,
    EMOTION_DISPLAY,
    main,
)


def _make_classifier(scores):
    """Return a mock classifier that produces the given score list."""
    mock = MagicMock()
    mock.return_value = [scores]
    return mock


SAMPLE_SCORES = [
    {"label": "joy", "score": 0.85},
    {"label": "neutral", "score": 0.10},
    {"label": "anger", "score": 0.02},
    {"label": "sadness", "score": 0.01},
    {"label": "fear", "score": 0.01},
    {"label": "disgust", "score": 0.005},
    {"label": "surprise", "score": 0.005},
]


class TestDetectEmotions:
    def test_returns_sorted_scores(self):
        clf = _make_classifier(SAMPLE_SCORES)
        results = detect_emotions("I am so happy!", classifier=clf)
        assert results[0]["label"] == "joy"
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]

    def test_raises_on_empty_string(self):
        clf = _make_classifier(SAMPLE_SCORES)
        with pytest.raises(ValueError, match="empty"):
            detect_emotions("", classifier=clf)

    def test_raises_on_whitespace_only(self):
        clf = _make_classifier(SAMPLE_SCORES)
        with pytest.raises(ValueError, match="empty"):
            detect_emotions("   ", classifier=clf)

    def test_all_labels_returned(self):
        clf = _make_classifier(SAMPLE_SCORES)
        results = detect_emotions("test", classifier=clf)
        labels = {r["label"] for r in results}
        assert labels == {"joy", "neutral", "anger", "sadness", "fear", "disgust", "surprise"}

    def test_scores_sum_near_one(self):
        clf = _make_classifier(SAMPLE_SCORES)
        results = detect_emotions("test", classifier=clf)
        total = sum(r["score"] for r in results)
        assert abs(total - 1.0) < 0.01


class TestFormatResults:
    def test_contains_header(self):
        output = format_results(SAMPLE_SCORES)
        assert "Detected emotions:" in output

    def test_contains_emotion_name(self):
        output = format_results(SAMPLE_SCORES)
        assert "Joy" in output
        assert "Neutral" in output

    def test_contains_percentage(self):
        output = format_results(SAMPLE_SCORES)
        assert "85.0%" in output

    def test_unknown_label_falls_back_to_capitalized(self):
        scores = [{"label": "confusion", "score": 1.0}]
        output = format_results(scores)
        assert "Confusion" in output


class TestAnalyzeText:
    def test_integration_with_mock(self):
        clf = _make_classifier(SAMPLE_SCORES)
        output = analyze_text("I am so happy!", classifier=clf)
        assert "Detected emotions:" in output
        assert "Joy" in output

    def test_dominant_emotion_appears_first(self):
        clf = _make_classifier(SAMPLE_SCORES)
        output = analyze_text("I am so happy!", classifier=clf)
        lines = output.splitlines()
        # First emotion line (index 1) should be Joy
        assert "Joy" in lines[1]


class TestMain:
    def test_exits_zero_with_text_argument(self, capsys):
        clf = _make_classifier(SAMPLE_SCORES)
        with patch("emotions_detection._get_classifier", return_value=clf):
            exit_code = main(["I feel amazing!"])
        assert exit_code == 0

    def test_output_includes_results(self, capsys):
        clf = _make_classifier(SAMPLE_SCORES)
        with patch("emotions_detection._get_classifier", return_value=clf):
            main(["I feel amazing!"])
        captured = capsys.readouterr()
        assert "Detected emotions:" in captured.out
        assert "Joy" in captured.out

    def test_interactive_quit(self, capsys):
        clf = _make_classifier(SAMPLE_SCORES)
        with patch("emotions_detection._get_classifier", return_value=clf):
            with patch("builtins.input", side_effect=["quit"]):
                exit_code = main([])
        assert exit_code == 0

    def test_interactive_eof_exits_gracefully(self, capsys):
        clf = _make_classifier(SAMPLE_SCORES)
        with patch("emotions_detection._get_classifier", return_value=clf):
            with patch("builtins.input", side_effect=EOFError):
                exit_code = main([])
        assert exit_code == 0


class TestEmotionDisplayMap:
    def test_all_model_labels_have_display_strings(self):
        expected = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}
        assert expected.issubset(set(EMOTION_DISPLAY.keys()))

    def test_display_strings_contain_emoji(self):
        for display in EMOTION_DISPLAY.values():
            # Each display string should start with an emoji (non-ASCII)
            assert display[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
