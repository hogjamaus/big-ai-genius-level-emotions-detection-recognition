#!/usr/bin/env python3
"""
Big AI Genius Level Emotions Detection Engine
Detects emotions in text so you know how your friends are feeling.
"""

import argparse
import sys

EMOTION_DISPLAY = {
    "joy": "😊 Joy",
    "sadness": "😢 Sadness",
    "anger": "😠 Anger",
    "fear": "😨 Fear",
    "surprise": "😲 Surprise",
    "disgust": "🤢 Disgust",
    "neutral": "😐 Neutral",
}

_classifier = None


def _get_classifier():
    """Lazily load the emotion classification pipeline."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline  # noqa: PLC0415

        _classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            device=-1,
        )
    return _classifier


def detect_emotions(text, classifier=None):
    """
    Detect emotions in the given text.

    Args:
        text: The text to analyze.
        classifier: Optional pre-loaded classifier (for testing).

    Returns:
        A list of dicts with 'label' and 'score' keys, sorted by score descending.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    clf = classifier if classifier is not None else _get_classifier()
    results = clf(text)
    # When return_all_scores=True the pipeline returns a list of lists.
    scores = results[0] if results and isinstance(results[0], list) else results
    return sorted(scores, key=lambda x: x["score"], reverse=True)


def format_results(results):
    """Format detected emotions as a human-readable string."""
    lines = ["Detected emotions:"]
    for item in results:
        label = item["label"]
        score = item["score"]
        display = EMOTION_DISPLAY.get(label, label.capitalize())
        bar = "█" * int(score * 30)
        lines.append(f"  {display:<22} {bar:<30} {score:.1%}")
    return "\n".join(lines)


def analyze_text(text, classifier=None):
    """Analyze text and return formatted emotion results."""
    results = detect_emotions(text, classifier=classifier)
    return format_results(results)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "🧠 Big AI Genius Level Emotions Detection Engine — "
            "detect how people are feeling from their text."
        )
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze. If omitted, enters interactive mode.",
    )
    args = parser.parse_args(argv)

    print("🧠 Big AI Genius Level Emotions Detection Engine")
    print("=" * 60)

    if args.text:
        print(f"\nAnalyzing: {args.text!r}\n")
        print(analyze_text(args.text))
        return 0

    print("Enter text to analyze emotions (type 'quit' to exit).\n")
    while True:
        try:
            text = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not text:
            continue

        try:
            print()
            print(analyze_text(text))
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
