# big-ai-genius-level-emotions-detection-recognition

🧠 **Big AI Genius Level Emotions Detection Engine**

Demon level intelligence. So simple you don't need to think. AI will know how your friends are feeling, so you don't have to!

## What it does

Paste in any piece of text and the engine tells you exactly what emotions are lurking inside it — joy, sadness, anger, fear, surprise, disgust, or neutral — complete with confidence scores and a friendly bar chart.

## Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Single-shot mode

Pass text directly as a command-line argument:

```bash
python emotions_detection.py "I just got promoted and I can't believe it!"
```

Example output:

```
🧠 Big AI Genius Level Emotions Detection Engine
============================================================

Analyzing: 'I just got promoted and I can't believe it!'

Detected emotions:
  😊 Joy                   ██████████████████████████    87.2%
  😲 Surprise              ███                            9.1%
  😐 Neutral               █                              2.3%
  😠 Anger                                                0.8%
  😢 Sadness                                              0.4%
  😨 Fear                                                 0.2%
  🤢 Disgust                                              0.0%
```

### Interactive mode

Run without arguments to enter a prompt loop:

```bash
python emotions_detection.py
```

Type any text and press Enter to analyze it. Type `quit` (or `exit`) to leave.

## Running tests

```bash
pip install pytest
pytest tests/
```
