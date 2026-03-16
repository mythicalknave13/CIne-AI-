# CineAI

CineAI is a cinematic storytelling app for the Gemini Live Agent Challenge.

Current flow:
- ask 5 questions: `Who -> World -> Sound / Mood -> Change -> Ending`
- for custom stories, show an image-led visual story with narration
- for preset demos, play a cached demo cut instantly

## Run Locally

```bash
pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

## Reproducible Testing

### Option A: No-API Demo Test

This is the most reliable judge path.

1. Start the server:

```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

2. Open the app in a browser.
3. Click the preset `The Father's Sacrifice`.

Expected result:
- the preset loads immediately
- the right panel switches to video mode
- the demo film is playable

### Option B: Custom Story Test

1. Export a valid Gemini key:

```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

2. Start the server:

```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

3. Answer the five prompts with this sample:

- `Who`: `My grandmother who taught me to make bread`
- `World`: `A small warm kitchen at sunrise`
- `Sound / Mood`: `Quiet, tender, flour in the air, soft morning light`
- `Change`: `The last morning we baked together, her hands were shaking`
- `Ending`: `Flour on my hands still feels like her blessing`

Expected result:
- the app advances through all five questions
- after the fifth answer, the custom image-story path starts
- the right panel presents a cinematic image-led story
- narration text appears with the images
- narration audio appears if TTS succeeds

If model quota is unavailable, use the preset demo path above for evaluation.

## Minimal Checks

```bash
python3 -m pytest tests/test_story.py tests/test_extraction.py
```
