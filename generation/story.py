from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import requests
from google import genai
from google.cloud import storage
from google.genai import types

from config.settings import Settings
from generation.vertex import vertex_client_kwargs

logger = logging.getLogger("uvicorn.error")
INTERLEAVED_SCENE_COUNT = 6
SCENES_WITH_NARRATION = {1, 3, 4, 5, 6}
DEMO_PRESET_CANDIDATES = ("fisherman", "sacrifice", "blue_meadow")


def _story_title(story_context: Dict[str, Any]) -> str:
    title = str(story_context.get("title", "")).strip()
    if title:
        return title
    source = (
        str(story_context.get("who", "")).strip()
        or str(story_context.get("character_essence", "")).strip()
        or str(story_context.get("character", "")).strip()
        or "Untold Memory"
    )
    words = [word for word in re.split(r"\s+", source) if word][:4]
    return " ".join(words)[:48] or "Untold Memory"


def _character_description(story_context: Dict[str, Any]) -> str:
    raw = (
        str(story_context.get("character_description", "")).strip()
        or str(story_context.get("character", "")).strip()
        or str(story_context.get("character_essence", "")).strip()
        or str(story_context.get("who", "")).strip()
    )
    return re.sub(r"\s+", " ", raw).strip()


def _story_prompt(story_context: Dict[str, Any]) -> str:
    who = str(story_context.get("who", story_context.get("character_essence", ""))).strip()
    turn = str(story_context.get("turn", story_context.get("the_turn", ""))).strip()
    remains = str(story_context.get("remains", story_context.get("residual_feeling", ""))).strip()
    return f"""You are CineAI, a cinematic visual storyteller.

You create intimate short stories by weaving narration
and scene illustrations together in one flowing response.

THE USER'S STORY:
Who: {who}
What changed: {turn}
What remains: {remains}

Create a 6-scene illustrated story.

CHARACTER:
- Invent one specific character based on the user's
  description
- Give them a distinct, memorable face, specific age,
  ethnicity, clothing, and physical presence
- Keep this character VISUALLY IDENTICAL across every
  image they appear in - same face, same clothes,
  same body, same hair
- You are generating all images in this single response
  so maintain perfect visual consistency throughout

THREAD OBJECT:
- Choose one small meaningful physical object that
  appears throughout the story
- Something the character makes, carries, or gives
- It should be visible in scenes 1, 4, and 6

NARRATION:
- First person "I" voice - the character remembering
- 5-12 words per line maximum
- Sparse, poetic, like a single breath
- Scene 2 has NO narration - just the image
- Not every scene needs words. Silence is powerful.

VISUAL STYLE:
- All images: cinematic, photorealistic, 16:9
  composition
- Warm natural lighting throughout
- Shallow depth of field
- Consistent color palette across all six scenes
- Each image should feel like a frame from the same
  film - same world, same light, same texture

THE SIX SCENES:

Scene 1 - WARMTH
An intimate image. The character doing something
that defines them. Show their face, their hands,
the action. Include the thread object. Warm light.
Write one narration line before the image.

Scene 2 - THE WORLD
Pull back. Show where this story lives. The character
in their environment.
No narration. Let the image breathe.

Scene 3 - THE CRACK
Something shifts. Show it in the character's
expression, in the light, in a new presence or
absence. The turn begins.
Write one narration line.

Scene 4 - THE WEIGHT
The hardest moment. Show it on their face, in their
body language, in how they hold the thread object -
differently now.
Write one narration line.

Scene 5 - THE MOMENT
The emotional peak. The most visually striking image.
Whatever composition serves this moment - let the
story decide.
Write one line, or write [silence].

Scene 6 - WHAT REMAINS
Echo scene 1. Same warmth, but changed. The
same character transformed, or the thread object
alone, or the space they filled.
Write the final line.

Begin. Alternate between narration text and
generated scene images."""


def _clean_text_block(text: str) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    lines = [
        re.sub(r"^\s*(scene\s*\d+[:.-]?\s*)", "", part.strip(), flags=re.IGNORECASE)
        for part in re.split(r"\n+", cleaned)
    ]
    blocks = [re.sub(r"\s+", " ", line).strip() for line in lines if line.strip()]
    return [block for block in blocks if block]


def load_demo_story(
    story_context: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    demo_root = Path(__file__).resolve().parents[1] / "app" / "static" / "demo_presets"
    preferred = str(story_context.get("preset_name", "")).strip().lower()
    candidates = [preferred] if preferred else []
    candidates.extend([name for name in DEMO_PRESET_CANDIDATES if name not in candidates])

    chosen_dir: Path | None = None
    for name in candidates:
        demo_dir = demo_root / name
        if (demo_dir / "manifest.json").exists():
            chosen_dir = demo_dir
            break
    if chosen_dir is None:
        raise RuntimeError("No cached demo story is available.")

    manifest = json.loads((chosen_dir / "manifest.json").read_text(encoding="utf-8"))
    scenes: List[Dict[str, Any]] = []
    story_parts: List[Dict[str, Any]] = []
    scene_narrations = list(manifest.get("scene_narrations", []) or [])

    for scene_index in range(1, INTERLEAVED_SCENE_COUNT + 1):
        source_path = chosen_dir / "scenes" / f"scene_{scene_index:02d}.png"
        dest_path = output_dir / f"scene_{scene_index:02d}.png"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        narration = str(scene_narrations[scene_index - 1]).strip() if scene_index - 1 < len(scene_narrations) else ""
        if narration:
            story_parts.append({"type": "narration", "text": narration})
        story_parts.append(
            {
                "type": "image",
                "scene_index": scene_index,
                "path": str(dest_path),
                "mime_type": "image/png",
            }
        )
        scenes.append(
            {
                "scene_number": scene_index,
                "narration": narration,
                "image_path": str(dest_path),
                "mime_type": "image/png",
            }
        )

    result = {
        "title": str(manifest.get("title", _story_title(story_context))).strip() or _story_title(story_context),
        "character_description": _character_description(story_context),
        "story_parts": story_parts,
        "scenes": scenes,
        "prompt": "cached-demo-fallback",
        "narration_lines": [scene["narration"] for scene in scenes if scene.get("narration")],
        "fallback": True,
        "fallback_source": chosen_dir.name,
    }
    (output_dir / "interleaved_story.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def generate_interleaved_story(
    story_context: Dict[str, Any],
    settings: Settings,
    output_dir: Path,
    scene_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(**vertex_client_kwargs(settings))
    prompt = _story_prompt(story_context)
    model_candidates = [settings.image_model]
    if settings.image_model != "gemini-2.5-flash-preview-04-17":
        model_candidates.append("gemini-2.5-flash-preview-04-17")
    if settings.image_model != "gemini-2.5-flash-image":
        model_candidates.append("gemini-2.5-flash-image")

    last_error: Exception | None = None
    response = None
    for model_name in model_candidates:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=[types.Modality.TEXT, types.Modality.IMAGE],
                    temperature=0.8,
                ),
            )
            break
        except Exception as exc:
            last_error = exc
            message = str(exc)
            if "only supports text output" in message and model_name != "gemini-2.5-flash-preview-04-17":
                logger.warning(
                    "Interleaved story model %s returned text-only error, retrying with preview model.",
                    model_name,
                )
                continue
            if model_name != model_candidates[-1]:
                logger.warning(
                    "Interleaved story generation failed on model %s, trying fallback model. (%s)",
                    model_name,
                    exc,
                )
                continue
            response = None
            break

    if response is None:
        exc = last_error or RuntimeError("Interleaved story generation failed.")
        logger.warning("Interleaved story generation failed, using cached demo fallback. (%s)", exc)
        result = load_demo_story(story_context, output_dir)
        if scene_callback is not None:
            for scene in result["scenes"]:
                scene_callback(dict(scene))
        return result

    parts = getattr(getattr((response.candidates or [None])[0], "content", None), "parts", None) or []
    story_parts: List[Dict[str, Any]] = []
    scenes: List[Dict[str, Any]] = []
    pending_narration: List[str] = []
    scene_index = 0

    for part in parts:
        text = getattr(part, "text", None)
        inline_data = getattr(part, "inline_data", None)
        if text:
            for block in _clean_text_block(text):
                story_parts.append({"type": "narration", "text": block})
                pending_narration.append(block)
            continue
        if not inline_data or scene_index >= INTERLEAVED_SCENE_COUNT:
            continue
        scene_index += 1
        mime_type = str(getattr(inline_data, "mime_type", "") or "image/png")
        suffix = ".png" if "png" in mime_type else ".jpg"
        image_path = output_dir / f"scene_{scene_index:02d}{suffix}"
        image_path.write_bytes(inline_data.data)

        narration = ""
        if scene_index in SCENES_WITH_NARRATION and pending_narration:
            narration = pending_narration.pop(0)
        story_parts.append({
            "type": "image",
            "scene_index": scene_index,
            "path": str(image_path),
            "mime_type": mime_type,
        })
        scenes.append(
            {
                "scene_number": scene_index,
                "narration": narration,
                "image_path": str(image_path),
                "mime_type": mime_type,
            }
        )
        if scene_callback is not None:
            scene_callback(dict(scenes[-1]))

    result = {
        "title": _story_title(story_context),
        "character_description": _character_description(story_context),
        "story_parts": story_parts,
        "scenes": scenes,
        "prompt": prompt,
        "narration_lines": [scene["narration"] for scene in scenes if scene.get("narration")],
    }
    if not scenes:
        logger.warning("Interleaved story response was empty, using cached demo fallback.")
        result = load_demo_story(story_context, output_dir)
        if scene_callback is not None:
            for scene in result["scenes"]:
                scene_callback(dict(scene))
        return result
    (output_dir / "interleaved_story.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _download_gcs_video(uri: str, output_path: Path) -> None:
    parsed = re.match(r"^gs://([^/]+)/(.+)$", uri)
    if not parsed:
        raise ValueError(f"Unsupported GCS URI: {uri}")
    bucket_name, blob_name = parsed.groups()
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(output_path))


def _write_video_blob(video: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    direct_bytes = getattr(video, "video_bytes", None)
    if direct_bytes:
        output_path.write_bytes(direct_bytes)
        return
    nested_video = getattr(video, "video", None)
    if nested_video is not None:
        nested_bytes = getattr(nested_video, "video_bytes", None)
        if nested_bytes:
            output_path.write_bytes(nested_bytes)
            return
        nested_uri = str(getattr(nested_video, "uri", "") or "").strip()
        if nested_uri.startswith("gs://"):
            _download_gcs_video(nested_uri, output_path)
            return
        if nested_uri.startswith("http://") or nested_uri.startswith("https://"):
            response = requests.get(nested_uri, timeout=120)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            return
    direct_uri = str(getattr(video, "uri", "") or "").strip()
    if direct_uri.startswith("gs://"):
        _download_gcs_video(direct_uri, output_path)
        return
    if direct_uri.startswith("http://") or direct_uri.startswith("https://"):
        response = requests.get(direct_uri, timeout=120)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return
    raise ValueError("Video result did not include downloadable data.")


def _hero_video_prompt(story_context: Dict[str, Any], scene: Dict[str, Any]) -> str:
    character_desc = _character_description(story_context) or "an adult figure"
    turn = str(story_context.get("turn", story_context.get("the_turn", ""))).strip()
    remains = str(story_context.get("remains", story_context.get("residual_feeling", ""))).strip()
    narration = str(scene.get("narration", "")).strip()
    return (
        "Medium profile shot, slow cinematic drift. "
        f"{character_desc}. "
        "They hold still at the emotional turning point of the story. "
        f"The moment is defined by: {turn or remains or narration}. "
        "Photorealistic 16:9 composition, warm earth tones, soft natural light, shallow depth of field, gentle film grain. "
        "No dialogue, no speech, no spoken words, no voices. "
        "Audio: wind, room tone, distant weather. "
        "No text, no subtitles, no logos, no title cards."
    )


def generate_hero_video(
    story_context: Dict[str, Any],
    scene: Dict[str, Any],
    settings: Settings,
    output_path: Path,
) -> Path | None:
    client = genai.Client(**vertex_client_kwargs(settings))
    operation = client.models.generate_videos(
        model=settings.veo_model,
        prompt=_hero_video_prompt(story_context, scene),
        config=types.GenerateVideosConfig(
            aspect_ratio="16:9",
            number_of_videos=1,
            person_generation="allow_adult",
        ),
    )

    deadline = time.time() + max(settings.veo_timeout_seconds, 60)
    while not operation.done:
        if time.time() >= deadline:
            raise TimeoutError("Hero video generation timed out.")
        time.sleep(settings.veo_poll_interval_seconds)
        operation = client.operations.get(operation)

    if operation.error:
        raise RuntimeError(str(operation.error))

    result = operation.result or operation.response
    generated_videos = list(getattr(result, "generated_videos", []) or [])
    if not generated_videos:
        return None
    _write_video_blob(generated_videos[0], output_path)
    return output_path
