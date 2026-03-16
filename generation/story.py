from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from google import genai
from google.cloud import storage
from google.genai import types

from config.settings import Settings
from generation.vertex import vertex_client_kwargs

logger = logging.getLogger("uvicorn.error")
INTERLEAVED_SCENE_COUNT = 6
SCENES_WITH_NARRATION = {1, 3, 4, 5, 6}


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
        or str(story_context.get("character_bible", "")).strip()
        or str(story_context.get("character", "")).strip()
        or str(story_context.get("character_essence", "")).strip()
        or str(story_context.get("who", "")).strip()
    )
    return re.sub(r"\s+", " ", raw).strip()


def _story_prompt(story_context: Dict[str, Any]) -> str:
    character_desc = _character_description(story_context) or "an adult protagonist whose hands carry the story"
    who = str(story_context.get("who", story_context.get("character_essence", ""))).strip()
    turn = str(story_context.get("turn", story_context.get("the_turn", ""))).strip()
    remains = str(story_context.get("remains", story_context.get("residual_feeling", ""))).strip()
    return f"""You are CineAI, a cinematic visual storyteller.

Create one intimate 6-scene story using interleaved text and images.

THE USER'S STORY:
Who they love: {who}
What changed everything: {turn}
What feeling remains: {remains}

CHARACTER DESCRIPTION (use EXACTLY these words in every image where the character appears):
{character_desc}

Choose one thread object and keep it visually consistent in scenes 1, 4, and 6.

NARRATION RULES:
- First person "I" voice only.
- 5-12 words maximum per narration line.
- Sparse. Poetic. One breath.
- Scene 2 must have no narration.
- Scene 6 is the final line.
- Do not explain. Do not summarize. Do not use labels.

IMAGE RULES:
- Every image must be cinematic 16:9 and photorealistic.
- Warm earth tones, soft natural light, shallow depth of field, gentle film grain.
- When the character appears, use the exact character description above.
- Show only hands, silhouette, back, profile, or over-the-shoulder views.
- Never show a clear frontal face with both eyes visible.
- One subject, one action, one moment per image.

Return the story in this exact sequence:
1. Scene 1 narration text.
2. Scene 1 image: close-up of hands doing the defining action, include the thread object.
3. Scene 2 image only: wide establishing shot, character small in frame.
4. Scene 3 narration text.
5. Scene 3 image: medium shot from behind or profile, something changes in the world.
6. Scene 4 narration text.
7. Scene 4 image: close-up of hands holding the thread object, dramatic lighting.
8. Scene 5 narration text or [silence].
9. Scene 5 image: the emotional peak, profile or silhouette against dramatic light.
10. Scene 6 narration text.
11. Scene 6 image: mirror scene 1's framing, same warm light, echo of the thread object.

Do not add headings, scene numbers, bullet points, captions, or explanations. Alternate narration and images only."""


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


def generate_interleaved_story(
    story_context: Dict[str, Any],
    settings: Settings,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(**vertex_client_kwargs(settings))
    prompt = _story_prompt(story_context)
    response = client.models.generate_content(
        model=settings.conversation_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=[types.Modality.TEXT, types.Modality.IMAGE],
            temperature=0.8,
            image_config=types.ImageConfig(aspect_ratio="16:9"),
        ),
    )

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

    result = {
        "title": _story_title(story_context),
        "character_description": _character_description(story_context),
        "story_parts": story_parts,
        "scenes": scenes,
        "prompt": prompt,
        "narration_lines": [scene["narration"] for scene in scenes if scene.get("narration")],
    }
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
