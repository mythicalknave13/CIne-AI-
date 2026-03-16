from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List
from urllib.parse import urlparse

import requests
from google import genai
from google.cloud import storage
from google.genai import types

from config.settings import Settings
from generation.extraction import enforce_character_bible, rewrite_for_veo_safety
from generation.vertex import vertex_client_kwargs

logger = logging.getLogger("uvicorn.error")


def _default_veo_prompt(scene_data: Dict[str, Any], story_context: Dict[str, Any]) -> str:
    setting = str(story_context.get("setting", "")).strip()
    visual_style = str(story_context.get("visual_style", "cinematic")).strip()
    narration = str(scene_data.get("narration", "")).strip()
    image_prompt = str(scene_data.get("image_prompt", "")).strip()
    audio_cue = str(scene_data.get("veo_audio_cue", "")).strip()
    return (
        "Medium shot, "
        f"{image_prompt or narration}. "
        f"Setting: {setting}. Visual style: {visual_style}. "
        "Cinematic lighting, shallow depth of field, subtle camera motion, film grain. "
        f"Ambient audio: {audio_cue or 'natural atmosphere and room tone'}. "
        "No text, no titles, no subtitles, no logos."
    )


def _compose_prompt(veo_prompt: str, audio_cue: str) -> str:
    prompt = re.sub(r"\s+", " ", str(veo_prompt or "")).strip()
    cue = re.sub(r"\s+", " ", str(audio_cue or "")).strip()
    if cue and cue.lower() not in prompt.lower():
        prompt = f"{prompt} Ambient audio: {cue}."
    return prompt


def _is_veo_safety_rejection(error: Exception | str) -> bool:
    message = str(error).lower()
    return any(
        pattern in message
        for pattern in (
            "blocked by your current safety settings",
            "sensitive words",
            "responsible ai practices",
            "safety",
            "person/face generation",
        )
    )


def _safe_feeling_from_narration(narration: str) -> str:
    lowered = narration.lower()
    if any(word in lowered for word in ("grief", "loss", "mourning", "absence", "goodbye")):
        return "grief, tenderness, and acceptance"
    if any(word in lowered for word in ("fear", "dread", "panic", "attack", "battle", "war")):
        return "dread turning into resolve"
    if any(word in lowered for word in ("wonder", "miracle", "god", "shadow", "awe")):
        return "awe, wonder, and disbelief"
    if any(word in lowered for word in ("hope", "saved", "belonging", "home", "peace")):
        return "relief, belonging, and hope"
    return "deep reflection and emotion"


def get_safe_fallback_prompt(scene_data: Dict[str, Any]) -> str:
    """Ultra-safe Veo fallback prompt focused on emotion rather than conflict."""
    narration = re.sub(r"\s+", " ", str(scene_data.get("narration", "")).strip())
    feeling = _safe_feeling_from_narration(narration) if narration else "quiet awe after something life-changing"
    return (
        "Cinematic detail shot of a meaningful object or empty space carrying deep emotion. "
        "Warm natural lighting, gentle camera drift, shallow depth of field where appropriate. "
        f"The feeling is: {feeling}. "
        "No violence, no weapons, no conflict shown. No people and no faces. "
        "Use objects, light, weather, texture, and absence only."
    )


def _download_from_gcs_uri(uri: str, output_path: Path) -> None:
    parsed = urlparse(uri)
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.download_to_filename(str(output_path))


def _write_generated_video(video: Any, output_path: Path) -> None:
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
            _download_from_gcs_uri(nested_uri, output_path)
            return
        if nested_uri.startswith("http://") or nested_uri.startswith("https://"):
            response = requests.get(nested_uri, timeout=120)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            return

    direct_uri = str(getattr(video, "uri", "") or "").strip()
    if direct_uri.startswith("gs://"):
        _download_from_gcs_uri(direct_uri, output_path)
        return
    if direct_uri.startswith("http://") or direct_uri.startswith("https://"):
        response = requests.get(direct_uri, timeout=120)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return

    raise ValueError("Veo response did not include downloadable video data.")


def _extract_preview_frame(video_path: Path, preview_path: Path) -> Path:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "select=eq(n\\,0)",
            "-frames:v",
            "1",
            str(preview_path),
        ],
        check=True,
        capture_output=True,
    )
    return preview_path


def _concat_video_parts(parts: List[Path], output_path: Path) -> Path:
    if len(parts) == 1:
        output_path.write_bytes(parts[0].read_bytes())
        return output_path

    concat_list = output_path.with_name(f"{output_path.stem}_parts.txt")
    concat_list.write_text(
        "".join(f"file '{part.resolve()}'\n" for part in parts),
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    return output_path


def generate_single_veo_scene(
    *,
    settings: Settings,
    scene_index: int,
    scene_data: Dict[str, Any],
    story_context: Dict[str, Any],
    character_reference_path: Path,
    output_dir: Path,
    preview_dir: Path,
) -> Dict[str, Any]:
    if str(scene_data.get("format", "video")).strip().lower() == "image":
        return {
            "scene_index": scene_index,
            "video_path": None,
            "preview_image_path": None,
            "duration": 0.0,
            "method": "image_requested",
            "has_native_audio": False,
        }

    client = genai.Client(**vertex_client_kwargs(settings))
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    blueprint = (
        story_context.get("film_blueprint", {})
        if isinstance(story_context.get("film_blueprint", {}), dict)
        else {}
    )
    veo_prompt = str(scene_data.get("veo_prompt", "")).strip() or _default_veo_prompt(scene_data, story_context)
    if blueprint:
        veo_prompt = enforce_character_bible(blueprint, {**scene_data, "veo_prompt": veo_prompt})
    audio_cue = str(scene_data.get("veo_audio_cue", "")).strip()
    rewritten_prompt = rewrite_for_veo_safety(veo_prompt, settings)
    if blueprint:
        rewritten_prompt = enforce_character_bible(blueprint, {**scene_data, "veo_prompt": rewritten_prompt})
    prompt = _compose_prompt(rewritten_prompt, audio_cue)
    safe_fallback_prompt = _compose_prompt(get_safe_fallback_prompt(scene_data), audio_cue)
    extension_seed = str(scene_data.get("veo_extension_prompt", "")).strip() or veo_prompt
    rewritten_extension = rewrite_for_veo_safety(extension_seed, settings)
    if blueprint:
        rewritten_extension = enforce_character_bible(blueprint, {**scene_data, "veo_prompt": rewritten_extension})
    extension_prompt = _compose_prompt(rewritten_extension, audio_cue)

    output_path = output_dir / f"scene_{scene_index:02d}_veo.mp4"
    base_output_path = output_dir / f"scene_{scene_index:02d}_veo_base.mp4"
    extension_output_path = output_dir / f"scene_{scene_index:02d}_veo_ext.mp4"
    preview_path = preview_dir / f"scene_{scene_index:02d}.png"
    base_duration = min(max(int(settings.veo_duration_seconds), 5), 8)
    target_duration = max(int(scene_data.get("duration_seconds", base_duration) or base_duration), base_duration)

    start = time.perf_counter()
    logger.info(
        "TIMING veo scene=%d start model=%s format=%s",
        scene_index,
        settings.veo_model,
        scene_data.get("format", "video"),
    )

    def _request_video(
        request_prompt: str,
        *,
        destination_path: Path,
        duration_seconds: int,
        source_video_path: Path | None = None,
    ) -> None:
        operation = client.models.generate_videos(
            model=settings.veo_model,
            prompt=request_prompt,
            video=types.Video.from_file(location=str(source_video_path)) if source_video_path else None,
            config=types.GenerateVideosConfig(
                aspect_ratio=settings.veo_aspect_ratio,
                duration_seconds=duration_seconds,
                number_of_videos=1,
                person_generation="allow_adult",
                fps=24,
                generate_audio=True,
            ),
        )

        deadline = time.time() + settings.veo_timeout_seconds
        while not operation.done:
            if time.time() >= deadline:
                raise TimeoutError(f"Veo scene {scene_index} timed out after {settings.veo_timeout_seconds}s")
            time.sleep(settings.veo_poll_interval_seconds)
            operation = client.operations.get(operation)

        if operation.error:
            raise RuntimeError(str(operation.error))

        result = operation.result or operation.response
        generated_videos = list(getattr(result, "generated_videos", []) or [])
        if not generated_videos:
            raise ValueError("Veo operation completed without generated videos.")

        _write_generated_video(generated_videos[0], destination_path)

    try:
        _request_video(
            prompt,
            destination_path=base_output_path,
            duration_seconds=base_duration,
        )
        parts = [base_output_path]
        if target_duration > base_duration:
            extension_duration = min(max(target_duration - base_duration, 5), 8)
            try:
                _request_video(
                    extension_prompt,
                    destination_path=extension_output_path,
                    duration_seconds=extension_duration,
                    source_video_path=base_output_path,
                )
                parts.append(extension_output_path)
            except Exception as extension_exc:
                logger.warning(
                    "Veo scene %d extension failed; keeping base clip only. (%s)",
                    scene_index,
                    extension_exc,
                )
        _concat_video_parts(parts, output_path)
        _extract_preview_frame(output_path, preview_path)
        duration = time.perf_counter() - start
        logger.info(
            "TIMING veo scene=%d complete duration=%.2fs output=%s preview=%s",
            scene_index,
            duration,
            output_path,
            preview_path,
        )
        return {
            "scene_index": scene_index,
            "video_path": output_path,
            "preview_image_path": preview_path,
            "duration": duration,
            "method": "veo",
            "has_native_audio": True,
        }
    except Exception as exc:
        if _is_veo_safety_rejection(exc):
            logger.warning(
                "Veo scene %d safety rejection; retrying with ultra-safe fallback prompt.",
                scene_index,
            )
            try:
                _request_video(
                    safe_fallback_prompt,
                    destination_path=output_path,
                    duration_seconds=base_duration,
                )
                _extract_preview_frame(output_path, preview_path)
                duration = time.perf_counter() - start
                return {
                    "scene_index": scene_index,
                    "video_path": output_path,
                    "preview_image_path": preview_path,
                    "duration": duration,
                    "method": "veo",
                    "has_native_audio": True,
                }
            except Exception as fallback_exc:
                duration = time.perf_counter() - start
                logger.warning(
                    "Veo scene %d failed after %.2fs even with ultra-safe fallback: %s",
                    scene_index,
                    duration,
                    fallback_exc,
                )
                return {
                    "scene_index": scene_index,
                    "video_path": None,
                    "preview_image_path": None,
                    "duration": duration,
                    "method": "failed",
                    "has_native_audio": False,
                    "error": str(fallback_exc),
                }
        duration = time.perf_counter() - start
        logger.warning("Veo scene %d failed after %.2fs: %s", scene_index, duration, exc)
        return {
            "scene_index": scene_index,
            "video_path": None,
            "preview_image_path": None,
            "duration": duration,
            "method": "failed",
            "has_native_audio": False,
            "error": str(exc),
        }


async def _maybe_call(callback: Callable[..., Any] | None, *args: Any) -> None:
    if callback is None:
        return
    result = callback(*args)
    if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
        await result


async def generate_all_veo_scenes(
    *,
    settings: Settings,
    script: List[Dict[str, Any]],
    story_context: Dict[str, Any],
    character_reference_path: Path,
    output_dir: Path,
    preview_dir: Path,
    progress_callback: Callable[[int, int], Any] | None = None,
    scene_ready_callback: Callable[[Dict[str, Any]], Any] | None = None,
) -> List[Dict[str, Any]]:
    if not script:
        return []

    semaphore = asyncio.Semaphore(min(settings.veo_max_parallel, len(script)))
    results: List[Dict[str, Any] | None] = [None] * len(script)

    async def _run(scene_index: int, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(
                generate_single_veo_scene,
                settings=settings,
                scene_index=scene_index,
                scene_data=scene_data,
                story_context=story_context,
                character_reference_path=character_reference_path,
                output_dir=output_dir,
                preview_dir=preview_dir,
            )

    tasks = {
        asyncio.create_task(_run(index, scene_data)): index
        for index, scene_data in enumerate(script, start=1)
    }

    completed = 0
    while tasks:
        done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            scene_index = tasks.pop(task)
            try:
                result = task.result()
            except Exception as exc:
                result = {
                    "scene_index": scene_index,
                    "video_path": None,
                    "preview_image_path": None,
                    "duration": 0.0,
                    "method": "failed",
                    "has_native_audio": False,
                    "error": str(exc),
                }
            results[scene_index - 1] = result
            completed += 1
            await _maybe_call(progress_callback, completed, len(script))
            await _maybe_call(scene_ready_callback, result)

    return [dict(result or {}) for result in results]
