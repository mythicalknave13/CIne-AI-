from __future__ import annotations

import base64
import io
import json
import logging
import random
import re
import shutil
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import ffmpeg
from google import genai
import requests
from google.genai import types

from config.settings import Settings
from generation.vertex import vertex_client_kwargs, vertex_credentials

logger = logging.getLogger("uvicorn.error")
ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class AudioAssets:
    narration_path: Path
    music_path: Path
    mixed_path: Path


@dataclass
class NarrationTiming:
    intro_duration_seconds: float
    scene_duration_seconds: float
    total_duration_seconds: float


@dataclass(frozen=True)
class NarratorProfile:
    voice_name: str
    style_prompt: str


NARRATOR_PROFILES: dict[str, NarratorProfile] = {
    "old_woman_remembering": NarratorProfile(
        voice_name="en-US-Studio-O",
        style_prompt="Female, warm, reflective, intimate, gently aged, with slow cinematic pauses and the softness of remembered love.",
    ),
    "old_man_remembering": NarratorProfile(
        voice_name="en-US-Studio-M",
        style_prompt="Male, warm, weathered, reflective, grounded, with slow cinematic pauses and the gravity of memory.",
    ),
    "young_woman_present": NarratorProfile(
        voice_name="en-US-Studio-O",
        style_prompt="Female, clear, immediate, present tense, emotionally grounded, close and natural.",
    ),
    "young_man_present": NarratorProfile(
        voice_name="en-US-Studio-M",
        style_prompt="Male, clear, immediate, present tense, emotionally grounded, close and natural.",
    ),
    "mythic_narrator": NarratorProfile(
        voice_name="en-US-Chirp3-HD-Laomedeia",
        style_prompt="Timeless, resonant, mythic, very slow and deliberate, like a legend spoken into stone and firelight.",
    ),
    "child_witnessing": NarratorProfile(
        voice_name="en-US-Chirp3-HD-Aoede",
        style_prompt="Soft, small, innocent, gently observant, with quiet wonder and vulnerability.",
    ),
    "older_memory": NarratorProfile(
        voice_name="en-US-Studio-O",
        style_prompt="Female, warm, reflective, intimate, gently aged, with slow cinematic pauses and the softness of remembered love.",
    ),
    "older_memory_aged": NarratorProfile(
        voice_name="en-US-Studio-O",
        style_prompt="Female, older and lower, slower, fragile but clear, carrying legacy, grief, and hard-earned tenderness.",
    ),
    "mythic_storyteller": NarratorProfile(
        voice_name="en-US-Chirp3-HD-Laomedeia",
        style_prompt="Timeless, resonant, mythic, very slow and deliberate, like a legend spoken into stone and firelight.",
    ),
    "quiet_child": NarratorProfile(
        voice_name="Leda",
        style_prompt="Young, soft, vulnerable, innocent, spoken with quiet wonder.",
    ),
    "urgent_witness": NarratorProfile(
        voice_name="Erinome",
        style_prompt="Urgent, tense, immediate, eyewitness narration with controlled fear.",
    ),
    "hacker_present_tense": NarratorProfile(
        voice_name="Kore",
        style_prompt="Cool, modern, close-mic, observant, present-tense, emotionally restrained.",
    ),
    "hacker_urgent": NarratorProfile(
        voice_name="Alnilam",
        style_prompt="Fast, high-stakes, sharp, urgent, like a system failure unfolding in real time.",
    ),
    "young_apprentice": NarratorProfile(
        voice_name="Leda",
        style_prompt="Bright, curious, slightly nervous, learning through awe and uncertainty.",
    ),
    "old_sage": NarratorProfile(
        voice_name="Sadaltager",
        style_prompt="Very slow, grounded, wise, peaceful, spoken by someone who has already accepted the end.",
    ),
    "sage_remembering": NarratorProfile(
        voice_name="Sulafat",
        style_prompt="Fond, bittersweet, thoughtful, carrying memory with warmth and distance.",
    ),
}

TTS_RESPONSE_TIMEOUT_MS = 120_000
DEFAULT_TTS_STYLE = "Cinematic voiceover, natural pacing, emotionally grounded, clear and intimate."
NARRATION_PRE_SILENCE_SECONDS = 0.5
NARRATION_POST_SILENCE_SECONDS = 1.0
NARRATION_GOODBYE_POST_SILENCE_SECONDS = 2.0
NARRATION_STILLNESS_PRE_SILENCE_SECONDS = 2.0
NARRATION_SCENE_CROSSFADE_SECONDS = 0.3
MUSIC_SEGMENT_CROSSFADE_SECONDS = 1.5
AMBIENT_BASE_VOLUME = 0.25
MUSIC_DUCK_BASE_VOLUME = 0.20
AMBIENT_DUCK_THRESHOLD = 0.03
AMBIENT_DUCK_RATIO = 4
AMBIENT_DUCK_ATTACK_MS = 35
AMBIENT_DUCK_RELEASE_MS = 250
AMBIENT_DUCK_LEVEL_SC = 0.5
MUSIC_DUCK_THRESHOLD = 0.02
MUSIC_DUCK_RATIO = 5
MUSIC_DUCK_ATTACK_MS = 50
MUSIC_DUCK_RELEASE_MS = 400
MUSIC_DUCK_LEVEL_SC = 0.5
FINAL_AUDIO_FADE_IN_SECONDS = 1.5
FINAL_AUDIO_FADE_OUT_SECONDS = 3.0
AMBIENT_PAUSE_VOLUME = 0.0
MUSIC_PAUSE_VOLUME = 0.02
TTS_VOICE_ALIASES = {
    "en-us-chirp3-hd-achernar": "Achernar",
    "en-us-chirp3-hd-aoede": "Aoede",
    "en-us-chirp3-hd-erinome": "Erinome",
    "en-us-chirp3-hd-laomedeia": "Laomedeia",
    "en-us-studio-o": "Sulafat",
    "en-us-studio-m": "Gacrux",
    "studio-o": "Sulafat",
    "studio-m": "Gacrux",
    "achernar": "Achernar",
    "aoede": "Aoede",
    "erinome": "Erinome",
    "laomedeia": "Laomedeia",
    "sulafat": "Sulafat",
    "gacrux": "Gacrux",
    "sadaltager": "Sadaltager",
    "leda": "Leda",
    "kore": "Kore",
    "alnilam": "Alnilam",
}


def clean_narration_text(text: str) -> str:
    """Strip formatting and labels so TTS speaks natural narration."""
    cleaned = str(text or "")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    cleaned = re.sub(r"_(.*?)_", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"(?im)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?im)\bscene\s*\d+\s*[:.-]?\s*", "", cleaned)
    cleaned = re.sub(r"(?im)\bnarration\s*:\s*", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*[-*+]\s+", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*\d+\.\s+", "", cleaned)
    cleaned = re.sub(r"[-*_]{3,}", " ", cleaned)
    cleaned = cleaned.replace("#", " ")
    cleaned = re.sub(r"[•●▪▶️🎬🎞️🌃🗡️🐉]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _fallback_music_asset_path() -> Path:
    return ROOT_DIR / "assets" / "fallback_music.wav"


def _copy_fallback_music(output_dir: Path) -> Path:
    fallback_path = _fallback_music_asset_path()
    if not fallback_path.exists():
        raise FileNotFoundError(f"Fallback music asset not found: {fallback_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    music_path = output_dir / "music.wav"
    shutil.copyfile(fallback_path, music_path)
    logger.warning("Using fallback music asset: %s", music_path)
    return music_path


def _scene_narration(scene: Any) -> str:
    if hasattr(scene, "narration"):
        raw = str(getattr(scene, "narration"))
    elif isinstance(scene, dict):
        raw = str(scene.get("narration", ""))
    else:
        raw = str(scene)
    return clean_narration_text(raw)


def _scene_narration_ssml(scene: Any) -> str:
    if hasattr(scene, "narration_ssml"):
        return str(getattr(scene, "narration_ssml") or "").strip()
    if isinstance(scene, dict):
        return str(scene.get("narration_ssml", "") or "").strip()
    return ""


def _strip_ssml_tags(ssml_text: str) -> str:
    without_breaks = re.sub(r"<break[^>]+/>", " ", ssml_text)
    without_tags = re.sub(r"</?[^>]+>", " ", without_breaks)
    return clean_narration_text(without_tags)


def _parse_ssml_sequence(ssml_text: str) -> List[tuple[str, str | float]]:
    normalized = re.sub(r"\s+", " ", str(ssml_text or "").strip())
    if not normalized:
        return []

    sequence: List[tuple[str, str | float]] = []
    pattern = re.compile(r'<break time="([0-9]+)ms"\s*/?>', re.IGNORECASE)
    cursor = 0
    for match in pattern.finditer(normalized):
        raw_text = normalized[cursor:match.start()]
        text = _strip_ssml_tags(raw_text)
        if text:
            sequence.append(("text", text))
        sequence.append(("break", int(match.group(1)) / 1000.0))
        cursor = match.end()

    tail_text = _strip_ssml_tags(normalized[cursor:])
    if tail_text:
        sequence.append(("text", tail_text))
    return sequence


def _scene_narrator_profile(scene: Any) -> str:
    if hasattr(scene, "narrator_profile"):
        return str(getattr(scene, "narrator_profile") or "").strip()
    if isinstance(scene, dict):
        return str(scene.get("narrator_profile", "") or "").strip()
    return ""


def _resolve_narrator_profile(profile_name: str, settings: Settings) -> NarratorProfile:
    key = str(profile_name or "").strip().lower()
    if key and key in NARRATOR_PROFILES:
        return NARRATOR_PROFILES[key]
    return NarratorProfile(
        voice_name=_normalize_tts_voice_name(settings.tts_voice_name),
        style_prompt=DEFAULT_TTS_STYLE,
    )


def _normalize_tts_voice_name(raw_name: str) -> str:
    key = str(raw_name or "").strip().lower()
    if key in TTS_VOICE_ALIASES:
        return TTS_VOICE_ALIASES[key]
    if "-" in key:
        suffix = key.split("-")[-1]
        if suffix in TTS_VOICE_ALIASES:
            return TTS_VOICE_ALIASES[suffix]
    normalized = str(raw_name or "").strip()
    return normalized or "Achernar"


def _extract_audio_bytes(response: Any) -> tuple[bytes, str]:
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            data = getattr(inline_data, "data", None)
            mime_type = str(getattr(inline_data, "mime_type", "") or "")
            if data:
                return data, mime_type
    raise ValueError("No audio bytes found in Gemini TTS response.")


def _pcm_to_wav_bytes(audio_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    return buffer.getvalue()


def _audio_bytes_to_wav(audio_bytes: bytes, mime_type: str, settings: Settings) -> bytes:
    normalized_mime = str(mime_type or "").lower()
    if "wav" in normalized_mime:
        return audio_bytes

    sample_rate = settings.audio_sample_rate
    rate_match = re.search(r"rate=(\d+)", normalized_mime)
    if rate_match:
        sample_rate = int(rate_match.group(1))

    channels = 1
    channel_match = re.search(r"channels=(\d+)", normalized_mime)
    if channel_match:
        channels = max(int(channel_match.group(1)), 1)

    return _pcm_to_wav_bytes(audio_bytes, sample_rate=sample_rate, channels=channels)


def _tts_prompt(text: str, profile: NarratorProfile) -> str:
    return (
        "Read the narration after NARRATION: exactly as written. "
        f"Delivery style: {profile.style_prompt} "
        "Honor punctuation and let pauses linger between short clauses. "
        "Do not add extra words, titles, or scene labels. "
        f"NARRATION: {text}"
    )


def _is_transient_tts_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return any(
        token in message
        for token in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "resource_exhausted",
            "timeout",
            "timed out",
            "deadline",
            "gateway timeout",
        )
    )


def _synthesize_speech(
    client: genai.Client,
    text: str,
    profile: NarratorProfile,
    settings: Settings,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=settings.tts_model,
                contents=_tts_prompt(text, profile),
                config=types.GenerateContentConfig(
                    response_modalities=[types.Modality.AUDIO],
                    speech_config=types.SpeechConfig(
                        language_code=settings.tts_language_code,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=_normalize_tts_voice_name(profile.voice_name),
                            )
                        ),
                    ),
                    http_options=types.HttpOptions(timeout=TTS_RESPONSE_TIMEOUT_MS),
                ),
            )
            audio_bytes, mime_type = _extract_audio_bytes(response)
            return _audio_bytes_to_wav(audio_bytes, mime_type, settings)
        except Exception as exc:
            if not _is_transient_tts_error(exc) or attempt == 2:
                raise
            last_error = exc
            wait_seconds = (2 ** attempt) + random.uniform(0, 0.75)
            logger.warning(
                "Transient TTS error for voice %s. Waiting %.1fs before retry %d/3. (%s)",
                profile.voice_name,
                wait_seconds,
                attempt + 1,
                exc,
            )
            time.sleep(wait_seconds)
    raise RuntimeError("TTS synthesis failed after retries") from last_error


def _synthesize_ssml_like_sequence(
    *,
    client: genai.Client,
    ssml_text: str,
    profile: NarratorProfile,
    settings: Settings,
    output_path: Path,
) -> Path:
    sequence = _parse_ssml_sequence(ssml_text)
    if not sequence:
        output_path.write_bytes(
            _synthesize_speech(
                client=client,
                text=_strip_ssml_tags(ssml_text),
                profile=profile,
                settings=settings,
            )
        )
        return output_path

    segment_paths: List[Path] = []
    for index, (kind, value) in enumerate(sequence, start=1):
        part_path = output_path.with_name(f"{output_path.stem}_part_{index:02d}.wav")
        if kind == "text":
            part_path.write_bytes(
                _synthesize_speech(
                    client=client,
                    text=str(value),
                    profile=profile,
                    settings=settings,
                )
            )
        else:
            _write_silence_wav(
                part_path,
                duration_seconds=max(float(value), 0.05),
                sample_rate=settings.audio_sample_rate,
            )
        segment_paths.append(part_path)

    return _concat_wav_files(segment_paths, output_path)


def _apply_tempo_adjustment(
    *,
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    rate: float,
) -> Path:
    if abs(rate - 1.0) < 0.01:
        shutil.copyfile(input_path, output_path)
        return output_path

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            f"atempo={rate:.3f}",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def _volume_shape_expression(
    windows: List[tuple[float, float]] | None,
    *,
    muted_value: float,
) -> str:
    if not windows:
        return "1"

    expression = "1"
    for start, end in sorted(windows):
        expression = f"if(between(t,{start:.3f},{end:.3f}),{muted_value:.3f},{expression})"
    return expression


def create_narration_audio(
    scenes: Iterable[Any],
    settings: Settings,
    output_dir: Path,
    intro_text: str | None = None,
    intro_duration_seconds: float = 0.0,
    intro_narrator_profile: str | None = None,
    scene_duration_seconds: float | None = None,
    scene_duration_map: dict[int, float] | None = None,
    total_duration_seconds: float | None = None,
    scene_pause_after: dict[int, float] | None = None,
    ending_silence_seconds: float = 0.0,
    scene_lead_in_map: dict[int, float] | None = None,
    scene_rate_map: dict[int, float] | None = None,
    preserve_intro_length: bool = True,
) -> Path:
    total_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_items = list(scenes)

    script_lines = [_scene_narration(scene).strip() for scene in scene_items]
    intro_line = clean_narration_text(intro_text or "")
    if not any(script_lines) and not intro_line:
        raise ValueError("No narration text available.")

    client = genai.Client(**vertex_client_kwargs(settings))
    scene_count = max(len(script_lines), 1)
    assembled_paths: List[Path] = []
    intro_duration_seconds = max(float(intro_duration_seconds or 0.0), 0.0)
    intro_target_duration = 0.0
    goodbye_scene_index = _goodbye_scene_index(len(script_lines))
    stillness_scene_index = _stillness_scene_index(len(script_lines))
    target_total_duration = max(float(total_duration_seconds or settings.film_duration_seconds), 1.0)
    pause_after_map = {int(k): float(v) for k, v in (scene_pause_after or {}).items()}
    lead_in_map = {int(k): float(v) for k, v in (scene_lead_in_map or {}).items()}
    rate_map = {int(k): float(v) for k, v in (scene_rate_map or {}).items()}
    duration_map = {int(k): max(float(v), 1.0) for k, v in (scene_duration_map or {}).items()}

    if intro_line and intro_duration_seconds > 0:
        intro_raw_path = output_dir / "narration_intro_raw.wav"
        intro_profile = _resolve_narrator_profile(intro_narrator_profile or "", settings)
        intro_start = time.perf_counter()
        try:
            intro_raw_path.write_bytes(
                _synthesize_speech(
                    client=client,
                    text=intro_line,
                    profile=intro_profile,
                    settings=settings,
                )
            )
            logger.info(
                "TIMING tts intro duration=%.2fs voice=%s",
                time.perf_counter() - intro_start,
                intro_profile.voice_name,
            )
        except Exception as exc:
            logger.warning("Intro narration synthesis failed, using silence. (%s)", exc)
            _write_silence_wav(
                intro_raw_path,
                duration_seconds=max(intro_duration_seconds, 0.25),
                sample_rate=settings.audio_sample_rate,
            )
        intro_target_duration = (
            max(intro_duration_seconds, _safe_wav_duration_seconds(intro_raw_path) + 0.25)
            if preserve_intro_length
            else intro_duration_seconds
        )
    elif intro_duration_seconds > 0:
        intro_target_duration = intro_duration_seconds

    remaining_duration = max(target_total_duration - intro_target_duration, 1.0)
    scene_duration = float(scene_duration_seconds or (remaining_duration / scene_count))

    if intro_target_duration > 0:
        intro_path = output_dir / "narration_intro.wav"
        if intro_line:
            (
                ffmpeg
                .output(
                    ffmpeg.input(str(intro_raw_path)).audio.filter("apad"),
                    str(intro_path),
                    ac=2,
                    ar=settings.audio_sample_rate,
                    t=intro_target_duration,
                )
                .overwrite_output()
                .run(quiet=True)
            )
        else:
            _write_silence_wav(intro_path, duration_seconds=intro_target_duration, sample_rate=settings.audio_sample_rate)
        assembled_paths.append(intro_path)

    for idx, line in enumerate(script_lines, start=1):
        raw_path = output_dir / f"narration_scene_{idx:02d}_raw.wav"
        adjusted_path = output_dir / f"narration_scene_{idx:02d}_adjusted.wav"
        padded_path = output_dir / f"narration_scene_{idx:02d}_padded.wav"
        target_scene_duration = duration_map.get(idx, scene_duration)
        scene_profile = _resolve_narrator_profile(
            _scene_narrator_profile(scene_items[idx - 1]),
            settings,
        )
        ssml_text = _scene_narration_ssml(scene_items[idx - 1])

        if line:
            synth_start = time.perf_counter()
            try:
                _synthesize_ssml_like_sequence(
                    client=client,
                    ssml_text=ssml_text or line,
                    profile=scene_profile,
                    settings=settings,
                    output_path=raw_path,
                )
                logger.info(
                    "TIMING tts scene=%d duration=%.2fs voice=%s",
                    idx,
                    time.perf_counter() - synth_start,
                    scene_profile.voice_name,
                )
            except Exception as exc:
                logger.warning(
                    "Narration synthesis failed for scene %d, using silence. (%s)",
                    idx,
                    exc,
                )
                _write_silence_wav(
                    raw_path,
                    duration_seconds=max(min(target_scene_duration, 0.5), 0.1),
                    sample_rate=settings.audio_sample_rate,
                )
            source_for_padding = _apply_tempo_adjustment(
                input_path=raw_path,
                output_path=adjusted_path,
                sample_rate=settings.audio_sample_rate,
                rate=max(rate_map.get(idx, 1.0), 0.5),
            )
            speech_duration = _safe_wav_duration_seconds(source_for_padding)
            if speech_duration > 5.5:
                logger.warning(
                    "Scene %d narration too long: %.1fs, may overlap scene timing.",
                    idx,
                    speech_duration,
                )
        else:
            _write_silence_wav(raw_path, duration_seconds=target_scene_duration, sample_rate=settings.audio_sample_rate)
            shutil.copyfile(raw_path, adjusted_path)
            source_for_padding = adjusted_path
            speech_duration = 0.0

        if line:
            pre_silence = max(lead_in_map.get(idx, 1.5), 1.5)
            post_silence = max(0.5, target_scene_duration - pre_silence - speech_duration)
        else:
            pre_silence = 0.0
            post_silence = 0.0
        _build_padded_narration_clip(
            input_path=source_for_padding,
            output_path=padded_path,
            sample_rate=settings.audio_sample_rate,
            target_duration=target_scene_duration,
            pre_silence_seconds=pre_silence,
            post_silence_seconds=post_silence,
        )
        assembled_paths.append(padded_path)
        pause_seconds = max(pause_after_map.get(idx, 0.0), 0.0)
        if pause_seconds > 0:
            pause_path = output_dir / f"narration_pause_after_{idx:02d}.wav"
            _write_silence_wav(pause_path, duration_seconds=pause_seconds, sample_rate=settings.audio_sample_rate)
            assembled_paths.append(pause_path)

    if ending_silence_seconds > 0:
        ending_silence_path = output_dir / "narration_end_silence.wav"
        _write_silence_wav(
            ending_silence_path,
            duration_seconds=ending_silence_seconds,
            sample_rate=settings.audio_sample_rate,
        )
        assembled_paths.append(ending_silence_path)

    narration_assembled_path = output_dir / "narration_assembled.wav"
    _concat_wav_files(assembled_paths, narration_assembled_path)

    output_path = output_dir / "narration.wav"
    _pad_audio_to_duration(
        input_path=narration_assembled_path,
        output_path=output_path,
        target_duration=target_total_duration,
        sample_rate=settings.audio_sample_rate,
    )
    logger.info(
        "TIMING tts complete scenes=%d intro_duration=%.2fs scene_duration=%.2fs total=%.2fs output=%s",
        scene_count,
        intro_target_duration,
        scene_duration,
        time.perf_counter() - total_start,
        output_path,
    )
    timing_path = output_dir / "narration_timing.json"
    timing_path.write_text(
        json.dumps(
            {
                "intro_duration_seconds": intro_target_duration,
                "scene_duration_seconds": scene_duration,
                "scene_duration_map": duration_map,
                "total_duration_seconds": target_total_duration,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def _extract_music_bytes(payload: dict) -> bytes:
    predictions = payload.get("predictions") or []
    if not predictions:
        raise ValueError("Lyria response has no predictions.")

    candidate = predictions[0]
    if isinstance(candidate, str):
        return base64.b64decode(candidate)

    keys_to_try = [
        "bytesBase64Encoded",
        "audioBytes",
        "audio_base64",
        "content",
    ]

    for key in keys_to_try:
        value = candidate.get(key) if isinstance(candidate, dict) else None
        if isinstance(value, str) and value.strip():
            return base64.b64decode(value)

    audio_obj = candidate.get("audio") if isinstance(candidate, dict) else None
    if isinstance(audio_obj, dict):
        for key in keys_to_try:
            value = audio_obj.get(key)
            if isinstance(value, str) and value.strip():
                return base64.b64decode(value)

    raise ValueError("Lyria audio bytes were not found in prediction payload.")


def _scene_music_moods(scenes: Iterable[Any]) -> List[str]:
    moods: List[str] = []
    for scene in scenes:
        if hasattr(scene, "music_mood"):
            moods.append(str(getattr(scene, "music_mood")))
        elif isinstance(scene, dict):
            moods.append(str(scene.get("music_mood", "")))
    return moods


def _lyria_endpoint(settings: Settings) -> str:
    return (
        f"https://{settings.gcp_location}-aiplatform.googleapis.com/v1/projects/"
        f"{settings.gcp_project_id}/locations/{settings.gcp_location}/publishers/google/models/"
        f"{settings.music_model}:predict"
    )


def _lyria_credentials() -> tuple[Any, Any]:
    return vertex_credentials()


def _generate_single_music_clip(
    *,
    prompt: str,
    settings: Settings,
    output_path: Path,
    duration_seconds: int,
    negative_prompt: str = "",
) -> Path:
    if not settings.gcp_project_id:
        raise ValueError("GCP_PROJECT_ID is required for Lyria generation.")

    credentials, _ = _lyria_credentials()
    final_prompt = str(prompt or "").strip()
    if negative_prompt.strip():
        final_prompt = f"{final_prompt} Avoid: {negative_prompt.strip()}."

    payload = {
        "instances": [
            {
                "prompt": final_prompt,
                "duration_seconds": int(duration_seconds),
            }
        ],
        "parameters": {
            "sample_rate_hertz": settings.audio_sample_rate,
        },
    }

    response = requests.post(
        _lyria_endpoint(settings),
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_extract_music_bytes(response.json()))
    return output_path


def _concat_with_crossfade(
    segment_paths: List[Path],
    output_path: Path,
    *,
    crossfade_seconds: float,
    sample_rate: int,
) -> Path:
    existing_segments = [path for path in segment_paths if path.exists()]
    if not existing_segments:
        raise ValueError("No music segments available to concatenate.")
    if len(existing_segments) == 1:
        shutil.copyfile(existing_segments[0], output_path)
        return output_path

    cmd = ["ffmpeg", "-y"]
    for path in existing_segments:
        cmd.extend(["-i", str(path)])

    filter_steps: List[str] = []
    previous_label = "[0:a]"
    for index in range(1, len(existing_segments)):
        next_label = f"[a{index}]"
        filter_steps.append(
            f"{previous_label}[{index}:a]acrossfade=d={crossfade_seconds}:c1=tri:c2=tri{next_label}"
        )
        previous_label = next_label

    cmd.extend(
        [
            "-filter_complex",
            ";".join(filter_steps),
            "-map",
            previous_label,
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            str(output_path),
        ]
    )
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def create_music_audio(
    scenes: Iterable[Any],
    settings: Settings,
    output_dir: Path,
    music_prompt_override: str | None = None,
    music_segments: List[dict[str, Any]] | None = None,
) -> Path:
    total_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        scene_items = list(scenes)
        moods = _scene_music_moods(scene_items)
        segment_items = [segment for segment in (music_segments or []) if isinstance(segment, dict)]

        if segment_items:
            logger.info("Generating %d music segments in parallel.", len(segment_items))
            segment_paths: List[Path] = []

            def _render_segment(index: int, segment: dict[str, Any]) -> Path:
                seg_path = output_dir / f"music_seg_{index:02d}.wav"
                try:
                    api_start = time.perf_counter()
                    result_path = _generate_single_music_clip(
                        prompt=str(segment.get("prompt", "")).strip(),
                        negative_prompt=str(segment.get("negative_prompt", "")).strip(),
                        duration_seconds=max(int(settings.film_duration_seconds / max(len(segment_items), 1)), 1),
                        output_path=seg_path,
                        settings=settings,
                    )
                    logger.info(
                        "TIMING lyria segment=%d duration=%.2fs output=%s",
                        index,
                        time.perf_counter() - api_start,
                        result_path,
                    )
                    return result_path
                except Exception as exc:
                    logger.warning("Music segment %d failed, using silence. (%s)", index, exc)
                    _write_silence_wav(
                        seg_path,
                        duration_seconds=max(int(settings.film_duration_seconds / max(len(segment_items), 1)), 1),
                        sample_rate=settings.audio_sample_rate,
                    )
                    return seg_path

            with ThreadPoolExecutor(max_workers=min(4, max(len(segment_items), 1))) as executor:
                futures = [
                    executor.submit(_render_segment, index, segment)
                    for index, segment in enumerate(segment_items, start=1)
                ]
                for future in futures:
                    segment_paths.append(future.result())

            music_path = _concat_with_crossfade(
                segment_paths,
                output_dir / "music.wav",
                crossfade_seconds=MUSIC_SEGMENT_CROSSFADE_SECONDS,
                sample_rate=settings.audio_sample_rate,
            )
            logger.info(
                "TIMING lyria segmented complete duration=%.2fs output=%s",
                time.perf_counter() - total_start,
                music_path,
            )
            return music_path

        if not settings.gcp_project_id:
            raise ValueError("GCP_PROJECT_ID is required for Lyria generation.")

        music_prompt = music_prompt_override or (
            "Compose cinematic instrumental score for a 2-minute short film. "
            f"Mood arc by scene: {', '.join([m for m in moods if m])}. "
            "No vocals, modern film score, emotional continuity across scenes."
        )

        api_start = time.perf_counter()
        music_path = _generate_single_music_clip(
            prompt=music_prompt,
            settings=settings,
            output_path=output_dir / "music.wav",
            duration_seconds=settings.film_duration_seconds,
        )
        logger.info(
            "TIMING lyria api_call duration=%.2fs endpoint=%s",
            time.perf_counter() - api_start,
            _lyria_endpoint(settings),
        )
        logger.info(
            "TIMING lyria complete duration=%.2fs output=%s",
            time.perf_counter() - total_start,
            music_path,
        )
        return music_path
    except Exception as exc:
        logger.warning("Lyria generation failed, attempting fallback music. (%s)", exc)
        music_path = _copy_fallback_music(output_dir)
        logger.info(
            "TIMING lyria complete duration=%.2fs output=%s fallback=true",
            time.perf_counter() - total_start,
            music_path,
        )
        return music_path


def mix_audio_tracks(
    narration_path: Path,
    music_path: Path,
    settings: Settings,
    output_dir: Path,
    ambient_path: Path | None = None,
    target_duration_seconds: float | None = None,
    silence_windows: List[tuple[float, float]] | None = None,
) -> Path:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    mixed_path = output_dir / "mixed_audio.wav"
    total_duration = float(target_duration_seconds or settings.film_duration_seconds)
    fade_out_start = max(total_duration - FINAL_AUDIO_FADE_OUT_SECONDS, 0.0)
    music_shape_expr = _volume_shape_expression(silence_windows, muted_value=MUSIC_PAUSE_VOLUME)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(narration_path),
        "-stream_loop",
        "-1",
        "-i",
        str(music_path),
        "-filter_complex",
        (
            f"[0:a]aformat=sample_fmts=fltp:sample_rates={settings.audio_sample_rate}:channel_layouts=stereo,"
            f"highpass=f=120,acompressor=threshold=0.05:ratio=3:attack=5:release=50,"
            f"volume=1.2,atrim=0:{total_duration}[voice];"
            f"[1:a]aformat=sample_fmts=fltp:sample_rates={settings.audio_sample_rate}:channel_layouts=stereo,"
            f"highpass=f=100,lowpass=f=6000,volume=0.18,atrim=0:{total_duration},volume='{music_shape_expr}'[mus_raw];"
            f"[mus_raw][voice]sidechaincompress=threshold=0.02:ratio=6:attack=40:release=350:level_sc=0.5[mus];"
            f"[voice][mus]amix=inputs=2:duration=first:dropout_transition=3,"
            f"highpass=f=60,alimiter=limit=0.9,"
            f"afade=t=in:st=0:d={FINAL_AUDIO_FADE_IN_SECONDS},"
            f"afade=t=out:st={fade_out_start}:d={FINAL_AUDIO_FADE_OUT_SECONDS}[out]"
        ),
        "-map",
        "[out]",
        "-ac",
        "2",
        "-ar",
        str(settings.audio_sample_rate),
        "-t",
        str(total_duration),
        str(mixed_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="ignore").strip().splitlines()
        preview = stderr[-1] if stderr else "unknown ffmpeg error"
        logger.warning("Sidechain mix failed, falling back to flat mix. (%s)", preview)
        fallback_inputs = [
            "ffmpeg",
            "-y",
            "-i",
            str(narration_path),
            "-stream_loop",
            "-1",
            "-i",
            str(music_path),
        ]
        fallback_filter = (
            f"[0:a]aformat=sample_fmts=fltp:sample_rates={settings.audio_sample_rate}:channel_layouts=stereo,"
            f"highpass=f=120,volume=1.1[voice];"
            f"[1:a]aformat=sample_fmts=fltp:sample_rates={settings.audio_sample_rate}:channel_layouts=stereo,"
            f"highpass=f=100,lowpass=f=6000,volume=0.16,volume='{music_shape_expr}'[music];"
            f"[voice][music]amix=inputs=2:duration=first:dropout_transition=3,"
            f"highpass=f=60,alimiter=limit=0.9,"
            f"afade=t=in:st=0:d={FINAL_AUDIO_FADE_IN_SECONDS},"
            f"afade=t=out:st={fade_out_start}:d={FINAL_AUDIO_FADE_OUT_SECONDS}[out]"
        )
        subprocess.run(
            fallback_inputs + [
                "-filter_complex",
                fallback_filter,
                "-map",
                "[out]",
                "-ac",
                "2",
                "-ar",
                str(settings.audio_sample_rate),
                "-t",
                str(total_duration),
                str(mixed_path),
            ],
            check=True,
            capture_output=True,
        )

    logger.info(
        "TIMING ffmpeg audio_mix duration=%.2fs output=%s",
        time.perf_counter() - start,
        mixed_path,
    )

    return mixed_path


def generate_audio_bundle(
    scenes: Iterable[Any],
    settings: Settings,
    output_dir: Path,
    music_segments: List[dict[str, Any]] | None = None,
) -> AudioAssets:
    narration = create_narration_audio(scenes=scenes, settings=settings, output_dir=output_dir)
    music = create_music_audio(
        scenes=scenes,
        settings=settings,
        output_dir=output_dir,
        music_segments=music_segments,
    )
    mixed = mix_audio_tracks(
        narration_path=narration,
        music_path=music,
        settings=settings,
        output_dir=output_dir,
    )
    return AudioAssets(narration_path=narration, music_path=music, mixed_path=mixed)


def _extract_interleaved_narrations(interleaved_response: Any) -> List[str]:
    narrations: List[str] = []
    candidates = getattr(interleaved_response, "candidates", []) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            cleaned = clean_narration_text(str(text)) if text else ""
            if cleaned:
                narrations.append(cleaned)
    return narrations


def _safe_wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if frame_rate <= 0:
                return 0.0
            return frame_count / frame_rate
    except Exception:
        return 0.0


def load_narration_timing(output_dir: Path) -> NarrationTiming:
    timing_path = output_dir / "narration_timing.json"
    if not timing_path.exists():
        return NarrationTiming(0.0, 0.0, 0.0)
    try:
        payload = json.loads(timing_path.read_text(encoding="utf-8"))
        return NarrationTiming(
            intro_duration_seconds=float(payload.get("intro_duration_seconds", 0.0) or 0.0),
            scene_duration_seconds=float(payload.get("scene_duration_seconds", 0.0) or 0.0),
            total_duration_seconds=float(payload.get("total_duration_seconds", 0.0) or 0.0),
        )
    except Exception:
        return NarrationTiming(0.0, 0.0, 0.0)


def _write_silence_wav(path: Path, duration_seconds: float, sample_rate: int) -> None:
    frame_count = max(int(duration_seconds * sample_rate), 1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


def _goodbye_scene_index(scene_count: int) -> int:
    if scene_count <= 0:
        return 1
    return max(1, min(scene_count, scene_count - 3))


def _stillness_scene_index(scene_count: int) -> int:
    if scene_count <= 0:
        return 1
    return max(1, min(scene_count, scene_count - 1))


def _build_padded_narration_clip(
    *,
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    target_duration: float,
    pre_silence_seconds: float,
    post_silence_seconds: float,
) -> Path:
    filter_chain = (
        f"adelay={int(pre_silence_seconds * 1000)}:all=1,"
        f"apad=pad_dur={post_silence_seconds}"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            filter_chain,
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-t",
            f"{target_duration:.3f}",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def _concat_wav_files(paths: List[Path], output_path: Path) -> Path:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        raise ValueError("No audio segments available to concatenate.")
    if len(existing_paths) == 1:
        shutil.copyfile(existing_paths[0], output_path)
        return output_path

    concat_list_path = output_path.with_name(f"{output_path.stem}_concat.txt")
    concat_list_path.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in existing_paths),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def _pad_audio_to_duration(
    *,
    input_path: Path,
    output_path: Path,
    target_duration: float,
    sample_rate: int,
) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            "apad",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-t",
            f"{target_duration:.3f}",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def generate_full_audio(
    interleaved_or_scenes: Any,
    settings: Settings,
    output_dir: Path,
    music_prompts: List[str] | None = None,
    music_segments: List[dict[str, Any]] | None = None,
) -> dict:
    """
    Generate narration, music, and mixed audio from either:
    - interleaved Gemini response object (text/image parts), or
    - iterable scene objects/dicts with narration/music_mood fields.
    """
    scenes: List[Any]
    if hasattr(interleaved_or_scenes, "candidates"):
        narrations = _extract_interleaved_narrations(interleaved_or_scenes)
        if not narrations:
            raise ValueError("No narration text found in interleaved response.")

        prompts = music_prompts or ["cinematic emotional score"]
        scenes = []
        for idx, narration in enumerate(narrations):
            scenes.append(
                {
                    "narration": narration,
                    "music_mood": prompts[min(idx, len(prompts) - 1)],
                }
            )
    else:
        scenes = list(interleaved_or_scenes)
        if not scenes:
            raise ValueError("No scenes provided for audio generation.")

    narration_path = create_narration_audio(scenes=scenes, settings=settings, output_dir=output_dir)
    music_path = create_music_audio(
        scenes=scenes,
        settings=settings,
        output_dir=output_dir,
        music_segments=music_segments,
    )
    mixed_audio_path = mix_audio_tracks(
        narration_path=narration_path,
        music_path=music_path,
        settings=settings,
        output_dir=output_dir,
    )

    return {
        "narration_path": str(narration_path),
        "music_path": str(music_path),
        "mixed_audio_path": str(mixed_audio_path),
        "total_duration": _safe_wav_duration_seconds(mixed_audio_path),
    }
