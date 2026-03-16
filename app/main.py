"""CineAI Director — 3-Beat Conversation Engine.

Handles both preset clicks and custom story input through a universal
3-beat conversation framework with real-time progress streaming.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

from app.health import run_startup_preflight
from app.cineai_agent.agent import (
    cineai_director,
    get_session_state,
    initialize_session_state,
    new_session_id,
    set_current_session,
    state_store,
)
from config.presets import PRESETS, get_preset
from config.settings import get_settings
from generation.audio import (
    create_music_audio,
    create_narration_audio,
    load_narration_timing,
    mix_audio_tracks,
)
from generation.cache import copy_cached_asset
from generation.export import generate_storyboard_pdf
from generation.extraction import (
    classify_veo_safety,
    enforce_character_bible,
    extract_conflict,
    extract_emotional_setup,
    extract_ending,
    extract_residual_emotion,
    extract_story_setup,
    extract_story_turn,
    generate_creative_direction,
    generate_film_blueprint,
    generate_emotional_script,
    generate_script_from_outline,
    generate_dynamic_script,
    generate_music_brief,
    generate_opening_line,
    generate_story_outline,
)
from generation.scenes import (
    INTERLEAVED_SCENE_MAX_WORKERS,
    GeneratedScene,
    build_interleaved_batch_specs,
    generate_single_scene,
    generate_scene_batch_interleaved,
)
from generation.story import (
    INTERLEAVED_SCENE_COUNT,
    generate_hero_video,
    generate_interleaved_story,
)
from generation.veo_scenes import generate_all_veo_scenes
from generation.video import assemble_video, create_black_card, create_ken_burns_clip

logger = logging.getLogger("uvicorn.error")
settings = get_settings()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
GENERATED_DIR = STATIC_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
DEMO_PRESETS_DIR = STATIC_DIR / "demo_presets"
DEMO_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
run_startup_preflight(settings=settings, writable_dirs=[GENERATED_DIR])

app = FastAPI(title="CineAI")


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Dict[str, Any]) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")

session_service = InMemorySessionService()
runner = Runner(app_name="cineai", agent=cineai_director, session_service=session_service)
ADK_APP_NAME = "cineai"
ADK_USER_ID = "browser"
BACKGROUND_MUSIC_TASKS: Dict[str, asyncio.Task[Path]] = {}
BACKGROUND_CHARACTER_TASKS: Dict[str, asyncio.Task[Path]] = {}
OPENING_CARD_SECONDS = 3.0
END_CARD_SECONDS = 4.0
CINEMATIC_SCENE_SECONDS = 8.0
MAJOR_STAGE_STAGGER_SECONDS = 2.0
STILL_FALLBACK_STAGGER_SECONDS = 20.0
SCENE_TRANSITIONS_AFTER: Dict[int, float] = {}
SCENE_BLACK_PAUSES = {5: 3.0, 6: 2.0, 7: 2.0}
SACRIFICE_SCENE_LEAD_INS = {
    1: 1.5,
    2: 2.0,
    3: 2.0,
    4: 1.5,
    5: 3.5,
    6: 2.5,
    7: 4.0,
    8: 1.5,
}
SACRIFICE_SCENE_RATES = {
    1: 0.86,
    2: 0.88,
    3: 0.84,
    4: 0.82,
    5: 0.82,
    6: 0.75,
    7: 0.70,
    8: 0.86,
}

PRESET_LABELS = {
    "sacrifice": "The Father's Sacrifice",
    "blue_meadow": "The Great Green Shadows",
    "escape": "The Escape",
    "discovery": "The Discovery",
}
DEFAULT_MUSIC_PREFERENCE = "Epic and orchestral"
ENABLE_VEO = False

# ── Opening greeting ─────────────────────────────────────────────────

OPENING_MESSAGE = (
    "Tell me about someone. Real or imagined. Who are they, and why do they matter?\n\n"
    "Pick a preset to start from a story frame, or describe the person at the center of yours."
)

GENERIC_CHARACTER_SEEDS = {
    "",
    "i wanna",
    "i want",
    "something",
    "story",
    "a story",
    "anything",
    "someone",
    "something cool",
}


def _planned_film_runtime_seconds(
    *,
    scene_durations: List[float],
    opening_card_seconds: float = OPENING_CARD_SECONDS,
    ending_card_seconds: float = END_CARD_SECONDS,
    pause_after: Dict[int, float] | None = None,
    transition_after: Dict[int, float] | None = None,
) -> float:
    pause_total = sum(float(value) for value in (pause_after or {}).values())
    transition_total = sum(float(value) for value in (transition_after or {}).values())
    return opening_card_seconds + sum(scene_durations) + transition_total + pause_total + ending_card_seconds


def _planned_silence_windows(
    *,
    scene_durations: List[float],
    opening_card_seconds: float = OPENING_CARD_SECONDS,
    pause_after: Dict[int, float] | None = None,
    transition_after: Dict[int, float] | None = None,
) -> List[tuple[float, float]]:
    windows: List[tuple[float, float]] = []
    elapsed = opening_card_seconds
    pauses = pause_after or {}
    transitions = transition_after or {}
    for index, scene_duration in enumerate(scene_durations, start=1):
        elapsed += float(scene_duration)
        elapsed += float(transitions.get(index, 0.0))
        pause_duration = float(pauses.get(index, 0.0))
        if pause_duration > 0:
            windows.append((elapsed, elapsed + pause_duration))
            elapsed += pause_duration
    return windows


@app.get("/")
async def index() -> FileResponse:
    response = FileResponse(INDEX_FILE)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ── WebSocket Helpers ────────────────────────────────────────────────


async def _send_progress(ws: WebSocket, message: str, percent: int) -> None:
    try:
        await ws.send_json({"type": "progress", "message": message, "percent": percent})
    except Exception:
        pass


async def _send_assistant(ws: WebSocket, text: str, **extra: Any) -> None:
    try:
        await ws.send_json({"type": "assistant", "content": text, "text": text, **extra})
    except Exception:
        pass


async def _send_scene(ws: WebSocket, session_id: str, scene: GeneratedScene) -> None:
    url = f"/static/generated/{session_id}/scenes/scene_{scene.index:02d}.png"
    try:
        await ws.send_json({
            "type": "scene_generated",
            "scene_index": scene.index,
            "scene_url": url,
            "narration": scene.narration,
        })
    except Exception:
        pass


async def _send_scene_payload(ws: WebSocket, scene_index: int, scene_url: str, narration: str) -> None:
    try:
        await ws.send_json({
            "type": "scene_generated",
            "scene_index": scene_index,
            "scene_url": scene_url,
            "narration": narration,
        })
    except Exception:
        pass


async def _send_character(ws: WebSocket, session_id: str) -> None:
    url = f"/static/generated/{session_id}/character/character_reference.png"
    try:
        await ws.send_json({"type": "character_generated", "character_url": url})
    except Exception:
        pass


async def _send_character_url(ws: WebSocket, character_url: str) -> None:
    try:
        await ws.send_json({"type": "character_generated", "character_url": character_url})
    except Exception:
        pass


async def _send_audio_ready(ws: WebSocket, audio_url: str) -> None:
    try:
        await ws.send_json({"type": "audio_ready", "audio_url": audio_url})
    except Exception:
        pass


def _story_title(story_context: Dict[str, Any]) -> str:
    title = str(story_context.get("title", "")).strip()
    if title:
        return title

    preset_name = str(story_context.get("preset_name", "")).strip().lower()
    if preset_name in PRESETS:
        preset_title = str(PRESETS[preset_name].get("title", "")).strip()
        if preset_title:
            return preset_title
    if preset_name in PRESET_LABELS:
        return PRESET_LABELS[preset_name]

    setting = str(story_context.get("setting", "")).strip()
    if setting:
        title = setting.split(",")[0].strip()
        return title[:48] if title else "Untold World"
    return "Untold World"


def _script_music_moods(script: List[Dict[str, Any]]) -> List[str]:
    return [
        str(scene.get("music_mood", "")).strip()
        for scene in script
        if isinstance(scene, dict) and str(scene.get("music_mood", "")).strip()
    ]


def _resolve_music_preference(user_input: str, fallback: str = DEFAULT_MUSIC_PREFERENCE) -> str:
    text = str(user_input or "").lower()
    if "mythic" in text or "ancient" in text:
        return "Mythic and ancient"
    if "dark" in text or "tense" in text:
        return "Dark and tense"
    if "bittersweet" in text or "reflective" in text:
        return "Bittersweet and reflective"
    if "tender" in text or "tragic" in text:
        return "Tender and tragic"
    if "epic" in text or "orchestral" in text:
        return "Epic and orchestral"
    return fallback or DEFAULT_MUSIC_PREFERENCE


def _story_preview_palette(story_context: Dict[str, Any]) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    mood = " ".join(
        [
            str(story_context.get("residual_feeling", "")),
            str(story_context.get("visual_style", "")),
            str(story_context.get("world", "")),
        ]
    ).lower()
    if any(word in mood for word in ("tragic", "grief", "mourning", "ache", "ash", "smoke")):
        return (36, 30, 42), (112, 84, 74), (246, 224, 196)
    if any(word in mood for word in ("cyber", "neon", "escape", "orbital", "space")):
        return (15, 24, 45), (39, 102, 185), (205, 236, 255)
    if any(word in mood for word in ("meadow", "grass", "nature", "wonder", "jungle", "observatory")):
        return (22, 52, 36), (118, 160, 92), (240, 243, 225)
    return (28, 30, 44), (108, 90, 146), (241, 235, 226)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _render_story_preview_card(story_context: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 675
    top, bottom, accent = _story_preview_palette(story_context)
    image = Image.new("RGB", (width, height), top)
    draw = ImageDraw.Draw(image)

    for y in range(height):
        blend = y / max(height - 1, 1)
        color = tuple(int(top[i] * (1 - blend) + bottom[i] * blend) for i in range(3))
        draw.line((0, y, width, y), fill=color)

    draw.rounded_rectangle((48, 48, width - 48, height - 48), radius=28, outline=(220, 214, 205), width=2)
    draw.rounded_rectangle((72, 72, width - 72, height - 72), radius=24, fill=(18, 20, 28))

    title_font = _load_font(44)
    label_font = _load_font(20)
    body_font = _load_font(28)

    title = _story_title(story_context)
    seeds = [
        ("Who", str(story_context.get("character_essence", story_context.get("character", ""))).strip()),
        ("Turn", str(story_context.get("the_turn", story_context.get("inciting_incident", ""))).strip()),
        ("Feeling", str(story_context.get("residual_feeling", story_context.get("resolution", ""))).strip()),
    ]

    draw.text((96, 96), title, font=title_font, fill=(255, 248, 236))
    draw.text((96, 152), "Story Card", font=label_font, fill=accent)

    y = 220
    for label, value in seeds:
        if not value:
            continue
        draw.text((96, y), label.upper(), font=label_font, fill=accent)
        y += 28
        wrapped = textwrap.wrap(value, width=42)[:3]
        for line in wrapped:
            draw.text((96, y), line, font=body_font, fill=(245, 239, 232))
            y += 36
        y += 20

    footer = "Generated from story seeds. Veo scenes use text prompts only."
    draw.text((96, height - 88), footer, font=label_font, fill=(215, 208, 198))
    image.save(output_path, format="PNG")
    return output_path


async def _generate_music_background(
    session_id: str,
    story_context: Dict[str, Any],
    script: List[Dict[str, Any]],
    render_settings: Any | None = None,
) -> Path:
    audio_dir = STATIC_DIR / "generated" / session_id / "audio"
    active_settings = render_settings or settings
    preset_name = str(story_context.get("preset_name", "")).strip().lower()
    music_segments = story_context.get("music_segments")
    if preset_name and not music_segments:
        cached_music_path = copy_cached_asset(
            STATIC_DIR,
            preset_name,
            "music",
            audio_dir / "music.wav",
        )
        if cached_music_path is not None:
            logger.info("Using cached preset music for %s: %s", preset_name, cached_music_path)
            return cached_music_path

    act_moods = _script_music_moods(script)
    music_brief = None
    if not music_segments:
        music_brief = await asyncio.to_thread(generate_music_brief, dict(story_context), act_moods, active_settings)
        logger.info("Music brief ready for session %s: %s", session_id, music_brief)
    return await asyncio.to_thread(
        create_music_audio,
        scenes=script,
        settings=active_settings,
        output_dir=audio_dir,
        music_prompt_override=music_brief,
        music_segments=music_segments if isinstance(music_segments, list) else None,
    )


def _start_background_music(session_id: str, story_context: Dict[str, Any], script: List[Dict[str, Any]]) -> None:
    existing_task = BACKGROUND_MUSIC_TASKS.get(session_id)
    if existing_task and not existing_task.done():
        return
    BACKGROUND_MUSIC_TASKS[session_id] = asyncio.create_task(
        _generate_music_background(session_id, dict(story_context), [dict(scene) for scene in script])
    )


async def _generate_character_background(
    ws: WebSocket,
    session_id: str,
    story_context: Dict[str, Any],
) -> Path:
    base = STATIC_DIR / "generated" / session_id
    await _send_progress(ws, "Preparing story card…", 10)
    character_path = await asyncio.to_thread(
        _render_story_preview_card,
        story_context,
        base / "character" / "character_reference.png",
    )

    state = state_store.load(session_id)
    assets = dict(state.get("generated_assets", {}))
    assets["character_ref"] = str(character_path)
    state_store.save(session_id, {
        "story_context": story_context,
        "generated_assets": assets,
    })
    await _send_character(ws, session_id)
    await _send_progress(ws, "Story card ready ✓", 25)
    return Path(character_path)


def _start_background_character(
    ws: WebSocket,
    session_id: str,
    story_context: Dict[str, Any],
) -> None:
    existing_task = BACKGROUND_CHARACTER_TASKS.get(session_id)
    if existing_task and not existing_task.done():
        return
    BACKGROUND_CHARACTER_TASKS[session_id] = asyncio.create_task(
        _generate_character_background(ws, session_id, dict(story_context))
    )


def _upload_video(local_path: Path, session_id: str) -> str:
    if not settings.gcs_bucket:
        return ""
    try:
        from google.cloud import storage
        client = storage.Client(project=settings.gcp_project_id or None)
        blob = client.bucket(settings.gcs_bucket).blob(f"{session_id}/final_film.mp4")
        blob.upload_from_filename(str(local_path), content_type="video/mp4")
        return f"gs://{settings.gcs_bucket}/{session_id}/final_film.mp4"
    except Exception:
        return ""


def _versioned_static_url(public_path: str, file_path: Path) -> str:
    try:
        version = int(file_path.stat().st_mtime_ns)
    except Exception:
        return public_path
    separator = "&" if "?" in public_path else "?"
    return f"{public_path}{separator}v={version}"


def _demo_preset_assets(preset_name: str) -> Dict[str, Any] | None:
    preset = get_preset(preset_name)
    preset_dir = DEMO_PRESETS_DIR / preset_name
    character_path = preset_dir / "character" / "character_reference.png"
    video_path = preset_dir / "video" / "final_film.mp4"
    scene_paths = [
        preset_dir / "scenes" / f"scene_{index:02d}.png"
        for index in range(1, settings.scene_count + 1)
    ]
    if not character_path.exists() or not video_path.exists() or not all(path.exists() for path in scene_paths):
        return None

    manifest_path = preset_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    scene_narrations = manifest.get("scene_narrations")
    if not isinstance(scene_narrations, list) or len(scene_narrations) != settings.scene_count:
        scene_narrations = [""] * settings.scene_count

    return {
        "title": str(manifest.get("title", preset.get("title", PRESET_LABELS.get(preset_name, preset_name)))),
        "character_url": _versioned_static_url(
            f"/static/demo_presets/{preset_name}/character/character_reference.png",
            character_path,
        ),
        "video_url": _versioned_static_url(
            f"/static/demo_presets/{preset_name}/video/final_film.mp4",
            video_path,
        ),
        "storyboard_url": _versioned_static_url(
            str(manifest.get("storyboard_url", f"/static/demo_presets/{preset_name}/storyboard/storyboard.pdf")),
            preset_dir / "storyboard" / "storyboard.pdf",
        ),
        "scene_urls": [
            _versioned_static_url(
                f"/static/demo_presets/{preset_name}/scenes/scene_{index:02d}.png",
                preset_dir / "scenes" / f"scene_{index:02d}.png",
            )
            for index in range(1, settings.scene_count + 1)
        ],
        "scene_narrations": scene_narrations,
        "video_subtitle": str(manifest.get("video_subtitle", "Pre-rendered preset cut for demo playback.")),
    }


def _promote_preset_demo(
    *,
    preset_name: str,
    session_id: str,
    story_title: str,
    script: List[Dict[str, Any]],
    final_path: Path,
    storyboard_path: Path | None,
) -> Path:
    source_dir = GENERATED_DIR / session_id
    dest_dir = DEMO_PRESETS_DIR / preset_name
    scene_count = min(settings.scene_count, len(script))

    required_paths = [
        source_dir / "character" / "character_reference.png",
        final_path,
    ]
    required_paths.extend(
        source_dir / "scenes" / f"scene_{index:02d}.png"
        for index in range(1, scene_count + 1)
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot promote preset demo. Missing assets:\n- " + "\n- ".join(missing)
        )

    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    (dest_dir / "character").mkdir(parents=True, exist_ok=True)
    (dest_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (dest_dir / "video").mkdir(parents=True, exist_ok=True)
    (dest_dir / "storyboard").mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        source_dir / "character" / "character_reference.png",
        dest_dir / "character" / "character_reference.png",
    )
    shutil.copy2(final_path, dest_dir / "video" / "final_film.mp4")
    for index in range(1, scene_count + 1):
        shutil.copy2(
            source_dir / "scenes" / f"scene_{index:02d}.png",
            dest_dir / "scenes" / f"scene_{index:02d}.png",
        )

    copied_storyboard = None
    if storyboard_path and storyboard_path.exists():
        copied_storyboard = dest_dir / "storyboard" / storyboard_path.name
        shutil.copy2(storyboard_path, copied_storyboard)

    manifest = {
        "preset_name": preset_name,
        "source_session_id": session_id,
        "title": story_title,
        "scene_narrations": [
            str(scene.get("narration", "")).strip()
            for scene in script[:scene_count]
        ],
        "video_subtitle": "Pre-rendered preset cut for demo playback.",
        "storyboard_url": (
            f"/static/demo_presets/{preset_name}/storyboard/{copied_storyboard.name}"
            if copied_storyboard else f"/static/demo_presets/{preset_name}/storyboard/storyboard.pdf"
        ),
    }
    (dest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dest_dir


async def _serve_demo_preset(
    ws: WebSocket,
    session_id: str,
    preset_name: str,
    story_context: Dict[str, Any],
) -> bool:
    assets = _demo_preset_assets(preset_name)
    if not assets:
        return False

    state_store.save(session_id, {
        "current_beat": 6,
        "story_type": "preset_demo",
        "story_context": story_context,
        "generated_assets": {
            "character_ref": assets["character_url"],
            "scenes_demo": assets["scene_urls"],
        },
        "video_url": assets["video_url"],
        "storyboard_url": assets["storyboard_url"],
    })

    await ws.send_json({
        "type": "demo_preset_loaded",
        "title": assets["title"],
        "video_url": assets["video_url"],
        "storyboard_url": assets["storyboard_url"],
    })
    await _send_progress(ws, "Loading demo preset…", 100)
    await _send_character_url(ws, assets["character_url"])
    for scene_index, (scene_url, narration) in enumerate(
        zip(assets["scene_urls"], assets["scene_narrations"]),
        start=1,
    ):
        await _send_scene_payload(ws, scene_index, scene_url, narration)
    await ws.send_json({
        "type": "film_ready",
        "video_url": assets["video_url"],
        "storyboard_url": assets["storyboard_url"],
        "video_title": assets["title"],
        "video_subtitle": assets["video_subtitle"],
        "demo_preset": True,
    })
    await _send_assistant(
        ws,
        "Demo preset loaded instantly. This cut is pre-rendered for the demo. **Press play.**",
        film_ready=True,
        step=6,
        video_subtitle=assets["video_subtitle"],
        demo_preset=True,
    )
    return True


async def _render_interleaved_story(
    ws: WebSocket,
    session_id: str,
    story_ctx: Dict[str, Any],
    *,
    story_type: str,
) -> None:
    render_start = time.perf_counter()
    loop = asyncio.get_running_loop()
    session_dir = STATIC_DIR / "generated" / session_id
    scenes_dir = session_dir / "scenes"
    audio_dir = session_dir / "audio"
    video_dir = session_dir / "video"
    session_dir.mkdir(parents=True, exist_ok=True)
    story_title = _story_title(story_ctx)

    await _send_assistant(ws, "I have what I need. Let me make your story.", step=3)
    await _generate_character_background(ws, session_id, story_ctx)

    try:
        (session_dir / "story_context.json").write_text(
            json.dumps(story_ctx, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to persist story context for session %s: %s", session_id, exc)

    await _send_progress(ws, "Imagining the story world…", 30)
    streamed_scene_numbers: set[int] = set()

    def _scene_ready(scene: Dict[str, Any]) -> None:
        scene_number = int(scene.get("scene_number", 0) or 0)
        scene_path = Path(str(scene.get("image_path", "")))
        if scene_number <= 0 or not scene_path.exists():
            return
        streamed_scene_numbers.add(scene_number)
        scene_url = _versioned_static_url(
            f"/static/generated/{session_id}/scenes/{scene_path.name}",
            scene_path,
        )
        narration = str(scene.get("narration", "")).strip()
        asyncio.run_coroutine_threadsafe(
            _send_scene_payload(
                ws,
                scene_index=scene_number,
                scene_url=scene_url,
                narration=narration,
            ),
            loop,
        )

    interleaved = await asyncio.to_thread(
        generate_interleaved_story,
        {
            **story_ctx,
            "title": story_title,
            "who": str(story_ctx.get("character_essence", story_ctx.get("who", ""))).strip(),
            "turn": str(story_ctx.get("the_turn", story_ctx.get("turn", ""))).strip(),
            "remains": str(story_ctx.get("residual_feeling", story_ctx.get("remains", ""))).strip(),
        },
        settings,
        scenes_dir,
        _scene_ready,
    )

    scenes = list(interleaved.get("scenes", []) or [])[:INTERLEAVED_SCENE_COUNT]
    if not scenes:
        raise ValueError("Gemini did not return any story images.")

    assets: Dict[str, Any] = {
        "scene_urls": [],
        "scene_narrations": [],
    }

    await _send_progress(ws, "Streaming your story…", 55)
    for scene in scenes:
        scene_path = Path(str(scene.get("image_path", "")))
        scene_url = _versioned_static_url(
            f"/static/generated/{session_id}/scenes/{scene_path.name}",
            scene_path,
        )
        narration = str(scene.get("narration", "")).strip()
        assets["scene_urls"].append(scene_url)
        assets["scene_narrations"].append(narration)
        scene_number = int(scene.get("scene_number", 0) or 0)
        if scene_number not in streamed_scene_numbers:
            await _send_scene_payload(
                ws,
                scene_index=scene_number,
                scene_url=scene_url,
                narration=narration,
            )
            await asyncio.sleep(0.1)

    narration_path: Path | None = None
    narration_scenes = [
        {
            "scene_number": idx,
            "narration": str(scene.get("narration", "")).strip(),
            "narrator_profile": str(story_ctx.get("narrator_profile", "older_memory")).strip() or "older_memory",
        }
        for idx, scene in enumerate(scenes, start=1)
    ]
    if any(
        str(scene.get("narration", "")).strip()
        and str(scene.get("narration", "")).strip().lower() != "[silence]"
        for scene in scenes
    ):
        await _send_progress(ws, "Recording the voiceover…", 72)
        try:
            narration_path = await asyncio.to_thread(
                create_narration_audio,
                scenes=narration_scenes,
                settings=settings,
                output_dir=audio_dir,
                intro_text="",
                intro_duration_seconds=0.0,
                scene_duration_seconds=6.0,
                total_duration_seconds=max(24.0, len(scenes) * 6.0),
                preserve_intro_length=False,
            )
            narration_url = _versioned_static_url(
                f"/static/generated/{session_id}/audio/{narration_path.name}",
                narration_path,
            )
            await _send_audio_ready(ws, narration_url)
            assets["audio_url"] = narration_url
        except Exception as exc:
            logger.warning("Narration generation failed for session %s: %s", session_id, exc)

    video_url = ""
    if ENABLE_VEO and len(scenes) >= 5:
        await _send_progress(ws, "Bringing the peak moment to life…", 84)
        try:
            hero_video_path = await asyncio.to_thread(
                generate_hero_video,
                story_ctx,
                scenes[4],
                settings,
                video_dir / "hero_video.mp4",
            )
            if hero_video_path and Path(hero_video_path).exists():
                video_url = _versioned_static_url(
                    f"/static/generated/{session_id}/video/{Path(hero_video_path).name}",
                    Path(hero_video_path),
                )
                await ws.send_json({
                    "type": "film_ready",
                    "video_url": video_url,
                    "storyboard_url": None,
                    "video_title": f"{story_title} — Hero Scene",
                    "video_subtitle": "One Veo scene at the emotional peak.",
                })
        except Exception as exc:
            logger.warning("Hero video generation failed for session %s: %s", session_id, exc)

    state_store.save(session_id, {
        "current_beat": 6,
        "story_type": story_type,
        "story_context": story_ctx,
        "generated_assets": assets,
        "video_url": video_url,
    })

    await _send_progress(ws, "Story ready ✓", 100)
    residual_feeling = re.sub(r"\s+", " ", str(story_ctx.get("residual_feeling", "")).strip())
    closing_line = (
        f"Your story is ready. {residual_feeling}. Scroll through it."
        if residual_feeling
        else "Your story is ready. Scroll through it."
    )
    await _send_assistant(
        ws,
        closing_line,
        film_ready=bool(video_url),
        step=6,
    )
    logger.info(
        "TIMING render_interleaved_story complete total=%.2fs session=%s scenes=%d hero_video=%s",
        time.perf_counter() - render_start,
        session_id,
        len(scenes),
        bool(video_url),
    )


async def _render_story(
    ws: WebSocket,
    session_id: str,
    story_ctx: Dict[str, Any],
    *,
    story_type: str,
) -> None:
    render_start = time.perf_counter()
    assets: Dict[str, Any] = {}
    preset_name = str(story_ctx.get("preset_name", "")).strip().lower()
    await _send_assistant(ws, "I have what I need. Rendering your film now.", step=2)

    stage_start = time.perf_counter()
    char_path = await _generate_character_background(ws, session_id, story_ctx)
    logger.info("TIMING stage character total=%.2fs session=%s", time.perf_counter() - stage_start, session_id)
    await asyncio.sleep(MAJOR_STAGE_STAGGER_SECONDS)

    await _send_progress(ws, "Shaping the emotional wave…", 28)
    script_stage_start = time.perf_counter()
    blueprint = await asyncio.to_thread(
        generate_film_blueprint,
        dict(story_ctx),
        settings,
    )
    blueprint = enforce_character_bible(blueprint)
    full_script = [
        dict(scene)
        for scene in list(blueprint.get("scenes", []) or [])[:settings.scene_count]
    ]
    if len(full_script) < settings.scene_count:
        logger.warning(
            "Film blueprint returned %d scenes for session %s; falling back to generic dynamic script.",
            len(full_script),
            session_id,
        )
        full_script = await asyncio.to_thread(
            generate_dynamic_script,
            story_ctx,
            settings.scene_count,
            "Full 8-scene film arc",
            settings,
        )
        full_script = [dict(scene) for scene in full_script[:settings.scene_count]]
        blueprint = {
            "character_bible": str(story_ctx.get("character_bible", "")).strip(),
            "thread_object": str(story_ctx.get("thread_object_hint", "")).strip(),
            "visual_style_anchor": str(story_ctx.get("visual_style", "")).strip(),
            "music_segments": list(story_ctx.get("music_segments", []) or []),
            "silence_after_scene_5": 3.0,
            "silence_after_scene_7": 2.0,
            "title": str(story_ctx.get("title", "")).strip(),
            "scenes": [dict(scene) for scene in full_script],
        }
    story_ctx["film_blueprint"] = blueprint
    if str(blueprint.get("character_bible", "")).strip():
        story_ctx["character_bible"] = str(blueprint.get("character_bible", "")).strip()
    if str(blueprint.get("thread_object", "")).strip():
        story_ctx["thread_object_hint"] = str(blueprint.get("thread_object", "")).strip()
    if str(blueprint.get("visual_style_anchor", "")).strip():
        story_ctx["visual_style_anchor"] = str(blueprint.get("visual_style_anchor", "")).strip()
    if str(blueprint.get("title", "")).strip():
        story_ctx["title"] = str(blueprint.get("title", "")).strip()
    if isinstance(blueprint.get("music_segments"), list) and blueprint.get("music_segments"):
        story_ctx["music_segments"] = list(blueprint.get("music_segments") or [])
    for scene_data in full_script:
        scene_data["format"] = "video"
        scene_data["duration_seconds"] = CINEMATIC_SCENE_SECONDS
        scene_data["narration_lead_in_seconds"] = float(
            scene_data.get("narration_pause_before", scene_data.get("narration_lead_in_seconds", 0.5)) or 0.5
        )
        scene_data["veo_prompt"] = str(scene_data.get("veo_prompt", "")).strip()
    logger.info(
        "TIMING stage blueprint total=%.2fs story_type=%s session=%s",
        time.perf_counter() - script_stage_start,
        story_type,
        session_id,
    )

    creative_direction = {
        "source": "film_blueprint",
        "intro_narrator_profile": str(story_ctx.get("intro_narrator_profile", "")).strip(),
        "scenes": [dict(scene) for scene in full_script],
        "music_segments": list(story_ctx.get("music_segments", []) or []),
    }
    story_ctx["creative_direction"] = creative_direction
    logger.info(
        "TIMING stage blueprint_resolved total=%.2fs session=%s",
        0.0,
        session_id,
    )

    scenes_dir = STATIC_DIR / "generated" / session_id / "scenes"
    audio_dir = STATIC_DIR / "generated" / session_id / "audio"
    video_dir = STATIC_DIR / "generated" / session_id / "video"
    story_title = _story_title(story_ctx)
    all_script = list(full_script)
    all_narrations = [str(scene.get("narration", "")).strip() for scene in all_script]
    all_moods = [str(scene.get("music_mood", "")).strip() for scene in all_script]
    all_narrator_profiles = [
        str(scene.get("narrator_profile", story_ctx.get("narrator_profile", ""))).strip()
        for scene in all_script
    ]
    scene_durations = [
        max(float(scene.get("duration_seconds", CINEMATIC_SCENE_SECONDS) or CINEMATIC_SCENE_SECONDS), 1.0)
        for scene in all_script
    ]
    scene_duration_map = {
        index: duration
        for index, duration in enumerate(scene_durations, start=1)
    }
    dynamic_black_pauses = {
        5: max(float(blueprint.get("silence_after_scene_5", 3.0) or 0.0), 0.0),
        7: max(float(blueprint.get("silence_after_scene_7", 2.0) or 0.0), 0.0),
    }
    scene_pause_after = dict(SCENE_TRANSITIONS_AFTER)
    scene_pause_after.update({index: value for index, value in dynamic_black_pauses.items() if value > 0})
    target_runtime_seconds = _planned_film_runtime_seconds(
        scene_durations=scene_durations,
        pause_after=dynamic_black_pauses,
        transition_after=SCENE_TRANSITIONS_AFTER,
    )
    render_settings = settings.model_copy(
        update={"film_duration_seconds": int(round(target_runtime_seconds))}
    )
    silence_windows = _planned_silence_windows(
        scene_durations=scene_durations,
        pause_after=dynamic_black_pauses,
        transition_after=SCENE_TRANSITIONS_AFTER,
    )
    scene_lead_in_map = {
        index: (1.5 if str(scene.get("narration", "")).strip() else 0.0)
        for index, scene in enumerate(all_script, start=1)
    }
    scene_rate_map = {
        index: max(
            float(scene.get("tts_rate", 1.0) or 1.0),
            0.5,
        )
        for index, scene in enumerate(all_script, start=1)
        if scene.get("tts_rate") is not None
    }

    async def _build_narration_with_intro() -> Path:
        return await asyncio.to_thread(
            create_narration_audio,
            scenes=all_script,
            settings=render_settings,
            output_dir=audio_dir,
            intro_text="",
            intro_duration_seconds=OPENING_CARD_SECONDS,
            intro_narrator_profile=(
                str(story_ctx.get("intro_narrator_profile", "")).strip()
                or str(story_ctx.get("narrator_profile", "")).strip()
            ),
            scene_duration_seconds=CINEMATIC_SCENE_SECONDS,
            scene_duration_map=scene_duration_map,
            total_duration_seconds=target_runtime_seconds,
            scene_pause_after=scene_pause_after,
            ending_silence_seconds=END_CARD_SECONDS,
            scene_lead_in_map=scene_lead_in_map,
            scene_rate_map=scene_rate_map,
            preserve_intro_length=False,
        )

    async def _resolve_music_path() -> Path:
        existing_task = BACKGROUND_MUSIC_TASKS.get(session_id)
        if existing_task is not None:
            try:
                return await existing_task
            finally:
                BACKGROUND_MUSIC_TASKS.pop(session_id, None)
        return await _generate_music_background(session_id, story_ctx, full_script, render_settings)

    async def _veo_progress(done: int, total: int) -> None:
        await _send_progress(
            ws,
            f"Bringing scene {done}/{total} to life…",
            20 + round((72 - 20) * done / max(total, 1)),
        )

    async def _veo_scene_ready(result: Dict[str, Any]) -> None:
        preview_raw = result.get("preview_image_path")
        if not preview_raw:
            return
        preview_path = Path(str(preview_raw))
        if not preview_path.exists():
            return
        scene_index = int(result.get("scene_index", 0) or 0)
        if scene_index <= 0 or scene_index > len(all_narrations):
            return
        await _send_scene_payload(
            ws,
            scene_index=scene_index,
            scene_url=f"/static/generated/{session_id}/scenes/{preview_path.name}",
            narration=all_narrations[scene_index - 1],
        )

    session_dir = STATIC_DIR / "generated" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    try:
        (session_dir / "story_context.json").write_text(
            json.dumps(story_ctx, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (session_dir / "script.json").write_text(
            json.dumps(all_script, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (session_dir / "film_blueprint.json").write_text(
            json.dumps(blueprint, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (session_dir / "creative_direction.json").write_text(
            json.dumps(creative_direction, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to persist debug artifacts for session %s: %s", session_id, exc)

    await _send_progress(ws, "Bringing your story to life…", 20)
    primary_stage_start = time.perf_counter()
    parallel_results = await asyncio.gather(
        generate_all_veo_scenes(
            settings=settings,
            script=all_script,
            story_context=story_ctx,
            character_reference_path=Path(char_path),
            output_dir=video_dir / "veo",
            preview_dir=scenes_dir,
            progress_callback=_veo_progress,
            scene_ready_callback=_veo_scene_ready,
        ),
        _build_narration_with_intro(),
        _resolve_music_path(),
        return_exceptions=True,
    )
    veo_result_raw, tts_result, music_result = parallel_results
    veo_results = veo_result_raw if isinstance(veo_result_raw, list) else []
    logger.info(
        "TIMING stage veo_primary total=%.2fs scenes=%d session=%s",
        time.perf_counter() - primary_stage_start,
        len(veo_results),
        session_id,
    )

    narration_path = tts_result if not isinstance(tts_result, Exception) else None
    music_path = music_result if not isinstance(music_result, Exception) else None
    narration_timing = load_narration_timing(audio_dir) if narration_path else None
    opening_card_duration = OPENING_CARD_SECONDS

    await _send_progress(ws, "Polishing scenes…", 75)
    scene_video_paths: List[Path] = []
    scene_preview_paths: List[Path] = []
    scene_has_native_audio: List[bool] = []
    fallback_dir = video_dir / "fallback"

    for index, scene_data in enumerate(all_script, start=1):
        result = (
            veo_results[index - 1]
            if index - 1 < len(veo_results) and isinstance(veo_results[index - 1], dict)
            else {"scene_index": index, "method": "failed"}
        )
        video_path_raw = result.get("video_path")
        preview_path_raw = result.get("preview_image_path")
        video_path = Path(str(video_path_raw)) if video_path_raw else None
        preview_path = Path(str(preview_path_raw)) if preview_path_raw else None

        if result.get("method") == "veo" and video_path and video_path.exists():
            scene_video_paths.append(video_path)
            scene_has_native_audio.append(bool(result.get("has_native_audio")))
            if preview_path and preview_path.exists():
                scene_preview_paths.append(preview_path)
            else:
                still_scene = await asyncio.to_thread(
                    generate_single_scene,
                    settings=settings,
                    scene_data=scene_data,
                    story_context=story_ctx,
                    character_reference_path=Path(char_path),
                    output_dir=scenes_dir,
                    scene_index=index,
                    use_character_reference=False,
                )
                scene_preview_paths.append(still_scene.image_path)
                await _send_scene(ws, session_id, still_scene)
                if index < len(all_script):
                    await asyncio.sleep(STILL_FALLBACK_STAGGER_SECONDS)
            continue

        still_scene = await asyncio.to_thread(
            generate_single_scene,
            settings=settings,
            scene_data=scene_data,
            story_context=story_ctx,
            character_reference_path=Path(char_path),
            output_dir=scenes_dir,
            scene_index=index,
            use_character_reference=False,
        )
        kb_path = await asyncio.to_thread(
            create_ken_burns_clip,
            image_path=still_scene.image_path,
            output_path=fallback_dir / f"scene_{index:02d}_kb.mp4",
            duration_seconds=scene_durations[index - 1],
            clip_index=index - 1,
        )
        scene_video_paths.append(Path(kb_path))
        scene_preview_paths.append(still_scene.image_path)
        scene_has_native_audio.append(False)
        await _send_scene(ws, session_id, still_scene)
        if index < len(all_script):
            await asyncio.sleep(STILL_FALLBACK_STAGGER_SECONDS)

    assets["character_ref"] = str(char_path)
    assets["scenes_act12"] = [str(path) for path in scene_preview_paths[:SCENES_ACT12]]
    assets["narrations_act12"] = all_narrations[:SCENES_ACT12]
    assets["moods_act12"] = all_moods[:SCENES_ACT12]
    assets["narrator_profiles_act12"] = all_narrator_profiles[:SCENES_ACT12]
    assets["scenes_act3"] = [str(path) for path in scene_preview_paths[SCENES_ACT12:SCENES_ACT12 + SCENES_ACT3]]
    assets["narrations_act3"] = all_narrations[SCENES_ACT12:SCENES_ACT12 + SCENES_ACT3]
    assets["moods_act3"] = all_moods[SCENES_ACT12:SCENES_ACT12 + SCENES_ACT3]
    assets["narrator_profiles_act3"] = all_narrator_profiles[SCENES_ACT12:SCENES_ACT12 + SCENES_ACT3]

    timeline_media_paths: List[Path] = []
    timeline_has_native_audio: List[bool] = []
    timeline_durations: List[float] = []
    for index, video_path in enumerate(scene_video_paths, start=1):
        timeline_media_paths.append(video_path)
        timeline_has_native_audio.append(scene_has_native_audio[index - 1])
        timeline_durations.append(scene_durations[index - 1])
        pause_duration = float(dynamic_black_pauses.get(index, 0.0))
        if pause_duration > 0:
            pause_clip = await asyncio.to_thread(
                create_black_card,
                text="",
                duration_seconds=pause_duration,
                output_path=video_dir / f"pause_after_scene_{index:02d}.mp4",
            )
            timeline_media_paths.append(Path(pause_clip))
            timeline_has_native_audio.append(False)
            timeline_durations.append(pause_duration)

    end_card_path = await asyncio.to_thread(
        create_black_card,
        text=story_title.upper(),
        duration_seconds=END_CARD_SECONDS,
        output_path=video_dir / "end_card.mp4",
    )
    timeline_media_paths.append(Path(end_card_path))
    timeline_has_native_audio.append(False)
    timeline_durations.append(END_CARD_SECONDS)

    ambient_path = None

    if narration_path and music_path:
        await _send_progress(ws, "Mixing audio tracks…", 90)
        mix_start = time.perf_counter()
        mixed_path = await asyncio.to_thread(
            mix_audio_tracks,
            narration_path=narration_path,
            music_path=music_path,
            ambient_path=ambient_path,
            settings=render_settings,
            output_dir=audio_dir,
            target_duration_seconds=target_runtime_seconds,
            silence_windows=silence_windows,
        )
        logger.info("TIMING stage audio_mix total=%.2fs session=%s", time.perf_counter() - mix_start, session_id)
    elif narration_path:
        mixed_path = narration_path
    elif music_path:
        mixed_path = music_path
    else:
        import wave
        await _send_progress(ws, "Continuing without audio…", 90)
        audio_dir.mkdir(parents=True, exist_ok=True)
        silent_path = audio_dir / "silence.wav"
        with wave.open(str(silent_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00" * 24000 * 2 * render_settings.film_duration_seconds)
        mixed_path = silent_path

    loop = asyncio.get_event_loop()

    def video_progress(msg: str, pct: int) -> None:
        asyncio.run_coroutine_threadsafe(_send_progress(ws, msg, pct), loop)

    opening_card_path = await asyncio.to_thread(
        create_black_card,
        text=story_title.upper(),
        duration_seconds=opening_card_duration,
        output_path=video_dir / "opening_card.mp4",
    )
    video_stage_start = time.perf_counter()
    final_path = await asyncio.to_thread(
        assemble_video,
        scene_images=timeline_media_paths,
        mixed_audio_path=mixed_path,
        settings=render_settings,
        output_dir=video_dir,
        progress_callback=video_progress,
        prefix_clips=[opening_card_path],
        scene_duration_seconds=CINEMATIC_SCENE_SECONDS,
        clip_durations=timeline_durations,
        transition_after=SCENE_TRANSITIONS_AFTER,
    )
    logger.info("TIMING stage video total=%.2fs session=%s", time.perf_counter() - video_stage_start, session_id)
    storyboard_dir = STATIC_DIR / "generated" / session_id / "storyboard"
    storyboard_path = None
    try:
        storyboard_path = await asyncio.to_thread(
            generate_storyboard_pdf,
            story_title=story_title,
            scenes=[
                {
                    "image_path": str(scene_path),
                    "narration": narration,
                }
                for scene_path, narration in zip(scene_preview_paths, all_narrations)
            ],
            output_path=storyboard_dir / "storyboard.pdf",
        )
    except Exception as exc:
        logger.warning("Storyboard export failed for session %s: %s", session_id, exc)

    await _send_progress(ws, "Film ready ✓", 100)
    video_url = f"/static/generated/{session_id}/video/{final_path.name}"
    storyboard_url = (
        f"/static/generated/{session_id}/storyboard/{storyboard_path.name}"
        if storyboard_path else None
    )
    preset_name = str(story_ctx.get("preset_name", "")).strip().lower()
    if story_type == "preset" and preset_name in PRESETS:
        try:
            demo_dir = await asyncio.to_thread(
                _promote_preset_demo,
                preset_name=preset_name,
                session_id=session_id,
                story_title=story_title,
                script=all_script,
                final_path=final_path,
                storyboard_path=storyboard_path,
            )
            logger.info("Promoted preset demo cache for %s to %s", preset_name, demo_dir)
        except Exception as exc:
            logger.warning("Preset demo promotion failed for %s: %s", preset_name, exc)
    gcs_uri = await asyncio.to_thread(_upload_video, final_path, session_id)
    state_store.save(session_id, {
        "current_beat": 3,
        "story_context": story_ctx,
        "generated_assets": assets,
        "video_url": video_url,
        "storyboard_url": storyboard_url,
        "video_gcs_uri": gcs_uri,
    })

    await ws.send_json({
        "type": "film_ready",
        "video_url": video_url,
        "storyboard_url": storyboard_url,
        "video_title": story_title,
    })
    residual_feeling = re.sub(r"\s+", " ", str(story_ctx.get("residual_feeling", "")).strip())
    closing_line = (
        f"Your film is ready. {residual_feeling}. Press play."
        if residual_feeling
        else "Your film is ready. Press play."
    )
    await _send_assistant(
        ws,
        closing_line,
        film_ready=True,
        step=3,
        video_title=story_title,
    )
    logger.info("TIMING render_story complete total=%.2fs session=%s", time.perf_counter() - render_start, session_id)


async def handle_custom_intake(ws: WebSocket, session_id: str, user_input: str, music_preference: str) -> None:
    await _send_progress(ws, "Understanding your story…", 4)
    story_ctx: Dict[str, Any] = {
        "who": user_input,
        "character_essence": user_input,
        "character": user_input,
        "character_description": user_input,
        "music_preference": _resolve_music_preference(user_input, music_preference),
    }
    state_store.save(session_id, {
        "current_beat": 2,
        "story_type": "custom",
        "story_context": story_ctx,
        "generated_assets": {},
    })
    await _render_interleaved_story(ws, session_id, story_ctx, story_type="custom")


async def handle_preset_intake(ws: WebSocket, session_id: str, user_input: str, music_preference: str) -> None:
    state = state_store.load(session_id)
    story_ctx = dict(state.get("story_context", {}))
    story_ctx["turn"] = user_input
    story_ctx["the_turn"] = user_input
    story_ctx["remains"] = user_input
    story_ctx["residual_feeling"] = user_input
    story_ctx["music_preference"] = _resolve_music_preference(user_input, music_preference)
    state_store.save(session_id, {
        "current_beat": 2,
        "story_type": "preset",
        "story_context": story_ctx,
        "generated_assets": {},
    })
    await _render_interleaved_story(ws, session_id, story_ctx, story_type="preset")


# ── Three-Beat Conversation Flow ────────────────────────────────────


async def handle_step1_preset(ws: WebSocket, session_id: str, preset_name: str) -> None:
    preset = get_preset(preset_name)
    story_context = {
        "preset_name": preset_name,
        "title": preset.get("title", ""),
        "character_essence": str(preset.get("character_essence", preset.get("character_description", ""))).strip(),
        "emotional_anchor": str(preset.get("emotional_anchor", "")).strip(),
        "world": str(preset.get("world", preset.get("setting", ""))).strip(),
        "the_turn": str(preset.get("the_turn", "")).strip(),
        "residual_feeling": str(preset.get("residual_feeling", "")).strip(),
        "setting": preset["setting"],
        "character": preset["character_description"],
        "visual_style": preset["visual_style"],
        "narrator_profile": str(preset.get("narrator_profile", "")).strip(),
        "intro_narrator_profile": str(preset.get("intro_narrator_profile", "")).strip(),
        "music_segments": list(preset.get("music_segments", []) or []),
        "thread_object_hint": str(preset.get("thread_object_hint", "")).strip(),
        "use_locked_script": bool(preset.get("use_locked_script", False)),
        "character_bible": str(preset.get("character_bible", "")).strip(),
    }
    if await _serve_demo_preset(ws, session_id, preset_name, story_context):
        return
    state_store.save(session_id, {
        "current_beat": 2,
        "story_type": "preset",
        "story_context": story_context,
        "generated_assets": {},
    })
    await _send_assistant(
        ws,
        "I can feel the world around them already.\n\n"
        "What's the one moment that changes everything for them? Not the whole story. Just the turn.",
        step=2,
    )


async def handle_step1_character(ws: WebSocket, session_id: str, user_input: str) -> None:
    cleaned = user_input.strip()
    if cleaned.lower() == "surprise me":
        await handle_step1_preset(ws, session_id, "sacrifice")
        return
    if cleaned.lower() in GENERIC_CHARACTER_SEEDS or len(cleaned.split()) < 2:
        await _send_assistant(
            ws,
            "Give me one clear protagonist, like: "
            "`a young scout ant`, `a grieving father`, or `a brilliant hacker`.",
            step=1,
        )
        return
    state_store.save(session_id, {
        "current_beat": 2,
        "story_type": "custom",
        "story_context": {
            "who": cleaned,
            "character_essence": cleaned,
            "character": cleaned,
            "character_description": cleaned,
            "beat1_seed": cleaned,
        },
        "generated_assets": {},
    })
    await _send_assistant(
        ws,
        "I know who this film is holding close.\n\n"
        "What's the one moment that changes everything for them? Not the whole story. Just the turn.",
        step=2,
    )


async def handle_step2_turn(ws: WebSocket, session_id: str, user_input: str) -> None:
    state = state_store.load(session_id)
    story_ctx = dict(state.get("story_context", {}))
    story_ctx["turn"] = user_input
    story_ctx["the_turn"] = user_input
    story_ctx["beat2_seed"] = user_input

    state_store.save(session_id, {
        "current_beat": 3,
        "story_context": story_ctx,
    })
    await _send_assistant(
        ws,
        "That turn gives the film its gravity.\n\n"
        "Last question. When the screen goes dark, what feeling stays in the room?",
        step=3,
    )


async def handle_step3_generate(ws: WebSocket, session_id: str, user_input: str) -> None:
    state = state_store.load(session_id)
    story_ctx = dict(state.get("story_context", {}))
    story_ctx["remains"] = user_input
    story_ctx["residual_feeling"] = user_input
    story_ctx["beat3_seed"] = user_input
    state_store.save(session_id, {
        "current_beat": 3,
        "story_context": story_ctx,
    })
    await _render_interleaved_story(
        ws,
        session_id,
        story_ctx,
        story_type=str(state.get("story_type", "custom")),
    )


# ── Interleaved Scene Generation Helper ──────────────────────────────


async def _generate_script_scenes_interleaved(
    ws: WebSocket,
    session_id: str,
    story_ctx: Dict[str, Any],
    char_path: Path,
    scenes_dir: Path,
    script_act12: List[Dict[str, Any]],
    script_act3: List[Dict[str, Any]],
) -> tuple[List[GeneratedScene], List[GeneratedScene]]:
    """Generate scenes in small interleaved batches with bounded concurrency."""
    await _send_progress(ws, "Generating scenes in interleaved batches…", 35)

    full_script = list(script_act12) + list(script_act3)
    raw_batches = build_interleaved_batch_specs(script=full_script, start_index=1)
    if not raw_batches:
        return ([], [])

    progress_start = 35
    progress_end = 78
    batch_total = len(raw_batches)
    batch_specs = []
    for batch_number, (batch_start, batch_script) in enumerate(raw_batches, start=1):
        batch_end = batch_start + len(batch_script) - 1
        progress_pct = progress_start + round((progress_end - progress_start) * batch_number / batch_total)
        batch_specs.append((batch_start, f"{batch_start}-{batch_end}", progress_pct, batch_script))

    semaphore = asyncio.Semaphore(INTERLEAVED_SCENE_MAX_WORKERS)

    async def _run_batch(batch_start: int, batch_script: List[Dict[str, Any]]) -> List[GeneratedScene]:
        async with semaphore:
            return await asyncio.to_thread(
                generate_scene_batch_interleaved,
                settings=settings,
                script=batch_script,
                story_context=story_ctx,
                character_reference_path=char_path,
                output_dir=scenes_dir,
                start_index=batch_start,
            )

    tasks = {}
    for index, (batch_start, label, progress_pct, batch_script) in enumerate(batch_specs):
        task = asyncio.create_task(_run_batch(batch_start, batch_script))
        tasks[task] = (batch_start, label, progress_pct)
        if index < len(batch_specs) - 1:
            await asyncio.sleep(MAJOR_STAGE_STAGGER_SECONDS)

    batch_results: Dict[int, List[GeneratedScene]] = {}
    while tasks:
        done, _pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            batch_start, label, progress_pct = tasks.pop(task)
            scenes = sorted(task.result(), key=lambda scene: scene.index)
            batch_results[batch_start] = scenes
            await _send_progress(ws, f"Scenes {label} ready…", progress_pct)
            for scene in scenes:
                await _send_scene(ws, session_id, scene)

    ordered_scenes: List[GeneratedScene] = []
    for batch_start, _label, _progress_pct, _batch_script in batch_specs:
        ordered_scenes.extend(batch_results.get(batch_start, []))

    return (
        [scene for scene in ordered_scenes if scene.index <= SCENES_ACT12],
        [scene for scene in ordered_scenes if scene.index > SCENES_ACT12],
    )


async def _generate_classified_stills(
    ws: WebSocket,
    session_id: str,
    story_ctx: Dict[str, Any],
    char_path: Path,
    scenes_dir: Path,
    script: List[Dict[str, Any]],
) -> Dict[int, GeneratedScene]:
    still_scene_numbers = [
        index
        for index, scene in enumerate(script, start=1)
        if str(scene.get("format", "video")).strip().lower() == "image"
    ]
    if not still_scene_numbers:
        return {}
    results: Dict[int, GeneratedScene] = {}
    for offset, scene_index in enumerate(still_scene_numbers):
        scene = await asyncio.to_thread(
            generate_single_scene,
            settings=settings,
            scene_data=script[scene_index - 1],
            story_context=story_ctx,
            character_reference_path=char_path,
            output_dir=scenes_dir,
            scene_index=scene_index,
            use_character_reference=False,
        )
        await _send_scene(ws, session_id, scene)
        results[scene.index] = scene
        if offset < len(still_scene_numbers) - 1:
            await asyncio.sleep(20)
    return results


# ── Final Generation (After Beat 3) ─────────────────────────────────


SCENES_ACT12 = 4
SCENES_ACT3 = 4


# ── WebSocket Endpoint ───────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = new_session_id()
    await session_service.create_session(
        app_name=ADK_APP_NAME, user_id=ADK_USER_ID,
        session_id=session_id, state={},
    )
    initialize_session_state(session_id)
    set_current_session(session_id)

    await websocket.send_json({
        "type": "session",
        "session_id": session_id,
        "step": 1,
    })
    await _send_assistant(websocket, OPENING_MESSAGE, step=1)

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            msg_type = payload.get("type")

            set_current_session(session_id)
            state = state_store.load(session_id)
            current_beat = state.get("current_beat", 1)

            # ── Preset click (always Beat 1) ──
            if msg_type == "preset" and current_beat == 1:
                preset_name = str(payload.get("preset", "")).strip().lower()
                if preset_name in PRESETS:
                    try:
                        await handle_step1_preset(websocket, session_id, preset_name)
                    except Exception as exc:
                        logger.exception("Step 1 (preset) failed")
                        await _send_assistant(websocket, f"I hit a snag setting the world: {exc}. Try again or pick a different preset.")
                continue

            if msg_type == "chat":
                user_message = str(payload.get("message", "")).strip()
                if not user_message:
                    continue

                try:
                    if current_beat == 1:
                        await handle_step1_character(websocket, session_id, user_message)
                    elif current_beat == 2:
                        await handle_step2_turn(websocket, session_id, user_message)
                    elif current_beat == 3:
                        await handle_step3_generate(websocket, session_id, user_message)
                    else:
                        await _send_assistant(websocket,
                            "Your film has been created! Refresh the page to start a new story."
                        )
                except Exception as exc:
                    logger.exception("Step %d failed", current_beat)
                    await _send_assistant(websocket, f"Something went wrong: {exc}. Please try sending your message again.")
                continue

            await websocket.send_json({"type": "error", "message": "Unsupported message type."})

    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.exception("WebSocket error for session %s", session_id)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
