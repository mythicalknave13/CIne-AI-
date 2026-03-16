"""AI-powered story data extraction for custom user stories.

Uses Gemini flash-lite to extract structured data from natural language
at each beat of the 3-beat conversation framework.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List
from xml.sax.saxutils import escape as xml_escape

from google import genai
from google.genai import types

from config.settings import Settings
from generation.vertex import vertex_client_kwargs

logger = logging.getLogger("uvicorn.error")
CREATIVE_DIRECTION_PROFILES = {
    "older_memory",
    "older_memory_aged",
    "quiet_child",
    "urgent_witness",
    "mythic_storyteller",
    "hacker_present_tense",
    "hacker_urgent",
    "young_apprentice",
    "old_sage",
    "sage_remembering",
}
MUSIC_SEGMENT_NEGATIVE_PROMPT = "vocals, singing, speech, voice, crowd noise"
VIDEO_ACTION_KEYWORDS = {
    "battle",
    "run",
    "running",
    "chase",
    "flee",
    "escape",
    "flying",
    "ride",
    "storm",
    "sirens",
    "debris",
    "guns",
    "blast",
    "explosion",
    "explode",
    "fire",
    "fighting",
    "attack",
    "crowd",
    "falling",
    "fall",
    "wind",
    "rain",
    "water",
    "waves",
    "opening",
    "opens",
    "shuttle",
    "machine",
    "gears",
    "turning",
    "moving",
    "movement",
    "comet",
    "glowing path",
}
IMAGE_STILLNESS_KEYWORDS = {
    "sleep",
    "sleeping",
    "rest",
    "still",
    "quiet",
    "peaceful",
    "peace",
    "reflection",
    "reflective",
    "remember",
    "memory",
    "grief",
    "hold",
    "holding",
    "reading",
    "watching",
    "waiting",
    "sits",
    "sitting",
    "kneeling",
    "goodbye",
    "death",
    "mourning",
    "dawn",
    "sunrise",
}
TECH_STORY_KEYWORDS = {
    "cyber",
    "cyberpunk",
    "hacker",
    "terminal",
    "code",
    "neon",
    "orbital",
    "station",
    "shuttle",
    "retro-futurist",
    "drone",
    "ai",
    "prison",
    "space",
    "spaceship",
    "observational",
}
MYTHIC_STORY_KEYWORDS = {
    "myth",
    "mythic",
    "legend",
    "dragon",
    "fantasy",
    "magic",
    "sacred",
    "queen",
    "kingdom",
    "ancient",
    "meadow",
    "shrine",
}
APPRENTICE_STORY_KEYWORDS = {
    "apprentice",
    "student",
    "novice",
    "astronomer",
    "learning",
    "curious",
    "discovery",
    "field notes",
    "notebooks",
}
MEMORY_STORY_KEYWORDS = {
    "older now",
    "many years",
    "looking back",
    "remember",
    "memory",
    "once",
    "when i was",
    "my father",
    "my mother",
    "i was",
}
SAGE_STORY_KEYWORDS = {
    "sage",
    "wisdom",
    "ancient",
    "legacy",
    "acceptance",
    "peaceful",
    "old",
    "elder",
}
GROUPISH_TERMS = {
    "ant colony",
    "colony",
    "ants",
    "villagers",
    "people",
    "rebels",
    "soldiers",
    "children",
    "family",
    "crew",
    "swarm",
    "hive",
    "kingdom",
    "tribe",
}
GENERIC_CHARACTER_VALUES = {
    "",
    "a protagonist",
    "protagonist",
    "main character",
    "someone",
    "a person",
}


def _parse_json_response(text: str) -> Any:
    """Robustly parse JSON from Gemini response text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _looks_groupish(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in GROUPISH_TERMS:
        return True
    return any(term in normalized for term in GROUPISH_TERMS)


def _normalize_story_setup(data: Dict[str, str], user_input: str) -> Dict[str, str]:
    setting = str(data.get("setting", "")).strip() or str(user_input).strip()
    character = str(data.get("character", "")).strip()
    visual_style = str(data.get("visual_style", "")).strip() or "Cinematic, dramatic lighting"

    normalized_character = character
    lowered_user = str(user_input).lower()
    lowered_setting = setting.lower()

    if normalized_character.lower() in GENERIC_CHARACTER_VALUES or len(normalized_character.split()) < 2:
        normalized_character = ""

    if _looks_groupish(normalized_character) or not normalized_character:
        if "ant" in lowered_user or "ant" in lowered_setting:
            normalized_character = "A young scout ant, small but determined, curious and brave"
        elif any(term in lowered_user or term in lowered_setting for term in {"hive", "swarm", "colony"}):
            normalized_character = "A lone scout from the colony, alert, vulnerable, and quietly brave"
        elif any(term in lowered_user or term in lowered_setting for term in {"village", "city", "kingdom", "empire"}):
            normalized_character = "A single young survivor, observant, resilient, and quietly brave"
        else:
            normalized_character = "One point-of-view protagonist, distinct, vulnerable, and determined"

    normalized_character = re.sub(r"\s+", " ", normalized_character).strip()
    return {
        "setting": setting,
        "character": normalized_character,
        "visual_style": visual_style,
    }


def extract_story_setup(user_input: str, settings: Settings) -> Dict[str, str]:
    """Beat 1: Extract setting, character, and visual_style from user input."""
    client = genai.Client(**vertex_client_kwargs(settings))
    prompt = (
        f'Extract story elements from this user input:\n"{user_input}"\n\n'
        "Return JSON with exactly these fields:\n"
        "- setting: Where/when? (era, location, genre atmosphere)\n"
        "- character: ONE point-of-view protagonist only (age/species/role, 2-3 key traits)\n"
        "- visual_style: Cinematic style? (lighting, palette, mood)\n\n"
        "Rules:\n"
        "- Always return a single protagonist, never a group, city, colony, family, or civilization.\n"
        "- If the user names a group or world instead of a person, infer one cinematic protagonist who belongs there.\n"
        "- Prefer film-ready wording that can drive a single consistent character reference image.\n\n"
        'Example: {{"setting":"2145 Neo-Tokyo, cyberpunk dystopia",'
        '"character":"Young hacker (mid-20s), brilliant but isolated",'
        '"visual_style":"Neon noir, rain-soaked, high contrast blue/pink"}}'
    )
    response = client.models.generate_content(
        model=settings.script_routing_model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        data = _parse_json_response(response.text)
    except Exception:
        data = {"setting": user_input, "character": "A protagonist", "visual_style": "Cinematic, dramatic lighting"}
    return _normalize_story_setup(
        {
            "setting": str(data.get("setting", user_input)),
            "character": str(data.get("character", "A protagonist")),
            "visual_style": str(data.get("visual_style", "Cinematic, dramatic lighting")),
        },
        user_input=user_input,
    )


def extract_conflict(user_input: str, settings: Settings) -> Dict[str, str]:
    """Beat 2: Extract inciting_incident and transformation from user input."""
    client = genai.Client(**vertex_client_kwargs(settings))
    prompt = (
        f'From this user input about a story\'s conflict:\n"{user_input}"\n\n'
        "Extract:\n1. inciting_incident — what kicks off the conflict?\n"
        "2. transformation — how does the protagonist change?\n\n"
        'Return JSON: {{"inciting_incident":"...","transformation":"..."}}'
    )
    response = client.models.generate_content(
        model=settings.script_routing_model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        data = _parse_json_response(response.text)
    except Exception:
        return {"inciting_incident": user_input, "transformation": "The character transforms through adversity"}
    return {
        "inciting_incident": str(data.get("inciting_incident", user_input)),
        "transformation": str(data.get("transformation", "The character transforms")),
    }


def extract_ending(user_input: str, settings: Settings) -> Dict[str, str]:
    """Beat 3: Extract climax and resolution from user input."""
    client = genai.Client(**vertex_client_kwargs(settings))
    prompt = (
        f'From this user input about a story\'s ending:\n"{user_input}"\n\n'
        "Extract:\n1. climax — peak moment of tension\n"
        "2. resolution — how it ends, what changes\n\n"
        'Return JSON: {{"climax":"...","resolution":"..."}}'
    )
    response = client.models.generate_content(
        model=settings.script_routing_model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        data = _parse_json_response(response.text)
    except Exception:
        return {"climax": user_input, "resolution": "The story reaches its conclusion"}
    return {
        "climax": str(data.get("climax", user_input)),
        "resolution": str(data.get("resolution", "The story concludes")),
    }


def extract_emotional_setup(user_input: str, settings: Settings) -> Dict[str, str]:
    """Beat 1: extract the person, why they matter, and the world around them."""
    client = genai.Client(**vertex_client_kwargs(settings))
    prompt = (
        f'Extract the emotional core of this story setup:\n"{user_input}"\n\n'
        "Return JSON with exactly these fields:\n"
        '- "character_essence": who this person is at their core\n'
        '- "emotional_anchor": why they matter emotionally\n'
        '- "world": where they exist\n'
        '- "character": a visual character description for image generation\n'
        '- "setting": a visual world description for image generation\n'
        '- "visual_style": cinematic lighting/palette/mood\n'
        "Rules:\n"
        "- Keep the character singular and specific.\n"
        "- Prefer emotionally grounded wording over plot summary.\n"
        "- Make the character and world ready for cinematic generation.\n"
    )
    response = client.models.generate_content(
        model=settings.script_routing_model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        data = _parse_json_response(response.text)
    except Exception:
        normalized = _normalize_story_setup({"setting": user_input, "character": "", "visual_style": ""}, user_input)
        return {
            "character_essence": normalized["character"],
            "emotional_anchor": "They matter because losing them would change the world around them.",
            "world": normalized["setting"],
            **normalized,
        }

    normalized = _normalize_story_setup(
        {
            "setting": str(data.get("setting", data.get("world", user_input))),
            "character": str(data.get("character", data.get("character_essence", ""))),
            "visual_style": str(data.get("visual_style", "Cinematic, dramatic lighting")),
        },
        user_input=user_input,
    )
    return {
        "character_essence": str(data.get("character_essence", normalized["character"])).strip() or normalized["character"],
        "emotional_anchor": str(data.get("emotional_anchor", "")).strip() or "They matter because they hold the story's emotional center.",
        "world": str(data.get("world", normalized["setting"])).strip() or normalized["setting"],
        **normalized,
    }


def extract_story_turn(user_input: str, settings: Settings) -> Dict[str, str]:
    """Beat 2: extract the single turning moment and emotional shift."""
    client = genai.Client(**vertex_client_kwargs(settings))
    prompt = (
        f'Extract the turning point from this answer:\n"{user_input}"\n\n'
        "Return JSON with exactly these fields:\n"
        '- "the_turn": the single moment that changes everything\n'
        '- "emotional_direction": the emotional shift it causes\n'
        '- "inciting_incident": a concise plot-facing version of the turn\n'
        '- "transformation": how the protagonist changes internally\n'
    )
    response = client.models.generate_content(
        model=settings.script_routing_model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        data = _parse_json_response(response.text)
    except Exception:
        return {
            "the_turn": user_input,
            "emotional_direction": "Their world shifts and cannot return to what it was.",
            "inciting_incident": user_input,
            "transformation": "The protagonist is changed by what the moment demands.",
        }
    return {
        "the_turn": str(data.get("the_turn", user_input)).strip() or user_input,
        "emotional_direction": str(data.get("emotional_direction", "")).strip() or "Everything tilts emotionally after this moment.",
        "inciting_incident": str(data.get("inciting_incident", user_input)).strip() or user_input,
        "transformation": str(data.get("transformation", "")).strip() or "The protagonist is changed by what the moment demands.",
    }


def extract_residual_emotion(user_input: str, settings: Settings) -> Dict[str, str]:
    """Beat 3: extract the feeling that remains and the final image it suggests."""
    client = genai.Client(**vertex_client_kwargs(settings))
    prompt = (
        f'Extract the ending feeling from this answer:\n"{user_input}"\n\n'
        "Return JSON with exactly these fields:\n"
        '- "residual_feeling": the emotion left in the room after the film ends\n'
        '- "final_image": what the last frame should evoke visually\n'
        '- "resolution": a concise emotional resolution for the story\n'
        '- "climax": the peak emotional moment the ending depends on\n'
    )
    response = client.models.generate_content(
        model=settings.script_routing_model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        data = _parse_json_response(response.text)
    except Exception:
        return {
            "residual_feeling": user_input,
            "final_image": "A lingering visual echo of what was loved and what remains.",
            "resolution": user_input,
            "climax": "The decisive emotional turn before the final stillness.",
        }
    return {
        "residual_feeling": str(data.get("residual_feeling", user_input)).strip() or user_input,
        "final_image": str(data.get("final_image", "")).strip() or "A final image that echoes the beginning with changed meaning.",
        "resolution": str(data.get("resolution", user_input)).strip() or user_input,
        "climax": str(data.get("climax", "")).strip() or "The decisive emotional turn before the ending.",
    }


EMOTIONAL_BEAT_ORDER = [
    "warmth",
    "context",
    "the_crack",
    "the_weight",
    "the_goodbye",
    "the_stand",
    "the_stillness",
    "what_remains",
]


def map_narrator_to_profile(narrator_voice: str) -> str:
    """Map a narrative voice description from the script model to a supported TTS profile."""
    voice_lower = str(narrator_voice or "").lower()

    if "old_woman_remembering" in voice_lower:
        return "old_woman_remembering"
    if "old_man_remembering" in voice_lower:
        return "old_man_remembering"
    if "young_woman_present" in voice_lower:
        return "young_woman_present"
    if "young_man_present" in voice_lower:
        return "young_man_present"
    if "mythic_narrator" in voice_lower:
        return "mythic_narrator"
    if "child_witnessing" in voice_lower:
        return "child_witnessing"

    if any(w in voice_lower for w in ["old", "elder", "aged", "grandmother", "grandfather", "looking back", "remembering", "years later"]):
        if any(w in voice_lower for w in ["woman", "mother", "grandmother", "daughter", "she", "her"]):
            if any(w in voice_lower for w in ["aged", "very slow", "whisper", "old woman", "barely"]):
                return "older_memory_aged"
            return "older_memory"
        return "old_sage"

    if any(w in voice_lower for w in ["child", "young", "boy", "girl", "kid", "small", "innocent", "little"]):
        return "quiet_child"

    if any(w in voice_lower for w in ["urgent", "desperate", "running", "panicked", "breathless", "fast"]):
        return "urgent_witness"

    if any(w in voice_lower for w in ["wise", "sage", "ancient", "mythic", "legend", "timeless"]):
        return "mythic_storyteller"

    if any(w in voice_lower for w in ["hacker", "digital", "cyber", "tech", "code"]):
        return "hacker_present_tense"

    if any(w in voice_lower for w in ["apprentice", "student", "learning", "curious"]):
        return "young_apprentice"

    if any(w in voice_lower for w in ["daughter", "mother", "woman", "female", "remembering"]):
        return "older_memory"

    return "sage_remembering"


BLUEPRINT_NARRATOR_VOICES = {
    "old_woman_remembering",
    "old_man_remembering",
    "young_woman_present",
    "young_man_present",
    "mythic_narrator",
    "child_witnessing",
}

BLUEPRINT_TTS_DEFAULTS = {
    "old_woman_remembering": {"rate": 0.84, "pitch": -3.0},
    "old_man_remembering": {"rate": 0.82, "pitch": -4.0},
    "young_woman_present": {"rate": 0.95, "pitch": -1.0},
    "young_man_present": {"rate": 0.95, "pitch": -1.0},
    "mythic_narrator": {"rate": 0.78, "pitch": -2.0},
    "child_witnessing": {"rate": 0.95, "pitch": 1.0},
    "older_memory": {"rate": 0.84, "pitch": -3.0},
    "older_memory_aged": {"rate": 0.78, "pitch": -4.0},
    "mythic_storyteller": {"rate": 0.78, "pitch": -2.0},
    "old_sage": {"rate": 0.80, "pitch": -3.0},
    "sage_remembering": {"rate": 0.86, "pitch": -2.0},
    "young_apprentice": {"rate": 0.95, "pitch": -1.0},
    "quiet_child": {"rate": 0.95, "pitch": 1.0},
    "urgent_witness": {"rate": 0.95, "pitch": -1.0},
}

DEFAULT_NARRATION_PAUSES = {
    1: 1.5,
    2: 0.0,
    3: 1.5,
    4: 1.5,
    5: 1.5,
    6: 0.0,
    7: 0.0,
    8: 1.5,
}


def _limit_words(text: str, limit: int) -> str:
    words = re.findall(r"\S+", str(text or "").strip())
    return " ".join(words[:limit]).strip()


def _normalize_character_bible(text: str, fallback: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        cleaned = re.sub(r"\s+", " ", str(fallback or "").strip())
    if not cleaned:
        cleaned = "an adult protagonist in their forties with a steady gaze, worn practical clothes, and work-marked hands"
    return _limit_words(cleaned, 40)


def _default_character_bible(story_context: Dict[str, Any]) -> str:
    explicit = str(story_context.get("character_bible", "")).strip()
    if explicit:
        return _normalize_character_bible(explicit, explicit)

    base = str(
        story_context.get("character", "")
        or story_context.get("character_essence", "")
        or "an adult protagonist"
    ).strip()
    world = str(story_context.get("world", story_context.get("setting", ""))).strip()
    candidate = (
        f"an adult over 25 from {world or 'their world'}, with medium-brown skin, a solid work-worn build, short practical hair, "
        f"wearing a faded textured work shirt, and thick weathered hands marked by years of ritual and labor"
    )
    return _normalize_character_bible(candidate, candidate)


def _default_thread_object(story_context: Dict[str, Any]) -> str:
    hint = re.sub(r"\s+", " ", str(story_context.get("thread_object_hint", "")).strip())
    if hint:
        return _limit_words(hint, 15)
    character = str(story_context.get("character_essence", "")).lower()
    world = str(story_context.get("world", story_context.get("setting", ""))).lower()
    if any(token in world or token in character for token in ("sea", "ocean", "harbor", "boat", "lighthouse")):
        return "a small hand-carved wooden float painted blue and smoothed by salt air"
    if any(token in world or token in character for token in ("book", "library", "scribe", "teacher", "ink")):
        return "a small hand-carved wooden bird with spread wings, pale unfinished wood"
    return "a small hand-held keepsake worn smooth by years of carrying"


def _default_visual_style_anchor(story_context: Dict[str, Any]) -> str:
    style = re.sub(r"\s+", " ", str(story_context.get("visual_style", "")).strip())
    if style:
        return style
    return "Muted cinematic tones, soft natural light, gentle film grain, and air filled with visible texture"


def _normalize_narrator_voice(raw_voice: str, fallback_voice: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(raw_voice or "").strip())
    if cleaned in BLUEPRINT_NARRATOR_VOICES:
        return cleaned
    mapped = map_narrator_to_profile(cleaned)
    if mapped in BLUEPRINT_NARRATOR_VOICES:
        return mapped
    return fallback_voice


def _tts_defaults_for_voice(narrator_voice: str) -> Dict[str, float]:
    return dict(BLUEPRINT_TTS_DEFAULTS.get(narrator_voice, {"rate": 0.86, "pitch": -2.0}))


def _clean_narration_line(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    cleaned = re.sub(r"(?im)\bnarration\s*:\s*", "", cleaned)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    return cleaned.strip()


def _build_narration_ssml(narration: str, narrator_voice: str) -> str:
    cleaned = _clean_narration_line(narration)
    if not cleaned:
        return ""
    defaults = _tts_defaults_for_voice(narrator_voice)
    fragments = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if not fragments:
        fragments = [cleaned]
    joined = f' <break time="400ms"/> '.join(xml_escape(fragment) for fragment in fragments)
    return (
        "<speak>"
        f'<prosody rate="{int(defaults["rate"] * 100)}%" pitch="{defaults["pitch"]:+.0f}st">'
        f"{joined}"
        "</prosody>"
        "</speak>"
    )


def _extract_audio_cue_from_prompt(veo_prompt: str, fallback: str = "") -> str:
    prompt = str(veo_prompt or "")
    match = re.search(r"audio:\s*(.+?)(?:no text|$)", prompt, flags=re.IGNORECASE)
    if match:
        cue = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        if cue:
            return cue
    return re.sub(r"\s+", " ", str(fallback or "").strip())


def _fallback_emotional_script(story_context: Dict[str, Any]) -> List[Dict[str, str]]:
    character_essence = str(story_context.get("character_essence", story_context.get("character", "someone beloved"))).strip()
    emotional_anchor = str(story_context.get("emotional_anchor", "they matter deeply to someone")).strip()
    world = str(story_context.get("world", story_context.get("setting", "a cinematic world"))).strip()
    the_turn = str(story_context.get("the_turn", story_context.get("inciting_incident", "something changes everything"))).strip()
    residual_feeling = str(story_context.get("residual_feeling", story_context.get("resolution", "bittersweet hope"))).strip()
    final_image = str(story_context.get("final_image", "the beginning echoed with changed meaning")).strip()
    visual_style = str(story_context.get("visual_style", "cinematic natural light")).strip()
    thread_object = str(story_context.get("thread_object_hint", "")).strip() or "a small hand-held keepsake"

    fallback = [
        {
            "scene_number": 1,
            "emotional_beat": "warmth",
            "narration": "My hands knew the ritual before dawn ever did.",
            "narrator_voice": "old_woman_remembering",
            "veo_prompt": f"Close-up, static camera. Hands of {character_essence} work through a familiar ritual with the thread object nearby. Primary workspace at dawn, soft natural light from one side. {visual_style}. Audio: cloth moving, distant birds, soft room tone. No text, no subtitles, no logos, no title cards.",
            "veo_audio_cue": "soft room tone, distant birds, gentle movement",
            "image_prompt": f"Close-up of the hands of {character_essence} doing a familiar task, warm natural light, intimate and calm, {world}",
            "music_mood": "tender intimate warmth",
            "thread_object": thread_object,
            "thread_object_role": "introduced as an ordinary object linked to the person at the center of the story",
            "has_character": True,
        },
        {
            "scene_number": 2,
            "emotional_beat": "context",
            "narration": "",
            "narrator_voice": "old_woman_remembering",
            "veo_prompt": f"Wide aerial shot, gentle descent. The wider world around the primary location with the character as a small figure in the distance. Same dawn conditions and weather as scene 1. {visual_style}. Audio: wind, distant life, natural ambience. No text, no subtitles, no logos, no title cards.",
            "veo_audio_cue": "wind, distant life, natural ambience",
            "image_prompt": f"Wide cinematic view of {world}, peaceful and fragile, detailed environment, {visual_style}",
            "music_mood": "worldbuilding fragile peace",
            "thread_object": thread_object,
            "thread_object_role": "visible somewhere in the world as part of the ordinary life about to change",
            "has_character": True,
        },
        {
            "scene_number": 3,
            "emotional_beat": "the_crack",
            "narration": "I heard the change before I found words for it.",
            "narrator_voice": "old_woman_remembering",
            "veo_prompt": "Medium shot from behind, slow push-in. The character faces away toward something shifting beyond the primary space while one hand goes still. Same time of day, same weather, light cooling as conditions change. Audio: distant rumble, wind tightening, sudden hush. No text, no subtitles, no logos, no title cards.",
            "veo_audio_cue": "distant rumble, wind tightening, sudden hush",
            "image_prompt": "A cinematic sign of change: paused hands, narrowing light, a room holding its breath, tension without visible conflict",
            "music_mood": "hairline crack tension",
            "thread_object": thread_object,
            "thread_object_role": "still present as the world shifts, now carrying unease instead of comfort",
            "has_character": True,
        },
        {
            "scene_number": 4,
            "emotional_beat": "the_weight",
            "narration": "I touched each thing like it might remember me.",
            "narrator_voice": "old_woman_remembering",
            "veo_prompt": "Close-up of hands, slow dolly-in. Hands close, lift, or place meaningful objects with the thread object clearly visible. Back in the primary location under the same anchored light and weather. Audio: cloth rustle, wood, quiet breathing. No text, no subtitles, no logos, no title cards.",
            "veo_audio_cue": "cloth rustle, wood, quiet breathing",
            "image_prompt": "Close-up of hands making a careful irreversible choice with meaningful objects, intimate light, physical gestures of resolve",
            "music_mood": "resolve under sorrow",
            "thread_object": thread_object,
            "thread_object_role": "handled deliberately as it takes on the meaning of a promise or farewell",
            "has_character": True,
        },
        {
            "scene_number": 5,
            "emotional_beat": "the_goodbye",
            "narration": "I left it where love would find it after me.",
            "narrator_voice": "old_woman_remembering",
            "veo_prompt": "Medium profile shot, static camera. The character pauses in dramatic side light while leaving the thread object behind. A place just beyond the primary location, same weather pressing in. Audio: one breath, soft footstep, near-silence. No text, no subtitles, no logos, no title cards.",
            "veo_audio_cue": "one breath, candle flicker, near-silence",
            "image_prompt": "A tender room with a candle, doorway light, and the feeling that someone has just left, intimate sorrow without showing faces",
            "music_mood": "near-silent heartbreak",
            "thread_object": thread_object,
            "thread_object_role": "left behind or passed on in silence at the point of goodbye",
            "has_character": True,
        },
        {
            "scene_number": 6,
            "emotional_beat": "the_stand",
            "narration": "",
            "narrator_voice": "old_woman_remembering",
            "veo_prompt": "Wide environmental shot, static camera. The character appears as a silhouette holding steady against the world, body turned away from camera. Same anchored weather, same light now harsher at the edge of day. Audio: wind, echo, distant surf. No text, no subtitles, no logos, no title cards.",
            "veo_audio_cue": "wind, echo, dust settling",
            "image_prompt": "Silhouette of a single figure in strong backlight, long shadow, monumental and still, transformation through posture only",
            "music_mood": "monumental tragic peak",
            "thread_object": thread_object,
            "thread_object_role": "absent from the figure, making the sacrifice legible through what is no longer carried",
            "has_character": True,
        },
        {
            "scene_number": 7,
            "emotional_beat": "the_stillness",
            "narration": "",
            "narrator_voice": "old_woman_remembering",
            "veo_prompt": "Static wide shot. The primary location stands empty under the same weather and light, with one object missing and the absence doing the storytelling. No people in frame. Audio: morning birds, gentle wind, profound stillness. No text, no subtitles, no logos, no title cards.",
            "veo_audio_cue": "morning birds, gentle wind, profound stillness",
            "image_prompt": "Empty place at dawn with one meaningful object left behind, no people, peaceful stillness that carries loss",
            "music_mood": "devastated earned quiet",
            "thread_object": thread_object,
            "thread_object_role": "missing or left behind, so its absence becomes the emotional center of the frame",
            "has_character": False,
        },
        {
            "scene_number": 8,
            "emotional_beat": "what_remains",
            "narration": "My hands still remember what the world asked of them.",
            "narrator_voice": "old_woman_remembering",
            "veo_prompt": f"Close-up, static camera. Hands echo scene 1 exactly, repeating the same gesture in the same primary location with changed time inside the same weather pattern. {visual_style}. Audio: the first birdsong returning, gentle room tone. No text, no subtitles, no logos, no title cards.",
            "veo_audio_cue": "the first birdsong returning, gentle room tone",
            "image_prompt": f"Echo of scene 1 with the same gesture transformed by time, cinematic closure, final image evoking {final_image}",
            "music_mood": "bittersweet full-circle resolution",
            "thread_object": thread_object,
            "thread_object_role": "returned in changed hands as proof that the legacy continues",
            "has_character": True,
        },
    ]
    for scene in fallback:
        scene["narrator_profile"] = map_narrator_to_profile(scene.get("narrator_voice", ""))
        scene["narration_pause_before"] = DEFAULT_NARRATION_PAUSES.get(int(scene["scene_number"]), 0.5)
        defaults = _tts_defaults_for_voice(scene["narrator_profile"])
        scene["tts_rate"] = defaults["rate"]
        scene["tts_pitch"] = defaults["pitch"]
        scene["narration_ssml"] = _build_narration_ssml(scene["narration"], scene["narrator_profile"])
        scene["format"] = "video"
    return fallback


def _fallback_film_blueprint(story_context: Dict[str, Any]) -> Dict[str, Any]:
    scenes = [dict(scene) for scene in _fallback_emotional_script(story_context)]
    character_bible = _default_character_bible(story_context)
    thread_object = _default_thread_object(story_context)
    visual_style_anchor = _default_visual_style_anchor(story_context)
    music_segments: List[Dict[str, str]] = []
    for segment_number in range(1, 5):
        start = (segment_number - 1) * 2
        segment_slice = scenes[start:start + 2]
        fallback_prompt = _segment_prompt_fallback(story_context, segment_slice, segment_number)
        music_segments.append(
            {
                "segment": segment_number,
                "prompt": _normalize_music_segment_prompt("", fallback_prompt),
                "negative_prompt": "vocals, singing, speech, voice",
            }
        )
    title = re.sub(r"\s+", " ", str(story_context.get("title", "")).strip()) or "Untitled Light"
    return {
        "character_bible": character_bible,
        "thread_object": thread_object,
        "visual_style_anchor": visual_style_anchor,
        "scenes": scenes,
        "music_segments": music_segments,
        "silence_after_scene_5": 3.0,
        "silence_after_scene_7": 2.0,
        "title": title,
    }


def _normalize_music_segments(raw_segments: Any, fallback_segments: List[Dict[str, str]]) -> List[Dict[str, str]]:
    normalized_segments: List[Dict[str, str]] = []
    for segment_number in range(1, 5):
        default_segment = fallback_segments[segment_number - 1]
        raw_segment = (
            raw_segments[segment_number - 1]
            if isinstance(raw_segments, list)
            and segment_number - 1 < len(raw_segments)
            and isinstance(raw_segments[segment_number - 1], dict)
            else {}
        )
        prompt_text = _normalize_music_segment_prompt(
            str(raw_segment.get("prompt", "")).strip(),
            str(default_segment.get("prompt", "")),
        )
        negative_prompt = re.sub(r"\s+", " ", str(raw_segment.get("negative_prompt", "")).strip()) or "vocals, singing, speech, voice"
        normalized_segments.append(
            {
                "segment": segment_number,
                "prompt": prompt_text,
                "negative_prompt": negative_prompt,
            }
        )
    return normalized_segments


def generate_film_blueprint(story_context: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    """Generate the full film blueprint in one call."""
    client = genai.Client(**vertex_client_kwargs(settings))
    prompt = f"""You are a film director creating a 90-second short film. You will output a complete production blueprint.

USER'S STORY:
Who: {story_context.get('character_essence', '')}
Why they matter: {story_context.get('emotional_anchor', '')}
World: {story_context.get('world', '')}
The turn: {story_context.get('the_turn', '')}
What remains: {story_context.get('residual_feeling', '')}

═══ STEP 1: CHARACTER BIBLE ═══

Write a PHYSICAL DESCRIPTION of the main character in exactly 40 words. This description will be copy-pasted into every scene prompt to maintain consistency.

Include: approximate age (must be 25+), ETHNICITY OR SKIN TONE explicitly, one facial feature, one hair detail, one clothing item with a color, one distinctive physical detail (scars, stains, jewelry, tattoo, glasses).

Example: "a South Asian man in his early 50s with deep brown skin, a short dark beard flecked with gray, kind brown eyes, wearing a sand-colored linen tunic with ink stains on the cuffs, strong calloused hands"

═══ STEP 2: THREAD OBJECT ═══

Choose ONE small physical object that appears throughout the film. It must be:
- Something the character MAKES, CARRIES, or GIVES
- Small enough to hold in one hand
- Visually distinctive enough to recognize in each scene
- Present in scenes 1, 4, 5, and 8 minimum

Write a 15-word description of this object.

Example: "a small hand-carved wooden bird with spread wings, pale unfinished wood"

═══ STEP 3: VISUAL STYLE ANCHOR ═══

Write ONE sentence that defines the visual look of the entire film. This sentence will be appended to every scene prompt.

Include: color palette, film reference or era, lighting quality, one texture detail.

Example: "Warm earth tones, Terrence Malick golden hour aesthetic, soft natural light, visible film grain and dust motes"

═══ STEP 4: EIGHT SCENES ═══

Write 8 scenes following this emotional structure. For EACH scene output ALL of these fields:

A) emotional_beat: one of warmth, context, crack, weight, goodbye, stand, stillness, remains

B) narration: 5-12 words MAXIMUM. Sparse. Poetic. Written to be spoken aloud. ALL narration must be FIRST PERSON. The narrator IS the main character, speaking from memory.

GOOD narration:
- "I carved it the night before she left."
- "The sea was different that morning."
- "I knew. I think I always knew."
- "My hands still remember the weight of it."

BAD narration:
- "He carved a bird for his daughter."
- "The old man walked to the shore."
- "She watched as the sun set over the water."
- "Their bond was unbreakable."

Each line must be speakable in under 5 seconds at a slow deliberate pace. If it cannot be said in one breath, it is too long.

Narration map:
- Scene 1: YES
- Scene 2: OPTIONAL, may be empty
- Scene 3: YES
- Scene 4: YES
- Scene 5: YES
- Scene 6: NO narration preferred
- Scene 7: NO narration
- Scene 8: YES

C) narrator_voice: who speaks this line. Pick from:
   - "old_woman_remembering"
   - "old_man_remembering"
   - "young_woman_present"
   - "young_man_present"
   Use maximum 3 different voices across 8 scenes.

D) narration_pause_before: seconds of silence before narration starts. Use 1.5 for scenes with narration and 0.0 for silent scenes.

E) veo_prompt: Follow this EXACT formula —

   [SHOT TYPE + CAMERA MOVEMENT]. [CHARACTER BIBLE text — paste the 40-word description from Step 1 if the character is in frame, or describe what IS in frame]. [ONE concrete action verb]. [SETTING with time of day and weather]. [VISUAL STYLE ANCHOR from Step 3]. Audio: [2-3 specific sounds]. No text, no subtitles, no logos, no title cards.

   CRITICAL VEO RULES:
   - Start EVERY prompt with shot type and camera movement
   - ONE subject, ONE action, ONE verb per scene
   - If the character is in frame, paste the FULL character bible (40 words) — do not abbreviate or rephrase it
   - Include the thread object when it appears
   - Action verbs must be PHYSICAL: carves, walks, closes, stands, places, turns, holds — never "feels" or "experiences"
   - Audio line must name SPECIFIC sounds: "knife on wood, distant birdsong" not "peaceful sounds"
   - Total prompt: 60-100 words
   - Scene 7 must have NO PERSON in frame — only objects, space, light
   - Scene 8 must MIRROR Scene 1's shot type and framing
   - Avoid frontal face shots. Use body, clothing, hands, silhouettes, profiles, backs, and over-the-shoulder compositions.

   CAMERA RULES FOR CHARACTER CONSISTENCY:
   - Allowed shots: close-up of hands, medium shot from behind, wide silhouette, over-the-shoulder, low angle with face in shadow, profile shot in dim lighting
   - Forbidden shots: frontal close-up of face, clearly lit front-facing medium shot, both eyes clearly visible, tracking shot from the front

   SPECIFIC SHOT ASSIGNMENTS:
   - Scene 1: close-up of hands doing the character's defining action
   - Scene 2: wide aerial or establishing, character is a small figure in landscape
   - Scene 3: medium from behind, character facing away
   - Scene 4: close-up of hands again, with thread object
   - Scene 5: profile in dramatic light
   - Scene 6: wide shot, character as silhouette
   - Scene 7: no person, empty space only
   - Scene 8: close-up of hands mirroring Scene 1 exactly

   LOCATION ANCHORING RULES:
   - Define ONE primary location and return to it in scenes 1, 4, and 8
   - Scene 1 and Scene 8 must happen in the SAME place with the SAME camera angle
   - Scene 2 must show the wider version of Scene 1's location
   - Scenes can leave the primary location for scenes 3, 5, and 6 only
   - Scene 7 is ALWAYS an empty space with no person
   - The thread object must be visible in scenes 1, 4, 5, and 8

   SHOT PROGRESSION RULES:
   - Scene 1: close-up or medium close-up
   - Scene 2: wide or aerial
   - Scene 3: medium shot
   - Scene 4: close-up of hands doing something
   - Scene 5: medium shot
   - Scene 6: wide or environmental
   - Scene 7: static wide shot, no person
   - Scene 8: SAME framing as Scene 1

   SETTING ANCHORING:
   - Choose a specific time of day and weather for the whole film
   - All 8 scenes happen in the same conditions unless the story clearly demands a time change
   - Put that same time-of-day and weather detail into every veo_prompt

F) image_prompt: A single-frame version of the veo_prompt for Ken Burns fallback if Veo fails. 30-40 words.

G) music_mood: 4-6 words describing this scene's music

H) has_character: true if the main character appears in this scene, false if it's objects/space/landscape only

═══ STEP 5: MUSIC DIRECTION ═══

Write 4 music segment prompts (30 seconds each). Each must include: specific instruments, tempo feel, emotional quality, and the word "instrumental" at the end.

Segment 1 (scenes 1-2): Establishing, gentle, sparse
Segment 2 (scenes 3-4): Building tension, more instruments
Segment 3 (scenes 5-6): Emotional peak OR silence
Segment 4 (scenes 7-8): Resolution, return to opening instrument, gentle fade

═══ STEP 6: SILENCE MAP ═══

Specify which scenes have BLACK SILENCE after them:
- After scene 5: how many seconds
- After scene 7: how many seconds

═══ OUTPUT FORMAT ═══

Return this exact JSON structure:

{{
  "character_bible": "40-word physical description...",
  "thread_object": "15-word object description...",
  "visual_style_anchor": "one sentence visual style...",
  "scenes": [
    {{
      "scene_number": 1,
      "emotional_beat": "warmth",
      "narration": "...",
      "narrator_voice": "...",
      "narration_pause_before": 1.5,
      "veo_prompt": "...",
      "image_prompt": "...",
      "music_mood": "...",
      "has_character": true
    }}
  ],
  "music_segments": [
    {{
      "segment": 1,
      "prompt": "full music generation prompt..., instrumental",
      "negative_prompt": "vocals, singing, speech, voice"
    }}
  ],
  "silence_after_scene_5": 3.0,
  "silence_after_scene_7": 2.0,
  "title": "short film title"
}}"""

    fallback = _fallback_film_blueprint(story_context)
    try:
        response = client.models.generate_content(
            model=settings.script_routing_model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        data = _parse_json_response(response.text)
    except Exception as exc:
        logger.warning("Film blueprint generation failed, using fallback blueprint. (%s)", exc)
        return fallback

    if not isinstance(data, dict):
        return fallback

    character_bible = _normalize_character_bible(
        str(data.get("character_bible", "")).strip(),
        fallback["character_bible"],
    )
    thread_object = _limit_words(str(data.get("thread_object", "")).strip(), 15) or fallback["thread_object"]
    visual_style_anchor = re.sub(r"\s+", " ", str(data.get("visual_style_anchor", "")).strip()) or fallback["visual_style_anchor"]
    title = re.sub(r"\s+", " ", str(data.get("title", "")).strip()) or fallback["title"]

    raw_scenes = data.get("scenes")
    scenes_by_number: Dict[int, Dict[str, Any]] = {}
    for index, item in enumerate(raw_scenes if isinstance(raw_scenes, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        try:
            scene_number = int(item.get("scene_number", index))
        except Exception:
            scene_number = index
        scenes_by_number[scene_number] = item

    normalized_scenes: List[Dict[str, Any]] = []
    for index, default_scene in enumerate(fallback["scenes"], start=1):
        item = scenes_by_number.get(index, {})
        narrator_voice = _normalize_narrator_voice(
            str(item.get("narrator_voice", "")).strip(),
            str(default_scene.get("narrator_voice", "old_woman_remembering")),
        )
        narrator_profile = map_narrator_to_profile(narrator_voice)
        tts_defaults = _tts_defaults_for_voice(narrator_profile)
        veo_prompt = re.sub(r"\s+", " ", str(item.get("veo_prompt", "")).strip()) or str(default_scene["veo_prompt"])
        veo_audio_cue = _extract_audio_cue_from_prompt(
            veo_prompt,
            str(default_scene.get("veo_audio_cue", "")),
        )
        if "narration" in item:
            item_narration = _clean_narration_line(item.get("narration", ""))
        elif index in {2, 6, 7}:
            item_narration = ""
        else:
            item_narration = _clean_narration_line(default_scene["narration"])

        scene = {
            "scene_number": index,
            "emotional_beat": str(item.get("emotional_beat", default_scene["emotional_beat"])).strip() or default_scene["emotional_beat"],
            "narration": _limit_words(item_narration, 12),
            "narrator_voice": narrator_voice,
            "narrator_profile": narrator_profile,
            "narration_pause_before": float(item.get("narration_pause_before", default_scene.get("narration_pause_before", DEFAULT_NARRATION_PAUSES.get(index, 0.5))) or DEFAULT_NARRATION_PAUSES.get(index, 0.5)),
            "veo_prompt": veo_prompt,
            "veo_audio_cue": veo_audio_cue,
            "image_prompt": re.sub(r"\s+", " ", str(item.get("image_prompt", "")).strip()) or str(default_scene["image_prompt"]),
            "music_mood": re.sub(r"\s+", " ", str(item.get("music_mood", "")).strip()) or str(default_scene["music_mood"]),
            "thread_object": thread_object,
            "thread_object_role": re.sub(r"\s+", " ", str(item.get("thread_object_role", "")).strip()) or str(default_scene.get("thread_object_role", "")),
            "has_character": bool(item.get("has_character", default_scene.get("has_character", index != 7))),
            "format": "video",
            "tts_rate": float(item.get("tts_rate", default_scene.get("tts_rate", tts_defaults["rate"])) or tts_defaults["rate"]),
            "tts_pitch": float(item.get("tts_pitch", default_scene.get("tts_pitch", tts_defaults["pitch"])) or tts_defaults["pitch"]),
        }
        scene["narration_ssml"] = _build_narration_ssml(scene["narration"], narrator_profile)
        normalized_scenes.append(scene)

    silence_after_scene_5 = float(data.get("silence_after_scene_5", fallback["silence_after_scene_5"]) or fallback["silence_after_scene_5"])
    silence_after_scene_7 = float(data.get("silence_after_scene_7", fallback["silence_after_scene_7"]) or fallback["silence_after_scene_7"])
    music_segments = _normalize_music_segments(data.get("music_segments"), fallback["music_segments"])

    return {
        "character_bible": character_bible,
        "thread_object": thread_object,
        "visual_style_anchor": visual_style_anchor,
        "scenes": normalized_scenes,
        "music_segments": music_segments,
        "silence_after_scene_5": silence_after_scene_5,
        "silence_after_scene_7": silence_after_scene_7,
        "title": title,
    }


def generate_emotional_script(story_context: Dict[str, Any], settings: Settings) -> List[Dict[str, Any]]:
    """Compatibility wrapper returning the scenes list from the film blueprint."""
    return list(generate_film_blueprint(story_context, settings).get("scenes", []))


def _enforce_character_bible_prompt(blueprint: Dict[str, Any], scene: Dict[str, Any]) -> str:
    bible = re.sub(r"\s+", " ", str(blueprint.get("character_bible", "")).strip())
    style = re.sub(r"\s+", " ", str(blueprint.get("visual_style_anchor", "")).strip())
    prompt = re.sub(r"\s+", " ", str(scene.get("veo_prompt", "")).strip())
    no_speech_clause = "No dialogue, no speech, no spoken words, no voices."
    if not prompt:
        return prompt

    if not bool(scene.get("has_character", False)):
        if style and style not in prompt:
            prompt = prompt.rstrip(". ") + ". " + style
        if "no dialogue" not in prompt.lower():
            if "no text" in prompt.lower():
                prompt = re.sub(
                    r"(?i)no text,\s*no subtitles",
                    f"{no_speech_clause} No text, no subtitles",
                    prompt,
                    count=1,
                )
            else:
                prompt = prompt.rstrip(". ") + f". {no_speech_clause}"
        if "no text" not in prompt.lower():
            prompt += " No text, no subtitles, no logos, no title cards."
        return re.sub(r"\s+", " ", prompt).strip()

    if bible:
        parts = prompt.split(".", 1)
        camera = parts[0].strip() + "."
        rest = parts[1].strip() if len(parts) > 1 else ""
        rest_sentences = [sentence.strip() for sentence in rest.split(".") if sentence.strip()]
        if len(rest_sentences) > 1:
            action_and_rest = ". ".join(rest_sentences[1:])
        else:
            action_and_rest = ". ".join(rest_sentences)
        if action_and_rest:
            prompt = f"{camera} {bible}. {action_and_rest}"
        else:
            prompt = f"{camera} {bible}."

    if style and style not in prompt:
        prompt = prompt.rstrip(". ") + ". " + style

    if "no dialogue" not in prompt.lower():
        if "no text" in prompt.lower():
            prompt = re.sub(
                r"(?i)no text,\s*no subtitles",
                f"{no_speech_clause} No text, no subtitles",
                prompt,
                count=1,
            )
        else:
            prompt = prompt.rstrip(". ") + f". {no_speech_clause}"

    if "no text" not in prompt.lower():
        prompt = f"{prompt} No text, no subtitles, no logos, no title cards."

    return re.sub(r"\s+", " ", prompt).strip()


def enforce_character_bible(blueprint: Dict[str, Any], scene: Dict[str, Any] | None = None) -> Dict[str, Any] | str:
    """Enforce the exact character bible across a whole blueprint or a single scene prompt."""
    if scene is not None:
        return _enforce_character_bible_prompt(blueprint, scene)

    scenes = blueprint.get("scenes", [])
    if not isinstance(scenes, list):
        return blueprint

    for item in scenes:
        if isinstance(item, dict):
            item["veo_prompt"] = _enforce_character_bible_prompt(blueprint, item)
    return blueprint


def _safe_alternative_for_scene(scene: Dict[str, Any]) -> str:
    image_prompt = str(scene.get("image_prompt", "")).strip()
    narration = str(scene.get("narration", "")).strip()
    source = image_prompt or narration or "a cinematic emotional moment"
    return (
        f"Close-up or wide cinematic detail implying this moment without visible faces or conflict: {source}. "
        "Use hands, silhouettes, empty space, meaningful objects, or nature. "
        "Safe for a strict video model, under warm or dramatic natural light, with subtle camera motion."
    )


def classify_veo_safety(script: List[Dict[str, Any]], settings: Settings) -> List[Dict[str, Any]]:
    """Predict which scenes are safe for Veo and attach a conservative fallback alternative."""
    if not script:
        return []

    scene_descriptions = "\n".join(
        [
            f"Scene {i + 1}: {s.get('veo_prompt', s.get('image_prompt', ''))}"
            for i, s in enumerate(script)
        ]
    )

    prompt = f"""You are a content safety classifier for a video generation model. The model will REJECT any scene that shows or implies:

WILL BE REJECTED:
- Any child, minor, or anyone who appears under 25
- People in distress, danger, injury, or conflict
- Weapons of any kind, even historical/decorative
- Physical confrontation or aggressive postures
- Death or injury shown on a person
- A person alone in a dark or threatening environment
- Military, armor, or combat-related clothing
- Crowds in tense or confrontational situations

WILL PASS:
- Adult faces and expressions in calm, non-violent situations
- Adults sitting, reading, working, watching, pausing, speaking, or standing still
- Landscapes, architecture, nature without people
- Objects alone
- Hands only
- Silhouettes where the face is not identifiable
- Backs of people, no face visible
- Wide shots where people are tiny or distant
- Empty rooms, courtyards, paths
- Sky, clouds, sunrise, sunset
- Animals, plants, water
- Abstract or metaphorical imagery

For each scene, classify as SAFE or UNSAFE for the video model.
Adult people are allowed if they are clearly adults and not in danger.
If UNSAFE, also provide a safe_alternative that shows the same emotional beat without minors, conflict, weapons, or danger.
Be conservative.

SCENES:
{scene_descriptions}

Return JSON array:
[
  {{"scene_number": 1, "veo_safe": true, "veo_needs_character_ref": false, "reason": "...", "safe_alternative": ""}}
]"""

    fallback: List[Dict[str, Any]] = []
    for index, scene in enumerate(script, start=1):
        scene_text = f"{scene.get('veo_prompt', '')} {scene.get('image_prompt', '')} {scene.get('narration', '')}".lower()
        unsafe = any(
            token in scene_text
            for token in (
                "child",
                "minor",
                "kid",
                "girl",
                "boy",
                "teen",
                "teenage",
                "school-age",
                "toddler",
                "infant",
                "baby",
                "kiss",
                "cry",
                "tears",
                "weapon",
                "sword",
                "gun",
                "army",
                "soldier",
                "attack",
                "fight",
                "battle",
                "dead",
                "death",
                "injury",
                "blood",
                "distress",
            )
        )
        fallback.append(
            {
                "scene_number": index,
                "veo_safe": not unsafe,
                "reason": "fallback heuristic",
                "safe_alternative": _safe_alternative_for_scene(scene),
            }
        )

    try:
        client = genai.Client(**vertex_client_kwargs(settings))
        response = client.models.generate_content(
            model=settings.script_routing_model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        data = _parse_json_response(response.text)
        if isinstance(data, list):
            by_number = {}
            for index, item in enumerate(data, start=1):
                if not isinstance(item, dict):
                    continue
                try:
                    scene_number = int(item.get("scene_number", index))
                except Exception:
                    scene_number = index
                by_number[scene_number] = item

            enriched: List[Dict[str, Any]] = []
            for index, scene in enumerate(script, start=1):
                info = by_number.get(index, {})
                safe_alternative = re.sub(r"\s+", " ", str(info.get("safe_alternative", "")).strip()) or fallback[index - 1]["safe_alternative"]
                enriched.append(
                    {
                        **scene,
                        "veo_safe": bool(info.get("veo_safe", False)),
                        "veo_needs_character_ref": bool(info.get("veo_needs_character_ref", False)),
                        "veo_reason": str(info.get("reason", "")).strip() or fallback[index - 1]["reason"],
                        "safe_alternative": safe_alternative,
                    }
                )
            return enriched
    except Exception as exc:
        logger.warning("Veo safety classification failed; using conservative fallback. (%s)", exc)

    enriched = []
    for index, scene in enumerate(script, start=1):
        info = fallback[index - 1]
        enriched.append(
            {
                **scene,
                "veo_safe": bool(info["veo_safe"]),
                "veo_needs_character_ref": False,
                "veo_reason": str(info["reason"]),
                "safe_alternative": str(info["safe_alternative"]),
            }
        )
    return enriched


def generate_dynamic_script(
    story_context: Dict[str, Any],
    num_scenes: int,
    act_label: str,
    settings: Settings,
) -> List[Dict[str, str]]:
    """Generate a dynamic scene script for custom stories.

    Returns list of dicts matching preset format:
    [{"narration": ..., "image_prompt": ..., "music_mood": ...}, ...]
    """
    client = genai.Client(**vertex_client_kwargs(settings))
    ctx_lines = []
    for key in ["setting", "character", "visual_style", "inciting_incident",
                 "transformation", "climax", "resolution"]:
        val = str(story_context.get(key, "")).strip()
        if val:
            ctx_lines.append(f"- {key}: {val}")

    prompt = (
        f"Create a {num_scenes}-scene script for {act_label} of this story:\n"
        + "\n".join(ctx_lines) + "\n\n"
        f"Return a JSON array with {num_scenes} objects. Each must have:\n"
        '- "narration": One evocative sentence (15-25 words)\n'
        '- "image_prompt": Cinematic image description (20-30 words)\n'
        '- "music_mood": Two-word mood (e.g., "tender sorrow")\n'
    )
    response = client.models.generate_content(
        model=settings.script_routing_model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        data = _parse_json_response(response.text)
    except Exception:
        # Fallback: generate simple placeholder scenes
        return [{"narration": f"Scene {i+1}", "image_prompt": story_context.get("setting", ""), "music_mood": "cinematic"} for i in range(num_scenes)]

    if isinstance(data, dict) and "scenes" in data:
        data = data["scenes"]
    scenes: List[Dict[str, str]] = []
    for item in (data if isinstance(data, list) else [])[:num_scenes]:
        scenes.append({
            "narration": str(item.get("narration", "")),
            "image_prompt": str(item.get("image_prompt", "")),
            "music_mood": str(item.get("music_mood", "cinematic")),
        })
    return scenes


def generate_story_outline(
    story_context: Dict[str, Any],
    num_scenes: int,
    settings: Settings,
) -> List[Dict[str, str]]:
    """Create a locked chronological scene outline for a custom story."""
    client = genai.Client(**vertex_client_kwargs(settings))
    ctx_lines = []
    for key in [
        "setting",
        "character",
        "visual_style",
        "inciting_incident",
        "transformation",
        "climax",
        "resolution",
        "music_preference",
    ]:
        val = str(story_context.get(key, "")).strip()
        if val:
            ctx_lines.append(f"- {key}: {val}")

    prompt = (
        f"Create a locked {num_scenes}-scene film outline from this story context:\n"
        + "\n".join(ctx_lines)
        + "\n\nReturn a JSON array with exactly "
        f"{num_scenes} objects.\n"
        "Each object must contain:\n"
        '- "scene_number": integer\n'
        '- "story_beat": one clear causal beat in chronological order\n'
        '- "visual_focus": what the camera must show in this scene\n'
        '- "music_mood": two-to-four words for the cue mood\n\n'
        "Rules:\n"
        "- Scene 1 begins the story; scene numbers strictly increase.\n"
        "- Every scene must happen after the previous scene. No flashbacks.\n"
        "- Do not repeat the same action beat in multiple scenes.\n"
        "- If the story is an escape, do not reach the escape vehicle until late scenes.\n"
        "- Preserve cause and effect: each beat must logically lead to the next.\n"
        "- The final scene must reflect the ending feeling or resolution.\n"
        "- Output JSON only."
    )
    response = client.models.generate_content(
        model=settings.script_routing_model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        data = _parse_json_response(response.text)
    except Exception:
        data = []

    outline: List[Dict[str, str]] = []
    for index, item in enumerate((data if isinstance(data, list) else [])[:num_scenes], start=1):
        outline.append({
            "scene_number": str(item.get("scene_number", index)),
            "story_beat": str(item.get("story_beat", "")).strip(),
            "visual_focus": str(item.get("visual_focus", "")).strip(),
            "music_mood": str(item.get("music_mood", "cinematic")).strip() or "cinematic",
        })

    if len(outline) == num_scenes:
        return outline

    # Fallback outline keeps chronology explicit even when the model response is unusable.
    fallback_beats = [
        "Introduce the protagonist in their ordinary world.",
        "Show the world tightening around them and hint at the coming disruption.",
        "The inciting incident breaks the ordinary routine.",
        "They choose to act and move toward danger or possibility.",
        "The plan deepens and the pressure escalates.",
        "The protagonist reaches the point of no return.",
        "The climax forces the defining choice or confrontation.",
        "Aftermath: reveal what remains and what has changed.",
    ]
    return [
        {
            "scene_number": str(index + 1),
            "story_beat": fallback_beats[index],
            "visual_focus": fallback_beats[index],
            "music_mood": "cinematic",
        }
        for index in range(num_scenes)
    ]


def generate_script_from_outline(
    story_context: Dict[str, Any],
    outline: List[Dict[str, str]],
    settings: Settings,
) -> List[Dict[str, str]]:
    """Turn a locked outline into the final narrated/image scene script."""
    client = genai.Client(**vertex_client_kwargs(settings))
    ctx_lines = []
    for key in [
        "setting",
        "character",
        "visual_style",
        "inciting_incident",
        "transformation",
        "climax",
        "resolution",
        "music_preference",
    ]:
        val = str(story_context.get(key, "")).strip()
        if val:
            ctx_lines.append(f"- {key}: {val}")

    outline_lines = [
        f"{item.get('scene_number', index + 1)}. "
        f"beat={item.get('story_beat', '')} | "
        f"visual_focus={item.get('visual_focus', '')} | "
        f"music_mood={item.get('music_mood', '')}"
        for index, item in enumerate(outline)
    ]

    prompt = (
        "Write the final scene script for this film using the locked outline below.\n\n"
        "Story context:\n"
        + "\n".join(ctx_lines)
        + "\n\nLocked outline:\n"
        + "\n".join(outline_lines)
        + "\n\nReturn a JSON array with one object per outline scene."
        "\nEach object must contain:\n"
        '- "scene_number": integer\n'
        '- "narration": One evocative sentence (15-25 words)\n'
        '- "image_prompt": Cinematic image description (20-35 words)\n'
        '- "music_mood": Two-to-four words\n\n'
        "Rules:\n"
        "- Preserve the exact scene order from the outline.\n"
        "- Do not introduce new major beats or reorder actions.\n"
        "- Make each scene visually distinct and causally connected.\n"
        "- Output JSON only."
    )
    response = client.models.generate_content(
        model=settings.script_routing_model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        data = _parse_json_response(response.text)
    except Exception:
        data = []

    scenes: List[Dict[str, str]] = []
    by_number: Dict[int, Dict[str, Any]] = {}
    for item in (data if isinstance(data, list) else []):
        try:
            scene_number = int(item.get("scene_number"))
        except Exception:
            continue
        by_number[scene_number] = item

    for index, outline_item in enumerate(outline, start=1):
        item = by_number.get(index, {})
        beat = str(outline_item.get("story_beat", "")).strip()
        visual_focus = str(outline_item.get("visual_focus", "")).strip()
        music_mood = str(item.get("music_mood", "")).strip() or str(outline_item.get("music_mood", "cinematic"))
        scenes.append({
            "narration": str(item.get("narration", "")).strip() or beat or f"Scene {index}",
            "image_prompt": str(item.get("image_prompt", "")).strip() or visual_focus or beat,
            "music_mood": music_mood,
        })
    return scenes


def _story_direction_text(story_context: Dict[str, Any], script: List[Dict[str, Any]]) -> str:
    chunks = [
        str(story_context.get("setting", "")),
        str(story_context.get("character", "")),
        str(story_context.get("visual_style", "")),
        str(story_context.get("inciting_incident", "")),
        str(story_context.get("resolution", "")),
    ]
    for scene in script:
        chunks.append(str(scene.get("narration", "")))
        chunks.append(str(scene.get("image_prompt", "")))
    return " ".join(chunks).lower()


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _primary_narrator_profile(story_context: Dict[str, Any], script: List[Dict[str, Any]]) -> str:
    configured = str(story_context.get("narrator_profile", "")).strip().lower()
    if configured in CREATIVE_DIRECTION_PROFILES:
        return configured

    full_text = _story_direction_text(story_context, script)
    if _contains_any(full_text, TECH_STORY_KEYWORDS):
        return "hacker_present_tense"
    if _contains_any(full_text, MEMORY_STORY_KEYWORDS):
        return "older_memory"
    if _contains_any(full_text, APPRENTICE_STORY_KEYWORDS):
        return "young_apprentice"
    if _contains_any(full_text, SAGE_STORY_KEYWORDS):
        return "sage_remembering"
    if _contains_any(full_text, MYTHIC_STORY_KEYWORDS):
        return "mythic_storyteller"
    if any(term in full_text for term in {"girl", "boy", "child", "daughter", "son"}):
        return "quiet_child"
    return "sage_remembering"


def _scene_is_urgent(scene: Dict[str, Any]) -> bool:
    scene_text = f"{scene.get('narration', '')} {scene.get('image_prompt', '')}".lower()
    return _contains_any(scene_text, VIDEO_ACTION_KEYWORDS) or any(
        term in scene_text for term in {"alarm", "danger", "panic", "climax", "high action", "adrenaline"}
    )


def _instrument_palette(story_context: Dict[str, Any]) -> str:
    full_text = _story_direction_text(story_context, [])
    if _contains_any(full_text, TECH_STORY_KEYWORDS):
        return "analog synths, pulse bass, restrained percussion"
    if "medieval" in full_text or "middle eastern" in full_text or "village" in full_text:
        return "oud, low strings, frame drums"
    if _contains_any(full_text, MYTHIC_STORY_KEYWORDS):
        return "flute, harp, low choir, drums"
    if any(term in full_text for term in {"jungle", "observatory", "mist", "valley", "forest"}):
        return "airy flute, marimba, warm strings"
    return "piano, strings, soft percussion"


def _fallback_music_mood(
    story_context: Dict[str, Any],
    scene: Dict[str, Any],
    scene_index: int,
    scene_count: int,
) -> str:
    palette = _instrument_palette(story_context)
    palette_prefix = palette.split(",", 1)[0]
    urgent = _scene_is_urgent(scene)

    if scene_index <= 2:
        return f"gentle {palette_prefix}, worldbuilding"
    if scene_index <= 4:
        return f"rising {palette_prefix}, tension"
    if scene_index <= 6:
        return f"{'urgent' if urgent else 'driving'} {palette_prefix}, climax"
    if scene_index == scene_count:
        return f"bittersweet {palette_prefix}, resolution"
    return f"settling {palette_prefix}, afterglow"


def _fallback_scene_profile(
    story_context: Dict[str, Any],
    scene: Dict[str, Any],
    scene_index: int,
    scene_count: int,
    primary_profile: str,
) -> str:
    scene_text = f"{scene.get('narration', '')} {scene.get('image_prompt', '')}".lower()

    if primary_profile == "hacker_present_tense":
        return "hacker_urgent" if _scene_is_urgent(scene) else "hacker_present_tense"
    if primary_profile == "young_apprentice":
        if scene_index == scene_count and any(term in scene_text for term in {"realized", "wisdom", "acceptance", "dawn"}):
            return "sage_remembering"
        return "urgent_witness" if _scene_is_urgent(scene) else "young_apprentice"
    if primary_profile == "older_memory":
        return "older_memory_aged" if scene_index == scene_count else "older_memory"
    if primary_profile == "sage_remembering":
        return "old_sage" if scene_index == scene_count else "sage_remembering"
    if primary_profile == "quiet_child":
        return "urgent_witness" if _scene_is_urgent(scene) else "quiet_child"
    return primary_profile


def _fallback_scene_formats(script: List[Dict[str, Any]]) -> List[str]:
    formats: List[str] = []
    for index, scene in enumerate(script, start=1):
        scene_text = f"{scene.get('narration', '')} {scene.get('image_prompt', '')}".lower()
        if _contains_any(scene_text, IMAGE_STILLNESS_KEYWORDS) and index not in {5, 6}:
            formats.append("image")
        elif _contains_any(scene_text, VIDEO_ACTION_KEYWORDS):
            formats.append("video")
        elif index in {5, 6}:
            formats.append("video")
        else:
            formats.append("video")

    return _rebalance_scene_formats(formats)


def _rebalance_scene_formats(formats: List[str]) -> List[str]:
    normalized = list(formats)
    if not normalized:
        return normalized

    video_count = sum(1 for value in normalized if value == "video")
    promote_order = [6, 5, 4, 7, 3, 8, 2, 1]
    demote_order = [1, len(normalized), 2, 7, 3, 8, 4]
    min_video_count = max(len(normalized) - 2, 1)
    while video_count < min_video_count:
        promoted = False
        for scene_number in promote_order:
            idx = scene_number - 1
            if 0 <= idx < len(normalized) and normalized[idx] != "video":
                normalized[idx] = "video"
                video_count += 1
                promoted = True
                break
        if not promoted:
            for idx, value in enumerate(normalized):
                if value != "video":
                    normalized[idx] = "video"
                    video_count += 1
                    promoted = True
                    break
        if not promoted:
            break
    max_video_count = len(normalized)
    while video_count > max_video_count:
        demoted = False
        for scene_number in demote_order:
            idx = scene_number - 1
            if 0 <= idx < len(normalized) and normalized[idx] == "video":
                normalized[idx] = "image"
                video_count -= 1
                demoted = True
                break
        if not demoted:
            break

    climax_idx = min(max(len(normalized) - 3, 0), len(normalized) - 1)
    normalized[climax_idx] = "video"
    return normalized


def _fallback_veo_audio_cue(scene: Dict[str, Any]) -> str:
    scene_text = f"{scene.get('narration', '')} {scene.get('image_prompt', '')}".lower()
    if _contains_any(scene_text, {"rain", "storm", "thunder"}):
        return "rain on stone, distant thunder"
    if _contains_any(scene_text, {"battle", "attack", "fighting", "sword", "guns", "fire"}):
        return "rising wind, distant impacts, urgent footsteps"
    if _contains_any(scene_text, {"space", "orbital", "shuttle", "station", "debris"}):
        return "low engine rumble, warning beeps, hull vibration"
    if _contains_any(scene_text, {"jungle", "forest", "meadow", "grass", "birds"}):
        return "wind through leaves, distant birds, soft insects"
    if _contains_any(scene_text, {"village", "courtyard", "home", "room", "lamp"}):
        return "soft room tone, fabric movement, faint distant life"
    if _contains_any(scene_text, {"observatory", "stars", "comet", "sky"}):
        return "night wind, turning gears, distant echo"
    if _contains_any(scene_text, {"quiet", "memory", "grief", "waiting", "watching"}):
        return "silence with gentle breathing and room tone"
    return "natural atmospheric ambience"


def _fallback_veo_prompt(
    story_context: Dict[str, Any],
    scene: Dict[str, Any],
    scene_index: int,
    scene_count: int,
) -> str:
    preset_override = re.sub(r"\s+", " ", str(scene.get("veo_prompt", "")).strip())
    if preset_override:
        return preset_override

    setting = str(story_context.get("setting", "")).strip()
    visual_style = str(story_context.get("visual_style", "cinematic")).strip()
    image_prompt = str(scene.get("image_prompt", "")).strip()
    narration = str(scene.get("narration", "")).strip()
    scene_text = f"{image_prompt} {narration}".lower()
    if scene_index == 1:
        shot = "Wide establishing shot"
        motion = "slow dolly-in"
    elif _contains_any(scene_text, {"battle", "attack", "chase", "run", "escape", "storm"}):
        shot = "Medium reaction shot"
        motion = "gentle push-in"
    elif _contains_any(scene_text, {"close", "hands", "bird", "compass", "lens"}):
        shot = "Close-up"
        motion = "subtle push-in"
    else:
        shot = "Medium shot"
        motion = "gentle lateral drift"

    lighting = "dramatic cinematic lighting"
    if _contains_any(scene_text, {"night", "lamp", "candle", "dawn", "sunrise", "moonlight"}):
        lighting = "moody practical light with strong contrast"

    subject_action = image_prompt or narration or "the protagonist moving through the moment"
    if _contains_any(scene_text, {"battle", "attack", "fighting", "sword", "guns", "burning", "invaders"}):
        subject_action = (
            "the protagonist reacting to distant upheaval as dust, shadows, and anxious faces gather around them"
        )
    elif _contains_any(scene_text, {"death", "mourning", "grave", "loss", "absence"}):
        subject_action = (
            "hands holding a keepsake while dawn light settles over an emptied space and everyone grows still"
        )
    elif _contains_any(scene_text, {"threat", "riders", "army", "commanders", "approaching"}):
        subject_action = (
            "figures watching the horizon as long shadows stretch across the ground and resolve settles into their posture"
        )
    audio_cue = _fallback_veo_audio_cue(scene)
    return (
        f"{shot}, {subject_action}, in {setting}. "
        f"{lighting}, {motion}, {visual_style}, film grain. "
        f"Ambient sound of {audio_cue}."
    )


def rewrite_for_veo_safety(veo_prompt: str, settings: Settings) -> str:
    """Rewrite a Veo prompt to imply conflict instead of depicting it directly."""
    original_prompt = re.sub(r"\s+", " ", str(veo_prompt or "")).strip()
    if not original_prompt:
        return original_prompt

    prompt = f"""You are a cinematic prompt rewriter. Your job is to take a scene description that may contain violence, battle, weapons, death, or conflict and REWRITE it so it conveys the same emotional story through IMPLICATION rather than depiction.

This is for a video generation model with strict safety filters. The rewritten prompt must NOT contain:
- Any child, minor, teenager, or anyone who appears under 25
- Weapons being used aggressively (swords swinging, guns firing)
- Physical combat or fighting
- Blood, injury, death shown directly
- Threatening poses or aggressive confrontation
- Characters in physical danger shown explicitly

Instead, use these cinematic techniques:
- Show the BEFORE: preparation, resolve, ritual, waiting
- Show the AFTER: peaceful aftermath, emptied spaces, dawn light, stillness
- Show the REACTION: adult faces in profile, hands, watching, listening, awe
- Show the METAPHOR: storm clouds, shadows, dust, candles, fallen objects, changing light
- Show the HANDS: holding, closing, placing, letting go
- Show the SILHOUETTE: backlit figures, long shadows, doorway shapes

Adults are allowed. Calm adult faces, profiles, reading, working, sitting, walking, or standing are safe if there is no violence or danger.
If the original prompt implies a child, rewrite that person as a young adult in their mid-twenties or shift to first-person perspective, hands, shadow, or what they see.

ORIGINAL PROMPT:
{original_prompt}

Rewrite the prompt so it preserves the same emotional beat but avoids explicit violence and minors. Keep it cinematic, under 100 words, and suitable for a safety-sensitive video model.

Return ONLY the rewritten prompt. No explanation."""

    try:
        client = genai.Client(**vertex_client_kwargs(settings))
        response = client.models.generate_content(
            model=settings.script_routing_model,
            contents=prompt,
        )
    except Exception as exc:
        logger.warning("Veo safety rewrite failed; using original prompt. (%s)", exc)
        return original_prompt

    rewritten = re.sub(r"\s+", " ", str(getattr(response, "text", "") or "").strip())
    return rewritten or original_prompt


def _limit_voice_variety(
    scenes: List[Dict[str, Any]],
    fallback_scenes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ordered_profiles: List[str] = []
    for scene in scenes:
        profile = str(scene.get("narrator_profile", "")).strip().lower()
        if profile and profile not in ordered_profiles:
            ordered_profiles.append(profile)
    if len(ordered_profiles) <= 3:
        return scenes

    keep_profiles = ordered_profiles[:3]
    replacement = keep_profiles[0]
    normalized: List[Dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        updated = dict(scene)
        profile = str(updated.get("narrator_profile", "")).strip().lower()
        if profile not in keep_profiles:
            fallback_profile = str(fallback_scenes[index].get("narrator_profile", replacement)).strip().lower()
            updated["narrator_profile"] = fallback_profile if fallback_profile in keep_profiles else replacement
        normalized.append(updated)
    return normalized


def _segment_prompt_fallback(
    story_context: Dict[str, Any],
    scene_slice: List[Dict[str, Any]],
    segment_number: int,
) -> str:
    palette = _instrument_palette(story_context)
    instruments = [part.strip() for part in palette.split(",") if part.strip()]
    lead_instruments = ", ".join(instruments[:2]) or "piano, strings"
    support_instrument = instruments[2] if len(instruments) > 2 else "subtle percussion"
    setting = str(story_context.get("setting", "")).strip() or "a cinematic world"
    mood_summary = ", ".join(
        str(scene.get("music_mood", "")).strip()
        for scene in scene_slice
        if str(scene.get("music_mood", "")).strip()
    ) or "emotional continuity"
    energy = {
        1: "soft, sparse, establishing energy",
        2: "building, unsettled, rising energy",
        3: "peak, full, high-stakes energy",
        4: "resolving, stripped-back, gentle landing energy",
    }.get(segment_number, "controlled energy")
    tempo = {
        1: "slow, around 60 BPM",
        2: "medium, around 90 BPM",
        3: "fast, around 120 BPM",
        4: "slow, around 70 BPM",
    }.get(segment_number, "medium, around 90 BPM")
    emotion = {
        1: "tender and watchful",
        2: "restless and tightening",
        3: "defiant and overwhelming",
        4: "bittersweet and accepting",
    }.get(segment_number, "cinematic")
    return (
        f"{lead_instruments} carry a cue for scenes {((segment_number - 1) * 2) + 1}-{min(segment_number * 2, 8)} "
        f"inspired by {setting}, with {support_instrument} underneath, {tempo}, {emotion}, "
        f"{mood_summary}, {energy}, instrumental"
    )


def _normalize_music_segment_prompt(prompt: str, fallback_prompt: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(prompt or "")).strip()
    if not cleaned:
        cleaned = fallback_prompt
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()
    if not cleaned.lower().endswith("instrumental"):
        cleaned = f"{cleaned}, instrumental"
    return cleaned


def _fallback_creative_direction(
    story_context: Dict[str, Any],
    script: List[Dict[str, Any]],
) -> Dict[str, Any]:
    primary_profile = _primary_narrator_profile(story_context, script)
    formats = _fallback_scene_formats(script)
    scenes: List[Dict[str, Any]] = []
    for index, scene in enumerate(script, start=1):
        scenes.append(
            {
                "scene_number": index,
                "narrator_profile": _fallback_scene_profile(
                    story_context,
                    scene,
                    index,
                    len(script),
                    primary_profile,
                ),
                "music_mood": _fallback_music_mood(story_context, scene, index, len(script)),
                "format": formats[index - 1] if index - 1 < len(formats) else "image",
                "veo_prompt": _fallback_veo_prompt(story_context, scene, index, len(script)),
                "veo_audio_cue": _fallback_veo_audio_cue(scene),
            }
        )

    scenes = _limit_voice_variety(scenes, scenes)
    intro_profile = str(story_context.get("intro_narrator_profile", "")).strip().lower()
    if intro_profile not in CREATIVE_DIRECTION_PROFILES:
        intro_profile = scenes[0]["narrator_profile"] if scenes else primary_profile

    music_segments: List[Dict[str, str]] = []
    for segment_number in range(1, 5):
        start = (segment_number - 1) * 2
        segment_slice = scenes[start:start + 2]
        fallback_prompt = _segment_prompt_fallback(story_context, segment_slice, segment_number)
        music_segments.append(
            {
                "segment": segment_number,
                "prompt": _normalize_music_segment_prompt("", fallback_prompt),
                "negative_prompt": MUSIC_SEGMENT_NEGATIVE_PROMPT,
            }
        )

    return {
        "intro_narrator_profile": intro_profile,
        "scenes": scenes,
        "music_segments": music_segments,
    }


def generate_creative_direction(
    story_context: Dict[str, Any],
    script: List[Dict[str, Any]],
    settings: Settings,
) -> Dict[str, Any]:
    """Analyze a complete script and return voice, music, and format direction."""
    if not script:
        return {
            "intro_narrator_profile": "sage_remembering",
            "scenes": [],
            "music_segments": [],
        }

    scene_summaries = "\n".join(
        [
            f"Scene {i + 1}: {s.get('narration', '')} | Visual: {s.get('image_prompt', '')}"
            for i, s in enumerate(script)
        ]
    )
    prompt = f"""You are the creative director for a 2-minute cinematic short film. Analyze this {len(script)}-scene script and make three decisions per scene.

STORY CONTEXT:
Setting: {story_context.get('setting', '')}
Character: {story_context.get('character', '')}
Visual style: {story_context.get('visual_style', '')}

SCRIPT:
{scene_summaries}

For each scene, decide:

1. NARRATOR VOICE — Pick from these profiles:
   - "older_memory": Warm female, slow, reflective. Use for adult looking back at past, maternal wisdom, gentle grief.
   - "older_memory_aged": Same voice but deeper/slower. Use for final scenes with deep emotion, endings, legacy moments.
   - "quiet_child": Young, soft, innocent. Use for childhood scenes, wonder, vulnerability, youth perspective.
   - "urgent_witness": Fast, tense, present-tense energy. Use for action scenes, chase, battle, time pressure.
   - "mythic_storyteller": Deep, resonant, timeless. Use for epic/fantasy narration, myth, legend, dragon/magic scenes.
   - "hacker_present_tense": Cool, modern, observational. Use for sci-fi, tech, cyberpunk, clinical observation.
   - "hacker_urgent": Same but faster/higher. Use for tech scenes with danger, system alerts, escape sequences.
   - "young_apprentice": Bright, curious, slightly nervous. Use for learning moments, discovery, coming-of-age.
   - "old_sage": Very slow, very deep, wise. Use for wisdom, endings told by someone who lived it, peaceful acceptance.
   - "sage_remembering": Medium pace, warm depth. Use for memories, looking back with fondness, bittersweet recall.

   Rules:
   - Match the narrator identity, not just the age shown in the frame.
   - Use at most 3 different narrator profiles across the full film.
   - Reflect the emotional arc in any voice shifts.

2. MUSIC MOOD — Write a short scene music description for each scene, 4-8 words.
   - Scenes 1-2 establish the world.
   - Scenes 3-4 build tension.
   - Scenes 5-6 are the climax.
   - Scenes 7-8 resolve into a changed calm.
   - Include instrument hints that fit the setting.
   - Use specific emotional language like grieving, defiant, tender, mournful, awestruck, or haunted.

3. FORMAT — Pick "video" or "image" for each scene.
   - Default to "video". Veo is the primary engine for this film.
   - Only choose "image" for very quiet, contemplative, or emotionally still scenes that truly benefit from painterly stillness.
   - Use video for most scenes, especially movement, scale, tension, weather, travel, and climax.
   - At most 2 scenes should be image.
   - At least one climax scene must be video.

Also decide:
4. INTRO VOICE — pick the opening narrator profile before scene 1.
5. MUSIC SEGMENTS — Group the scene moods into exactly 4 segment prompts, two scenes per segment.
   Each prompt must be one full sentence for a music generation model and MUST include:
   - 1-2 specific lead instruments, not just "orchestral"
   - an approximate tempo feel such as slow/60 BPM, medium/90 BPM, or fast/120 BPM
   - a specific emotional quality, not a generic word like "emotional"
   - instrument choices that reflect the story's cultural or genre setting
   - dynamic contrast across the film:
     * Segment 1 soft and sparse
     * Segment 2 building and more energized
     * Segment 3 the loudest, fullest climax
     * Segment 4 stripped back and resolving
   - Every prompt must end with "instrumental"
6. VEO_PROMPT — Write a cinematic Veo 3.1 video prompt for each scene.
   - Start with camera framing such as wide shot, medium shot, close-up, aerial, or tracking shot.
   - Describe one specific visible action.
   - Include lighting description.
   - Include camera movement such as dolly, pan, static, handheld, drift, or push-in.
   - End with one ambient sound idea.
   - Keep it under 100 words.
   - No text, titles, subtitles, or logos.
   - SAFETY RULES FOR VEO PROMPTS:
     * Never describe weapons being used, combat, fighting, or physical violence.
     * Never show death, injury, blood, or physical harm.
     * Never show threatening poses or aggressive confrontation.
     * Instead, imply conflict through preparation, aftermath, reaction shots, metaphor, hands, silhouettes, dust, shadows, weather, and changing light.
     * The goal is to imply sacrifice and danger without depicting violence directly.
7. VEO_AUDIO_CUE — A short ambient audio description for the scene, such as "rain on stone, distant thunder".
8. Every music segment must include "negative_prompt": "vocals, singing, speech, voice, crowd noise".

Return JSON in exactly this format:
{{
  "intro_narrator_profile": "profile_name",
  "scenes": [
    {{
      "scene_number": 1,
      "narrator_profile": "profile_name",
      "music_mood": "gentle strings, peaceful morning",
      "format": "video",
      "veo_prompt": "Wide shot, the protagonist crossing a lantern-lit bridge at dawn, soft amber light, slow dolly-in, film grain. Ambient sound of distant bells and wind.",
      "veo_audio_cue": "wind, distant bells"
    }}
  ],
  "music_segments": [
    {{
      "segment": 1,
      "prompt": "Full music generation prompt for scenes 1-2, instrumental",
      "negative_prompt": "vocals, singing, speech, voice, crowd noise"
    }}
  ]
}}"""

    fallback = _fallback_creative_direction(story_context, script)

    try:
        client = genai.Client(**vertex_client_kwargs(settings))
        response = client.models.generate_content(
            model=settings.script_routing_model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        data = _parse_json_response(response.text)
    except Exception as exc:
        logger.warning("Creative direction generation failed, using fallback direction. (%s)", exc)
        return fallback

    if not isinstance(data, dict):
        return fallback

    raw_scenes = data.get("scenes")
    scenes_by_number: Dict[int, Dict[str, Any]] = {}
    if isinstance(raw_scenes, list):
        for index, item in enumerate(raw_scenes, start=1):
            if not isinstance(item, dict):
                continue
            try:
                scene_number = int(item.get("scene_number", index))
            except Exception:
                scene_number = index
            scenes_by_number[scene_number] = item

    normalized_scenes: List[Dict[str, Any]] = []
    for index, default_scene in enumerate(fallback["scenes"], start=1):
        raw_scene = scenes_by_number.get(index, {})
        narrator_profile = str(raw_scene.get("narrator_profile", "")).strip().lower()
        if narrator_profile not in CREATIVE_DIRECTION_PROFILES:
            narrator_profile = default_scene["narrator_profile"]

        music_mood = re.sub(r"\s+", " ", str(raw_scene.get("music_mood", "")).strip())
        if not music_mood:
            music_mood = default_scene["music_mood"]

        format_value = str(raw_scene.get("format", "")).strip().lower()
        if format_value not in {"video", "image"}:
            format_value = default_scene["format"]

        veo_prompt = re.sub(r"\s+", " ", str(raw_scene.get("veo_prompt", "")).strip())
        if not veo_prompt:
            veo_prompt = default_scene["veo_prompt"]

        veo_audio_cue = re.sub(r"\s+", " ", str(raw_scene.get("veo_audio_cue", "")).strip())
        if not veo_audio_cue:
            veo_audio_cue = default_scene["veo_audio_cue"]

        normalized_scenes.append(
            {
                "scene_number": index,
                "narrator_profile": narrator_profile,
                "music_mood": music_mood,
                "format": format_value,
                "veo_prompt": veo_prompt,
                "veo_audio_cue": veo_audio_cue,
            }
        )

    normalized_scenes = _limit_voice_variety(normalized_scenes, fallback["scenes"])
    balanced_formats = _rebalance_scene_formats([scene["format"] for scene in normalized_scenes])
    for scene, format_value in zip(normalized_scenes, balanced_formats):
        scene["format"] = format_value

    intro_profile = str(data.get("intro_narrator_profile", "")).strip().lower()
    if intro_profile not in CREATIVE_DIRECTION_PROFILES:
        intro_profile = fallback["intro_narrator_profile"]

    raw_segments = data.get("music_segments")
    normalized_segments: List[Dict[str, str]] = []
    for segment_number in range(1, 5):
        default_segment = fallback["music_segments"][segment_number - 1]
        raw_segment = (
            raw_segments[segment_number - 1]
            if isinstance(raw_segments, list)
            and segment_number - 1 < len(raw_segments)
            and isinstance(raw_segments[segment_number - 1], dict)
            else {}
        )
        prompt_text = _normalize_music_segment_prompt(
            str(raw_segment.get("prompt", "")).strip(),
            str(default_segment.get("prompt", "")),
        )
        normalized_segments.append(
            {
                "segment": segment_number,
                "prompt": prompt_text,
                "negative_prompt": MUSIC_SEGMENT_NEGATIVE_PROMPT,
            }
        )

    return {
        "intro_narrator_profile": intro_profile,
        "scenes": normalized_scenes,
        "music_segments": normalized_segments,
    }


def generate_music_brief(
    story_context: Dict[str, Any],
    act_moods: List[str],
    settings: Settings,
) -> str:
    """Write a one-line Lyria brief based on the story arc."""
    client = genai.Client(**vertex_client_kwargs(settings))
    mood_arc = " -> ".join([m for m in act_moods if m]) or "cinematic emotional arc"
    prompt = (
        "You are a film composer's assistant. "
        "Write one sentence for a music generation model.\n\n"
        f"Setting: {story_context.get('setting', '')}\n"
        f"Character: {story_context.get('character', '')}\n"
        f"Visual style: {story_context.get('visual_style', '')}\n"
        f"Conflict: {story_context.get('inciting_incident', '')}\n"
        f"Ending: {story_context.get('resolution', '')}\n"
        f"User soundtrack preference: {story_context.get('music_preference', '')}\n"
        f"Emotional arc: {mood_arc}\n\n"
        "Requirements:\n"
        "- Include 1-2 specific lead instruments.\n"
        "- Include an approximate tempo feel.\n"
        "- Use a precise emotional description.\n"
        "- Reflect the cultural or genre setting in the instrument choices.\n"
        '- End with "instrumental".\n'
        "Output only the music prompt. No explanation."
    )
    response = client.models.generate_content(
        model=settings.script_routing_model,
        contents=prompt,
    )
    text = (response.text or "").strip()
    if text:
        return text
    preference = str(story_context.get("music_preference", "")).strip()
    if preference:
        return (
            f"Solo piano and low strings, slow 70 BPM, {preference.lower()}, emotional arc {mood_arc}, instrumental."
        )
    return f"Solo piano and warm strings, medium 90 BPM, emotionally evolving arc {mood_arc}, instrumental."


def generate_opening_line(story_context: Dict[str, Any], settings: Settings) -> str:
    """Write a single opening narration line for the film."""
    client = genai.Client(**vertex_client_kwargs(settings))
    prompt = (
        "Write one cinematic narrator opening line for a short film.\n"
        f"Setting: {story_context.get('setting', '')}\n"
        f"Character: {story_context.get('character', '')}\n"
        f"Visual style: {story_context.get('visual_style', '')}\n"
        "One sentence. Evocative. Present tense. No character names. No title."
    )
    response = client.models.generate_content(
        model=settings.script_routing_model,
        contents=prompt,
    )
    text = (response.text or "").strip()
    if text:
        return text
    return "Night gathers around a life that is about to change forever."
