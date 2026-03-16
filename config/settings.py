from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

VERTEX_MODEL_ALIASES = {
    "gemini-3.1-flash-image-preview": "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview": "gemini-2.5-flash-image",
    "gemini-3.1-flash-lite-preview": "gemini-2.5-flash-lite",
    "gemini-3.1-pro-preview": "gemini-2.5-pro",
    "gemini-2.5-flash-preview-tts": "gemini-2.5-flash-tts",
}


class Settings(BaseModel):
    gemini_api_key: str = Field(default="")
    gcp_project_id: str = Field(default="")
    gcp_location: str = Field(default="us-central1")
    image_regions: str = Field(default="us-central1,us-east4,europe-west4")
    gcs_bucket: str = Field(default="")
    firestore_collection: str = Field(default="sessions")
    firestore_database_id: str = Field(default="(default)")

    image_model: str = Field(default="gemini-2.5-flash")
    character_image_model: str = Field(default="gemini-2.5-flash")
    conversation_model: str = Field(default="gemini-2.5-flash")
    script_routing_model: str = Field(default="gemini-2.5-flash-lite")
    script_writer_model: str = Field(default="gemini-2.5-pro")
    tts_model: str = Field(default="gemini-2.5-flash-tts")
    music_model: str = Field(default="lyria-002")
    veo_model: str = Field(default="veo-3.1-generate-preview")
    veo_fast_model: str = Field(default="veo-3.1-fast-generate-preview")

    film_duration_seconds: int = Field(default=120)
    scene_count: int = Field(default=8)
    character_timeout_ms: int = Field(default=45000)
    character_hedge_delay_seconds: float = Field(default=4.0)

    audio_sample_rate: int = Field(default=24000)
    tts_language_code: str = Field(default="en-US")
    tts_voice_name: str = Field(default="en-US-Studio-O")
    tts_speaking_rate: float = Field(default=1.0)
    tts_pitch: float = Field(default=0.0)
    narration_volume: float = Field(default=1.0)
    music_volume: float = Field(default=0.12)
    veo_duration_seconds: int = Field(default=8)
    veo_aspect_ratio: str = Field(default="16:9")
    veo_max_parallel: int = Field(default=8)
    veo_timeout_seconds: int = Field(default=120)
    veo_poll_interval_seconds: int = Field(default=5)


def _resolve_model_name(env_name: str, default: str) -> str:
    raw_value = os.getenv(env_name, default).strip() or default
    return VERTEX_MODEL_ALIASES.get(raw_value, raw_value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
        gcp_project_id=os.getenv("GCP_PROJECT_ID", ""),
        gcp_location=os.getenv("GCP_LOCATION", "us-central1"),
        image_regions=os.getenv("IMAGE_REGIONS", "us-central1,us-east4,europe-west4"),
        gcs_bucket=os.getenv("GCS_BUCKET", ""),
        firestore_collection=os.getenv("FIRESTORE_COLLECTION", "sessions"),
        firestore_database_id=os.getenv("FIRESTORE_DATABASE_ID", "(default)"),
        image_model=_resolve_model_name("IMAGE_MODEL", "gemini-2.5-flash"),
        character_image_model=_resolve_model_name("CHARACTER_IMAGE_MODEL", "gemini-2.5-flash"),
        conversation_model=_resolve_model_name("CONVERSATION_MODEL", "gemini-2.5-flash"),
        script_routing_model=_resolve_model_name("SCRIPT_ROUTING_MODEL", "gemini-2.5-flash-lite"),
        script_writer_model=_resolve_model_name("SCRIPT_WRITER_MODEL", "gemini-2.5-pro"),
        tts_model=_resolve_model_name("TTS_MODEL", "gemini-2.5-flash-tts"),
        music_model=os.getenv("MUSIC_MODEL", "lyria-002"),
        veo_model=os.getenv("VEO_MODEL", "veo-3.1-generate-preview"),
        veo_fast_model=os.getenv("VEO_FAST_MODEL", "veo-3.1-fast-generate-preview"),
        film_duration_seconds=int(os.getenv("FILM_DURATION_SECONDS", "120")),
        scene_count=int(os.getenv("SCENE_COUNT", "8")),
        character_timeout_ms=int(os.getenv("CHARACTER_TIMEOUT_MS", "45000")),
        character_hedge_delay_seconds=float(os.getenv("CHARACTER_HEDGE_DELAY_SECONDS", "4.0")),
        audio_sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "24000")),
        tts_language_code=os.getenv("TTS_LANGUAGE_CODE", "en-US"),
        tts_voice_name=os.getenv("TTS_VOICE_NAME", "en-US-Studio-O"),
        tts_speaking_rate=float(os.getenv("TTS_SPEAKING_RATE", "1.0")),
        tts_pitch=float(os.getenv("TTS_PITCH", "0.0")),
        narration_volume=float(os.getenv("NARRATION_VOLUME", "1.0")),
        music_volume=float(os.getenv("MUSIC_VOLUME", "0.12")),
        veo_duration_seconds=int(os.getenv("VEO_DURATION_SECONDS", "8")),
        veo_aspect_ratio=os.getenv("VEO_ASPECT_RATIO", "16:9"),
        veo_max_parallel=int(os.getenv("VEO_MAX_PARALLEL", "8")),
        veo_timeout_seconds=int(os.getenv("VEO_TIMEOUT_SECONDS", "120")),
        veo_poll_interval_seconds=int(os.getenv("VEO_POLL_INTERVAL_SECONDS", "5")),
    )
