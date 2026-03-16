from __future__ import annotations

import itertools
import os
import random
import threading
import time
from typing import Callable, TypeVar

import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest

from config.settings import Settings

VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
T = TypeVar("T")
DEFAULT_IMAGE_REGIONS = ("us-central1", "us-east4", "europe-west4")
_IMAGE_REGION_LOCK = threading.Lock()
_IMAGE_REGION_CYCLES: dict[tuple[str, ...], object] = {}


class RegionRateLimiter:
    """Space image calls per region so round-robin routing does not burst a quota bucket."""

    def __init__(self, min_interval_seconds: float = 25.0) -> None:
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()
        self._min_interval_seconds = max(float(min_interval_seconds), 0.0)

    def wait_if_needed(self, region: str) -> None:
        if self._min_interval_seconds <= 0:
            return
        with self._lock:
            now = time.time()
            last_call = self._last_call.get(region, 0.0)
            elapsed = now - last_call
            if elapsed < self._min_interval_seconds:
                wait_seconds = self._min_interval_seconds - elapsed
                time.sleep(wait_seconds)
                now = time.time()
            self._last_call[region] = now


_IMAGE_RATE_LIMITER = RegionRateLimiter()


def vertex_genai_location(settings: Settings) -> str:
    requested = str(settings.gcp_location or "").strip().lower()
    if requested in {"", "us-central1", "global"}:
        return "global"
    return requested


def vertex_rest_location(settings: Settings, *, prefer_global: bool = False) -> str:
    requested = str(settings.gcp_location or "").strip().lower()
    if prefer_global and requested in {"", "us-central1", "global"}:
        return "global"
    return requested or "us-central1"


def vertex_rest_host(location: str) -> str:
    normalized = str(location or "").strip().lower()
    if normalized == "global":
        return "aiplatform.googleapis.com"
    return f"{normalized}-aiplatform.googleapis.com"


def is_rate_limited_error(exc: Exception) -> bool:
    message = str(exc or "")
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def is_retry_exhausted_error(exc: Exception) -> bool:
    return "Max retries exceeded for 429 errors" in str(exc or "")


def call_with_retry(
    fn: Callable[[], T],
    *,
    logger,
    description: str,
    max_retries: int = 3,
) -> T:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if not is_rate_limited_error(exc):
                raise
            last_error = exc
            if attempt == max_retries - 1:
                break
            wait_seconds = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "Rate limited (429) during %s. Waiting %.1fs before retry %d/%d",
                description,
                wait_seconds,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait_seconds)
    raise RuntimeError("Max retries exceeded for 429 errors") from last_error


def vertex_client_kwargs(settings: Settings) -> dict[str, object]:
    return {
        "api_key": settings.gemini_api_key or os.getenv("GEMINI_API_KEY", ""),
    }


def vertex_image_regions(settings: Settings) -> tuple[str, ...]:
    configured = [
        region.strip().lower()
        for region in str(getattr(settings, "image_regions", "") or "").split(",")
        if region.strip()
    ]
    if configured:
        return tuple(configured)
    requested = str(settings.gcp_location or "").strip().lower()
    if requested and requested != "global":
        return (requested,)
    return DEFAULT_IMAGE_REGIONS


def _next_image_region(regions: tuple[str, ...]) -> str:
    with _IMAGE_REGION_LOCK:
        cycle = _IMAGE_REGION_CYCLES.get(regions)
        if cycle is None:
            cycle = itertools.cycle(regions)
            _IMAGE_REGION_CYCLES[regions] = cycle
        return next(cycle)


def vertex_image_client_config(settings: Settings) -> tuple[dict[str, object], str]:
    if not settings.gcp_project_id:
        raise ValueError("GCP_PROJECT_ID is required for Vertex AI generation.")
    region = _next_image_region(vertex_image_regions(settings))
    _IMAGE_RATE_LIMITER.wait_if_needed(region)
    return (
        {
            "vertexai": True,
            "project": settings.gcp_project_id,
            "location": region,
        },
        region,
    )


def vertex_credentials() -> tuple[object, str | None]:
    credentials, project = google.auth.default(scopes=[VERTEX_SCOPE])
    if not getattr(credentials, "valid", False) or not getattr(credentials, "token", None):
        credentials.refresh(GoogleAuthRequest())
    return credentials, project
