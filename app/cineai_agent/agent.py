from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger("uvicorn.error")
_current_session_id: str = ""

if settings.gemini_api_key:
    os.environ.setdefault("GOOGLE_API_KEY", settings.gemini_api_key)
os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

from google.adk.agents import Agent
from google.cloud import firestore


# ── Session State Store ──────────────────────────────────────────────


class FirestoreStateStore:
    def __init__(self) -> None:
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._client: firestore.Client | None = None
        self._warned_runtime = False
        try:
            self._client = firestore.Client(
                project=settings.gcp_project_id or None,
                database=settings.firestore_database_id,
            )
        except Exception as exc:
            self._client = None
            logger.warning("Firestore unavailable, using local storage. (%s)", exc)

    @staticmethod
    def _new_session_payload() -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "current_beat": 1,
            "story_type": None,   # "preset" or "custom"
            "story_context": {},
            "generated_assets": {},
            "created_at": now,
            "updated_at": now,
        }

    def initialize_session(self, session_id: str) -> Dict[str, Any]:
        existing = self.load(session_id)
        if existing:
            return existing
        payload = self._new_session_payload()
        self._memory[session_id] = dict(payload)
        if self._client is not None:
            try:
                self._client.collection(settings.firestore_collection).document(session_id).set(payload, merge=True)
            except Exception as exc:
                if not self._warned_runtime:
                    logger.warning("Firestore write failed, using local. (%s)", exc)
                    self._warned_runtime = True
        return dict(payload)

    def load(self, session_id: str) -> Dict[str, Any]:
        if self._client is None:
            return dict(self._memory.get(session_id, {}))
        try:
            doc = self._client.collection(settings.firestore_collection).document(session_id).get()
            data = doc.to_dict() or {}
            if data:
                self._memory[session_id] = dict(data)
                return dict(data)
        except Exception:
            pass
        return dict(self._memory.get(session_id, {}))

    def save(self, session_id: str, payload: Dict[str, Any]) -> None:
        current = self._memory.get(session_id, {})
        merged = {**current, **payload, "updated_at": datetime.now(timezone.utc).isoformat()}
        self._memory[session_id] = merged
        if self._client is not None:
            try:
                self._client.collection(settings.firestore_collection).document(session_id).set(merged, merge=True)
            except Exception:
                pass


state_store = FirestoreStateStore()


# ── Session Helpers ──────────────────────────────────────────────────


def set_current_session(sid: str) -> None:
    global _current_session_id
    _current_session_id = sid

set_current_session_id = set_current_session


def _require_current_session() -> str:
    if not _current_session_id:
        raise ValueError("Active session is not set.")
    return _current_session_id


def initialize_session_state(session_id: str) -> Dict[str, Any]:
    set_current_session(session_id)
    return state_store.initialize_session(session_id)


def get_session_state(session_id: str) -> Dict[str, Any]:
    return state_store.load(session_id)


def new_session_id() -> str:
    return str(uuid.uuid4())


# ── ADK Agent (Minimal — used for fallback freeform chat) ────────────


def build_agent() -> Agent:
    instruction = """You are CineAI — not a chatbot and not a tool. You are a film director who finds the feeling inside someone's idea and turns it into two minutes of cinema.

You ask three questions. Each one goes deeper.

Beat 1: "Tell me about someone. Real or imagined. Who are they, and why do they matter?"
Beat 2: "What's the one moment that changes everything for them? Not the whole story. Just the turn."
Beat 3: "Last question. When the screen goes dark, what feeling stays in the room?"

Rules:
- Between their answer and your next question, say one reflective sentence that shows you heard them.
- Stay calm, brief, and emotionally precise.
- If they ask a general question, answer helpfully but keep the same directorial presence."""

    return Agent(
        name="cineai_director",
        model=os.getenv("AGENT_MODEL", settings.conversation_model),
        instruction=instruction,
        tools=[],
    )


cineai_director = build_agent()
