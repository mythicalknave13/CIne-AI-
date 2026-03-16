from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from config.settings import Settings
from generation.story import generate_interleaved_story


class _InlineData:
    def __init__(self, data: bytes, mime_type: str = "image/png") -> None:
        self.data = data
        self.mime_type = mime_type


class _Part:
    def __init__(self, *, text: str | None = None, image: bytes | None = None) -> None:
        self.text = text
        self.inline_data = _InlineData(image) if image is not None else None


class TestInterleavedStory(unittest.TestCase):
    @patch("generation.story.genai.Client")
    def test_generate_interleaved_story_saves_images_and_pairs_narration(self, mock_client_cls: MagicMock) -> None:
        image_bytes = b"\x89PNG\r\n\x1a\nfake"
        fake_response = MagicMock()
        fake_response.candidates = [
            MagicMock(
                content=MagicMock(
                    parts=[
                        _Part(text="I still feel the flour on my hands."),
                        _Part(image=image_bytes),
                        _Part(image=image_bytes),
                        _Part(text="The room changed before I did."),
                        _Part(image=image_bytes),
                        _Part(text="I carried the bowl like a promise."),
                        _Part(image=image_bytes),
                        _Part(text="[silence]"),
                        _Part(image=image_bytes),
                        _Part(text="The flour remembers."),
                        _Part(image=image_bytes),
                    ]
                )
            )
        ]
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response
        mock_client_cls.return_value = fake_client

        with tempfile.TemporaryDirectory() as tmp:
            result = generate_interleaved_story(
                {
                    "character_description": "a woman in her sixties with flour on her sleeves",
                    "who": "My grandmother",
                    "turn": "She stopped baking",
                    "remains": "The kitchen still remembers her",
                    "title": "Flour Memory",
                },
                Settings(gemini_api_key="test-key"),
                Path(tmp),
            )

            self.assertEqual(result["title"], "Flour Memory")
            self.assertEqual(len(result["scenes"]), 6)
            self.assertEqual(result["scenes"][0]["narration"], "I still feel the flour on my hands.")
            self.assertEqual(result["scenes"][1]["narration"], "")
            self.assertEqual(result["scenes"][4]["narration"], "[silence]")
            self.assertTrue(Path(result["scenes"][0]["image_path"]).exists())


if __name__ == "__main__":
    unittest.main()
