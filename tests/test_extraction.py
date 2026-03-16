from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from config.settings import Settings
from generation.extraction import (
    classify_veo_safety,
    enforce_character_bible,
    generate_creative_direction,
    generate_emotional_script,
    generate_film_blueprint,
    generate_script_from_outline,
    generate_story_outline,
    map_narrator_to_profile,
    rewrite_for_veo_safety,
)


class TestStoryExtraction(unittest.TestCase):
    def test_map_narrator_to_profile_maps_memory_voice(self) -> None:
        self.assertEqual(map_narrator_to_profile("adult daughter remembering"), "older_memory")
        self.assertEqual(map_narrator_to_profile("old sage looking back"), "old_sage")
        self.assertEqual(map_narrator_to_profile("mythic narrator"), "mythic_storyteller")
        self.assertEqual(map_narrator_to_profile("old woman, very slow, barely a whisper"), "older_memory_aged")

    @patch("generation.extraction.genai.Client")
    def test_generate_emotional_script_normalizes_eight_scenes(self, mock_genai_client: MagicMock) -> None:
        settings = Settings(gcp_project_id="test-project", scene_count=8)
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value.text = """
        {
          "character_bible": "a weathered man in his early 50s with kind brown eyes, short gray-flecked beard, sand-colored linen tunic, and ink-stained cuffs, strong gentle hands, and a calm patient posture shaped by years of teaching and careful work",
          "thread_object": "a folded letter",
          "visual_style_anchor": "Warm earth tones, soft natural light, visible film grain and dust motes",
          "title": "What Remains",
          "scenes": [
            {"scene_number": 1, "emotional_beat": "warmth", "narration": "Hands remember what love feels like.", "narrator_voice": "adult daughter remembering", "veo_prompt": "Close-up, slow dolly-in. a weathered man in his early 50s with kind brown eyes, short gray-flecked beard, sand-colored linen tunic, and ink-stained cuffs, strong gentle hands, and a calm patient posture shaped by years of teaching and careful work. He turns old pages at a stone table in warm morning light. Warm earth tones, soft natural light, visible film grain and dust motes. Audio: pages turning, distant birdsong, gentle breeze. No text, no subtitles, no logos, no title cards.", "image_prompt": "Hands turning pages.", "music_mood": "gentle warmth", "thread_object_role": "introduced in the drawer", "has_character": true},
            {"scene_number": 8, "emotional_beat": "what_remains", "narration": "The gesture returns, changed by time.", "narrator_voice": "adult daughter remembering", "veo_prompt": "Close-up, slow dolly-in. a weathered man in his early 50s with kind brown eyes, short gray-flecked beard, sand-colored linen tunic, and ink-stained cuffs, strong gentle hands, and a calm patient posture shaped by years of teaching and careful work. He turns old pages at the same table at sunset. Warm earth tones, soft natural light, visible film grain and dust motes. Audio: pages turning, distant birdsong, gentle breeze. No text, no subtitles, no logos, no title cards.", "image_prompt": "Younger hands turning pages.", "music_mood": "bittersweet hope", "thread_object_role": "opened at last", "has_character": true}
          ],
          "music_segments": [
            {"segment": 1, "prompt": "Solo oud and soft strings, gentle and reflective, instrumental"}
          ]
        }
        """
        mock_genai_client.return_value = fake_client

        scenes = generate_emotional_script(
            {
                "character_essence": "A father made for books",
                "emotional_anchor": "He loved knowledge and his daughter",
                "world": "A stone courtyard",
                "the_turn": "Invaders approach",
                "residual_feeling": "quiet ache",
            },
            settings,
        )

        self.assertEqual(len(scenes), 8)
        self.assertEqual(scenes[0]["narrator_profile"], "old_woman_remembering")
        self.assertEqual(scenes[7]["emotional_beat"], "what_remains")
        self.assertTrue(scenes[3]["image_prompt"])
        self.assertEqual(scenes[0]["thread_object"], "a folded letter")
        self.assertTrue(scenes[4]["thread_object_role"])

    @patch("generation.extraction.genai.Client")
    def test_generate_film_blueprint_normalizes_top_level_fields(self, mock_genai_client: MagicMock) -> None:
        settings = Settings(gcp_project_id="test-project", scene_count=8)
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value.text = """
        {
          "character_bible": "a weathered fisherman in his late 60s with white stubble, sea-gray eyes, windblown silver hair, wearing a navy wool sweater with salt stains, rope-scarred hands, stooped shoulders from decades at sea, a brass ring on one finger, and a slow deliberate gait",
          "thread_object": "a blue wooden float carved by hand and tied to his oldest net",
          "visual_style_anchor": "Cool maritime blues, natural dawn light, salt texture in the air, fine film grain",
          "title": "The Sea Remembers",
          "silence_after_scene_5": 2.5,
          "silence_after_scene_7": 1.5,
          "scenes": [
            {
              "scene_number": 1,
              "emotional_beat": "warmth",
              "narration": "He greeted the sea before he touched the nets.",
              "narrator_voice": "old_man_remembering",
              "narration_pause_before": 1.5,
              "veo_prompt": "Close-up, slow dolly-in. a weathered fisherman in his late 60s with white stubble, sea-gray eyes, windblown silver hair, wearing a navy wool sweater with salt stains, rope-scarred hands, stooped shoulders from decades at sea, a brass ring on one finger, and a slow deliberate gait. He knots a blue wooden float onto an old net at dawn on a weathered dock. Cool maritime blues, natural dawn light from the east, salt texture in the air, fine film grain. Audio: waves against wood, gulls calling, rope creaking. No text, no subtitles, no logos, no title cards."
            }
          ],
          "music_segments": [
            {"segment": 1, "prompt": "Solo cello and soft piano, gentle tide rhythm, reflective and sparse, instrumental"}
          ]
        }
        """
        mock_genai_client.return_value = fake_client

        blueprint = generate_film_blueprint(
            {
                "character_essence": "An old fisherman who talks to the sea every morning before casting his nets",
                "emotional_anchor": "The sea has been his companion for a lifetime",
                "world": "A small harbor at the edge of a cold sea",
                "the_turn": "One morning the sea doesn't answer",
                "residual_feeling": "That the sea remembers everyone who ever loved it",
            },
            settings,
        )

        self.assertEqual(blueprint["title"], "The Sea Remembers")
        self.assertEqual(len(blueprint["scenes"]), 8)
        self.assertEqual(len(blueprint["music_segments"]), 4)
        self.assertEqual(blueprint["scenes"][0]["narrator_profile"], "old_man_remembering")
        self.assertIn("Audio:", blueprint["scenes"][0]["veo_prompt"])
        self.assertTrue(blueprint["scenes"][0]["narration_ssml"].startswith("<speak>"))
        self.assertEqual(blueprint["silence_after_scene_5"], 2.5)
        self.assertEqual(blueprint["silence_after_scene_7"], 1.5)

    def test_enforce_character_bible_injects_exact_text_and_suffix(self) -> None:
        blueprint = {
            "character_bible": "a weathered man in his early 50s with kind brown eyes, short gray-flecked beard, sand-colored linen tunic, and ink-stained cuffs",
            "visual_style_anchor": "Warm earth tones, soft natural light, visible film grain and dust motes",
        }
        prompt = enforce_character_bible(
            blueprint,
            {
                "has_character": True,
                "veo_prompt": "Close-up, slow dolly-in. He carves a wooden bird at a stone table. Audio: knife on wood, distant birdsong.",
            },
        )
        self.assertIn(blueprint["character_bible"], prompt)
        self.assertIn(blueprint["visual_style_anchor"], prompt)
        self.assertIn("No text, no subtitles, no logos, no title cards.", prompt)

    @patch("generation.extraction.genai.Client")
    def test_classify_veo_safety_attaches_flags_and_safe_alternatives(self, mock_genai_client: MagicMock) -> None:
        settings = Settings(gcp_project_id="test-project", scene_count=8)
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value.text = """
        [
          {"scene_number": 1, "veo_safe": false, "reason": "shows child face", "safe_alternative": "Close-up of ink-stained hands turning pages in warm light."},
          {"scene_number": 2, "veo_safe": true, "reason": "adult reading calmly", "safe_alternative": ""}
        ]
        """
        mock_genai_client.return_value = fake_client

        classified = classify_veo_safety(
            [
                {"veo_prompt": "Close-up of a child face in tears.", "image_prompt": "Child face in tears."},
                {"veo_prompt": "Close-up of a middle-aged man reading by a window in warm light.", "image_prompt": "Middle-aged man reading by a window."},
            ],
            settings,
        )

        self.assertFalse(classified[0]["veo_safe"])
        self.assertTrue(classified[1]["veo_safe"])
        self.assertIn("hands", classified[0]["safe_alternative"].lower())

    @patch("generation.extraction.genai.Client")
    def test_classify_veo_safety_fallback_heuristic_allows_adult_faces(self, mock_genai_client: MagicMock) -> None:
        settings = Settings(gcp_project_id="test-project", scene_count=8)
        mock_genai_client.side_effect = RuntimeError("classifier unavailable")

        classified = classify_veo_safety(
            [
                {
                    "veo_prompt": "Medium shot of a middle-aged woman reading at a desk in morning light.",
                    "image_prompt": "Middle-aged woman reading at a desk.",
                    "narration": "She keeps the ritual of reading alive.",
                }
            ],
            settings,
        )

        self.assertTrue(classified[0]["veo_safe"])

    @patch("generation.extraction.genai.Client")
    def test_rewrite_for_veo_safety_returns_rewritten_prompt(self, mock_genai_client: MagicMock) -> None:
        settings = Settings(gcp_project_id="test-project")
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value.text = (
            "Wide shot of villagers watching storm clouds gather over a distant ridge, warm dusk light, slow push-in."
        )
        mock_genai_client.return_value = fake_client

        rewritten = rewrite_for_veo_safety(
            "Wide shot of villagers fighting invaders with swords at the gate.",
            settings,
        )

        self.assertIn("storm clouds", rewritten)
        self.assertNotIn("fighting", rewritten.lower())

    @patch("generation.extraction.genai.Client")
    def test_generate_story_outline_preserves_scene_count_and_order(self, mock_genai_client: MagicMock) -> None:
        settings = Settings(gcp_project_id="test-project", scene_count=8)
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value.text = """
        [
          {"scene_number": 1, "story_beat": "Locked in a prison cell.", "visual_focus": "A man alone in a stark cell.", "music_mood": "grim stillness"},
          {"scene_number": 2, "story_beat": "The prison shifts into violent unrest.", "visual_focus": "A mess hall erupts in chaos.", "music_mood": "rising threat"},
          {"scene_number": 3, "story_beat": "He chooses to move through maintenance corridors.", "visual_focus": "He slips into a narrow corridor.", "music_mood": "tense motion"},
          {"scene_number": 4, "story_beat": "He steals access credentials.", "visual_focus": "A terminal glows in shadow.", "music_mood": "cold focus"},
          {"scene_number": 5, "story_beat": "He reaches the hangar threshold.", "visual_focus": "A sealed hangar door opens.", "music_mood": "held breath"},
          {"scene_number": 6, "story_beat": "He mounts the prototype bike.", "visual_focus": "A bike gleams under hard light.", "music_mood": "ignition"},
          {"scene_number": 7, "story_beat": "Security pursues him across the station exterior.", "visual_focus": "A chase through open space.", "music_mood": "adrenaline chase"},
          {"scene_number": 8, "story_beat": "He breaks free into the void.", "visual_focus": "The bike races into open space.", "music_mood": "hard-won release"}
        ]
        """
        mock_genai_client.return_value = fake_client

        outline = generate_story_outline(
            {
                "setting": "Orbital prison above a blue planet",
                "character": "A prisoner",
                "inciting_incident": "A fight breaks out",
                "resolution": "He escapes",
            },
            8,
            settings,
        )

        self.assertEqual(len(outline), 8)
        self.assertEqual(outline[0]["scene_number"], "1")
        self.assertEqual(outline[-1]["scene_number"], "8")
        self.assertIn("hangar", outline[4]["story_beat"].lower())

    @patch("generation.extraction.genai.Client")
    def test_generate_script_from_outline_reorders_model_output_to_match_outline(self, mock_genai_client: MagicMock) -> None:
        settings = Settings(gcp_project_id="test-project", scene_count=4)
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value.text = """
        [
          {"scene_number": 4, "narration": "He escapes into open space.", "image_prompt": "A bike races into the stars.", "music_mood": "release"},
          {"scene_number": 2, "narration": "The prison erupts into chaos.", "image_prompt": "A riot in the mess hall.", "music_mood": "panic"},
          {"scene_number": 1, "narration": "He waits in silence in his cell.", "image_prompt": "A lone prisoner under cold light.", "music_mood": "grim stillness"},
          {"scene_number": 3, "narration": "He slips through maintenance tunnels.", "image_prompt": "A narrow tunnel lit by warning lights.", "music_mood": "tense motion"}
        ]
        """
        mock_genai_client.return_value = fake_client

        outline = [
            {"scene_number": "1", "story_beat": "Cell.", "visual_focus": "Cell.", "music_mood": "grim stillness"},
            {"scene_number": "2", "story_beat": "Riot.", "visual_focus": "Riot.", "music_mood": "panic"},
            {"scene_number": "3", "story_beat": "Tunnels.", "visual_focus": "Tunnels.", "music_mood": "tense motion"},
            {"scene_number": "4", "story_beat": "Escape.", "visual_focus": "Escape.", "music_mood": "release"},
        ]

        scenes = generate_script_from_outline(
            {
                "setting": "Orbital prison",
                "character": "A prisoner",
            },
            outline,
            settings,
        )

        self.assertEqual(len(scenes), 4)
        self.assertEqual(scenes[0]["narration"], "He waits in silence in his cell.")
        self.assertEqual(scenes[1]["narration"], "The prison erupts into chaos.")
        self.assertEqual(scenes[2]["narration"], "He slips through maintenance tunnels.")
        self.assertEqual(scenes[3]["narration"], "He escapes into open space.")

    @patch("generation.extraction.genai.Client")
    def test_generate_creative_direction_backfills_missing_scenes_and_segments(
        self,
        mock_genai_client: MagicMock,
    ) -> None:
        settings = Settings(gcp_project_id="test-project", scene_count=8)
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value.text = """
        {
          "intro_narrator_profile": "hacker_present_tense",
          "scenes": [
            {"scene_number": 1, "narrator_profile": "hacker_present_tense", "music_mood": "restrained synth pulses", "format": "image"},
            {"scene_number": 6, "narrator_profile": "hacker_urgent", "music_mood": "urgent bass propulsion", "format": "video"}
          ],
          "music_segments": [
            {"segment": 1, "prompt": "Cold neon synth pulses under a wary escape plan, instrumental", "negative_prompt": "vocals, singing, speech, voice"}
          ]
        }
        """
        mock_genai_client.return_value = fake_client

        script = [
            {"narration": f"Scene {index} narration", "image_prompt": f"Scene {index} visual", "music_mood": "placeholder"}
            for index in range(1, 9)
        ]
        direction = generate_creative_direction(
            {
                "setting": "Orbital prison over a blue planet",
                "character": "A former pilot hacker",
                "visual_style": "Retro-futurist thriller",
            },
            script,
            settings,
        )

        self.assertEqual(direction["intro_narrator_profile"], "hacker_present_tense")
        self.assertEqual(len(direction["scenes"]), 8)
        self.assertEqual(direction["scenes"][0]["narrator_profile"], "hacker_present_tense")
        self.assertEqual(direction["scenes"][5]["narrator_profile"], "hacker_urgent")
        self.assertIn(direction["scenes"][5]["format"], {"video", "image"})
        self.assertTrue(direction["scenes"][0]["veo_prompt"])
        self.assertTrue(direction["scenes"][0]["veo_audio_cue"])
        video_count = sum(1 for scene in direction["scenes"] if scene["format"] == "video")
        self.assertGreaterEqual(video_count, 6)
        self.assertLessEqual(video_count, 8)
        self.assertEqual(len(direction["music_segments"]), 4)
        self.assertTrue(direction["music_segments"][0]["prompt"].endswith("instrumental"))
        self.assertEqual(
            direction["music_segments"][0]["negative_prompt"],
            "vocals, singing, speech, voice, crowd noise",
        )


if __name__ == "__main__":
    unittest.main()
