import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from paper2slides.llm.client import (
    LLMClient,
    LLMSettings,
    _extract_output_text,
    _to_responses_input,
)


class TestLLMClient(unittest.TestCase):
    def test_codex_gateway_settings_select_responses_transport(self):
        with patch.dict(os.environ, {
            "RAG_LLM_PROVIDER": "codex_gateway",
            "CODEX_API_KEY": "test-key",
            "CODEX_BASE_URL": "https://gateway.example/backend-api/codex/",
            "CODEX_MODEL": "test-model",
        }, clear=True):
            from paper2slides.llm.client import get_llm_settings
            settings = get_llm_settings()
        self.assertEqual(settings.provider, "codex_gateway")
        self.assertEqual(settings.api_style, "responses")
        self.assertEqual(settings.base_url, "https://gateway.example/backend-api/codex/v1")
        self.assertEqual(settings.model, "test-model")

    def test_messages_convert_to_responses_input(self):
        converted = _to_responses_input([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]},
        ])
        self.assertEqual(converted[0]["content"][0]["type"], "input_text")
        self.assertEqual(converted[1]["content"][1]["type"], "input_image")
        self.assertTrue(converted[1]["content"][1]["image_url"].startswith("data:image/png"))

    def test_extract_output_text_from_response_object(self):
        response = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(type="output_text", text="OK")])]
        )
        self.assertEqual(_extract_output_text(response), "OK")

    def test_gateway_transport_collects_sse(self):
        settings = LLMSettings("codex_gateway", "secret", "https://gateway/v1", "model", "responses")
        client = LLMClient(settings=settings, client=object())

        class FakeResponse:
            is_error = False
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def iter_lines(self):
                yield "data: " + json.dumps({"type": "response.output_text.delta", "delta": "O"})
                yield "data: " + json.dumps({"type": "response.output_text.delta", "delta": "K"})
                yield "data: [DONE]"

        with patch("paper2slides.llm.client.httpx.stream", return_value=FakeResponse()):
            answer = client.complete_text([{"role": "user", "content": "test"}], max_tokens=8)
        self.assertEqual(answer, "OK")
