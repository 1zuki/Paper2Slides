import base64
import json
import unittest
from unittest.mock import patch

from paper2slides.generator.image_generator import ImageGenerator


class FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "candidates": [{
                "content": {
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64.b64encode(b"png-bytes").decode("ascii"),
                        }
                    }]
                }
            }]
        }


class TestGoogleImageGenerator(unittest.TestCase):
    def test_image_model_uses_modalities_and_api_key_header(self):
        generator = ImageGenerator(
            api_key="test-key",
            provider="google",
            model="models/gemini-2.5-flash-image",
            response_mime_type="image/png",
            google_api_base_url="https://generativelanguage.googleapis.com/v1beta",
        )

        with patch("paper2slides.generator.image_generator.requests.post", return_value=FakeResponse()) as post:
            image_bytes, mime_type = generator._call_model_google("Make a test image", [])

        self.assertEqual(image_bytes, b"png-bytes")
        self.assertEqual(mime_type, "image/png")
        args, kwargs = post.call_args
        self.assertEqual(
            args[0],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent",
        )
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "test-key")
        self.assertNotIn("params", kwargs)
        request = kwargs["json"]
        self.assertEqual(request["generationConfig"]["responseModalities"], ["TEXT", "IMAGE"])
        self.assertNotIn("responseMimeType", request["generationConfig"])

    def test_non_image_model_is_rejected_before_request(self):
        generator = ImageGenerator(api_key="test-key", provider="google", model="models/gemini-1.5-flash")
        with patch("paper2slides.generator.image_generator.requests.post") as post:
            with self.assertRaisesRegex(ValueError, "image-generation model"):
                generator._call_model_google("Make a test image", [])
        post.assert_not_called()

    def test_permanent_http_error_is_not_retried(self):
        generator = ImageGenerator(api_key="test-key", provider="google", model="models/gemini-2.5-flash-image")
        response = FakeResponse()
        response.status_code = 400
        response.text = json.dumps({"error": {"message": "invalid generation config"}})
        with patch("paper2slides.generator.image_generator.requests.post", return_value=response) as post:
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                generator._call_model_google("Make a test image", [])
        post.assert_called_once()

    def test_quota_http_429_fails_fast_with_actionable_message(self):
        generator = ImageGenerator(api_key="test-key", provider="google", model="models/gemini-2.5-flash-image")
        response = FakeResponse()
        response.status_code = 429
        response.text = '{"error":{"message":"quota exceeded"}}'
        with patch("paper2slides.generator.image_generator.requests.post", return_value=response) as post:
            with self.assertRaisesRegex(RuntimeError, "image quota/rate limit"):
                generator._call_model_google("Make a test image", [])
        post.assert_called_once()
