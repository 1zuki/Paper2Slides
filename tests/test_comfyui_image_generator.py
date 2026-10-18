import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from paper2slides.generator.image_generator import ImageGenerator


class FakeResponse:
    def __init__(self, payload=None, content=b"", headers=None):
        self._payload = payload or {}
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestComfyUIImageGenerator(unittest.TestCase):
    def _workflow_file(self, directory):
        path = Path(directory) / "workflow.json"
        path.write_text(json.dumps({
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
            "2": {"class_type": "SaveImage", "inputs": {}},
        }), encoding="utf-8")
        return path

    def test_comfyui_queue_poll_and_download(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = self._workflow_file(directory)
            env = {
                "IMAGE_GEN_PROVIDER": "comfyui",
                "IMAGE_GEN_COMFYUI_BASE_URL": "http://127.0.0.1:8188",
                "IMAGE_GEN_COMFYUI_WORKFLOW": str(workflow),
                "IMAGE_GEN_COMFYUI_PROMPT_NODE": "1",
                "IMAGE_GEN_COMFYUI_PROMPT_INPUT": "text",
                "IMAGE_GEN_COMFYUI_OUTPUT_NODE": "2",
                "IMAGE_GEN_COMFYUI_TIMEOUT_SECONDS": "1",
            }
            responses = [
                FakeResponse({"prompt_id": "prompt-1"}),
                FakeResponse({"prompt-1": {"outputs": {
                    "2": {"images": [{"filename": "slide.png", "type": "output"}]}
                }}}),
                FakeResponse(content=b"png-data", headers={"Content-Type": "image/png"}),
            ]
            with patch.dict(os.environ, env, clear=False), patch(
                "paper2slides.generator.image_generator.requests.post", return_value=responses[0]
            ) as post, patch(
                "paper2slides.generator.image_generator.requests.get", side_effect=responses[1:]
            ) as get:
                generator = ImageGenerator()
                image_bytes, mime_type = generator._call_model("scientific wavefield slide", [])

            self.assertEqual(image_bytes, b"png-data")
            self.assertEqual(mime_type, "image/png")
            body = post.call_args.kwargs["json"]["prompt"]
            self.assertEqual(body["1"]["inputs"]["text"], "scientific wavefield slide")
            self.assertEqual(get.call_count, 2)
            self.assertTrue(get.call_args.kwargs["params"]["filename"], "slide.png")

    def test_missing_workflow_fails_with_configuration_hint(self):
        with patch.dict(os.environ, {
            "IMAGE_GEN_PROVIDER": "comfyui",
            "IMAGE_GEN_COMFYUI_WORKFLOW": "",
        }, clear=False):
            with self.assertRaisesRegex(ValueError, "API-format workflow JSON"):
                ImageGenerator()

    def test_unreachable_comfyui_is_reported_without_response_details(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = self._workflow_file(directory)
            with patch.dict(os.environ, {
                "IMAGE_GEN_PROVIDER": "comfyui",
                "IMAGE_GEN_COMFYUI_WORKFLOW": str(workflow),
                "IMAGE_GEN_COMFYUI_PROMPT_NODE": "1",
            }, clear=False), patch(
                "paper2slides.generator.image_generator.requests.post",
                side_effect=requests.ConnectionError("private transport detail"),
            ):
                generator = ImageGenerator()
                with self.assertRaisesRegex(RuntimeError, "Confirm ComfyUI is running") as caught:
                    generator._call_model("test", [])
                self.assertNotIn("private transport detail", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
