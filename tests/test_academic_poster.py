import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from paper2slides.generator.academic_compositor import render_academic_poster
from paper2slides.generator.image_generator import GeneratedImage, ImageGenerator


class TestAcademicPoster(unittest.TestCase):
    def _plan(self):
        section = SimpleNamespace(
            id="poster_title",
            title="A structured scientific poster",
            content="Exact paper evidence belongs in native-rendered text.",
            figures=[],
            tables=[],
        )
        return SimpleNamespace(
            output_type="poster",
            sections=[section],
            tables_index={},
            figures_index={},
        )

    def test_academic_poster_is_landscape_native_render(self):
        data = render_academic_poster(self._plan(), [], density="dense")
        image = Image.open(io.BytesIO(data))
        self.assertEqual(image.size, (3200, 1800))
        self.assertEqual(image.mode, "RGB")

    def test_academic_poster_bypasses_diffusion_model(self):
        generator = object.__new__(ImageGenerator)
        generator.compose_academic = True
        generator.provider = "comfyui"

        config = SimpleNamespace(
            style=SimpleNamespace(value="academic"),
            custom_style=None,
            poster_density=SimpleNamespace(value="dense"),
        )
        gen_input = SimpleNamespace(
            config=config,
            origin=SimpleNamespace(base_path=""),
        )
        expected = GeneratedImage("poster", b"poster", "image/png")
        with patch.object(generator, "_generate_academic_poster", return_value=[expected]) as native, patch.object(
            generator, "_generate_poster"
        ) as diffusion:
            result = generator.generate(self._plan(), gen_input)

        self.assertEqual(result, [expected])
        native.assert_called_once()
        diffusion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
