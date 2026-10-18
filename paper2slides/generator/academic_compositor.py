"""Evidence-preserving rendering for academic paper slides.

Diffusion models are useful for illustrations, but they are a poor renderer for
scientific titles, tables, labels, and numerical results.  This module keeps
those elements deterministic and uses the original paper figures as evidence.
"""

from __future__ import annotations

import base64
import io
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 900

_FONT_CANDIDATES = {
    "regular": (
        "/usr/share/fonts/carlito/Carlito-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    ),
    "bold": (
        "/usr/share/fonts/carlito/Carlito-Bold.ttf",
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    ),
    "italic": (
        "/usr/share/fonts/carlito/Carlito-Italic.ttf",
        "/usr/share/fonts/noto/NotoSans-Italic.ttf",
    ),
}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES[kind]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _clean_text(value: str) -> str:
    """Turn the planner's markdown/LaTeX-ish text into readable slide text."""
    text = unicodedata.normalize("NFKC", value or "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = re.sub(r"\\(?:mathcal|mathrm|mathbf|text|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    commands = {
        "alpha": "alpha",
        "beta": "beta",
        "eta": "eta",
        "rho": "rho",
        "sigma": "sigma",
        "times": "x",
        "to": "to",
        "cdot": "x",
        "leq": "<=",
        "geq": ">=",
    }
    text = re.sub(r"\\([A-Za-z]+)", lambda match: commands.get(match.group(1), match.group(1)), text)
    text = re.sub(r"([A-Za-z]+)\s*_\s*\{\s*([^{}]+?)\s*\}", r"\1\2", text)
    text = re.sub(r"([A-Za-z]+)\s*_\s*([A-Za-z0-9]+)", r"\1\2", text)
    text = text.replace("\\,", " ").replace("\\%", "%")
    text = text.replace("{", "").replace("}", "").replace("$", "")
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = text.replace("\u2192", "to").replace("\u2248", "~").replace("\u00d7", "x")
    text = text.replace("\u00b1", "+/-").replace("\u2264", "<=").replace("\u2265", ">=")
    text = text.replace("₀", "0").replace("₁", "1").replace("₂", "2").replace("₃", "3")
    text = text.replace("₄", "4").replace("₅", "5").replace("₆", "6").replace("₇", "7")
    text = text.replace("₈", "8").replace("₉", "9")
    text = text.replace("\u202f", " ").replace("\u00a0", " ")
    # MinerU sometimes inserts spaces inside numbers such as ``13 500``.
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in _clean_text(text).splitlines() or [""]:
        paragraph = paragraph.strip()
        if not paragraph:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and draw.textbbox((0, 0), candidate, font=font)[2] > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: Tuple[int, int],
    width: int,
    font,
    fill,
    line_gap: int = 8,
    max_lines: Optional[int] = None,
) -> int:
    lines = _wrap(draw, text, font, width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            last = lines[-1].rstrip(" .")
            while draw.textbbox((0, 0), last + "…", font=font)[2] > width and last:
                last = last.rsplit(" ", 1)[0]
            lines[-1] = last + "…"
    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3]
    for line in lines:
        if line:
            draw.text((x, y), line, font=font, fill=fill)
        y += line_height + line_gap
    return y


def _fit_title(draw: ImageDraw.ImageDraw, title: str, width: int):
    for size in (50, 46, 42, 38, 34):
        font = _font("bold", size)
        if len(_wrap(draw, title, font, width)) <= 2:
            return font
    return _font("bold", 32)


def _decode_image(image_info: dict) -> Optional[Image.Image]:
    encoded = image_info.get("base64")
    if not encoded:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    except Exception:
        return None


def _fit_image(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    x, y, width, height = box
    scale = min(width / image.width, height / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _figure_for_section(section, section_index: int, figure_images: Sequence[dict]) -> Optional[dict]:
    by_id = {item.get("figure_id"): item for item in figure_images}
    requested = [ref.figure_id for ref in getattr(section, "figures", [])]
    for figure_id in requested:
        if figure_id in by_id:
            return by_id[figure_id]

    title = _clean_text(getattr(section, "title", "")).lower()
    # Prefer figures that explain the slide's subject. Do not fill empty
    # slides with an unrelated map merely because a figure exists in the paper.
    title_map = (
        (("field-wise accuracy",), "Figure 3"),
        (("windowed forecasting", "real-bathymetry"), "Figure 4"),
        (("solver gap", "cross-reference"), "Figure 5"),
        (("ensemble reliability", "coverage", "calibration"), "Figure 8"),
        (("additional diagnostics", "data-size"), "Figure 9"),
        (("resolution transfer",), "Figure 9"),
        (("source-family", "benchmark construction", "research gap"), "Figure 1"),
        (("dispersion",), "Figure 7"),
        (("high-error", "outlier"), "Figure 11"),
        (("conclusions",), "Figure 8"),
    )
    for keywords, figure_id in title_map:
        if any(keyword in title for keyword in keywords) and figure_id in by_id:
            return by_id[figure_id]
    if section_index == 0 and "Figure 2" in by_id:
        return by_id["Figure 2"]
    return None


def _evidence_values(text: str, limit: int = 3) -> List[str]:
    normalized = _clean_text(text)
    patterns = (
        r"\b\d[\d,]*(?:\.\d+)?\s*(?:×|x|%)",
        r"\b\d[\d,]*(?:\.\d+)?\s*(?:m|ms|s|h|M|cases?|scenarios?|checks?|pairs?|criteria|frames?|samples?|iterations?)\b",
        r"\b0\.\d{2,6}\b",
    )
    values: List[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, normalized, flags=re.IGNORECASE):
            value = re.sub(r"\s+", " ", match).strip()
            if value not in values:
                values.append(value)
            if len(values) >= limit:
                return values
    return values


def _draw_figure(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    figure_info: Optional[dict],
    box: Tuple[int, int, int, int],
) -> None:
    x, y, width, height = box
    draw.rounded_rectangle((x, y, x + width, y + height), radius=12, fill="#ffffff", outline="#d3dce5", width=2)
    if not figure_info:
        draw.text((x + 28, y + 28), "Paper evidence", font=_font("bold", 24), fill="#17324d")
        draw.text((x + 28, y + 72), "No source figure was assigned to this section.", font=_font("regular", 20), fill="#566575")
        return
    image = _decode_image(figure_info)
    if image is None:
        return
    inner = (x + 16, y + 16, width - 32, height - 74)
    fitted = _fit_image(image, inner)
    ix = x + (width - fitted.width) // 2
    iy = y + 16 + (height - 74 - fitted.height) // 2
    canvas.paste(fitted, (ix, iy))
    caption = f"{figure_info.get('figure_id', 'Figure')}: {_clean_text(figure_info.get('caption', ''))}"
    _draw_wrapped(draw, caption, (x + 16, y + height - 48), width - 32, _font("regular", 14), "#566575", line_gap=2, max_lines=2)


def render_academic_slide(section, index: int, total: int, figure_images: Sequence[dict]) -> bytes:
    """Render one 16:9 evidence-preserving academic slide as PNG bytes."""
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#f6f8fb")
    draw = ImageDraw.Draw(canvas)
    navy = "#17324d"
    teal = "#168c93"
    muted = "#566575"

    draw.rectangle((0, 0, WIDTH, 14), fill=teal)
    title = _clean_text(getattr(section, "title", "Untitled slide"))
    title_font = _fit_title(draw, title, 1450)
    _draw_wrapped(draw, title, (70, 44), 1450, title_font, navy, line_gap=5, max_lines=2)
    draw.line((70, 158, WIDTH - 70, 158), fill="#d3dce5", width=2)

    figure_info = _figure_for_section(section, index, figure_images)
    section_content = _clean_text(getattr(section, "content", ""))
    if index == 0:
        # Keep the cover quiet and readable. The authors are at the front of the content.
        paragraphs = section_content.splitlines()
        cover_text = "\n".join(paragraphs[:3]) if paragraphs else section_content
        _draw_wrapped(draw, cover_text, (78, 225), 760, _font("regular", 26), muted, line_gap=10, max_lines=10)
        _draw_figure(canvas, draw, figure_info, (930, 215, 570, 425))
    else:
        has_figure = figure_info is not None
        text_width = 720 if has_figure else 1430
        body = section_content
        if index == total - 1:
            body = "\n\n".join(_clean_text(section_content).split("**Key contributions**"))
        body_font = _font("regular", 22 if has_figure else 24)
        _draw_wrapped(
            draw,
            body,
            (78, 192),
            text_width,
            body_font,
            muted,
            line_gap=7,
            max_lines=12 if has_figure else 14,
        )
        if has_figure:
            _draw_figure(canvas, draw, figure_info, (855, 190, 680, 470))

    values = _evidence_values(section_content)
    if values:
        y = 720
        draw.line((70, y - 18, WIDTH - 70, y - 18), fill="#d3dce5", width=2)
        draw.text((70, y), "EVIDENCE FROM THE PAPER", font=_font("bold", 17), fill=teal)
        x = 390
        for value in values:
            draw.line((x - 22, y + 2, x - 22, y + 42), fill=teal, width=3)
            draw.text((x, y), value, font=_font("bold", 25), fill=navy)
            x += 330
    draw.text((70, 850), "Reference-aware neural PDE surrogate benchmark", font=_font("regular", 17), fill="#8794a0")
    draw.text((1480, 850), f"{index + 1:02d} / {total:02d}", font=_font("bold", 17), fill="#8794a0")

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
