"""Evidence-preserving rendering for academic paper slides.

Diffusion models are useful for illustrations, but they are a poor renderer for
scientific titles, tables, labels, and numerical results.  This module keeps
those elements deterministic and uses the original paper figures as evidence.
"""

from __future__ import annotations

import base64
import io
from html.parser import HTMLParser
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 900
POSTER_WIDTH = 3200
POSTER_HEIGHT = 1800

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


class _TableParser(HTMLParser):
    """Extract a small, presentation-safe row model from an HTML table."""

    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: List[str] | None = None
        self._cell: List[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"th", "td"} and self._row is not None and self._cell is not None:
            value = _clean_text("".join(self._cell))
            self._row.append(value)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _table_rows(html_content: str) -> List[List[str]]:
    if not html_content:
        return []
    parser = _TableParser()
    try:
        parser.feed(html_content)
    except Exception:
        return []
    return parser.rows


def _draw_table_preview(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    table_id: str,
    table_html: str,
    box: Tuple[int, int, int, int],
    max_rows: int = 4,
) -> None:
    """Draw a compact exact-data preview rather than asking diffusion to redraw it."""
    x, y, width, height = box
    rows = _table_rows(table_html)
    if not rows:
        return
    rows = rows[:max_rows]
    columns = max(len(row) for row in rows)
    if columns == 0:
        return
    draw.rounded_rectangle((x, y, x + width, y + height), radius=10, fill="#ffffff", outline="#cbd7e2", width=2)
    draw.text((x + 16, y + 10), table_id, font=_font("bold", 22), fill="#168c93")
    top = y + 42
    row_height = max(38, min(58, (height - 48) // max(1, len(rows))))
    col_width = width // columns
    for row_index, row in enumerate(rows):
        row_top = top + row_index * row_height
        fill = "#e8f2f3" if row_index == 0 else "#ffffff"
        draw.rectangle((x, row_top, x + width, row_top + row_height), fill=fill)
        for col_index in range(columns):
            cell_x = x + col_index * col_width
            cell = row[col_index] if col_index < len(row) else ""
            _draw_wrapped(
                draw,
                cell,
                (cell_x + 10, row_top + 7),
                col_width - 20,
                _font("bold" if row_index == 0 else "regular", 16),
                "#17324d" if row_index == 0 else "#566575",
                line_gap=1,
                max_lines=2,
            )
            draw.line((cell_x, row_top, cell_x, row_top + row_height), fill="#d5e0e8", width=1)
        draw.line((x, row_top, x + width, row_top), fill="#d5e0e8", width=1)
    draw.line((x, top + len(rows) * row_height, x + width, top + len(rows) * row_height), fill="#d5e0e8", width=1)


def _section_height_estimate(section) -> int:
    text = _clean_text(getattr(section, "content", ""))
    return max(320, 220 + len(text) // 9 + 170 * len(getattr(section, "figures", [])))


def _draw_poster_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    section,
    plan,
    figure_images: Sequence[dict],
    box: Tuple[int, int, int, int],
    section_index: int,
    density: str,
) -> None:
    x, y, width, height = box
    navy = "#17324d"
    teal = "#168c93"
    muted = "#566575"
    draw.rounded_rectangle((x, y, x + width, y + height), radius=18, fill="#ffffff", outline="#cbd7e2", width=3)
    draw.rectangle((x, y, x + width, y + 12), fill=teal)
    title = _clean_text(getattr(section, "title", "Untitled section"))
    title_font = _font("bold", 34 if width >= 900 else 30)
    _draw_wrapped(draw, title, (x + 28, y + 30), width - 56, title_font, navy, line_gap=3, max_lines=2)
    title_lines = _wrap(draw, title, title_font, width - 56)
    body_y = y + 42 + len(title_lines) * 40
    inner_x = x + 28
    inner_width = width - 56
    figure_info = _figure_for_section(section, section_index, figure_images)
    figure_height = min(330, max(210, height // 3)) if figure_info else 0

    if figure_info and height >= 620:
        text_width = int(inner_width * 0.54)
        figure_box = (x + width - int(inner_width * 0.42) - 28, body_y, int(inner_width * 0.42), figure_height)
        _draw_wrapped(
            draw,
            getattr(section, "content", ""),
            (inner_x, body_y),
            text_width,
            _font("regular", 22),
            muted,
            line_gap=6,
            max_lines=max(8, min(22, (height - (body_y - y) - 40) // 32)),
        )
        _draw_figure(canvas, draw, figure_info, figure_box)
        asset_y = max(body_y + figure_height + 18, y + height - 245)
    else:
        _draw_wrapped(
            draw,
            getattr(section, "content", ""),
            (inner_x, body_y),
            inner_width,
            _font("regular", 22 if density == "dense" else 24),
            muted,
            line_gap=6,
            max_lines=max(8, min(27, (height - (body_y - y) - 40) // 32)),
        )
        asset_y = y + height - 245
        if figure_info and height >= 430:
            figure_box = (x + 28, asset_y, inner_width, min(210, height - (asset_y - y) - 20))
            _draw_figure(canvas, draw, figure_info, figure_box)

    table_refs = getattr(section, "tables", [])
    if table_refs and height >= 500:
        table_ref = table_refs[0]
        table = plan.tables_index.get(table_ref.table_id)
        if table:
            table_box = (inner_x, y + height - 225, inner_width, 190)
            _draw_table_preview(canvas, draw, table_ref.table_id, table_ref.extract or table.html_content, table_box)
        if len(table_refs) > 1:
            extra = ", ".join(ref.table_id for ref in table_refs[1:])
            draw.text((inner_x, y + height - 28), f"Additional evidence: {extra}", font=_font("regular", 16), fill=muted)


def render_academic_poster(plan, figure_images: Sequence[dict], density: str = "medium") -> bytes:
    """Render a readable 16:9 academic poster from exact plan evidence.

    The poster is deliberately native-rendered.  A diffusion model can make a
    compelling background, but it cannot be trusted with scientific labels,
    equations, table values, or figure identity.
    """
    canvas = Image.new("RGB", (POSTER_WIDTH, POSTER_HEIGHT), "#f3f6f9")
    draw = ImageDraw.Draw(canvas)
    navy = "#17324d"
    teal = "#168c93"
    muted = "#566575"
    margin = 64
    gap = 34

    draw.rectangle((0, 0, POSTER_WIDTH, 20), fill=teal)
    sections = list(getattr(plan, "sections", []))
    title_section = sections[0] if sections else None
    title = _clean_text(getattr(title_section, "title", "Academic poster")) if title_section else "Academic poster"
    title_font = _font("bold", 64)
    _draw_wrapped(draw, title, (margin, 52), POSTER_WIDTH - 2 * margin, title_font, navy, line_gap=6, max_lines=2)
    title_lines = _wrap(draw, title, title_font, POSTER_WIDTH - 2 * margin)
    title_body = _clean_text(getattr(title_section, "content", "")) if title_section else ""
    title_y = 72 + len(title_lines) * 74
    _draw_wrapped(draw, title_body, (margin, title_y), POSTER_WIDTH - 2 * margin, _font("regular", 25), muted, line_gap=6, max_lines=4)
    header_bottom = max(270, title_y + 150)
    draw.line((margin, header_bottom, POSTER_WIDTH - margin, header_bottom), fill="#cbd7e2", width=3)

    body_sections = sections[1:]
    columns = [[], [], []]
    heights = [0, 0, 0]
    for section in body_sections:
        target = min(range(3), key=lambda index: heights[index])
        columns[target].append(section)
        heights[target] += _section_height_estimate(section)

    body_top = header_bottom + 32
    body_bottom = POSTER_HEIGHT - 60
    column_width = (POSTER_WIDTH - 2 * margin - 2 * gap) // 3
    for column_index, column_sections in enumerate(columns):
        if not column_sections:
            continue
        x = margin + column_index * (column_width + gap)
        total_estimate = sum(_section_height_estimate(section) for section in column_sections)
        available = body_bottom - body_top - gap * (len(column_sections) - 1)
        cursor = body_top
        for section_index, section in enumerate(column_sections):
            estimate = _section_height_estimate(section)
            height = max(250, int(available * estimate / max(1, total_estimate)))
            height = min(height, body_bottom - cursor)
            _draw_poster_card(
                canvas,
                draw,
                section,
                plan,
                figure_images,
                (x, cursor, column_width, height),
                section_index + 1,
                density,
            )
            cursor += height + gap

    draw.text((margin, POSTER_HEIGHT - 38), "Evidence-preserving academic poster • source text, tables, and figures rendered locally", font=_font("regular", 18), fill="#8794a0")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


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
