from __future__ import annotations

import base64
import html
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import fitz
from deep_translator import GoogleTranslator


ROOT = Path(__file__).resolve().parent
PAPERS = [
    ROOT / "1805.00909v3.pdf",
    ROOT / "2005.01643v3.pdf",
]

TITLE_MAP = {
    "1805.00909v3.pdf": "Reinforcement Learning and Control as Probabilistic Inference",
    "2005.01643v3.pdf": "Offline Reinforcement Learning",
}

PROTECTED_TERMS = [
    "Offline Reinforcement Learning",
    "online Reinforcement Learning",
    "Reinforcement Learning",
    "optimal control",
    "probabilistic inference",
    "probabilistic graphical models",
    "probabilistic graphical model",
    "variational inference",
    "stochastic dynamics",
    "Maximum Entropy Reinforcement Learning",
    "maximum entropy",
    "dynamic programming",
    "importance sampling",
    "off-policy",
    "on-policy",
    "policy gradient",
    "policy gradients",
    "policy search",
    "policy",
    "Actor-Critic",
    "actor-critic",
    "Soft Q-Learning",
    "Q-Learning",
    "Q-learning",
    "Q-function",
    "value function",
    "model-based",
    "uncertainty estimation",
    "distribution shift",
    "support constraint",
    "distribution constraint",
    "conservative Q-learning",
    "pessimistic value functions",
    "robotics",
    "autonomous driving",
    "dialogue",
    "language",
    "healthcare",
    "Kalman",
    "KL-divergence",
    "MDP",
    "POMDP",
]

FORMAT_NOTE = "\ud55c\uad6d\uc5b4\ub85c \uc77d\uae30 \uc27d\uac8c \uc7ac\uad6c\uc131\ud55c note\ud615 \ubc88\uc5ed\ubcf8"
STYLE_NOTE = "\ud575\uc2ec technical term\uc740 English\ub85c \uc720\uc9c0"
FIGURE_NOTE = "figure\ub294 `_ko.html`, `_ko.pdf`\ud310\uc5d0 \ud3ec\ud568"
EQUATION_NOTE = "\uc218\uc2dd \ube14\ub85d\uc740 PDF \ud14d\uc2a4\ud2b8 \ucd94\ucd9c \ud55c\uacc4\ub85c \uc77c\ubd80 \uc6d0\ubb38 \ud45c\uae30 \ub610\ub294 \uc0dd\ub7b5"


def normalize_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_match_key(text: str) -> str:
    text = normalize_text(text).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def is_noise_block(text: str) -> bool:
    if not text:
        return True
    if text.startswith("arXiv:"):
        return True
    if re.fullmatch(r"\d+", text):
        return True
    return False


def looks_like_sentence(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{1,}", text)
    return len(words) >= 6 or bool(re.search(r"[.!?]", text))


def is_math_heavy(text: str) -> bool:
    if len(text) < 20:
        return False
    math_chars = sum(ch in "=<>|[]{}_^~/*+%$#&\\" for ch in text)
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
    return math_chars >= 4 or digit_ratio > 0.28


def is_formula_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    alpha_words = re.findall(r"[A-Za-z]{2,}", stripped)
    digit_ratio = sum(ch.isdigit() for ch in stripped) / max(1, len(stripped))
    symbol_ratio = sum(ch in "=<>|[]{}_^~/*+%$#&\\" for ch in stripped) / max(1, len(stripped))

    if re.fullmatch(r"[\W\d_]+", stripped):
        return True
    if stripped.startswith(("#", "##", "!", "*")) and len(alpha_words) < 4:
        return True
    if symbol_ratio > 0.12 and len(alpha_words) < 6:
        return True
    if digit_ratio > 0.35 and len(alpha_words) < 6:
        return True
    if is_math_heavy(stripped) and not looks_like_sentence(stripped) and len(alpha_words) < 7:
        return True
    return False


def escape_markdown_paragraph(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if text[0] in "#-*+>|":
        return "\\" + text
    return text


def slugify(text: str, seen: dict[str, int]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", normalize_match_key(text)).strip("-") or "section"
    seen[base] = seen.get(base, 0) + 1
    if seen[base] == 1:
        return base
    return f"{base}-{seen[base]}"


@dataclass
class Block:
    kind: str
    text: str
    page_index: int
    bbox: tuple[float, float, float, float]
    font_max: float


class ParagraphTranslator:
    def __init__(self) -> None:
        self.engine = GoogleTranslator(source="en", target="ko")
        self.cache: dict[str, str] = {}

    def protect_terms(self, text: str) -> tuple[str, dict[str, str]]:
        placeholders: dict[str, str] = {}
        result = text
        for idx, term in enumerate(sorted(PROTECTED_TERMS, key=len, reverse=True)):
            token = f"ZZTERM{idx:03d}ZZ"
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(result):
                result = pattern.sub(token, result)
                placeholders[token] = term
        return result, placeholders

    def restore_terms(self, text: str, placeholders: dict[str, str]) -> str:
        for token, term in placeholders.items():
            text = text.replace(token, term)
        return text

    def postprocess(self, text: str) -> str:
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text.strip()

    def translate(self, text: str) -> str:
        text = normalize_text(text)
        if not text:
            return ""
        if text in self.cache:
            return self.cache[text]

        if is_math_heavy(text) and not looks_like_sentence(text):
            self.cache[text] = text
            return text

        protected, placeholders = self.protect_terms(text)
        translated: str | None = None

        try:
            if len(protected) <= 2400:
                translated = self.engine.translate(protected)
            else:
                pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", protected) if piece.strip()]
                translated_pieces: list[str] = []
                for piece in pieces:
                    translated_pieces.append(self.engine.translate(piece) or piece)
                translated = " ".join(translated_pieces)
        except Exception:
            time.sleep(1.0)
            try:
                translated = self.engine.translate(protected)
            except Exception:
                translated = protected

        if not translated:
            translated = protected

        translated = self.restore_terms(translated, placeholders)
        translated = self.postprocess(translated)
        self.cache[text] = translated
        return translated


def collect_blocks(doc: fitz.Document) -> tuple[str, list[tuple[int, str]], list[Block]]:
    title = ""
    blocks_out: list[Block] = []
    toc = doc.get_toc(simple=True)
    toc_map = {normalize_match_key(item[1]): item[0] for item in toc}

    for page_index, page in enumerate(doc):
        for blk in page.get_text("dict", sort=True)["blocks"]:
            if "lines" not in blk:
                continue
            lines: list[str] = []
            font_sizes: list[float] = []
            for line in blk["lines"]:
                spans = line["spans"]
                font_sizes.extend(span["size"] for span in spans)
                lines.append("".join(span["text"] for span in spans))

            text = normalize_text(" ".join(lines))
            font_max = max(font_sizes) if font_sizes else 0.0

            if is_noise_block(text):
                continue
            if page_index == 0 and font_max >= 16 and not title:
                title = text
                blocks_out.append(Block("title", text, page_index, tuple(blk["bbox"]), font_max))
                continue
            if text in {"Abstract", "References"}:
                blocks_out.append(Block("heading", text, page_index, tuple(blk["bbox"]), font_max))
                continue

            key = normalize_match_key(text)
            if key in toc_map:
                blocks_out.append(Block("heading", text, page_index, tuple(blk["bbox"]), font_max))
                continue

            if re.match(r"^(Figure|Table)\s+\d+:", text):
                blocks_out.append(Block("caption", text, page_index, tuple(blk["bbox"]), font_max))
                continue

            if re.match(r"^(\d+(\.\d+)*)\s+.+", text) and font_max >= 10.5:
                blocks_out.append(Block("heading", text, page_index, tuple(blk["bbox"]), font_max))
                continue

            blocks_out.append(Block("paragraph", text, page_index, tuple(blk["bbox"]), font_max))

    return title, doc.get_toc(simple=True), blocks_out


def toc_level_for(text: str, toc_entries: list[tuple[int, str]]) -> int:
    key = normalize_match_key(text)
    for level, title in toc_entries:
        if normalize_match_key(title) == key:
            return level
    if text in {"Abstract", "References"}:
        return 1
    match = re.match(r"^(\d+)(\.\d+)?", text)
    if match and match.group(2):
        return 2
    return 1


def figure_data_uri(doc: fitz.Document, caption_block: Block, prev_block: Block | None) -> str:
    page = doc[caption_block.page_index]
    _x0, y0, _x1, y1 = caption_block.bbox
    if prev_block is not None:
        top = max(40.0, prev_block.bbox[3] + 8.0)
        if top > y0 - 30:
            top = max(40.0, y0 - 220.0)
    else:
        top = max(40.0, y0 - 220.0)
    clip = fitz.Rect(40.0, top, page.rect.width - 40.0, y1 + 6.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
    data = base64.b64encode(pix.tobytes("png")).decode("ascii")
    return f"data:image/png;base64,{data}"


def caption_to_korean(text: str, translator: ParagraphTranslator) -> str:
    match = re.match(r"^(Figure|Table)\s+(\d+):\s*(.*)$", text)
    if not match:
        return translator.translate(text)
    kind, num, rest = match.groups()
    translated = translator.translate(rest)
    return f"{kind} {num}. {translated}"


def should_skip_paragraph(block: Block) -> bool:
    text = block.text
    if is_noise_block(text):
        return True
    if len(text) <= 2:
        return True
    if block.font_max <= 8.5 and len(text) < 80 and re.search(r"\b(arXiv|Proc\.|Proceedings|doi)\b", text):
        return True
    if is_formula_noise(text):
        return True
    return False


def render_outputs(
    doc: fitz.Document,
    title: str,
    filename: str,
    toc_entries: list[tuple[int, str]],
    blocks: list[Block],
) -> tuple[str, str]:
    translator = ParagraphTranslator()
    md_parts: list[str] = [
        f"# {title}",
        "",
        f"- Original PDF: `{filename}`",
        f"- Format: {FORMAT_NOTE}",
        f"- Style: {STYLE_NOTE}",
        f"- Figures: {FIGURE_NOTE}",
        f"- Notes: {EQUATION_NOTE}",
        "",
    ]
    body_html: list[str] = [
        f"<h1>{html.escape(title)}</h1>",
        "<ul class=\"meta\">",
        f"<li><strong>Original PDF</strong>: <code>{html.escape(filename)}</code></li>",
        f"<li><strong>Format</strong>: {html.escape(FORMAT_NOTE)}</li>",
        f"<li><strong>Style</strong>: {html.escape(STYLE_NOTE)}</li>",
        f"<li><strong>Figures</strong>: {html.escape(FIGURE_NOTE)}</li>",
        f"<li><strong>Notes</strong>: {html.escape(EQUATION_NOTE)}</li>",
        "</ul>",
    ]

    toc_items: list[tuple[int, str, str]] = []
    slug_seen: dict[str, int] = {}
    seen_headings: set[tuple[int, str]] = set()
    stop = False

    for idx, block in enumerate(blocks):
        if stop:
            break
        if block.kind == "title":
            continue

        if block.kind == "heading":
            if block.text == "References":
                stop = True
                continue

            level = toc_level_for(block.text, toc_entries)
            key = (level, block.text)
            if key in seen_headings:
                continue
            seen_headings.add(key)

            anchor = slugify(block.text, slug_seen)
            heading_tag = min(level + 1, 6)
            md_parts.extend([f"{'#' * heading_tag} {block.text}", ""])
            toc_items.append((heading_tag, block.text, anchor))
            body_html.append(f"<h{heading_tag} id=\"{anchor}\">{html.escape(block.text)}</h{heading_tag}>")
            continue

        if block.kind == "caption":
            prev_block = None
            for j in range(idx - 1, -1, -1):
                if blocks[j].page_index != block.page_index:
                    break
                if blocks[j].kind == "paragraph":
                    prev_block = blocks[j]
                    break

            caption_ko = caption_to_korean(block.text, translator)
            md_parts.extend([f"*{caption_ko}*", ""])
            data_uri = figure_data_uri(doc, block, prev_block)
            body_html.append(
                "<figure>"
                f"<img src=\"{data_uri}\" alt=\"{html.escape(caption_ko)}\" />"
                f"<figcaption>{html.escape(caption_ko)}</figcaption>"
                "</figure>"
            )
            continue

        if block.kind == "paragraph":
            if should_skip_paragraph(block):
                continue
            translated = translator.translate(block.text)
            translated = escape_markdown_paragraph(translated)
            if not translated:
                continue
            md_parts.extend([translated, ""])
            body_html.append(f"<p>{html.escape(translated)}</p>")

    toc_html = ""
    if toc_items:
        toc_lines = ["<nav class=\"toc\"><h2>Contents</h2><ul>"]
        for level, text, anchor in toc_items:
            depth = max(0, level - 2)
            toc_lines.append(
                f"<li class=\"toc-level-{depth}\"><a href=\"#{anchor}\">{html.escape(text)}</a></li>"
            )
        toc_lines.append("</ul></nav>")
        toc_html = "".join(toc_lines)

    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --paper: #ffffff;
      --text: #1c1f24;
      --muted: #5b6470;
      --line: #d6dde5;
      --accent: #eef3f8;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
      line-height: 1.78;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      background: var(--paper);
      padding: 48px 56px 72px;
      box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
    }}
    h1, h2, h3, h4, h5, h6 {{
      line-height: 1.35;
      margin: 1.8rem 0 0.8rem;
      page-break-after: avoid;
    }}
    h1 {{
      margin-top: 0;
      font-size: 2rem;
    }}
    h2 {{
      font-size: 1.45rem;
      border-bottom: 1px solid var(--line);
      padding-bottom: 0.35rem;
    }}
    h3 {{
      font-size: 1.18rem;
    }}
    p {{
      margin: 0.85rem 0;
      text-align: left;
      word-break: keep-all;
    }}
    code {{
      background: #f2f4f7;
      padding: 0.08rem 0.3rem;
      border-radius: 0.25rem;
    }}
    .meta {{
      margin: 1rem 0 1.5rem;
      padding: 1rem 1.15rem;
      background: var(--accent);
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    .meta li + li {{
      margin-top: 0.35rem;
    }}
    .toc {{
      margin: 1.4rem 0 2rem;
      padding: 1rem 1.15rem;
      background: #fbfcfd;
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    .toc h2 {{
      margin: 0 0 0.7rem;
      border: 0;
      padding: 0;
      font-size: 1.05rem;
    }}
    .toc ul {{
      list-style: none;
      margin: 0;
      padding: 0;
    }}
    .toc li + li {{
      margin-top: 0.38rem;
    }}
    .toc-level-1 {{
      padding-left: 1rem;
    }}
    .toc-level-2 {{
      padding-left: 2rem;
    }}
    .toc a {{
      color: inherit;
      text-decoration: none;
    }}
    figure {{
      margin: 1.35rem 0 1.75rem;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 1rem;
      background: #fafbfc;
      break-inside: avoid;
    }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 10px;
    }}
    figcaption {{
      margin-top: 0.65rem;
      color: var(--muted);
      font-size: 0.96rem;
    }}
    @page {{
      size: A4;
      margin: 16mm 14mm 18mm;
    }}
    @media print {{
      body {{
        background: #ffffff;
      }}
      main {{
        max-width: none;
        margin: 0;
        padding: 0;
        box-shadow: none;
      }}
      .toc, .meta, figure {{
        break-inside: avoid;
      }}
    }}
  </style>
</head>
<body>
  <main>
    {toc_html}
    {"".join(body_html)}
  </main>
</body>
</html>
"""

    md_text = "\n".join(md_parts).strip() + "\n"
    return md_text, html_text


def build_pdf_from_html(html_path: Path, pdf_path: Path) -> None:
    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    if not chrome.exists():
        raise FileNotFoundError(chrome)

    subprocess.run(
        [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--virtual-time-budget=8000",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
    )


def main() -> None:
    selected = {name.lower() for name in sys.argv[1:]}
    papers = [pdf_path for pdf_path in PAPERS if not selected or pdf_path.name.lower() in selected]

    for pdf_path in papers:
        doc = fitz.open(pdf_path)
        title, toc_raw, blocks = collect_blocks(doc)
        title = TITLE_MAP.get(pdf_path.name, title or pdf_path.stem)
        toc_entries = [(level, heading) for level, heading, _page in toc_raw]

        out_md = pdf_path.with_name(pdf_path.stem + "_ko.md")
        out_html = pdf_path.with_name(pdf_path.stem + "_ko.html")
        out_pdf = pdf_path.with_name(pdf_path.stem + "_ko.pdf")

        md_text, html_text = render_outputs(doc, title, pdf_path.name, toc_entries, blocks)
        out_md.write_text(md_text, encoding="utf-8")
        out_html.write_text(html_text, encoding="utf-8")
        build_pdf_from_html(out_html, out_pdf)

        print(f"created {out_md.name}")
        print(f"created {out_html.name}")
        print(f"created {out_pdf.name}")


if __name__ == "__main__":
    main()
