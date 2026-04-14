from __future__ import annotations

import html
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

try:
    from IPython.display import HTML
except ImportError:
    class HTML(str):
        def __new__(cls, data: str):
            return str.__new__(cls, data)


VIDEO_EXTENSIONS = {".webm", ".mp4", ".mkv", ".mov", ".avi"}
SUBTITLE_EXTENSIONS = {".vtt", ".srt"}
SKIP_DIR_NAMES = {".youtube_subs_env", "__pycache__", ".ipynb_checkpoints", "audio", "models"}


def list_media_files(root: str | Path = ".") -> dict[str, list[Path]]:
    root = Path(root).resolve()
    videos: list[Path] = []
    subtitles: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
        elif path.suffix.lower() in SUBTITLE_EXTENSIONS:
            subtitles.append(path)

    return {
        "videos": sorted(videos),
        "subtitles": sorted(subtitles),
    }


def extract_youtube_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]

    if host in {"youtu.be", "www.youtu.be"} and path_parts:
        return path_parts[0]

    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        video_ids = parse_qs(parsed.query).get("v")
        if video_ids:
            return video_ids[0]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "live", "embed"}:
            return path_parts[1]

    return None


def load_saved_metadata(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def choose_subtitle_for_stem(base_stem: Path, preferred_language: str = "ko") -> Path | None:
    candidates = [
        base_stem.with_suffix(f".{preferred_language}.vtt"),
        base_stem.with_suffix(f".{preferred_language}.srt"),
        base_stem.with_suffix(".vtt"),
        base_stem.with_suffix(".srt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in sorted(base_stem.parent.glob(base_stem.name + ".*")):
        if candidate.suffix.lower() in SUBTITLE_EXTENSIONS:
            return candidate
    return None


def find_matching_video(video_id: str | None, title: str | None, root: str | Path = ".") -> Path | None:
    candidates = list_media_files(root)["videos"]
    if not candidates:
        return None

    if video_id:
        by_id = [path for path in candidates if f"[{video_id}]" in path.name]
        if by_id:
            return sorted(by_id)[0]

    if title:
        expected = normalize_name_for_match(Path(title))
        for candidate in candidates:
            if normalize_name_for_match(candidate) == expected:
                return candidate

    return None


def find_outputs_for_url(
    url: str,
    root: str | Path = ".",
    preferred_language: str = "ko",
) -> dict[str, Path | dict | str | None] | None:
    root = Path(root).resolve()
    video_id = extract_youtube_id(url)

    for json_path in sorted(root.glob("*.json")):
        payload = load_saved_metadata(json_path)
        source = payload.get("source", {})
        if not isinstance(source, dict):
            continue

        source_id = source.get("id")
        source_url = source.get("webpage_url")
        if video_id:
            matched = source_id == video_id
        else:
            matched = source_url == url

        if not matched:
            continue

        base_stem = json_path.with_suffix("")
        subtitle = choose_subtitle_for_stem(base_stem, preferred_language=preferred_language)
        video = find_matching_video(source_id, source.get("title"), root=root)

        return {
            "json": json_path,
            "subtitle": subtitle,
            "video": video,
            "payload": payload,
            "video_id": source_id,
            "title": source.get("title"),
        }

    return None


def normalize_name_for_match(path: Path) -> str:
    name = path.stem.lower()
    name = re.sub(r"\.(ko|en|kr|eng|kor)$", "", name)
    name = re.sub(r"[\W_]+", "", name)
    return name


def find_best_subtitle(video_path: str | Path, root: str | Path = ".") -> Path | None:
    video_path = Path(video_path).resolve()
    candidates = list_media_files(root)["subtitles"]
    if not candidates:
        return None

    video_key = normalize_name_for_match(video_path)

    exact: list[Path] = []
    fuzzy: list[tuple[int, Path]] = []
    for subtitle in candidates:
        subtitle_key = normalize_name_for_match(subtitle)
        if subtitle_key == video_key:
            exact.append(subtitle)
        else:
            overlap = len(set(video_key) & set(subtitle_key))
            fuzzy.append((overlap, subtitle))

    if exact:
        return sorted(exact)[0]

    fuzzy.sort(key=lambda item: item[0], reverse=True)
    return fuzzy[0][1] if fuzzy and fuzzy[0][0] > 0 else None


def find_best_video(root: str | Path = ".") -> Path:
    videos = list_media_files(root)["videos"]
    if not videos:
        raise FileNotFoundError(
            "No video file found. Put a .webm/.mp4 file in the notebook folder or pass video_path explicitly."
        )
    return videos[0]


def srt_to_vtt(srt_path: str | Path, vtt_path: str | Path | None = None) -> Path:
    srt_path = Path(srt_path).resolve()
    if not srt_path.exists():
        raise FileNotFoundError(srt_path)

    vtt_path = Path(vtt_path).resolve() if vtt_path else srt_path.with_suffix(".vtt")
    text = srt_path.read_text(encoding="utf-8")

    # Convert only timestamp commas to dots.
    text = re.sub(
        r"(\d{2}:\d{2}:\d{2}),(\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}),(\d{3})",
        r"\1.\2 --> \3.\4",
        text,
    )
    vtt_path.write_text("WEBVTT\n\n" + text.strip() + "\n", encoding="utf-8")
    return vtt_path


def as_notebook_src(path: str | Path, base_dir: str | Path = ".") -> str:
    path = Path(path).resolve()
    base_dir = Path(base_dir).resolve()
    try:
        rel = path.relative_to(base_dir)
        return quote(rel.as_posix())
    except ValueError:
        return quote(path.as_posix())


def show_video(
    video_path: str | Path | None = None,
    subtitle_path: str | Path | None = None,
    root: str | Path = ".",
    width: int = 960,
    base_dir: str | Path | None = None,
):
    root = Path(root).resolve()
    base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()

    if video_path is None:
        video_path = find_best_video(root)
    else:
        video_path = Path(video_path).resolve()

    if subtitle_path is None:
        subtitle_path = find_best_subtitle(video_path, root)
    elif subtitle_path:
        subtitle_path = Path(subtitle_path).resolve()

    track_html = ""
    if subtitle_path:
        subtitle_path = Path(subtitle_path)
        if subtitle_path.suffix.lower() == ".srt":
            subtitle_path = srt_to_vtt(subtitle_path)
        track_src = as_notebook_src(subtitle_path, base_dir)
        track_html = (
            f'<track src="{track_src}" kind="subtitles" srclang="ko" label="Subtitle" default>'
        )

    video_src = as_notebook_src(video_path, base_dir)
    video_type = f"video/{video_path.suffix.lower().lstrip('.')}"

    return HTML(
        f"""
<div style="max-width:{width}px;">
  <video controls width="{width}" style="max-width:100%; border-radius: 12px;">
    <source src="{video_src}" type="{html.escape(video_type)}">
    {track_html}
    Your browser does not support embedded video playback.
  </video>
</div>
""".strip()
    )


def ensure_outputs_for_url(
    url: str,
    root: str | Path = ".",
    preferred_language: str = "ko",
    model: str = "medium",
    language: str = "",
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    prompt_hint: str = "",
    translate_to: str = "ko",
    concept_style: str = "auto",
    glossary_path: str | Path | None = None,
) -> dict[str, Path | dict | str | None]:
    existing = find_outputs_for_url(url, root=root, preferred_language=preferred_language)
    if existing and existing.get("subtitle"):
        print(f"Reusing saved subtitles for URL: {url}")
        return existing

    from youtube_subtitle_cli import run_job

    print(f"Generating subtitles for URL: {url}")
    result = run_job(
        url=url,
        output_dir=root,
        model=model,
        language=language,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        prompt_hint=prompt_hint,
        translate_to=translate_to,
        concept_style=concept_style,
        glossary_path=glossary_path,
    )
    refreshed = find_outputs_for_url(url, root=root, preferred_language=preferred_language)
    if refreshed:
        return refreshed

    outputs = result["outputs"]
    return {
        "json": outputs.get("json"),
        "subtitle": outputs.get(f"srt_{preferred_language}") or outputs.get("srt"),
        "video": None,
        "payload": {"source": result["info"], "meta": result["meta"]},
        "video_id": result["info"].get("id"),
        "title": result["info"].get("title"),
    }


def show_video_for_url(
    url: str,
    root: str | Path = "youtube_subs",
    width: int = 960,
    preferred_language: str = "ko",
    model: str = "medium",
    language: str = "",
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    prompt_hint: str = "",
    translate_to: str = "ko",
    concept_style: str = "auto",
    glossary_path: str | Path | None = None,
    base_dir: str | Path | None = None,
):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()

    saved = ensure_outputs_for_url(
        url=url,
        root=root,
        preferred_language=preferred_language,
        model=model,
        language=language,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        prompt_hint=prompt_hint,
        translate_to=translate_to,
        concept_style=concept_style,
        glossary_path=glossary_path,
    )

    video_path = saved.get("video")
    if not video_path:
        from youtube_subtitle_cli import download_video

        print(f"Downloading video for URL: {url}")
        video_path, _, _ = download_video(url, root)

    subtitle_path = saved.get("subtitle")
    if not subtitle_path:
        raise FileNotFoundError(f"No subtitle file found for URL: {url}")

    print(f"Video file      : {Path(video_path).resolve()}")
    print(f"Subtitle file   : {Path(subtitle_path).resolve()}")

    return show_video(
        video_path=video_path,
        subtitle_path=subtitle_path,
        root=root,
        width=width,
        base_dir=base_dir,
    )


def show_available_media(root: str | Path = ".") -> None:
    media = list_media_files(root)
    print("Videos:")
    for path in media["videos"]:
        print(" -", path)
    print("Subtitles:")
    for path in media["subtitles"]:
        print(" -", path)
