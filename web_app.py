from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, abort, render_template, request, send_from_directory, url_for

from jupyter_video_helper import choose_subtitle_for_stem, find_matching_video, srt_to_vtt
from youtube_subtitle_cli import CONCEPT_STYLE_CHOICES, download_video, run_job


APP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = APP_ROOT / "youtube_subs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_DEVICES = {"auto", "cpu", "cuda"}
ALLOWED_COMPUTE_TYPES = {"auto", "int8", "int8_float16", "float16", "float32"}
DEFAULT_FORM = {
    "urls_text": "",
    "model": "medium",
    "language": "",
    "translate_to": "ko",
    "concept_style": "ko_en",
    "device": "cuda",
    "compute_type": "float16",
    "beam_size": "5",
    "prompt_hint": "",
    "keep_audio": False,
    "continue_on_error": True,
}

app = Flask(__name__)


def parse_urls_text(urls_text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for raw in urls_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        seen.add(line)
        urls.append(line)
    return urls


def capture_output(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def media_relpath(path: str | Path) -> str:
    candidate = Path(path).resolve()
    return candidate.relative_to(OUTPUT_DIR).as_posix()


def build_download_links(entry: dict[str, Any]) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for path in entry["related_files"]:
        relpath = media_relpath(path)
        links.append(
            {
                "label": path.name,
                "url": url_for("media_file", relpath=relpath),
            }
        )
    if entry["video_path"]:
        links.insert(
            0,
            {
                "label": entry["video_path"].name,
                "url": url_for("media_file", relpath=media_relpath(entry["video_path"])),
            },
        )
    return links


def detect_video_mime(path: Path | None) -> str:
    if path is None:
        return "video/mp4"
    suffix = path.suffix.lower()
    if suffix == ".webm":
        return "video/webm"
    if suffix == ".mkv":
        return "video/x-matroska"
    if suffix == ".mov":
        return "video/quicktime"
    return "video/mp4"


def build_library_entry(json_path: Path) -> dict[str, Any]:
    payload = load_json(json_path)
    source = payload.get("source", {}) if isinstance(payload.get("source", {}), dict) else {}
    title = source.get("title") or json_path.stem
    video_id = source.get("id") or json_path.stem
    source_url = source.get("webpage_url") or ""
    base_stem = json_path.with_suffix("")
    preferred_subtitle = choose_subtitle_for_stem(base_stem, preferred_language="ko")
    video_path = find_matching_video(video_id, title, root=OUTPUT_DIR)
    related_files = sorted(
        [
            path
            for path in base_stem.parent.glob(base_stem.name + "*")
            if path.is_file()
            and path.suffix.lower() in {".json", ".srt", ".vtt", ".txt"}
        ]
    )

    return {
        "title": title,
        "video_id": video_id,
        "source_url": source_url,
        "json_path": json_path,
        "json_mtime": json_path.stat().st_mtime,
        "preferred_subtitle": preferred_subtitle,
        "video_path": video_path,
        "related_files": related_files,
        "duration": payload.get("meta", {}).get("duration"),
    }


def list_library_entries() -> list[dict[str, Any]]:
    entries = [build_library_entry(path) for path in OUTPUT_DIR.glob("*.json")]
    entries.sort(key=lambda item: item["json_mtime"], reverse=True)
    return entries


def find_entry_by_video_id(video_id: str) -> dict[str, Any] | None:
    for entry in list_library_entries():
        if entry["video_id"] == video_id:
            return entry
    return None


def normalize_form_data(form: Any) -> dict[str, Any]:
    beam_size = str(form.get("beam_size", DEFAULT_FORM["beam_size"])).strip() or DEFAULT_FORM["beam_size"]
    device = str(form.get("device", DEFAULT_FORM["device"])).strip().lower()
    compute_type = str(form.get("compute_type", DEFAULT_FORM["compute_type"])).strip().lower()
    concept_style = (
        str(form.get("concept_style", DEFAULT_FORM["concept_style"])).strip().lower()
        or DEFAULT_FORM["concept_style"]
    )

    if device not in ALLOWED_DEVICES:
        device = DEFAULT_FORM["device"]
    if compute_type not in ALLOWED_COMPUTE_TYPES:
        compute_type = DEFAULT_FORM["compute_type"]
    if concept_style not in CONCEPT_STYLE_CHOICES:
        concept_style = DEFAULT_FORM["concept_style"]

    return {
        "urls_text": str(form.get("urls_text", "")).strip(),
        "model": str(form.get("model", DEFAULT_FORM["model"])).strip() or DEFAULT_FORM["model"],
        "language": str(form.get("language", DEFAULT_FORM["language"])).strip(),
        "translate_to": str(form.get("translate_to", DEFAULT_FORM["translate_to"])).strip(),
        "concept_style": concept_style,
        "device": device,
        "compute_type": compute_type,
        "beam_size": beam_size,
        "prompt_hint": str(form.get("prompt_hint", DEFAULT_FORM["prompt_hint"])).strip(),
        "keep_audio": form.get("keep_audio") == "on",
        "continue_on_error": form.get("continue_on_error") == "on",
    }


def run_batch_download(form_data: dict[str, Any]) -> dict[str, Any]:
    urls = parse_urls_text(form_data["urls_text"])
    report = {
        "urls": urls,
        "successes": [],
        "failures": [],
        "log_text": "",
    }
    if not urls:
        report["failures"].append({"url": "", "error": "Paste at least one YouTube URL."})
        report["log_text"] = "No URLs were provided."
        return report

    all_logs: list[str] = []
    for index, url in enumerate(urls, start=1):
        all_logs.append(f"[{index}/{len(urls)}] {url}")
        try:
            result, logs = capture_output(
                run_job,
                url=url,
                output_dir=OUTPUT_DIR,
                model=form_data["model"],
                language=form_data["language"],
                device=form_data["device"],
                compute_type=form_data["compute_type"],
                beam_size=int(form_data["beam_size"]),
                prompt_hint=form_data["prompt_hint"],
                translate_to=form_data["translate_to"],
                concept_style=form_data["concept_style"],
                keep_audio=form_data["keep_audio"],
            )
            title = result["info"].get("title") or url
            report["successes"].append(
                {
                    "url": url,
                    "title": title,
                    "video_id": result["info"].get("id") or title,
                }
            )
            all_logs.append(logs.strip() or f"Finished: {title}")
        except Exception as exc:
            report["failures"].append({"url": url, "error": str(exc)})
            all_logs.append(f"ERROR: {exc}")
            if not form_data["continue_on_error"]:
                break

    report["log_text"] = "\n\n".join(part for part in all_logs if part).strip()
    return report


@app.route("/", methods=["GET", "POST"])
def index():
    form_data = dict(DEFAULT_FORM)
    report = None

    if request.method == "POST":
        form_data = normalize_form_data(request.form)
        report = run_batch_download(form_data)
    else:
        sample_urls = [entry["source_url"] for entry in list_library_entries()[:3] if entry["source_url"]]
        if sample_urls:
            form_data["urls_text"] = "\n".join(sample_urls[:2])

    entries = list_library_entries()
    return render_template(
        "index.html",
        entries=entries,
        form_data=form_data,
        report=report,
    )


@app.route("/player/<video_id>")
def player(video_id: str):
    entry = find_entry_by_video_id(video_id)
    if entry is None:
        abort(404)

    player_log = ""
    if entry["video_path"] is None and entry["source_url"]:
        _, player_log = capture_output(download_video, entry["source_url"], OUTPUT_DIR)
        entry = find_entry_by_video_id(video_id)

    if entry is None:
        abort(404)

    subtitle_path = entry["preferred_subtitle"]
    if subtitle_path and subtitle_path.suffix.lower() == ".srt":
        subtitle_path = srt_to_vtt(subtitle_path)

    video_url = None
    video_mime = "video/mp4"
    subtitle_url = None
    if entry["video_path"] is not None:
        video_url = url_for("media_file", relpath=media_relpath(entry["video_path"]))
        video_mime = detect_video_mime(entry["video_path"])
    if subtitle_path is not None:
        subtitle_url = url_for("media_file", relpath=media_relpath(subtitle_path))

    return render_template(
        "player.html",
        entry=entry,
        video_url=video_url,
        video_mime=video_mime,
        subtitle_url=subtitle_url,
        download_links=build_download_links(entry),
        player_log=player_log.strip(),
    )


@app.route("/media/<path:relpath>")
def media_file(relpath: str):
    return send_from_directory(OUTPUT_DIR, relpath)


@app.route("/health")
def health():
    return {
        "status": "ok",
        "library_size": len(list_library_entries()),
        "output_dir": str(OUTPUT_DIR),
    }


def main() -> None:
    host = os.environ.get("YTSUB_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("YTSUB_WEB_PORT", "8765"))
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
