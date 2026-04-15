from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, abort, render_template, request, send_from_directory, url_for

from jupyter_video_helper import choose_subtitle_for_stem, find_matching_video, srt_to_vtt
from youtube_subtitle_cli import (
    CONCEPT_STYLE_CHOICES,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    TRANSLATION_BACKEND_CHOICES,
    download_video,
    load_saved_job,
    run_job,
)


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
    "translation_backend": "ollama",
    "llm_model": DEFAULT_OLLAMA_MODEL,
    "ollama_host": DEFAULT_OLLAMA_HOST,
    "device": "cuda",
    "compute_type": "float16",
    "beam_size": "5",
    "prompt_hint": "",
    "keep_audio": False,
    "continue_on_error": True,
}

app = Flask(__name__)

JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict[str, Any]] = {}
MAX_LOG_CHARS = 120_000
MAX_STORED_JOBS = 18


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
    translation_backend = (
        str(form.get("translation_backend", DEFAULT_FORM["translation_backend"])).strip().lower()
        or DEFAULT_FORM["translation_backend"]
    )
    llm_model = str(form.get("llm_model", DEFAULT_FORM["llm_model"])).strip() or DEFAULT_FORM["llm_model"]
    ollama_host = str(form.get("ollama_host", DEFAULT_FORM["ollama_host"])).strip() or DEFAULT_FORM["ollama_host"]

    if device not in ALLOWED_DEVICES:
        device = DEFAULT_FORM["device"]
    if compute_type not in ALLOWED_COMPUTE_TYPES:
        compute_type = DEFAULT_FORM["compute_type"]
    if concept_style not in CONCEPT_STYLE_CHOICES:
        concept_style = DEFAULT_FORM["concept_style"]
    if translation_backend not in TRANSLATION_BACKEND_CHOICES:
        translation_backend = DEFAULT_FORM["translation_backend"]

    return {
        "urls_text": str(form.get("urls_text", "")).strip(),
        "model": str(form.get("model", DEFAULT_FORM["model"])).strip() or DEFAULT_FORM["model"],
        "language": str(form.get("language", DEFAULT_FORM["language"])).strip(),
        "translate_to": str(form.get("translate_to", DEFAULT_FORM["translate_to"])).strip(),
        "concept_style": concept_style,
        "translation_backend": translation_backend,
        "llm_model": llm_model,
        "ollama_host": ollama_host,
        "device": device,
        "compute_type": compute_type,
        "beam_size": beam_size,
        "prompt_hint": str(form.get("prompt_hint", DEFAULT_FORM["prompt_hint"])).strip(),
        "keep_audio": form.get("keep_audio") == "on",
        "continue_on_error": form.get("continue_on_error") == "on",
    }


def create_report(urls: list[str]) -> dict[str, Any]:
    return {
        "job_id": "",
        "status": "idle",
        "urls": urls,
        "total_urls": len(urls),
        "current_index": 0,
        "current_url": "",
        "successes": [],
        "failures": [],
        "log_text": "",
        "started_at": None,
        "finished_at": None,
    }


def copy_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": report.get("job_id", ""),
        "status": report.get("status", "idle"),
        "urls": list(report.get("urls", [])),
        "total_urls": int(report.get("total_urls", 0)),
        "current_index": int(report.get("current_index", 0)),
        "current_url": report.get("current_url", ""),
        "successes": [dict(item) for item in report.get("successes", [])],
        "failures": [dict(item) for item in report.get("failures", [])],
        "log_text": report.get("log_text", ""),
        "started_at": report.get("started_at"),
        "finished_at": report.get("finished_at"),
    }


def append_log_text(report: dict[str, Any], text: str) -> None:
    if not text:
        return
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    report["log_text"] = (report.get("log_text", "") + normalized)[-MAX_LOG_CHARS:]


def get_job_report(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        report = JOBS.get(job_id)
        return copy_report(report) if report else None


def update_job_report(job_id: str, callback) -> dict[str, Any] | None:
    with JOBS_LOCK:
        report = JOBS.get(job_id)
        if report is None:
            return None
        callback(report)
        return copy_report(report)


def prune_finished_jobs() -> None:
    with JOBS_LOCK:
        if len(JOBS) <= MAX_STORED_JOBS:
            return
        removable = sorted(
            JOBS.items(),
            key=lambda item: item[1].get("finished_at") or item[1].get("started_at") or 0,
        )
        while len(JOBS) > MAX_STORED_JOBS and removable:
            job_id, report = removable.pop(0)
            if report.get("status") == "running":
                continue
            JOBS.pop(job_id, None)


def build_cli_command(url: str, form_data: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(APP_ROOT / "youtube_subtitle_cli.py"),
        "--url",
        url,
        "--output-dir",
        str(OUTPUT_DIR),
        "--model",
        form_data["model"],
        "--device",
        form_data["device"],
        "--compute-type",
        form_data["compute_type"],
        "--beam-size",
        str(int(form_data["beam_size"])),
        "--translation-backend",
        form_data["translation_backend"],
        "--concept-style",
        form_data["concept_style"],
        "--llm-model",
        form_data["llm_model"],
        "--ollama-host",
        form_data["ollama_host"],
    ]
    if form_data["language"]:
        command += ["--language", form_data["language"]]
    if form_data["translate_to"]:
        command += ["--translate-to", form_data["translate_to"]]
    if form_data["prompt_hint"]:
        command += ["--prompt-hint", form_data["prompt_hint"]]
    if form_data["keep_audio"]:
        command.append("--keep-audio")
    return command


def run_cli_download(job_id: str, url: str, form_data: dict[str, Any]) -> dict[str, Any]:
    command = build_cli_command(url, form_data)
    process = subprocess.Popen(
        command,
        cwd=str(APP_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None
    try:
        for chunk in process.stdout:
            update_job_report(job_id, lambda report, part=chunk: append_log_text(report, part))
    finally:
        process.stdout.close()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Subtitle job exited with code {return_code}.")

    saved_job = load_saved_job(url, OUTPUT_DIR)
    if not saved_job:
        raise RuntimeError("The download finished, but no saved subtitle entry was found.")
    return saved_job


def run_background_job(job_id: str, form_data: dict[str, Any]) -> None:
    urls = parse_urls_text(form_data["urls_text"])

    def start_report(report: dict[str, Any]) -> None:
        report["status"] = "running"
        report["started_at"] = time.time()
        append_log_text(report, f"Queued {len(urls)} URL(s).\n")

    update_job_report(job_id, start_report)

    if not urls:
        def fail_empty(report: dict[str, Any]) -> None:
            report["status"] = "failed"
            report["finished_at"] = time.time()
            report["failures"].append({"url": "", "error": "Paste at least one YouTube URL."})
            append_log_text(report, "No URLs were provided.\n")

        update_job_report(job_id, fail_empty)
        return

    for index, url in enumerate(urls, start=1):
        def mark_current(report: dict[str, Any], current_index=index, current_url=url) -> None:
            report["current_index"] = current_index
            report["current_url"] = current_url
            append_log_text(report, f"\n[{current_index}/{report['total_urls']}] {current_url}\n")

        update_job_report(job_id, mark_current)

        try:
            saved_job = run_cli_download(job_id, url, form_data)
            info = saved_job["info"]
            title = info.get("title") or url
            video_id = info.get("id") or title

            def mark_success(report: dict[str, Any], item_title=title, item_url=url, item_video_id=video_id) -> None:
                report["successes"].append(
                    {
                        "url": item_url,
                        "title": item_title,
                        "video_id": item_video_id,
                    }
                )
                append_log_text(report, f"Finished: {item_title}\n")

            update_job_report(job_id, mark_success)
        except Exception as exc:
            error_text = str(exc)

            def mark_failure(report: dict[str, Any], item_url=url, item_error=error_text) -> None:
                report["failures"].append({"url": item_url, "error": item_error})
                append_log_text(report, f"ERROR: {item_error}\n")

            update_job_report(job_id, mark_failure)
            if not form_data["continue_on_error"]:
                break

    def finish_report(report: dict[str, Any]) -> None:
        report["finished_at"] = time.time()
        report["current_url"] = ""
        report["current_index"] = report["total_urls"]
        if report["failures"] and report["successes"]:
            report["status"] = "completed_with_errors"
        elif report["failures"]:
            report["status"] = "failed"
        else:
            report["status"] = "completed"

    update_job_report(job_id, finish_report)
    prune_finished_jobs()


def run_batch_download(form_data: dict[str, Any]) -> dict[str, Any]:
    urls = parse_urls_text(form_data["urls_text"])
    report = create_report(urls)
    if not urls:
        report["status"] = "failed"
        report["failures"].append({"url": "", "error": "Paste at least one YouTube URL."})
        report["log_text"] = "No URLs were provided."
        return report

    report["status"] = "running"
    report["started_at"] = time.time()
    for index, url in enumerate(urls, start=1):
        report["current_index"] = index
        report["current_url"] = url
        append_log_text(report, f"\n[{index}/{len(urls)}] {url}\n")
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
                translation_backend=form_data["translation_backend"],
                llm_model=form_data["llm_model"],
                ollama_host=form_data["ollama_host"],
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
            append_log_text(report, (logs.strip() + "\n") if logs.strip() else f"Finished: {title}\n")
        except Exception as exc:
            report["failures"].append({"url": url, "error": str(exc)})
            append_log_text(report, f"ERROR: {exc}\n")
            if not form_data["continue_on_error"]:
                break

    report["finished_at"] = time.time()
    report["current_url"] = ""
    report["current_index"] = report["total_urls"]
    if report["failures"] and report["successes"]:
        report["status"] = "completed_with_errors"
    elif report["failures"]:
        report["status"] = "failed"
    else:
        report["status"] = "completed"
    report["log_text"] = report["log_text"].strip()
    return report


@app.route("/", methods=["GET", "POST"])
def index():
    form_data = dict(DEFAULT_FORM)
    report = None

    if request.method == "POST":
        form_data = normalize_form_data(request.form)
        report = run_batch_download(form_data)
    else:
        requested_job_id = request.args.get("job_id", "").strip()
        if requested_job_id:
            report = get_job_report(requested_job_id)
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


@app.route("/api/jobs", methods=["POST"])
def create_job():
    form_data = normalize_form_data(request.form)
    urls = parse_urls_text(form_data["urls_text"])
    report = create_report(urls)
    job_id = uuid.uuid4().hex[:12]
    report["job_id"] = job_id

    with JOBS_LOCK:
        JOBS[job_id] = report

    worker = threading.Thread(target=run_background_job, args=(job_id, form_data), daemon=True)
    worker.start()
    prune_finished_jobs()

    return {
        "job_id": job_id,
        "status_url": url_for("job_status", job_id=job_id),
        "report_url": url_for("index", job_id=job_id),
    }, 202


@app.route("/api/jobs/<job_id>")
def job_status(job_id: str):
    report = get_job_report(job_id)
    if report is None:
        abort(404)
    return report


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
