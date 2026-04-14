from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DLL_DIR_HANDLES: list[Any] = []
VIDEO_EXTENSIONS = {".webm", ".mp4", ".mkv", ".mov", ".avi", ".m4v"}
DEFAULT_GLOSSARY_PATH = Path(__file__).with_name("concept_glossary.json")
CONCEPT_STYLE_CHOICES = {"auto", "plain", "ko", "ko_en", "en_ko"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download YouTube audio with yt-dlp and generate subtitles with faster-whisper."
    )
    parser.add_argument("--url", help="YouTube URL or a local audio/video file path.")
    parser.add_argument("--output-dir", default="youtube_subs", help="Folder to store results.")
    parser.add_argument("--model", default="medium", help="Whisper model name. Example: medium, large-v3")
    parser.add_argument("--language", default="", help="Language code like ko, en. Leave empty for auto-detect.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "int8", "int8_float16", "float16", "float32"],
        help="faster-whisper compute type.",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for transcription.")
    parser.add_argument("--prompt-hint", default="", help="Optional hint to improve terminology.")
    parser.add_argument("--translate-to", default="", help="Translate transcript to target language code, e.g. ko.")
    parser.add_argument(
        "--concept-style",
        default="auto",
        choices=sorted(CONCEPT_STYLE_CHOICES),
        help="How to render glossary concepts during translation. "
        "Examples: ko_en -> 강화학습(reinforcement learning), en_ko -> reinforcement learning(강화학습).",
    )
    parser.add_argument(
        "--glossary-path",
        default="",
        help="Optional JSON glossary path. Defaults to concept_glossary.json next to this script.",
    )
    parser.add_argument("--keep-audio", action="store_true", help="Keep the downloaded audio file.")
    parser.add_argument("--self-check", action="store_true", help="Check required packages and exit.")
    return parser


def self_check() -> int:
    import ctranslate2
    import faster_whisper
    import yt_dlp

    dll_dirs = activate_cuda_dll_search_path()
    print(f"Python       : {sys.executable}")
    print(f"yt-dlp       : {getattr(yt_dlp, 'version', yt_dlp).__dict__.get('__version__', 'installed')}")
    print(f"ctranslate2  : {getattr(ctranslate2, '__version__', 'installed')}")
    print(f"faster-whisper: {getattr(faster_whisper, '__version__', 'installed')}")
    try:
        import deep_translator  # noqa: F401
        translator_state = "installed"
    except Exception:
        translator_state = "missing"
    print(f"deep-translator: {translator_state}")
    cuda_device_count = 0
    try:
        cuda_device_count = int(ctranslate2.get_cuda_device_count())
    except Exception:
        cuda_device_count = 0
    print(f"cuda-devices : {cuda_device_count}")
    print(f"cuda-ready   : {'yes' if cuda_runtime_ready() else 'no'}")
    if dll_dirs:
        print("cuda-dll-dirs:")
        for path in dll_dirs:
            print(f"  - {path}")
    return 0


def try_load_library(name: str) -> bool:
    try:
        if sys.platform == "win32":
            ctypes.WinDLL(name)
        else:
            ctypes.CDLL(name)
        return True
    except OSError:
        return False


def candidate_cuda_dll_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_keys = ("CUDA_PATH", "CUDA_HOME")
    for key in env_keys:
        value = os.environ.get(key)
        if value:
            dirs.extend([Path(value) / "bin", Path(value) / "lib" / "x64"])

    roots = []
    for raw in {sys.prefix, sys.base_prefix}:
        if raw:
            roots.append(Path(raw))

    for root in roots:
        dirs.extend(
            [
                root / "Lib" / "site-packages" / "torch" / "lib",
                root / "Library" / "bin",
            ]
        )

    seen: set[str] = set()
    out: list[Path] = []
    for path in dirs:
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if path.exists() and resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def activate_cuda_dll_search_path() -> list[Path]:
    if sys.platform != "win32":
        return []

    added: list[Path] = []
    path_entries = os.environ.get("PATH", "").split(";")

    for directory in candidate_cuda_dll_dirs():
        dir_str = str(directory)
        if dir_str not in path_entries:
            os.environ["PATH"] = dir_str + ";" + os.environ.get("PATH", "")
            path_entries.insert(0, dir_str)
        try:
            DLL_DIR_HANDLES.append(os.add_dll_directory(dir_str))
        except (FileNotFoundError, OSError, AttributeError):
            pass
        added.append(directory)

    return added


def ctranslate2_cuda_ready() -> bool:
    try:
        import ctranslate2
    except Exception:
        return False

    try:
        device_count = int(ctranslate2.get_cuda_device_count())
        if device_count > 0:
            return True
    except Exception:
        pass

    try:
        supported = ctranslate2.get_supported_compute_types("cuda")
        if supported:
            return True
    except Exception:
        pass

    return False


def cuda_runtime_ready() -> bool:
    activate_cuda_dll_search_path()
    if ctranslate2_cuda_ready():
        return True

    if not shutil.which("nvidia-smi"):
        return False

    candidates = [
        "cublas64_12.dll",
        "cublas64_11.dll",
        "libcublas.so.12",
        "libcublas.so.11",
        "libcublas.dylib",
    ]
    return any(try_load_library(name) for name in candidates)


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if cuda_runtime_ready():
        return "cuda"
    return "cpu"


def detect_compute_type(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


def is_cuda_runtime_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = [
        "cublas",
        "cudnn",
        "cuda",
        "cannot be loaded",
        "failed to load library",
        "runtime version",
    ]
    return any(marker in text for marker in markers)


def format_timestamp(seconds: float) -> str:
    millis = max(0, int(round(seconds * 1000)))
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segment_text(text: str) -> str:
    text = " ".join(text.split()).strip()
    if not text:
        return ""
    return text


def sanitize_output_name(raw: str) -> str:
    base = "".join(ch for ch in raw if ch not in '<>:"/\\|?*').strip(" .")
    return base or "youtube_subtitle"


def build_output_stem(output_dir: Path, info: dict[str, Any]) -> Path:
    slug = info.get("title") or info.get("id") or "youtube_subtitle"
    return output_dir / sanitize_output_name(str(slug))


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


def write_srt(segments: list[dict[str, Any]], path: Path) -> None:
    parts: list[str] = []
    for idx, item in enumerate(segments, 1):
        text = segment_text(item["text"])
        if not text:
            continue
        parts.append(str(idx))
        parts.append(f"{format_timestamp(item['start'])} --> {format_timestamp(item['end'])}")
        parts.append(text)
        parts.append("")
    path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")


def write_txt(segments: list[dict[str, Any]], path: Path) -> None:
    lines = [segment_text(item["text"]) for item in segments]
    text = "\n".join(line for line in lines if line).strip() + "\n"
    path.write_text(text, encoding="utf-8")


def chunk_for_translation(text: str, max_len: int = 4000) -> list[str]:
    text = segment_text(text)
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    pieces: list[str] = []
    current = ""
    for sentence in text.split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not current else current + ". " + sentence
        if len(candidate) > max_len and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [text[:max_len]]


def normalize_concept_style(target_language: str, concept_style: str) -> str:
    style = (concept_style or "auto").strip().lower()
    if style not in CONCEPT_STYLE_CHOICES:
        raise ValueError(f"Unsupported concept style: {concept_style}")
    if style == "auto":
        return "ko_en" if target_language.strip().lower().startswith("ko") else "plain"
    return style


def load_concept_glossary(glossary_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(glossary_path).resolve() if glossary_path else DEFAULT_GLOSSARY_PATH
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Glossary file must contain a JSON list: {path}")

    out: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        english = segment_text(str(item.get("en", "")))
        korean = segment_text(str(item.get("ko", "")))
        if not english or not korean:
            continue

        aliases = [
            segment_text(str(alias))
            for alias in item.get("aliases", [])
            if segment_text(str(alias))
        ]
        max_length = max(len(term) for term in [english, *aliases])
        out.append(
            {
                "en": english,
                "ko": korean,
                "aliases": aliases,
                "max_length": max_length,
            }
        )

    out.sort(key=lambda entry: entry["max_length"], reverse=True)
    return out


def render_concept_term(entry: dict[str, Any], concept_style: str, matched_surface: str | None = None) -> str:
    english_surface = segment_text(matched_surface or entry["en"])
    korean = entry["ko"]
    if concept_style == "ko":
        return korean
    if concept_style == "ko_en":
        return f"{korean}({english_surface})"
    if concept_style == "en_ko":
        return f"{english_surface}({korean})"
    return english_surface


def compile_concept_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


def protect_concept_terms(
    text: str,
    glossary: list[dict[str, Any]],
    concept_style: str,
) -> tuple[str, dict[str, str]]:
    protected = text
    replacements: dict[str, str] = {}
    counter = 0

    for entry in glossary:
        variants = [entry["en"], *entry["aliases"]]
        for variant in variants:
            pattern = compile_concept_pattern(variant)

            def replace(match: re.Match[str]) -> str:
                nonlocal counter
                token = f"YTSUBTERM{counter}TOKEN"
                replacements[token] = render_concept_term(entry, concept_style, match.group(0))
                counter += 1
                return token

            protected = pattern.sub(replace, protected)

    return protected, replacements


def restore_concept_tokens(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for token, value in replacements.items():
        restored = re.sub(re.escape(token), value, restored, flags=re.IGNORECASE)
    return restored


def build_translation_metadata(
    target_language: str,
    concept_style: str,
    glossary_path: str | Path | None = None,
) -> dict[str, str]:
    target = target_language.strip().lower()
    style = normalize_concept_style(target, concept_style)
    glossary_name = ""
    if style != "plain":
        glossary_candidate = Path(glossary_path).resolve() if glossary_path else DEFAULT_GLOSSARY_PATH
        if glossary_candidate.exists():
            glossary_name = glossary_candidate.name
    return {
        "target_language": target,
        "concept_style": style,
        "glossary_name": glossary_name,
    }


def existing_output_paths(stem: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    base_files = {
        "srt": stem.with_suffix(".srt"),
        "txt": stem.with_suffix(".txt"),
        "json": stem.with_suffix(".json"),
        "vtt": stem.with_suffix(".vtt"),
    }
    for key, path in base_files.items():
        if path.exists():
            outputs[key] = path

    for candidate in sorted(stem.parent.glob(stem.name + ".*")):
        if not candidate.is_file():
            continue
        suffixes = candidate.suffixes
        if len(suffixes) >= 2 and suffixes[-1].lower() in {".srt", ".txt", ".vtt"}:
            tag = suffixes[-2].lstrip(".").lower()
            ext = suffixes[-1].lstrip(".").lower()
            if tag:
                outputs[f"{ext}_{tag}"] = candidate

    return outputs


def load_saved_job(url: str, output_dir: str | Path) -> dict[str, Any] | None:
    output_dir = Path(output_dir).resolve()
    video_id = extract_youtube_id(url)

    for json_path in sorted(output_dir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        source = payload.get("source", {})
        if not isinstance(source, dict):
            continue

        source_id = source.get("id")
        source_url = source.get("webpage_url")
        matched = source_id == video_id if video_id else source_url == url
        if not matched:
            continue

        stem = json_path.with_suffix("")
        info = {
            "title": source.get("title"),
            "id": source.get("id"),
            "webpage_url": source.get("webpage_url"),
        }
        meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
        segments = payload.get("segments", []) if isinstance(payload.get("segments", []), list) else []
        return {
            "info": info,
            "meta": meta,
            "segments": segments,
            "stem": stem,
            "outputs": existing_output_paths(stem),
        }

    return None


def has_base_outputs(outputs: dict[str, Path]) -> bool:
    return all(key in outputs for key in ("srt", "txt", "json"))


def has_translation_outputs(outputs: dict[str, Path], target_language: str) -> bool:
    tag = target_language.strip().lower()
    return all(key in outputs for key in (f"srt_{tag}", f"txt_{tag}"))


def translation_metadata_matches(
    saved_meta: dict[str, Any],
    target_language: str,
    concept_style: str,
    glossary_path: str | Path | None = None,
) -> bool:
    requested = build_translation_metadata(target_language, concept_style, glossary_path)
    saved_translation = saved_meta.get("translation", {}) if isinstance(saved_meta.get("translation", {}), dict) else {}
    if not saved_translation:
        return False
    return (
        saved_translation.get("target_language") == requested["target_language"]
        and saved_translation.get("concept_style") == requested["concept_style"]
        and (saved_translation.get("glossary_name") or "") == requested["glossary_name"]
    )


def find_existing_downloaded_media(
    url: str,
    output_dir: str | Path,
    media_kind: str,
) -> tuple[Path, dict[str, Any]] | None:
    output_dir = Path(output_dir).resolve()
    media_dir = output_dir / media_kind
    if not media_dir.exists():
        return None

    saved_job = load_saved_job(url, output_dir)
    candidate_ids: list[str] = []
    if saved_job and saved_job["info"].get("id"):
        candidate_ids.append(str(saved_job["info"]["id"]))
    parsed_id = extract_youtube_id(url)
    if parsed_id and parsed_id not in candidate_ids:
        candidate_ids.append(parsed_id)

    for video_id in candidate_ids:
        matches = [path for path in media_dir.iterdir() if path.is_file() and f"[{video_id}]" in path.name]
        if matches:
            selected = sorted(matches)[0].resolve()
            info = saved_job["info"] if saved_job else {"title": selected.stem, "id": video_id, "webpage_url": url}
            return selected, info

    return None


def translate_segments(
    segments: list[dict[str, Any]],
    source_language: str,
    target_language: str,
    concept_style: str = "auto",
    glossary_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    try:
        from deep_translator import GoogleTranslator
    except Exception as exc:
        raise RuntimeError(
            "Translation requested but `deep-translator` is not installed. "
            "Run make_youtube_subs.ps1 once, or install it with pip."
        ) from exc

    if not target_language:
        return segments

    src = source_language or "auto"
    target = target_language.strip().lower()
    if src.lower() == target:
        return segments

    resolved_concept_style = normalize_concept_style(target, concept_style)
    glossary = load_concept_glossary(glossary_path) if resolved_concept_style != "plain" else []

    translator = GoogleTranslator(source=src, target=target)
    cache: dict[str, str] = {}
    out: list[dict[str, Any]] = []

    print(f"Translating subtitles: {src} -> {target}")
    if glossary:
        glossary_origin = Path(glossary_path).resolve() if glossary_path else DEFAULT_GLOSSARY_PATH
        print(f"Concept style    : {resolved_concept_style}")
        print(f"Concept glossary : {glossary_origin}")

    for item in segments:
        text = segment_text(item["text"])
        if not text:
            out.append(dict(item))
            continue

        if text in cache:
            translated = cache[text]
        else:
            protected_text = text
            replacements: dict[str, str] = {}
            if glossary:
                protected_text, replacements = protect_concept_terms(
                    text,
                    glossary=glossary,
                    concept_style=resolved_concept_style,
                )

            translated_parts: list[str] = []
            for chunk in chunk_for_translation(protected_text):
                try:
                    translated_piece = translator.translate(chunk)
                except Exception:
                    translated_piece = chunk
                restored_piece = restore_concept_tokens(
                    segment_text(translated_piece or chunk),
                    replacements,
                )
                translated_parts.append(restored_piece)

            translated = " ".join(part for part in translated_parts if part).strip() or text
            translated = restore_concept_tokens(translated, replacements)
            cache[text] = translated

        translated_item = dict(item)
        translated_item["text"] = translated
        out.append(translated_item)

    return out


def download_media(url: str, output_dir: str | Path, media_kind: str) -> tuple[Path, dict[str, Any], bool]:
    from yt_dlp import YoutubeDL

    output_dir = Path(output_dir).resolve()
    local_candidate = Path(url)
    if local_candidate.exists() and local_candidate.is_file():
        return local_candidate.resolve(), {"title": local_candidate.stem, "id": local_candidate.stem}, False

    existing_media = find_existing_downloaded_media(url, output_dir, media_kind)
    if existing_media is not None:
        media_path, info = existing_media
        print(f"Reusing existing {media_kind}: {media_path}")
        return media_path, info, False

    if media_kind == "audio":
        download_dir = output_dir / "audio"
        opts = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "outtmpl": str(download_dir / "%(title).180B [%(id)s].%(ext)s"),
            "windowsfilenames": True,
            "retries": 10,
            "quiet": False,
        }
    elif media_kind == "video":
        download_dir = output_dir / "video"
        opts = {
            "format": "best[ext=mp4]/best[ext=webm]/best",
            "noplaylist": True,
            "outtmpl": str(download_dir / "%(title).180B [%(id)s].%(ext)s"),
            "windowsfilenames": True,
            "retries": 10,
            "quiet": False,
        }
    else:
        raise ValueError(f"Unsupported media kind: {media_kind}")

    download_dir.mkdir(parents=True, exist_ok=True)

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    if "entries" in info:
        raise RuntimeError("Playlist URL is not supported yet. Pass a single video URL.")

    media_path: Path | None = None
    for item in info.get("requested_downloads", []) or []:
        filepath = item.get("filepath")
        if filepath and Path(filepath).exists():
            media_path = Path(filepath)
            break

    if media_path is None:
        prepared = Path(YoutubeDL(opts).prepare_filename(info))
        if prepared.exists():
            media_path = prepared

    if media_path is None:
        video_id = info.get("id") or ""
        matches = [path for path in download_dir.iterdir() if path.is_file() and f"[{video_id}]" in path.name]
        if matches:
            media_path = sorted(matches)[-1]

    if media_path is None:
        raise FileNotFoundError(f"Downloaded {media_kind} file could not be located.")

    return media_path.resolve(), info, True


def download_audio(url: str, output_dir: Path) -> tuple[Path, dict[str, Any], bool]:
    return download_media(url, output_dir, media_kind="audio")


def download_video(url: str, output_dir: Path) -> tuple[Path, dict[str, Any], bool]:
    return download_media(url, output_dir, media_kind="video")


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
    beam_size: int,
    prompt_hint: str,
    output_dir: Path,
    allow_cpu_fallback: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from faster_whisper import WhisperModel

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    activate_cuda_dll_search_path()

    print(f"Audio file       : {audio_path}")

    def run_once(run_device: str, run_compute_type: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        print(f"Using model      : {model_name}")
        print(f"Using device     : {run_device}")
        print(f"Using compute    : {run_compute_type}")

        model = WhisperModel(
            model_name,
            device=run_device,
            compute_type=run_compute_type,
            download_root=str(models_dir),
        )

        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language or None,
            beam_size=beam_size,
            vad_filter=True,
            condition_on_previous_text=True,
            initial_prompt=prompt_hint or None,
        )

        segments: list[dict[str, Any]] = []
        for seg in segments_iter:
            text = segment_text(seg.text)
            if not text:
                continue
            segments.append(
                {
                    "id": seg.id,
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                }
            )

        meta = {
            "language": getattr(info, "language", ""),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "model": model_name,
            "device": run_device,
            "compute_type": run_compute_type,
            "beam_size": beam_size,
        }
        return segments, meta

    try:
        return run_once(device, compute_type)
    except RuntimeError as exc:
        if device == "cuda" and allow_cpu_fallback and is_cuda_runtime_error(exc):
            print("")
            print("CUDA runtime was not usable on this machine. Falling back to CPU int8 mode.")
            print("If you want to skip GPU detection entirely, run with: -Device cpu")
            print("")
            return run_once("cpu", "int8")
        if device == "cuda" and not allow_cpu_fallback and is_cuda_runtime_error(exc):
            raise RuntimeError(
                "CUDA was explicitly requested, but faster-whisper could not start on GPU. "
                "Fix the CUDA/ctranslate2 runtime in this notebook environment, or switch DEVICE to 'auto'/'cpu'."
            ) from exc
        raise


def save_outputs(
    output_dir: Path,
    info: dict[str, Any],
    segments: list[dict[str, Any]],
    meta: dict[str, Any],
    translated_segments: list[dict[str, Any]] | None = None,
    translate_to: str = "",
    concept_style: str = "auto",
    glossary_path: str | Path | None = None,
    translation_metadata: dict[str, str] | None = None,
) -> dict[str, Path]:
    stem = build_output_stem(output_dir, info)
    srt_path = stem.with_suffix(".srt")
    txt_path = stem.with_suffix(".txt")
    json_path = stem.with_suffix(".json")

    write_srt(segments, srt_path)
    write_txt(segments, txt_path)
    json_meta = dict(meta)
    json_meta.pop("translation", None)
    if translation_metadata:
        json_meta["translation"] = dict(translation_metadata)
    elif translated_segments and translate_to:
        json_meta["translation"] = build_translation_metadata(
            translate_to,
            concept_style=concept_style,
            glossary_path=glossary_path,
        )

    json_payload = {
        "source": {
            "title": info.get("title"),
            "id": info.get("id"),
            "webpage_url": info.get("webpage_url"),
        },
        "meta": json_meta,
        "segments": segments,
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    outputs: dict[str, Path] = {"srt": srt_path, "txt": txt_path, "json": json_path}

    if translated_segments and translate_to:
        translated_tag = translate_to.lower()
        srt_translated = stem.with_suffix(f".{translated_tag}.srt")
        txt_translated = stem.with_suffix(f".{translated_tag}.txt")
        write_srt(translated_segments, srt_translated)
        write_txt(translated_segments, txt_translated)
        outputs[f"srt_{translated_tag}"] = srt_translated
        outputs[f"txt_{translated_tag}"] = txt_translated

    return outputs


def run_job(
    url: str,
    output_dir: str | Path = "youtube_subs",
    model: str = "medium",
    language: str = "",
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    prompt_hint: str = "",
    translate_to: str = "",
    concept_style: str = "auto",
    glossary_path: str | Path | None = None,
    keep_audio: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_job = load_saved_job(url, output_dir)

    if saved_job and has_base_outputs(saved_job["outputs"]):
        if not translate_to:
            print(f"Reusing existing subtitles for URL: {url}")
            return {
                "info": saved_job["info"],
                "meta": saved_job["meta"],
                "outputs": saved_job["outputs"],
                "output_dir": output_dir,
            }

        if has_translation_outputs(saved_job["outputs"], translate_to):
            requested_translation = build_translation_metadata(
                translate_to,
                concept_style,
                glossary_path,
            )
            if translation_metadata_matches(
                saved_job["meta"],
                translate_to,
                concept_style,
                glossary_path,
            ):
                print(f"Reusing existing translated subtitles for URL: {url}")
                return {
                    "info": saved_job["info"],
                    "meta": saved_job["meta"],
                    "outputs": saved_job["outputs"],
                    "output_dir": output_dir,
                }

            saved_translation = (
                saved_job["meta"].get("translation", {})
                if isinstance(saved_job["meta"].get("translation", {}), dict)
                else {}
            )
            if not saved_translation:
                print(f"Reusing existing translated subtitles for URL: {url}")
                refreshed_job = None
                outputs = saved_job["outputs"]
                if saved_job["segments"]:
                    print("Backfilling missing translation metadata for future runs.")
                    outputs = save_outputs(
                        output_dir,
                        saved_job["info"],
                        saved_job["segments"],
                        saved_job["meta"],
                        translate_to=translate_to,
                        concept_style=concept_style,
                        glossary_path=glossary_path,
                        translation_metadata=requested_translation,
                    )
                    refreshed_job = load_saved_job(url, output_dir)
                return {
                    "info": saved_job["info"],
                    "meta": refreshed_job["meta"] if refreshed_job else {**saved_job["meta"], "translation": requested_translation},
                    "outputs": refreshed_job["outputs"] if refreshed_job else outputs,
                    "output_dir": output_dir,
                }

    resolved_device = detect_device(device)
    resolved_compute_type = detect_compute_type(resolved_device, compute_type)
    allow_cpu_fallback = device == "auto"

    if saved_job and saved_job["segments"]:
        print(f"Reusing saved transcript for URL: {url}")
        translated_segments = None
        if translate_to:
            print("Generating only the missing translation outputs.")
            translated_segments = translate_segments(
                saved_job["segments"],
                source_language=saved_job["meta"].get("language", "") or language,
                target_language=translate_to,
                concept_style=concept_style,
                glossary_path=glossary_path,
            )

        outputs = save_outputs(
            output_dir,
            saved_job["info"],
            saved_job["segments"],
            saved_job["meta"],
            translated_segments=translated_segments,
            translate_to=translate_to,
            concept_style=concept_style,
            glossary_path=glossary_path,
        )
        refreshed_job = load_saved_job(url, output_dir)
        return {
            "info": saved_job["info"],
            "meta": refreshed_job["meta"] if refreshed_job else saved_job["meta"],
            "outputs": refreshed_job["outputs"] if refreshed_job else outputs,
            "output_dir": output_dir,
        }

    downloaded_audio = False
    audio_path: Path | None = None
    try:
        audio_path, info, downloaded_audio = download_audio(url, output_dir)
        segments, meta = transcribe_audio(
            audio_path=audio_path,
            model_name=model,
            language=language,
            device=resolved_device,
            compute_type=resolved_compute_type,
            beam_size=beam_size,
            prompt_hint=prompt_hint,
            output_dir=output_dir,
            allow_cpu_fallback=allow_cpu_fallback,
        )

        translated_segments = None
        if translate_to:
            translated_segments = translate_segments(
                segments,
                source_language=meta.get("language", "") or language,
                target_language=translate_to,
                concept_style=concept_style,
                glossary_path=glossary_path,
            )

        outputs = save_outputs(
            output_dir,
            info,
            segments,
            meta,
            translated_segments=translated_segments,
            translate_to=translate_to,
            concept_style=concept_style,
            glossary_path=glossary_path,
        )
        return {
            "info": info,
            "meta": meta,
            "outputs": outputs,
            "output_dir": output_dir,
        }
    finally:
        if audio_path and downloaded_audio and not keep_audio and audio_path.exists():
            try:
                audio_path.unlink()
                print(f"Removed audio file: {audio_path}")
            except OSError:
                pass


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.self_check:
        return self_check()

    if not args.url:
        parser.error("--url is required unless --self-check is used.")

    result = run_job(
        url=args.url,
        output_dir=args.output_dir,
        model=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        prompt_hint=args.prompt_hint,
        translate_to=args.translate_to,
        concept_style=args.concept_style,
        glossary_path=args.glossary_path,
        keep_audio=args.keep_audio,
    )
    print("")
    print("Finished.")
    for name, path in result["outputs"].items():
        print(f"{name.upper():4s}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
