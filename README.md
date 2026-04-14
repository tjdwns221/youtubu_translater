# youtubu_translater
영어강의 들어야 되는데 영어를 못하는 사람을 위해 만든 toy-project codex랑 만들었음
A small Codex-assisted toy project for downloading YouTube lecture media, generating subtitles with `faster-whisper`, and previewing the result inside Jupyter.

## Files

- `down.ipynb`: process one YouTube URL and create subtitle files.
- `play.ipynb`: open a saved lecture video with subtitles inside Jupyter.
- `down_seriese.ipynb`: process multiple YouTube URLs in one batch.
- `web_app.py`: local Flask web app for browser-based download and playback.
- `run_web_app.ps1`: launch the browser UI with dependency setup.
- `start_web_ui.bat`: double-click launcher for Windows.
- `youtube_subtitle_cli.py`: shared download / transcription logic.
- `jupyter_video_helper.py`: notebook helpers for playback and subtitle matching.
- `make_youtube_subs.ps1`: PowerShell entry point for setting up dependencies and running the CLI.

## Basic workflow

1. Open `down.ipynb` for one lecture or `down_seriese.ipynb` for a batch.
2. Replace the URL value with your target YouTube link or links.
3. Run all cells.
4. Open `play.ipynb` to preview a lecture with subtitles.

## Browser UI

1. Run `start_web_ui.bat` or `.\run_web_app.ps1`.
2. Open the local address shown in the terminal.
3. Paste one or many URLs in the browser and run the batch.
4. Open saved lectures from the library section.

## Notes

- Default notebook settings use `cuda` and `float16` so GPU inference is preferred.
- Generated outputs are written under `youtube_subs/`.
- Large generated assets and local virtual environments are ignored by Git.
