[CmdletBinding()]
param(
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8765,
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-PythonCommand {
    foreach ($name in @("python", "python3")) {
        try {
            $cmd = Get-Command $name -ErrorAction Stop
            if ($cmd.Source) {
                return $cmd.Source
            }
        }
        catch {
        }
    }

    throw "Python 3 executable was not found on PATH."
}

function Ensure-Venv {
    param(
        [string]$BasePython,
        [string]$VenvDir
    )

    $venvPython = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Host "Creating virtual environment at $VenvDir"
        & $BasePython -m venv $VenvDir
    }

    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw "Virtual environment setup failed: $venvPython"
    }

    return $venvPython
}

function Test-Dependencies {
    param([string]$PythonExe)

    $check = "import importlib.util, sys; mods = ('yt_dlp', 'faster_whisper', 'deep_translator', 'flask'); missing = [m for m in mods if importlib.util.find_spec(m) is None]; sys.exit(0 if not missing else 1)"
    & $PythonExe -c $check
    return $LASTEXITCODE -eq 0
}

function Install-Dependencies {
    param([string]$PythonExe)

    Write-Host "Installing web UI dependencies..."
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install yt-dlp faster-whisper deep-translator flask
}

$basePython = Get-PythonCommand
$venvDir = Join-Path $scriptRoot ".youtube_subs_env"
$venvPython = Ensure-Venv -BasePython $basePython -VenvDir $venvDir
$webAppPath = Join-Path $scriptRoot "web_app.py"

if (-not (Test-Path -LiteralPath $webAppPath)) {
    throw "web_app.py was not found: $webAppPath"
}

if (-not $SkipInstall) {
    if (-not (Test-Dependencies -PythonExe $venvPython)) {
        Install-Dependencies -PythonExe $venvPython
    }
}

$env:YTSUB_WEB_HOST = $ListenHost
$env:YTSUB_WEB_PORT = "$Port"

Write-Host "Starting web app at http://$ListenHost`:$Port"
& $venvPython $webAppPath
