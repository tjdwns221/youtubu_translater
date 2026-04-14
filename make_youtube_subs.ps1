[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Url,

    [string]$OutputDir = "",

    [string]$Model = "medium",

    [string]$Language = "",

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",

    [ValidateSet("auto", "int8", "int8_float16", "float16", "float32")]
    [string]$ComputeType = "auto",

    [int]$BeamSize = 5,

    [string]$PromptHint = "",

    [string]$TranslateTo = "",

    [switch]$KeepAudio,

    [switch]$SetupOnly,

    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $scriptRoot "youtube_subs"
}

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

    throw "Python을 찾지 못했습니다. Python 3가 설치되어 있고 PATH에 잡혀 있어야 합니다."
}

function Ensure-Venv {
    param(
        [string]$BasePython,
        [string]$VenvDir
    )

    $venvPython = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Host "가상환경을 만듭니다: $VenvDir"
        & $BasePython -m venv $VenvDir
    }

    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw "가상환경 생성에 실패했습니다: $venvPython"
    }

    return $venvPython
}

function Test-Dependencies {
    param([string]$PythonExe)

    $check = "import importlib.util, sys; mods = ('yt_dlp', 'faster_whisper', 'deep_translator'); missing = [m for m in mods if importlib.util.find_spec(m) is None]; sys.exit(0 if not missing else 1)"
    & $PythonExe -c $check
    return $LASTEXITCODE -eq 0
}

function Install-Dependencies {
    param([string]$PythonExe)

    Write-Host "필요 패키지를 설치합니다. 첫 실행은 몇 분 걸릴 수 있습니다."
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install yt-dlp faster-whisper deep-translator
}

$basePython = Get-PythonCommand
$venvDir = Join-Path $scriptRoot ".youtube_subs_env"
$venvPython = Ensure-Venv -BasePython $basePython -VenvDir $venvDir
$cliPath = Join-Path $scriptRoot "youtube_subtitle_cli.py"

if (-not (Test-Path -LiteralPath $cliPath)) {
    throw "헬퍼 스크립트를 찾을 수 없습니다: $cliPath"
}

if (-not $SkipInstall) {
    if (-not (Test-Dependencies -PythonExe $venvPython)) {
        Install-Dependencies -PythonExe $venvPython
    }
}

if ($SetupOnly) {
    & $venvPython $cliPath --self-check
    exit $LASTEXITCODE
}

if ([string]::IsNullOrWhiteSpace($Url)) {
    throw "URL이 필요합니다. 예: .\make_youtube_subs.ps1 'https://www.youtube.com/watch?v=...'"
}

$arguments = @(
    $cliPath
    "--url", $Url
    "--output-dir", $OutputDir
    "--model", $Model
    "--device", $Device
    "--compute-type", $ComputeType
    "--beam-size", $BeamSize
)

if (-not [string]::IsNullOrWhiteSpace($Language)) {
    $arguments += @("--language", $Language)
}

if (-not [string]::IsNullOrWhiteSpace($PromptHint)) {
    $arguments += @("--prompt-hint", $PromptHint)
}

if (-not [string]::IsNullOrWhiteSpace($TranslateTo)) {
    $arguments += @("--translate-to", $TranslateTo)
}

if ($KeepAudio) {
    $arguments += "--keep-audio"
}

& $venvPython @arguments
exit $LASTEXITCODE
