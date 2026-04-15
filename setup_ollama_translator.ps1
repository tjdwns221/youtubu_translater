[CmdletBinding()]
param(
    [string]$Model = "translategemma:4b",
    [string]$OllamaHost = "http://127.0.0.1:11434",
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-OllamaCommand {
    try {
        return (Get-Command ollama -ErrorAction Stop).Source
    }
    catch {
        return $null
    }
}

function Install-Ollama {
    Write-Host "Installing Ollama from the official Windows installer script..."
    $installScript = Invoke-RestMethod -Uri "https://ollama.com/install.ps1"
    Invoke-Expression $installScript
}

function Wait-OllamaApi {
    param(
    [string]$ApiHost,
        [int]$TimeoutSeconds = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-RestMethod -Uri ($ApiHost.TrimEnd("/") + "/api/version") -Method Get | Out-Null
            return
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }

    throw "Ollama API did not become ready at $ApiHost within $TimeoutSeconds seconds."
}

$ollamaExe = Get-OllamaCommand
if (-not $ollamaExe) {
    if ($SkipInstall) {
        throw "Ollama is not installed and -SkipInstall was used."
    }
    Install-Ollama
    $ollamaExe = Get-OllamaCommand
}

if (-not $ollamaExe) {
    throw "Ollama installation finished but the `ollama` command was still not found."
}

try {
    Invoke-RestMethod -Uri ($OllamaHost.TrimEnd("/") + "/api/version") -Method Get | Out-Null
}
catch {
    Write-Host "Starting the local Ollama server..."
    Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden | Out-Null
}

Wait-OllamaApi -ApiHost $OllamaHost

Write-Host "Pulling model: $Model"
& $ollamaExe pull $Model

Write-Host ""
Write-Host "Ollama translation setup is ready."
Write-Host "Model : $Model"
Write-Host "Host  : $OllamaHost"
