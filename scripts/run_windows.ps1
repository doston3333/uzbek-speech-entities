# Start the local app on Windows (after setup_windows.ps1).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing .venv. Run .\scripts\setup_windows.ps1 first."
}
$env:PYTORCH_ENABLE_MPS_FALLBACK = "0"
& $VenvPython -m uzbek_speech_entities.api.server
