# Windows bootstrap: venv, Python deps, FFmpeg check, NER + Whisper downloads.
# Run from repo root in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Resolve-Python {
    if ($env:PYTHON) { return $env:PYTHON }
    foreach ($candidate in @("py -3.11", "py -3.12", "python3.11", "python3.12", "python")) {
        try {
            if ($candidate -like "py *") {
                $parts = $candidate.Split(" ")
                & $parts[0] $parts[1] -c "import sys; assert sys.version_info[:2] in {(3,11),(3,12)}" 2>$null
                if ($LASTEXITCODE -eq 0) { return $candidate }
            } else {
                & $candidate -c "import sys; assert sys.version_info[:2] in {(3,11),(3,12)}" 2>$null
                if ($LASTEXITCODE -eq 0) { return $candidate }
            }
        } catch { }
    }
    throw "Python 3.11 or 3.12 is required. Install from https://www.python.org/downloads/ and re-run."
}

$Python = Resolve-Python
Write-Host "Using Python launcher: $Python"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtualenv..."
    if ($Python -like "py *") {
        $parts = $Python.Split(" ")
        & $parts[0] $parts[1] -m venv .venv
    } else {
        & $Python -m venv .venv
    }
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt
& $VenvPython -m pip install --no-deps -e .

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host "Installing FFmpeg via winget..."
        winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
    } else {
        Write-Warning "FFmpeg not found. Install with: winget install --id Gyan.FFmpeg -e"
        Write-Warning "Or: choco install ffmpeg"
    }
}

& $VenvPython -m uzbek_speech_entities.setup
Write-Host ""
Write-Host "Done. Start the app with: .\scripts\run_windows.ps1"
Write-Host "Open http://127.0.0.1:8000"
