# GenSRT Packaging Script
# Builds a standalone executable package with PyInstaller
# Run from the GenSRT project root with the venv activated:
#   .\Pack-gensrt.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GenSRT Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Sanity checks ──────────────────────────────────────────────────────────

# Must be run from project root
if (-not (Test-Path ".\gensrt\__main__.py")) {
    Write-Host "ERROR: Run this script from the GenSRT project root." -ForegroundColor Red
    Write-Host "       Expected to find .\gensrt\__main__.py" -ForegroundColor Red
    exit 1
}

# Check PyInstaller
Write-Host "Checking PyInstaller..." -ForegroundColor Yellow
$pyinstallerCheck = python -m pip show pyinstaller 2>$null
if (-not $pyinstallerCheck) {
    Write-Host "PyInstaller not found. Installing..." -ForegroundColor Yellow
    python -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install PyInstaller" -ForegroundColor Red
        exit 1
    }
    Write-Host "PyInstaller installed successfully" -ForegroundColor Green
} else {
    Write-Host "PyInstaller found." -ForegroundColor Green
}

# ── Clean previous builds ──────────────────────────────────────────────────

Write-Host ""
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
foreach ($path in @(".\build", ".\dist", ".\gensrt.spec")) {
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
        Write-Host "  - Removed $path" -ForegroundColor Gray
    }
}

# ── Build ──────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "Building executable..." -ForegroundColor Yellow
Write-Host "(This will take several minutes — torch + ctranslate2 are large)" -ForegroundColor Gray
Write-Host ""

pyinstaller --noconfirm --clean --onedir --name gensrt `
    `
    --add-data "gensrt\static;gensrt\static" `
    --add-data "gensrt\templates;gensrt\templates" `
    `
    --hidden-import=flask `
    --hidden-import=flask.json `
    --hidden-import=werkzeug `
    --hidden-import=werkzeug.serving `
    --hidden-import=jinja2 `
    --hidden-import=webview `
    --hidden-import=webview.platforms.winforms `
    --hidden-import=clr `
    --hidden-import=ctranslate2 `
    --hidden-import=faster_whisper `
    --hidden-import=tokenizers `
    --hidden-import=huggingface_hub `
    --hidden-import=requests `
    --hidden-import=srt `
    --hidden-import=ffmpeg `
    `
    --collect-all ctranslate2 `
    --collect-all faster_whisper `
    --collect-all tokenizers `
    --collect-all nvidia.cuda_runtime `
    --collect-all nvidia.cublas `
    --collect-all nvidia.cudnn `
    `
    --exclude-module silero_vad `
    --exclude-module torchaudio `
    --exclude-module torchvision `
    --exclude-module matplotlib `
    --exclude-module notebook `
    --exclude-module IPython `
    --exclude-module PIL `
    `
    gensrt\__main__.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: PyInstaller build failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Build completed successfully" -ForegroundColor Green

# ── Copy supporting files ──────────────────────────────────────────────────

Write-Host ""
Write-Host "Copying supporting files..." -ForegroundColor Yellow

# Default config (optional — user can regenerate with --init-config)
if (Test-Path ".\gensrt-config.json") {
    Copy-Item ".\gensrt-config.json" ".\dist\gensrt\gensrt-config.json" -Force
    Write-Host "  - gensrt-config.json copied" -ForegroundColor Gray
} else {
    Write-Host "  - gensrt-config.json not found, skipping (user can run --init-config)" -ForegroundColor Yellow
}

if (Test-Path ".\README.md") {
    Copy-Item ".\README.md" ".\dist\gensrt\README.md" -Force
    Write-Host "  - README.md copied" -ForegroundColor Gray
}

# ── Size report ────────────────────────────────────────────────────────────

Write-Host ""
$distSize = [math]::Round((Get-ChildItem ".\dist\gensrt" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
Write-Host "Distribution size: ${distSize} MB" -ForegroundColor White

# ── Self-extracting installer (optional, requires 7-Zip) ──────────────────

Write-Host ""
Write-Host "Creating self-extracting installer..." -ForegroundColor Yellow
$7zCheck = Get-Command 7z -ErrorAction SilentlyContinue
if ($7zCheck) {
    $installerName = "gensrt-install.exe"
    7z a -sfx "$installerName" ".\dist\gensrt"

    if ($LASTEXITCODE -eq 0) {
        $size = [math]::Round((Get-Item $installerName).Length / 1MB, 1)
        Write-Host "Self-extracting installer created: $installerName (${size} MB)" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Failed to create self-extracting installer" -ForegroundColor Yellow
    }
} else {
    Write-Host "7z not found - skipping self-extracting installer" -ForegroundColor Yellow
    Write-Host "Install 7-Zip to enable: https://www.7-zip.org/" -ForegroundColor Gray
}

# ── Summary ────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Build Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output location: .\dist\gensrt\" -ForegroundColor White
Write-Host ""
Write-Host "To test the build:" -ForegroundColor Yellow
Write-Host "  .\dist\gensrt\gensrt.exe --version" -ForegroundColor Gray
Write-Host "  .\dist\gensrt\gensrt.exe --input video.mkv" -ForegroundColor Gray
Write-Host ""
Write-Host "Notes:" -ForegroundColor Yellow
Write-Host "  - Whisper model (~800MB) downloads on first run to:" -ForegroundColor Gray
Write-Host "    %USERPROFILE%\.cache\huggingface\hub" -ForegroundColor Gray
Write-Host "  - CUDA 12.8 runtime must be installed on the target machine" -ForegroundColor Gray
Write-Host "  - FFmpeg must be on PATH on the target machine" -ForegroundColor Gray
Write-Host ""
