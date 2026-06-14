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

# ── Pre-flight: bundled binaries must be present ───────────────────────────
#
# GenSRT bundles ffmpeg.exe + ffprobe.exe in gensrt\bin\ so end users don't
# need to install ffmpeg.  If they're missing, the installer would ship
# broken — better to fail here than at user launch time.

Write-Host ""
Write-Host "Checking bundled binaries..." -ForegroundColor Yellow
$missingBinaries = @()
foreach ($binary in @("ffmpeg.exe", "ffprobe.exe")) {
    $path = ".\gensrt\bin\$binary"
    if (-not (Test-Path $path)) {
        $missingBinaries += $binary
    } else {
        $size = [math]::Round((Get-Item $path).Length / 1MB, 1)
        Write-Host "  - gensrt\bin\$binary ($size MB)" -ForegroundColor Gray
    }
}
if ($missingBinaries.Count -gt 0) {
    Write-Host ""
    Write-Host "ERROR: Missing bundled binaries in gensrt\bin\:" -ForegroundColor Red
    foreach ($b in $missingBinaries) { Write-Host "  - $b" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Download ffmpeg-release-essentials.7z from:" -ForegroundColor Yellow
    Write-Host "  https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor Gray
    Write-Host "and copy ffmpeg.exe and ffprobe.exe from its bin\ folder" -ForegroundColor Yellow
    Write-Host "into gensrt\bin\.  Then re-run this script." -ForegroundColor Yellow
    exit 1
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
    --add-data "gensrt\bin;gensrt\bin" `
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

# User Guide — self-contained HTML (embedded CSS + base64 images).
# Opens in any browser; no Word, no internet, no plugins required.
if (Test-Path ".\user_guide.html") {
    Copy-Item ".\user_guide.html" ".\dist\gensrt\user_guide.html" -Force
    $ugSize = [math]::Round((Get-Item ".\user_guide.html").Length / 1KB, 1)
    Write-Host "  - user_guide.html copied (${ugSize} KB)" -ForegroundColor Gray
} else {
    Write-Host "  - user_guide.html not found" -ForegroundColor Yellow
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
Write-Host "  - FFmpeg is bundled (gensrt\bin\ffmpeg.exe + ffprobe.exe);" -ForegroundColor Gray
Write-Host "    no separate install needed on target machines." -ForegroundColor Gray
Write-Host ""
