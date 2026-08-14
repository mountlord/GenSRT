# GenSRT Packaging Script
# Builds a standalone executable package with PyInstaller.
#
# Run from the GenSRT project root with the venv activated:
#
#   .\Pack-gensrt.ps1                    # CUDA build (default)
#   .\Pack-gensrt.ps1 -Variant cpu       # CPU-only build — much smaller
#
# ─────────────────────────────────────────────────────────────────────────
# Why there are two variants
# ─────────────────────────────────────────────────────────────────────────
# GenSRT no longer bundles PyTorch (see gensrt/gpu_probe.py).  That removes
# the single largest component of the distribution, and it also means the
# CUDA libraries are now an explicit, separable choice rather than something
# torch dragged in.
#
# Two consequences worth building around:
#
#   * The CPU-only build needs no CUDA libraries at all.  For a user on a
#     machine without an NVIDIA GPU, downloading ~1.2 GB of cuBLAS and cuDNN
#     is pure waste — and in a lot of the world, bandwidth is metered and
#     expensive.  Those users should be able to download only what runs.
#
#   * The CUDA build still needs cuBLAS + cuDNN, but not torch, so it lands
#     at roughly a third of the previous size.
#
# Ship both.  Point CPU-only users (and anyone on a slow link) at the CPU
# installer, and note in the release that it is the one to take if in doubt:
# GenSRT falls back to CPU cleanly, so a CPU build on a GPU machine works,
# just slowly.
# ─────────────────────────────────────────────────────────────────────────

[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Variant = "cuda"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GenSRT Build Script" -ForegroundColor Cyan
Write-Host "  Variant: $Variant" -ForegroundColor Cyan
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

# ── Variant pre-flight ─────────────────────────────────────────────────────
#
# A CUDA build needs the nvidia-* wheels present in the venv, or
# --collect-all fails partway through a long build.  Check up front.

if ($Variant -eq "cuda") {
    Write-Host ""
    Write-Host "Checking CUDA libraries..." -ForegroundColor Yellow
    $missingCuda = @()
    foreach ($pkg in @("nvidia-cublas-cu12", "nvidia-cudnn-cu12",
                       "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12")) {
        $found = python -m pip show $pkg 2>$null
        if (-not $found) { $missingCuda += $pkg } else { Write-Host "  - $pkg" -ForegroundColor Gray }
    }
    if ($missingCuda.Count -gt 0) {
        Write-Host ""
        Write-Host "ERROR: CUDA build requested but these are not installed:" -ForegroundColor Red
        foreach ($p in $missingCuda) { Write-Host "  - $p" -ForegroundColor Red }
        Write-Host ""
        Write-Host "Run:  pip install -r requirements-cuda.txt" -ForegroundColor Yellow
        Write-Host "Or build the CPU variant:  .\Pack-gensrt.ps1 -Variant cpu" -ForegroundColor Yellow
        exit 1
    }
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

# ── Assemble PyInstaller arguments ─────────────────────────────────────────

$pyiArgs = @(
    "--noconfirm", "--clean", "--onedir", "--name", "gensrt",

    "--add-data", "gensrt\static;gensrt\static",
    "--add-data", "gensrt\templates;gensrt\templates",
    "--add-data", "gensrt\bin;gensrt\bin",

    "--hidden-import=flask",
    "--hidden-import=flask.json",
    "--hidden-import=werkzeug",
    "--hidden-import=werkzeug.serving",
    "--hidden-import=jinja2",
    "--hidden-import=webview",
    "--hidden-import=webview.platforms.winforms",
    "--hidden-import=clr",
    "--hidden-import=ctranslate2",
    "--hidden-import=faster_whisper",
    "--hidden-import=tokenizers",
    "--hidden-import=huggingface_hub",
    # hf_xet gives much faster model downloads from the Hub.  It is a
    # dependency of huggingface_hub but is imported dynamically, so
    # PyInstaller does not find it and the packaged build silently falls back
    # to plain HTTP on first run.
    "--hidden-import=hf_xet",
    "--collect-all", "hf_xet",
    "--hidden-import=onnxruntime",
    "--hidden-import=requests",
    "--hidden-import=srt",
    "--hidden-import=ffmpeg",

    # GenSRT imports its own engines, translation backends and helpers inside
    # functions, deliberately, to keep CLI startup fast.  PyInstaller's static
    # analysis does not reliably follow that, and a single missed submodule
    # kills the run at first use.  Collect the whole package rather than
    # relying on the graph walk.
    "--collect-submodules", "gensrt",

    "--collect-all", "ctranslate2",
    "--collect-all", "faster_whisper",
    "--collect-all", "tokenizers",
    # certifi ships the CA bundle requests and huggingface_hub verify TLS
    # against. PyInstaller's requests hook normally collects it, but every
    # model download depends on it, so it is named explicitly rather than
    # relied upon.
    "--collect-all", "certifi",

    # Nothing in GenSRT imports torch or transformers as of v1.2.5 — the
    # offline translation engines that needed them were removed. Excluded
    # unconditionally so a venv that happens to have them (from unrelated
    # work) cannot silently add gigabytes to the installer.
    "--exclude-module", "torch",
    "--exclude-module", "torchvision",
    "--exclude-module", "torchaudio",
    "--exclude-module", "transformers",
    "--exclude-module", "accelerate",

    "--exclude-module", "matplotlib",
    "--exclude-module", "notebook",
    "--exclude-module", "IPython",
    "--exclude-module", "PIL"
)

if ($Variant -eq "cuda") {
    # CUDA DLLs are added individually rather than with --collect-all, for
    # two reasons.
    #
    # 1. --collect-all preserves the wheel layout, putting the DLLs under
    #    _internal\nvidia\<component>\bin\.  Nothing places those directories
    #    on the Windows DLL search path — the nvidia-*-cu12 wheels ship no
    #    __init__.py and no .pth hook, and ctranslate2 registers only its own
    #    package directory.  A dev box works because a CUDA Toolkit install,
    #    a stale PATH entry, or another application (PyTorch, usually) has
    #    already made the libraries findable.  A clean machine with only the
    #    display driver has none of that, and every inference call fails with
    #    "Library cublas64_12.dll is not found or cannot be loaded".
    #    Placing the DLLs flat in _internal\ puts them where the bootloader
    #    already searches.  gensrt/_cuda_dlls.py covers this at runtime too.
    #
    # 2. The wheels are mostly not runtime files.  include\ holds C headers
    #    and lib\ holds import libraries — needed to *compile* against CUDA,
    #    useless to ship.  Taking bin\*.dll only cuts a large amount of dead
    #    weight from the installer.
    Write-Host ""
    Write-Host "Collecting CUDA runtime DLLs..." -ForegroundColor Yellow

    $sitePackages = python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    $nvidiaRoot = Join-Path $sitePackages "nvidia"

    if (-not (Test-Path $nvidiaRoot)) {
        Write-Host "ERROR: $nvidiaRoot not found." -ForegroundColor Red
        Write-Host "Run:  pip install -r requirements-cuda.txt" -ForegroundColor Yellow
        exit 1
    }

    $cudaDlls = Get-ChildItem -Path $nvidiaRoot -Filter "*.dll" -Recurse |
                Where-Object { $_.DirectoryName -like "*\bin" }

    if ($cudaDlls.Count -eq 0) {
        Write-Host "ERROR: No DLLs found under $nvidiaRoot\*\bin\" -ForegroundColor Red
        exit 1
    }

    # cublas64_12.dll is the one that failed on a clean machine, so its
    # presence is asserted by name rather than trusted to the glob.
    $required = @("cublas64_12.dll", "cudart64_12.dll")
    foreach ($req in $required) {
        if (-not ($cudaDlls | Where-Object { $_.Name -eq $req })) {
            Write-Host "ERROR: required $req not found under $nvidiaRoot" -ForegroundColor Red
            Write-Host "Run:  pip install -r requirements-cuda.txt" -ForegroundColor Yellow
            exit 1
        }
    }
    if (-not ($cudaDlls | Where-Object { $_.Name -like "cudnn*" })) {
        Write-Host "ERROR: no cuDNN DLLs found under $nvidiaRoot" -ForegroundColor Red
        exit 1
    }

    $cudaMB = [math]::Round(($cudaDlls | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
    Write-Host "  - $($cudaDlls.Count) DLL(s), ${cudaMB} MB" -ForegroundColor Gray

    foreach ($dll in $cudaDlls) {
        # ";." places the file at the root of _internal\, which the
        # PyInstaller bootloader adds to the DLL search path.
        $pyiArgs += @("--add-binary", "$($dll.FullName);.")
    }
} else {
    # CPU build: make sure no stray CUDA payload sneaks in from a venv that
    # happens to have the wheels installed.  CTranslate2's own DLL loads its
    # CUDA dependencies lazily, so a CPU build runs fine without them.
    $pyiArgs += @(
        "--exclude-module", "nvidia"
    )
}

$pyiArgs += "gensrt\__main__.py"

# ── Build ──────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "Building executable..." -ForegroundColor Yellow
if ($Variant -eq "cuda") {
    Write-Host "(This will take a few minutes — the cuDNN payload is large)" -ForegroundColor Gray
} else {
    Write-Host "(CPU build — this should be quick)" -ForegroundColor Gray
}
Write-Host ""

pyinstaller @pyiArgs

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

if (Test-Path ".\gensrt-config.json") {
    Copy-Item ".\gensrt-config.json" ".\dist\gensrt\gensrt-config.json" -Force
    Write-Host "  - gensrt-config.json copied" -ForegroundColor Gray
} else {
    Write-Host "  - gensrt-config.json not found, skipping (user can run --init-config)" -ForegroundColor Yellow
}

# CPU builds ship a config that already says device=cpu, so the app never
# even probes for a GPU it cannot use.
if ($Variant -eq "cpu" -and (Test-Path ".\dist\gensrt\gensrt-config.json")) {
    $cfg = Get-Content ".\dist\gensrt\gensrt-config.json" -Raw | ConvertFrom-Json
    $cfg.device = "cpu"
    $cfg.backend = "cpu"
    $cfg.compute_type = "int8"
    $cfg | ConvertTo-Json -Depth 5 | Set-Content ".\dist\gensrt\gensrt-config.json" -Encoding UTF8
    Write-Host "  - gensrt-config.json set to CPU mode" -ForegroundColor Gray
}

if (Test-Path ".\README.md") {
    Copy-Item ".\README.md" ".\dist\gensrt\README.md" -Force
    Write-Host "  - README.md copied" -ForegroundColor Gray
}

if (Test-Path ".\LICENSE") {
    Copy-Item ".\LICENSE" ".\dist\gensrt\LICENSE" -Force
    Write-Host "  - LICENSE copied" -ForegroundColor Gray
}

# User Guide — self-contained HTML (embedded CSS + base64 images).
if (Test-Path ".\user_guide.html") {
    Copy-Item ".\user_guide.html" ".\dist\gensrt\user_guide.html" -Force
    $ugSize = [math]::Round((Get-Item ".\user_guide.html").Length / 1KB, 1)
    Write-Host "  - user_guide.html copied (${ugSize} KB)" -ForegroundColor Gray
} else {
    Write-Host "  - user_guide.html not found" -ForegroundColor Yellow
}

# ── Models directory ───────────────────────────────────────────────────────
#
# Ship the convention so the folder is visible on extraction rather than only
# described in a message. GenSRT also creates it at startup, so this is
# belt-and-braces.

Write-Host ""
Write-Host "Creating models directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path ".\dist\gensrt\models" | Out-Null
Write-Host "  - dist\gensrt\models\" -ForegroundColor Gray

# ── Size report ────────────────────────────────────────────────────────────

# ── Post-build verification ────────────────────────────────────────────────
#
# A CUDA build that is missing cublas64_12.dll looks completely healthy: it
# starts, detects the GPU, prints "Device : cuda (cuda)", loads the model, and
# only then fails on every chunk.  Checking the file is present here is far
# cheaper than discovering it on a user's machine.

if ($Variant -eq "cuda") {
    Write-Host ""
    Write-Host "Verifying CUDA payload..." -ForegroundColor Yellow
    $internal = ".\dist\gensrt\_internal"
    $missing = @()
    foreach ($req in @("cublas64_12.dll", "cudart64_12.dll")) {
        if (-not (Test-Path (Join-Path $internal $req))) { $missing += $req }
    }
    if (-not (Get-ChildItem $internal -Filter "cudnn*.dll" -ErrorAction SilentlyContinue)) {
        $missing += "cudnn*.dll"
    }
    if ($missing.Count -gt 0) {
        Write-Host "ERROR: CUDA build is missing:" -ForegroundColor Red
        foreach ($m in $missing) { Write-Host "  - $m" -ForegroundColor Red }
        Write-Host "This build would fail on any machine without a system CUDA install." -ForegroundColor Red
        exit 1
    }
    Write-Host "  - cublas, cudart and cudnn present in _internal\" -ForegroundColor Gray
}

Write-Host ""
$distSize = [math]::Round((Get-ChildItem ".\dist\gensrt" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
Write-Host "Distribution size: ${distSize} MB  (variant: $Variant)" -ForegroundColor White

# ── Post-build self-check ──────────────────────────────────────────────────
#
# `gensrt.exe --version` proves almost nothing: GenSRT defers nearly every
# import, so the executable starts happily with a bundle that cannot
# transcribe.  --self-check does eagerly what a real run does lazily —
# imports every module, executes the bundled ffmpeg, and loads the CUDA
# libraries by name through the OS loader.
#
# Both shipped bugs (a CUDA DLL in an unsearched directory, and a missing
# gensrt submodule) fail this check in under two seconds.

Write-Host ""
Write-Host "Running self-check on the built executable..." -ForegroundColor Yellow

$selfCheckArgs = @("--self-check")
if ($Variant -eq "cuda") { $selfCheckArgs += "--require-cuda" }

& ".\dist\gensrt\gensrt.exe" @selfCheckArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Self-check FAILED. This build is not shippable." -ForegroundColor Red
    Write-Host "The errors above are what a user would hit at runtime." -ForegroundColor Red
    exit 1
}
Write-Host "Self-check passed." -ForegroundColor Green

# ── Self-extracting installer ──────────────────────────────────────────────

Write-Host ""
Write-Host "Creating self-extracting installer..." -ForegroundColor Yellow
$7zCheck = Get-Command 7z -ErrorAction SilentlyContinue
if ($7zCheck) {
    if ($Variant -eq "cpu") {
        $installerName = "gensrt-install-cpu.exe"
    } else {
        $installerName = "gensrt-install.exe"
    }
    if (Test-Path $installerName) { Remove-Item $installerName -Force }

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
Write-Host "  .\dist\gensrt\gensrt.exe --dump-config" -ForegroundColor Gray
Write-Host "  .\dist\gensrt\gensrt.exe --input video.mkv" -ForegroundColor Gray
Write-Host ""
Write-Host "Notes:" -ForegroundColor Yellow
Write-Host "  - Whisper model (~800MB) downloads on first run to:" -ForegroundColor Gray
Write-Host "    %USERPROFILE%\.cache\huggingface\hub" -ForegroundColor Gray
if ($Variant -eq "cuda") {
    Write-Host "  - cuBLAS + cuDNN are bundled; an NVIDIA driver supporting" -ForegroundColor Gray
    Write-Host "    CUDA 12 is required on the target machine." -ForegroundColor Gray
    Write-Host "  - If CUDA init fails at runtime, GenSRT falls back to CPU" -ForegroundColor Gray
    Write-Host "    with a warning rather than failing the job." -ForegroundColor Gray
} else {
    Write-Host "  - CPU-only build: no CUDA libraries, no PyTorch." -ForegroundColor Gray
    Write-Host "    Runs on any x64 Windows machine, including Intel/AMD iGPU" -ForegroundColor Gray
    Write-Host "    laptops.  Substantially slower than the CUDA build." -ForegroundColor Gray
}
Write-Host "  - FFmpeg is bundled (gensrt\bin\ffmpeg.exe + ffprobe.exe);" -ForegroundColor Gray
Write-Host "    no separate install needed on target machines." -ForegroundColor Gray
Write-Host ""
