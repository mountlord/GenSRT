<#
.SYNOPSIS
    Create a GenSRT patch by diffing the released installer against a new one.

.DESCRIPTION
    Downloads the currently-published installer from GitHub, extracts it,
    extracts the newly-built one, and diffs the two.

    The downloaded release is the point of this script. It is what users
    actually have on disk. A local dist\gensrt tree is not: it may have been
    touched since the release, and it is not necessarily what the
    self-extracting archive produces when a user runs it. Diffing the real
    artifacts removes a class of error that stays invisible until somebody
    reports it.

    This is deliberately separate from Pack-gensrt.ps1. Packaging builds a
    release; this compares two of them. Keeping them apart means a broken
    patch step can never break a build.

    Nothing is published. The patch lands in the project directory for you to
    inspect and test before it goes anywhere.

.PARAMETER Variant
    cuda (default) or cpu. Selects which release asset to download and which
    locally-built installer to compare against.

.PARAMETER NewInstaller
    The freshly built installer. Defaults to .\gensrt-install.exe, or
    .\gensrt-install-cpu.exe for the cpu variant.

.PARAMETER FromVersion
    Version of the published release. Read from the release tag when omitted.

.PARAMETER ToVersion
    Version of the new build. Read from gensrt.__version__ when omitted.

.PARAMETER WorkDir
    Scratch space for downloads and extractions. Default .\patch-work.
    Reused between runs; the download is cached.

.PARAMETER Tag
    Diff against a specific release tag instead of the latest.

.EXAMPLE
    .\Create-gensrt-patch.ps1
    .\Create-gensrt-patch.ps1 -Variant cpu
    .\Create-gensrt-patch.ps1 -Tag Release-1.2.5 -FromVersion 1.2.5
#>
[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Variant = "cuda",
    [string]$NewInstaller = "",
    [string]$FromVersion = "",
    [string]$ToVersion = "",
    [string]$WorkDir = ".\patch-work",
    [string]$Tag = "",
    [string]$Repo = "mountlord/GenSRT"
)

$ErrorActionPreference = "Stop"

$assetName = if ($Variant -eq "cpu") { "gensrt-install-cpu.exe" } else { "gensrt-install.exe" }
if (-not $NewInstaller) { $NewInstaller = ".\$assetName" }

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GenSRT patch builder — $Variant" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ── Preflight ──────────────────────────────────────────────────────────────

if (-not (Test-Path ".\tools\make_patch.py")) {
    Write-Host "ERROR: run this from the GenSRT project root." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $NewInstaller)) {
    Write-Host "ERROR: new installer not found: $NewInstaller" -ForegroundColor Red
    Write-Host "Build it first:  .\Pack-gensrt.ps1 -Variant $Variant" -ForegroundColor Yellow
    exit 1
}
if (-not (Get-Command 7z -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 7z is required to extract the installers." -ForegroundColor Red
    Write-Host "Install 7-Zip and put it on PATH: https://www.7-zip.org/" -ForegroundColor Yellow
    exit 1
}

# ── Which release are we patching from? ────────────────────────────────────

Write-Host ""
Write-Host "Looking up the published release..." -ForegroundColor Yellow

$apiUrl = if ($Tag) {
    "https://api.github.com/repos/$Repo/releases/tags/$Tag"
} else {
    "https://api.github.com/repos/$Repo/releases/latest"
}

try {
    $release = Invoke-RestMethod -Uri $apiUrl -Headers @{ "User-Agent" = "GenSRT-patch-builder" }
} catch {
    Write-Host "ERROR: could not reach GitHub: $_" -ForegroundColor Red
    exit 1
}

$asset = $release.assets | Where-Object { $_.name -eq $assetName }
if (-not $asset) {
    Write-Host "ERROR: release $($release.tag_name) has no asset named $assetName" -ForegroundColor Red
    Write-Host "Assets present: $(($release.assets | ForEach-Object { $_.name }) -join ', ')" -ForegroundColor Yellow
    exit 1
}

if (-not $FromVersion) {
    # Release-1.2.5 -> 1.2.5
    $FromVersion = ($release.tag_name -replace '^[A-Za-z\-]*', '')
}
if (-not $ToVersion) {
    $ToVersion = (python -c "import gensrt; print(gensrt.__version__)" 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $ToVersion) {
        Write-Host "ERROR: could not read gensrt.__version__; pass -ToVersion." -ForegroundColor Red
        exit 1
    }
    $ToVersion = $ToVersion.Trim()
}

Write-Host "  published : $($release.tag_name)  ($assetName, $([math]::Round($asset.size / 1MB, 1)) MB)"
Write-Host "  patching  : $FromVersion  ->  $ToVersion"

if ($FromVersion -eq $ToVersion) {
    Write-Host ""
    Write-Host "ERROR: both versions are $ToVersion. Bump gensrt.__version__ first." -ForegroundColor Red
    Write-Host "A patch that does not change the version cannot be verified on the" -ForegroundColor Yellow
    Write-Host "user's side, and leaves two different installs claiming to be the same." -ForegroundColor Yellow
    exit 1
}

# ── Fetch and extract ──────────────────────────────────────────────────────

$work = Join-Path (Get-Location) ($WorkDir -replace '^\.\\', '')
$dlPath  = Join-Path $work "$($release.tag_name)-$assetName"
$oldDir  = Join-Path $work "old-$FromVersion-$Variant"
$newDir  = Join-Path $work "new-$ToVersion-$Variant"

New-Item -ItemType Directory -Force -Path $work | Out-Null

if (Test-Path $dlPath) {
    Write-Host ""
    Write-Host "Using cached download: $dlPath" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "Downloading $assetName ..." -ForegroundColor Yellow
    # Invoke-WebRequest's progress bar makes large downloads dramatically
    # slower in Windows PowerShell; suppressing it is not cosmetic.
    $prev = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $dlPath
    } finally {
        $ProgressPreference = $prev
    }
    Write-Host "  done" -ForegroundColor Gray
}

foreach ($pair in @(@($dlPath, $oldDir, "published"), @((Resolve-Path $NewInstaller).Path, $newDir, "new build"))) {
    $src, $dst, $label = $pair
    Write-Host ""
    Write-Host "Extracting $label ..." -ForegroundColor Yellow
    if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    & 7z x "$src" "-o$dst" -y | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: extraction failed for $src" -ForegroundColor Red
        exit 1
    }
    # The SFX contains a top-level gensrt\ folder; diff the install tree
    # itself so paths in the patch match what Apply-Patch.ps1 will see.
    $inner = Join-Path $dst "gensrt"
    if (Test-Path (Join-Path $inner "gensrt.exe")) {
        Get-ChildItem $inner -Force | Move-Item -Destination $dst -Force
        Remove-Item $inner -Recurse -Force
    }
    if (-not (Test-Path (Join-Path $dst "gensrt.exe"))) {
        Write-Host "ERROR: no gensrt.exe found after extracting $src" -ForegroundColor Red
        exit 1
    }
    $n = (Get-ChildItem $dst -Recurse -File).Count
    Write-Host "  $n files" -ForegroundColor Gray
}

# ── Diff ───────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "Diffing..." -ForegroundColor Yellow

$outZip = ".\gensrt-patch-$FromVersion-to-$ToVersion-$Variant.zip"

python tools\make_patch.py `
    --from "$oldDir" --to "$newDir" `
    --from-version $FromVersion --to-version $ToVersion `
    --variant $Variant --out $outZip

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: patch generation failed." -ForegroundColor Red
    exit 1
}

# ── What to do next ────────────────────────────────────────────────────────

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Patch built — NOT yet verified" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Test it before publishing:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Copy a clean $FromVersion install somewhere:" -ForegroundColor Gray
Write-Host "       Copy-Item `"$oldDir`" C:\Temp\patch-test -Recurse" -ForegroundColor Gray
Write-Host "  2. Unpack the patch and apply it:" -ForegroundColor Gray
Write-Host "       Expand-Archive $outZip C:\Temp\patch" -ForegroundColor Gray
Write-Host "       C:\Temp\patch\Apply-Patch.ps1 -InstallDir C:\Temp\patch-test" -ForegroundColor Gray
Write-Host "  3. Run it and confirm the fix is present:" -ForegroundColor Gray
Write-Host "       C:\Temp\patch-test\gensrt.exe --version" -ForegroundColor Gray
Write-Host ""
Write-Host "Working files are in $work (download is cached; delete to re-fetch)." -ForegroundColor Gray
Write-Host ""
