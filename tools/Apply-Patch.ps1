<#
    GenSRT patch installer.

    Verifies the installation matches what this patch was built against,
    backs up every file it will touch, applies the changes, verifies the
    result, and runs the self-check.

    Refuses to run against an unexpected version: a half-patched install
    produces bug reports nobody can reproduce. Use -Force only if you know
    exactly why the check is failing.
#>
[CmdletBinding()]
param(
    [string]$InstallDir = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$patchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifest  = Get-Content (Join-Path $patchRoot "patch-manifest.json") -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "GenSRT patch  $($manifest.from_version) -> $($manifest.to_version)  ($($manifest.variant))" -ForegroundColor Cyan
Write-Host ""

if (-not $InstallDir) {
    $guess = Split-Path -Parent $patchRoot
    if (Test-Path (Join-Path $guess "gensrt.exe")) { $InstallDir = $guess }
}
if (-not $InstallDir -or -not (Test-Path (Join-Path $InstallDir "gensrt.exe"))) {
    Write-Host "Could not find gensrt.exe." -ForegroundColor Red
    Write-Host "Run with:  .\Apply-Patch.ps1 -InstallDir C:\path\to\gensrt" -ForegroundColor Yellow
    exit 1
}
Write-Host "Installation: $InstallDir"

# A running instance holds gensrt.exe open and the copy would fail halfway.
if (Get-Process -Name "gensrt" -ErrorAction SilentlyContinue) {
    Write-Host "GenSRT is running. Close it and run this again." -ForegroundColor Red
    exit 1
}

# -- Verify ----------------------------------------------------------------
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
$mismatch = @()
foreach ($p in $manifest.expect.PSObject.Properties) {
    $target = Join-Path $InstallDir $p.Name
    if (-not (Test-Path $target)) { $mismatch += "$($p.Name)  (missing)"; continue }
    $actual = (Get-FileHash -Algorithm SHA256 $target).Hash.ToLower()
    if ($actual -ne $p.Value) { $mismatch += "$($p.Name)  (modified or wrong version)" }
}
if ($mismatch.Count -gt 0) {
    Write-Host ""
    Write-Host "This patch expects GenSRT $($manifest.from_version). These files do not match:" -ForegroundColor Red
    $mismatch | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    if ($mismatch.Count -gt 10) { Write-Host "  ... and $($mismatch.Count - 10) more" -ForegroundColor Red }
    Write-Host ""
    if (-not $Force) {
        Write-Host "Download the full installer instead, or re-run with -Force." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "-Force given; continuing anyway." -ForegroundColor Yellow
}
Write-Host "  OK" -ForegroundColor Gray

# -- Back up ---------------------------------------------------------------
$backup = Join-Path $InstallDir ("backup-" + $manifest.from_version)
Write-Host ""
Write-Host "Backing up to $backup ..." -ForegroundColor Yellow
foreach ($rel in @($manifest.changed) + @($manifest.removed)) {
    $src = Join-Path $InstallDir $rel
    if (Test-Path $src) {
        $dst = Join-Path $backup $rel
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
        Copy-Item $src $dst -Force
    }
}
Write-Host "  OK" -ForegroundColor Gray

# -- Apply -----------------------------------------------------------------
Write-Host ""
Write-Host "Applying..." -ForegroundColor Yellow
foreach ($rel in @($manifest.changed) + @($manifest.added)) {
    $src = Join-Path $patchRoot (Join-Path "files" $rel)
    $dst = Join-Path $InstallDir $rel
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Copy-Item $src $dst -Force
}
foreach ($rel in @($manifest.removed)) {
    $dst = Join-Path $InstallDir $rel
    if (Test-Path $dst) { Remove-Item $dst -Force }
}
Write-Host "  $(@($manifest.changed).Count) changed, $(@($manifest.added).Count) added, $(@($manifest.removed).Count) removed" -ForegroundColor Gray

# -- Verify the result -----------------------------------------------------
Write-Host ""
Write-Host "Verifying result..." -ForegroundColor Yellow
$bad = @()
foreach ($p in $manifest.result.PSObject.Properties) {
    $target = Join-Path $InstallDir $p.Name
    if (-not (Test-Path $target)) { $bad += $p.Name; continue }
    if ((Get-FileHash -Algorithm SHA256 $target).Hash.ToLower() -ne $p.Value) { $bad += $p.Name }
}
if ($bad.Count -gt 0) {
    Write-Host "Patch did not apply cleanly:" -ForegroundColor Red
    $bad | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host "Restore from $backup and use the full installer." -ForegroundColor Yellow
    exit 1
}
Write-Host "  OK" -ForegroundColor Gray

# -- Self-check ------------------------------------------------------------
Write-Host ""
Write-Host "Running self-check..." -ForegroundColor Yellow
$selfArgs = @("--self-check")
if ($manifest.variant -eq "cuda") { $selfArgs += "--require-cuda" }
& (Join-Path $InstallDir "gensrt.exe") @selfArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Self-check FAILED after patching. Restore from $backup." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Patched to $($manifest.to_version)." -ForegroundColor Green
Write-Host "The backup in $backup can be deleted once you are happy." -ForegroundColor Gray
Write-Host ""
