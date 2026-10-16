<#
.SYNOPSIS
    Run GenSRT on one file or a folder with the profile that fits the material.

.DESCRIPTION
    Picks the GenSRT profile from the source language (ko -> broadcast,
    anything else -> soft-speech), writes the per-run diagnostics next to the
    material (Dump\ and Heuristics\), and passes everything else straight to
    gensrt.  Values you set in gensrt-config.json win over the profile;
    explicit gensrt flags given through -Extra win over both (for example
    the translation engine for an A/B).

.PARAMETER InputPath
    (-i) A video file, or a folder of them.  Required.

.PARAMETER Source
    (-s) Source language code: ja, ko, ml, ...  Required.  Also chooses the
    model inside the soft-speech profile (kotoba-whisper for ja).

.PARAMETER Target
    (-t) Target language code.  Default en.

.PARAMETER KeepSource
    (-k) Also write the source-language subtitles (<name>.<source>.srt) from
    the SAME transcription the translation is made from, so the two files
    line up cue for cue.  Default ON; -KeepSource:$false turns it off.
    Needs a gensrt with --source-srt; an older gensrt gets a warning and
    runs without it.

.PARAMETER Extra
    (-x) Extra gensrt flags, one array element per token:
        -x --translation-engine,nllb
        -x "--translation-engine","nllb","--no-snap-onsets"
    These override the profile.  A single string with spaces in it is ONE
    argument to gensrt and will be rejected -- use commas.

.EXAMPLE
    .\SRTGen.ps1 -i .\clip.ts -s ja
    .\SRTGen.ps1 -i "F:\Live" -s ko -t en
    .\SRTGen.ps1 -i .\clip.ts -s ja -x --translation-engine,nllb
    .\SRTGen.ps1 -i .\clip.ts -s ja -KeepSource:$false

.NOTES
    Precedence inside gensrt: built-in defaults < profile < values set in
    gensrt-config.json < explicit flags.  A profile only fills in what the
    config file leaves at the default, so translation_engine in the config
    file is what runs; -Extra --translation-engine,<x> overrides it for one
    run.  The built-in profiles no longer name a translation engine.

    To compare engines on ONE transcription, translate the source file the
    run left behind instead of transcribing twice:
        gensrt --translate-srt .\clip.ja.srt --translation-engine nllb --output-filename clip.nllb.srt
#>
# Parameter names avoid PowerShell's automatic variables ($Input, $Args,
# $Profile) on purpose; shadowing them breaks in subtle ways.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [Alias("i")]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [Alias("s")]
    [string]$Source,

    [Alias("t")]
    [string]$Target = "en",

    [Alias("k")]
    [switch]$KeepSource = $true,

    [Alias("x")]
    [string[]]$Extra = @()
)

$ErrorActionPreference = "Stop"

# Resolve the input first: a relative path or a trailing backslash would
# otherwise make the folder logic below guess wrong.
$Item = Get-Item -LiteralPath $InputPath

# Dump\ and Heuristics\ live in the folder being processed: the folder itself
# when -InputPath is one, the file's parent when it is a single video.
$BaseDir    = if ($Item.PSIsContainer) { $Item.FullName } else { $Item.DirectoryName }
$Dump       = Join-Path -Path $BaseDir -ChildPath "Dump"
$Heuristics = Join-Path -Path $BaseDir -ChildPath "Heuristics"

$ProfileName = if ($Source -eq "ko") { "broadcast" } else { "soft-speech" }

$GensrtArgs = @(
    "--input", $Item.FullName,
    "--source-language", $Source,
    "--target-language", $Target,
    "--profile", $ProfileName,
    "--heuristics-report",
    "--dump-segments", $Dump,
    "--heuristics-report-dir", $Heuristics
)

if ($KeepSource) {
    # Only pass the flag to a gensrt that knows it; an unknown flag aborts
    # the run before any work.
    $help = (& gensrt --help 2>&1) -join "`n"
    if ($help -match "--source-srt") {
        $GensrtArgs += "--source-srt"
    } else {
        Write-Warning "This gensrt has no --source-srt; running without the source-language subtitles."
    }
}

# Explicit flags last, so they win over the profile and the config file.
$GensrtArgs += $Extra

Write-Host "Input      : $($Item.FullName)"
Write-Host "Profile    : $ProfileName  (source $Source -> $Target)"
Write-Host "Dump       : $Dump"
Write-Host "Heuristics : $Heuristics"
if ($Extra.Count) { Write-Host "Extra      : $($Extra -join ' ')" }
Write-Host ""

& gensrt @GensrtArgs
exit $LASTEXITCODE
