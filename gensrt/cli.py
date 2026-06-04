"""Command-line interface for GenSRT.

Usage examples::

    # GUI mode (default — opens desktop window)
    gensrt
    gensrt video.mkv          # open GUI pre-loaded with a file

    # Headless batch transcription
    gensrt --input video.mkv
    gensrt --input /media/shows/ --recurse
    gensrt --input video.mkv --output /subs/ --translation-engine nllb

    # Configuration helpers
    gensrt --init-config      # write default gensrt-config.json and exit
    gensrt --dump-config      # print resolved config and exit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from gensrt import __version__

logger = logging.getLogger(__name__)


# ── Argument parser ────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    from gensrt.config import BUILTIN_DEFAULTS

    bd = BUILTIN_DEFAULTS

    parser = argparse.ArgumentParser(
        prog="gensrt",
        description="GPU-accelerated subtitle (SRT) generation using Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  gensrt                                         # launch GUI\n"
            "  gensrt video.mkv                              # GUI, pre-load file\n"
            "  gensrt --input video.mkv                      # headless transcription\n"
            "  gensrt --input /shows/ --recurse              # batch, recurse dirs\n"
            "  gensrt --input ep01.mkv --no-translate        # keep source language\n"
            "  gensrt --input ep01.mkv --translation-engine nllb  # offline engine\n"
            "  gensrt --init-config                          # write default config\n"
            "  gensrt --dump-config                          # show resolved config\n"
        ),
    )

    # Positional (alternative to --input)
    parser.add_argument(
        "inputs_pos",
        nargs="*",
        type=Path,
        metavar="FILE_OR_DIR",
        help="Media file(s) or directories to process (positional --input alternative).",
    )

    # ── Input / Output ────────────────────────────────────────────────────
    io = parser.add_argument_group("input / output")
    io.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        default=None,
        metavar="FILE_OR_DIR",
        help="Media file or directory to process. Repeatable.",
    )
    io.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output folder (default: same directory as input).",
    )
    io.add_argument(
        "--output-filename",
        type=str,
        default=None,
        metavar="FILENAME",
        help=(
            "Override output .srt filename (single-file input only). "
            "When set, --output is ignored with a warning."
        ),
    )
    io.add_argument(
        "--recurse",
        action="store_true",
        default=None,
        help="Recurse into subdirectories when input is a directory.",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    mdl = parser.add_argument_group("model")
    mdl.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help=f"Whisper model name (default: {bd['model']!r}).",
    )
    mdl.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help=f"Compute device (default: {bd['device']!r}; overridden by GPU probe).",
    )
    mdl.add_argument(
        "--compute-type",
        dest="compute_type",
        choices=["float16", "int8_float16", "int8"],
        default=None,
        help=f"CTranslate2 compute type (default: {bd['compute_type']!r}).",
    )
    mdl.add_argument(
        "--gpu-id",
        dest="gpu_id",
        type=int,
        default=None,
        metavar="N",
        help=f"CUDA device ordinal (default: {bd['gpu_id']}).",
    )

    # ── Translation ───────────────────────────────────────────────────────
    tr = parser.add_argument_group("translation")
    tr.add_argument(
        "--translation-engine",
        dest="translation_engine",
        choices=["google", "nllb", "marian", "none"],
        default=None,
        help=f"Translation engine (default: {bd['translation_engine']!r}).",
    )
    tr.add_argument(
        "--source-language",
        dest="source_language",
        default=None,
        metavar="LANG",
        help=(
            f"Source language code or 'auto' (default: {bd['source_language']!r}). "
            "Examples: ja, ko, ml, fr, auto."
        ),
    )
    tr.add_argument(
        "--no-translate",
        dest="translate",
        action="store_false",
        default=None,
        help="Disable translation; output subtitles in the source language.",
    )

    # ── VAD ───────────────────────────────────────────────────────────────
    vad = parser.add_argument_group("voice activity detection")
    vad.add_argument(
        "--no-vad",
        dest="vad_enabled",
        action="store_false",
        default=None,
        help="Disable VAD; pass full audio to Whisper (more segments, may include silence noise).",
    )
    vad.add_argument(
        "--vad-threshold",
        dest="vad_threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help=f"Speech probability threshold 0–1 (default: {bd['vad_threshold']}). "
             "Lower = more speech detected, fewer missed lines.",
    )
    vad.add_argument(
        "--vad-min-speech-ms",
        dest="vad_min_speech_ms",
        type=int,
        default=None,
        metavar="MS",
        help=f"Minimum speech segment duration in ms (default: {bd['vad_min_speech_ms']}). "
             "Lower = shorter utterances are kept.",
    )
    vad.add_argument(
        "--vad-min-silence-ms",
        dest="vad_min_silence_ms",
        type=int,
        default=None,
        metavar="MS",
        help=f"Minimum silence gap that splits segments in ms (default: {bd['vad_min_silence_ms']}). "
             "Lower = more splits, shorter segments.",
    )
    vad.add_argument(
        "--vad-speech-pad-ms",
        dest="vad_speech_pad_ms",
        type=int,
        default=None,
        metavar="MS",
        help=f"Padding added before/after detected speech in ms (default: {bd['vad_speech_pad_ms']}; "
             "faster-whisper library default is 400). Lower = subtitles align closer to actual "
             "speech onset; higher = safer against clipping the first syllable.",
    )

    # ── SRT output ────────────────────────────────────────────────────────
    srt_grp = parser.add_argument_group("srt output")
    srt_grp.add_argument(
        "--max-subtitle-duration",
        dest="max_subtitle_duration_s",
        type=float,
        default=None,
        metavar="SEC",
        help=f"Cap maximum subtitle display time in seconds (default: {bd['max_subtitle_duration_s']}). "
             "Set to 0 to disable. Fixes subtitles that hang on screen for minutes.",
    )
    srt_grp.add_argument(
        "--min-subtitle-duration",
        dest="min_subtitle_duration_s",
        type=float,
        default=None,
        metavar="SEC",
        help=f"Floor minimum subtitle display time in seconds (default: {bd['min_subtitle_duration_s']}). "
             "Set to 0 to disable. Fixes subtitles that disappear too fast.",
    )

    # ── Configuration ─────────────────────────────────────────────────────
    cfg = parser.add_argument_group("configuration")
    cfg.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to gensrt-config.json (default: auto-discover).",
    )
    cfg.add_argument(
        "--init-config",
        action="store_true",
        default=False,
        help="Write a default gensrt-config.json to CWD and exit.",
    )
    cfg.add_argument(
        "--dump-config",
        action="store_true",
        default=False,
        help="Print the resolved configuration and exit.",
    )

    # ── UI ────────────────────────────────────────────────────────────────
    ui = parser.add_argument_group("ui")
    ui.add_argument(
        "--console",
        action="store_true",
        default=False,
        help="Open pyWebView DevTools on startup (for UI debugging).",
    )

    # ── Logging ───────────────────────────────────────────────────────────
    log = parser.add_argument_group("logging")
    log.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Log level (default: INFO).",
    )
    log.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Alias for --log-level DEBUG.",
    )
    log.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Alias for --log-level WARNING.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve_log_level(args: argparse.Namespace) -> str:
    if args.verbose:
        return "DEBUG"
    if args.quiet:
        return "WARNING"
    return args.log_level or "INFO"


def _resolve_settings(args: argparse.Namespace) -> dict:
    """Load config file and merge with CLI args.

    Returns the merged dict.
    """
    from gensrt.config import load_config, merge_config

    file_config = load_config(args.config, strict=True)
    cli_dict = vars(args)
    return merge_config(file_config, cli_dict)


def _dump_resolved_config(merged: dict) -> None:
    # Only print keys that are part of BUILTIN_DEFAULTS
    from gensrt.config import BUILTIN_DEFAULTS
    printable = {k: merged.get(k) for k in BUILTIN_DEFAULTS}
    print(json.dumps(printable, indent=2, default=str))


def _cli_status(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr)


def _make_tqdm_progress():
    """Return a (callback, pbar_ref) pair using tqdm if available."""
    try:
        from tqdm import tqdm

        pbar_holder: list = []

        def _progress(current: int, total: int) -> None:
            if not pbar_holder:
                pbar_holder.append(
                    tqdm(total=total, unit="step", dynamic_ncols=True, file=sys.stderr)
                )
            pbar = pbar_holder[0]
            pbar.total = total
            pbar.n = current
            pbar.refresh()

        return _progress, pbar_holder

    except ImportError:

        def _progress(current: int, total: int) -> None:
            print(f"  [{current}/{total}]", file=sys.stderr, end="\r")

        return _progress, []


def _print_banner(merged: dict) -> None:
    import sys
    print("", file=sys.stderr)
    print("═" * 60, file=sys.stderr)
    print("  GenSRT", file=sys.stderr)
    print(f"  Model  : {merged.get('model')}", file=sys.stderr)
    print(f"  Device : {merged.get('device')} ({merged.get('backend', 'auto')})", file=sys.stderr)
    print(f"  Compute: {merged.get('compute_type')}", file=sys.stderr)
    print(f"  Engine : {merged.get('translation_engine')}", file=sys.stderr)
    print(f"  VAD    : {'on' if merged.get('vad_enabled') else 'off'}", file=sys.stderr)
    print("═" * 60, file=sys.stderr)
    print("", file=sys.stderr)


# ── Headless runner ────────────────────────────────────────────────────────

def _run_headless(args: argparse.Namespace) -> int:
    """Execute the headless transcription pipeline.

    Returns:
        Exit code (0 = success, 1 = partial errors).
    """
    from gensrt.exceptions import ConfigError, ConfigParseError, GenSRTError
    from gensrt.operations import (
        build_transcription_config,
        resolve_output_path,
        run_transcription,
    )
    from gensrt.utils.media_files import collect_media_files

    # Resolve config
    try:
        merged = _resolve_settings(args)
    except ConfigParseError as exc:
        logger.error("%s", exc)
        return 2
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    # Build config (auto-detect GPU)
    config = build_transcription_config(merged, auto_detect_backend=True)
    _print_banner(merged)

    # Collect input files
    all_inputs: list[Path] = list(args.inputs or [])
    recurse: bool = bool(merged.get("recurse", False))
    output_dir: Path | None = args.output
    output_filename: str | None = args.output_filename

    media_files: list[Path] = []
    for inp in all_inputs:
        try:
            found = collect_media_files(inp, recurse=recurse)
            media_files.extend(found)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1

    if not media_files:
        logger.error("No media files found in the specified input(s).")
        return 1

    logger.info("Files to process: %d", len(media_files))

    if output_filename and len(media_files) > 1:
        logger.warning(
            "--output-filename is ignored when processing multiple files. "
            "Each file will use its own stem."
        )
        output_filename = None

    # Process each file
    errors = 0
    t0 = time.perf_counter()

    for i, media_path in enumerate(media_files, 1):
        out_path = resolve_output_path(media_path, output_dir, output_filename)

        print(f"\n[{i}/{len(media_files)}] {media_path.name}", file=sys.stderr)
        print(f"  → {out_path}", file=sys.stderr)

        progress_cb, pbar_holder = _make_tqdm_progress()

        try:
            run_transcription(
                input_path=media_path,
                output_path=out_path,
                config=config,
                progress=progress_cb,
                status=_cli_status,
            )
        except GenSRTError as exc:
            logger.error("Error processing %s: %s", media_path.name, exc)
            errors += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.", file=sys.stderr)
            return 130
        finally:
            if pbar_holder:
                pbar_holder[0].close()

    total_time = time.perf_counter() - t0
    print(f"\n{'═' * 60}", file=sys.stderr)
    print(
        f"  Finished {len(media_files)} file(s) in {total_time:.1f}s "
        f"({errors} error(s))",
        file=sys.stderr,
    )
    print(f"{'═' * 60}\n", file=sys.stderr)

    return 1 if errors else 0


# ── Main entry point ───────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``gensrt`` command."""
    from gensrt.utils.logging_config import setup_logging

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Merge positional inputs into --input list
    if not args.inputs and getattr(args, "inputs_pos", None):
        args.inputs = list(args.inputs_pos)

    # Early logging setup (before config is fully resolved)
    setup_logging(_resolve_log_level(args))

    # ── --init-config ──────────────────────────────────────────────────────
    if args.init_config:
        from gensrt.config import generate_default_config
        path = generate_default_config()
        print(f"Default config written: {path}", file=sys.stderr)
        sys.exit(0)

    # ── --dump-config ──────────────────────────────────────────────────────
    if args.dump_config:
        from gensrt.exceptions import ConfigError, ConfigParseError
        try:
            merged = _resolve_settings(args)
            _dump_resolved_config(merged)
            sys.exit(0)
        except (ConfigParseError, ConfigError) as exc:
            logger.error("%s", exc)
            sys.exit(2)

    # ── Mode dispatch ─────────────────────────────────────────────────────
    has_inputs = bool(args.inputs)

    if not has_inputs:
        # No inputs → launch GUI (no-op in headless/server environments)
        from gensrt.server import launch_server
        launch_server(console=args.console)
        sys.exit(0)

    # Inputs with no headless flag: launch GUI, pre-load first file if it's a file
    # For GenSRT, providing --input *always* means headless (transcription job).
    # The GUI is launched only when no --input is given.
    setup_logging(_resolve_log_level(args))
    sys.exit(_run_headless(args))


if __name__ == "__main__":
    main()
