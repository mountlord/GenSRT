#!/usr/bin/env python3
"""Build a patch from one GenSRT distribution to another.

Why
---
A GenSRT release is mostly things that do not change.  In the CUDA build,
cuBLAS and cuDNN alone are about 1.4 GB and are byte-identical between
versions; ffmpeg, the Python runtime and every third-party wheel are the same.
What actually changes for a typical fix is the executable and a handful of
template and static files.

Asking someone to re-download 1.5 GB to pick up a corrected error message is
the kind of thing that stops people updating at all — and users who do not
update are the ones who report bugs that were fixed two releases ago.

What it does
------------
Hashes every file in both trees, works out what changed, and writes a zip
containing only those files plus an apply script and a manifest.

Deletions are included.  A patch that only adds and overwrites leaves stale
files behind, and a stale module that still imports is worse than a missing
one — it runs.

Safety
------
The patch records the SHA-256 of every file it will overwrite, as they were in
the source distribution.  The apply script refuses to run if the target does
not match, because a half-patched install produces bug reports nobody can
reproduce.  ``--force`` exists for when you know better.

Both sides are *extracted installers*, not build trees.  That is deliberate:
the released installer is what users actually have on disk, and a local
``dist/`` tree may have been touched since, or may differ from what the
self-extracting archive produces.  Diffing the real artifacts removes a class
of error that is invisible until someone reports it.

``Create-gensrt-patch.ps1`` does the downloading and extracting; this script
only diffs two folders.

Usage
-----
    python tools/make_patch.py --from  .\\patch-work\\old \\
                               --to    .\\patch-work\\new \\
                               --from-version 1.2.5 --to-version 1.2.6 \\
                               --variant cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PATCH_MANIFEST = "patch-manifest.json"

# Files that legitimately differ per-installation and must never be patched:
# the user's own settings, and anything they put in models/.
EXCLUDE_PREFIXES = ("models/", "logs/")
EXCLUDE_NAMES = ("gensrt-config.json",)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _excluded(rel: str) -> bool:
    r = rel.replace("\\", "/")
    return r.startswith(EXCLUDE_PREFIXES) or Path(r).name in EXCLUDE_NAMES


def hash_tree(root: Path) -> dict[str, str]:
    """Map every file under *root* to its SHA-256, keyed by relative path."""
    root = Path(root).resolve()
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if _excluded(rel):
            continue
        out[rel] = _sha256(p)
    return out


def diff(old: dict[str, str], new: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """Return ``(changed, added, removed)`` relative paths."""
    changed = sorted(r for r in new if r in old and old[r] != new[r])
    added = sorted(r for r in new if r not in old)
    removed = sorted(r for r in old if r not in new)
    return changed, added, removed


def build_patch(
    old_tree: dict[str, str],
    new_root: Path,
    new_tree: dict[str, str],
    out_zip: Path,
    *,
    from_version: str,
    to_version: str,
    variant: str,
    apply_script: str,
) -> dict:
    changed, added, removed = diff(old_tree, new_tree)
    payload = changed + added

    manifest = {
        "from_version": from_version,
        "to_version": to_version,
        "variant": variant,
        # Expected pre-patch hashes, so the target can be verified.
        "expect": {r: old_tree[r] for r in changed},
        # Post-patch hashes, so the result can be verified.
        "result": {r: new_tree[r] for r in payload},
        "changed": changed,
        "added": added,
        "removed": removed,
    }

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in payload:
            z.write(Path(new_root) / rel, f"files/{rel}")
        z.writestr(PATCH_MANIFEST, json.dumps(manifest, indent=2))
        z.writestr("Apply-Patch.ps1", apply_script)

    manifest["_bytes"] = sum(
        (Path(new_root) / r).stat().st_size for r in payload
    )
    manifest["_zip_bytes"] = out_zip.stat().st_size
    return manifest


APPLY_SCRIPT = Path(__file__).with_name("Apply-Patch.ps1")


def _apply_script() -> str:
    """Read the apply script that ships inside every patch.

    A real file rather than a string literal in this module.  It was embedded
    at first so it could not drift from the manifest format, but the cost was
    worse than the risk: 120 lines of PowerShell inside a Python string get no
    highlighting, no linting, and cannot be read or diffed in the repository —
    and this is the script that overwrites files on a user's machine, so it is
    the one that most deserves review.

    It must stay ASCII-only.  ``zipfile`` writes with no BOM, and Windows
    PowerShell 5.1 decodes a BOM-less .ps1 as cp1252, so any non-ASCII
    character arrives mangled — and a byte that lands on a quote breaks
    parsing outright.  Checked below rather than trusted.
    """
    if not APPLY_SCRIPT.is_file():
        raise SystemExit(
            f"Missing {APPLY_SCRIPT}. It ships inside every patch; without it "
            f"the patch cannot be applied."
        )
    text = APPLY_SCRIPT.read_text(encoding="utf-8")
    non_ascii = sorted({c for c in text if ord(c) > 127})
    if non_ascii:
        raise SystemExit(
            f"{APPLY_SCRIPT} contains non-ASCII characters: {non_ascii}\n"
            f"Windows PowerShell 5.1 reads a BOM-less .ps1 as cp1252 and would "
            f"mangle them. Replace with ASCII equivalents."
        )
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="old", required=True, type=Path,
                    help="Folder holding the EXTRACTED previous installer — "
                         "what users actually have on disk")
    ap.add_argument("--to", dest="new", required=True, type=Path,
                    help="Folder holding the EXTRACTED new installer")
    ap.add_argument("--from-version", required=True)
    ap.add_argument("--to-version", required=True)
    ap.add_argument("--variant", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    for label, d in (("--from", args.old), ("--to", args.new)):
        if not d.is_dir():
            print(f"{label} is not a directory: {d}", file=sys.stderr)
            return 1

    old_tree = hash_tree(args.old)
    new_tree = hash_tree(args.new)

    out = args.out or Path(
        f"gensrt-patch-{args.from_version}-to-{args.to_version}-{args.variant}.zip"
    )
    m = build_patch(old_tree, args.new, new_tree, out,
                    from_version=args.from_version,
                    to_version=args.to_version,
                    variant=args.variant,
                    apply_script=_apply_script())

    total_new = sum((args.new / r).stat().st_size for r in new_tree)
    print()
    print(f"  {len(m['changed']):>5} changed")
    print(f"  {len(m['added']):>5} added")
    print(f"  {len(m['removed']):>5} removed")
    print()
    print(f"  patch payload : {m['_bytes'] / 1048576:>9.1f} MB")
    print(f"  patch zip     : {m['_zip_bytes'] / 1048576:>9.1f} MB")
    print(f"  extracted new : {total_new / 1048576:>9.1f} MB")
    print()
    print(f"  -> {out}")
    if m["changed"] or m["added"]:
        print()
        print("  contents:")
        sized = sorted(
            ((args.new / r).stat().st_size, r) for r in m["changed"] + m["added"]
        )
        for size, rel in reversed(sized):
            print(f"    {size / 1048576:>8.2f} MB  {rel}")
    if m["removed"]:
        print()
        print("  removed by this patch:")
        for rel in m["removed"]:
            print(f"               {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
