#!/usr/bin/env python3
"""Rewrite film_profiles.py with the FilmProfile blocks physically sorted by name.

WHY THIS EXISTS
---------------
film_profiles.py applies sorted() at import time, so anything that *consumes*
the database -- --list, the C++ table, -p all -- already sees alphabetical
order. But the source literal stays grouped by manufacturer and era, because
that is the sane way to maintain it: you want VISION3 50D next to VISION3 500T
while editing them.

This closes that gap when you want the source itself alphabetical too. It is a
source-to-source transform, NOT a generator that invents content: every block
moves verbatim, byte for byte, including its comments.

SAFETY -- nothing is written until all of these hold:
  1. block count identical before and after
  2. set of stock names identical
  3. concatenated blocks are a permutation of the original (same character
     multiset -- catches any truncation)
  4. the rewritten module imports and validate_all() passes
  5. every profile compares equal, field for field, to the original
A timestamped backup is written first. Any failure restores the original and
exits non-zero having changed nothing.

USAGE
    python sort_profiles.py              # sort in place
    python sort_profiles.py --check      # report order, change nothing
    python sort_profiles.py -f other.py  # a different file
    python sort_profiles.py --no-backup
Then regenerate the C++ tables so they stay in step:
    python cpp_codegen.py -o .
"""

from __future__ import annotations

import argparse
import collections
import pickle
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

NAME_RE = re.compile(r'name="([A-Za-z0-9_]+)"')


def split_blocks(text: str) -> tuple[str, list[tuple[str, str]], str]:
    """Cut the FILM_PROFILES literal into (header, [(name, block)], footer).

    A block runs from its `    FilmProfile(` line to the matching `    ),` at
    the same indent, and carries any comment or blank lines sitting directly
    above it -- those describe the block, so they travel with it.
    """
    lines = text.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("FILM_PROFILES"))
    end = next(i for i in range(start, len(lines)) if lines[i].rstrip() == ")")

    blocks: list[tuple[str, str]] = []
    pending: list[str] = []
    current: list[str] | None = None

    for line in lines[start + 1:end]:
        if current is None:
            if line.rstrip() == "    FilmProfile(":
                current = pending + [line]
                pending = []
            else:
                pending.append(line)
        else:
            current.append(line)
            if line.rstrip() == "    ),":
                blob = "".join(current)
                m = NAME_RE.search(blob)
                if not m:
                    raise SystemExit("a FilmProfile block has no name= field")
                blocks.append((m.group(1), blob))
                current = None

    if current is not None:
        raise SystemExit("unterminated FilmProfile block")

    header = "".join(lines[:start + 1])
    footer = "".join(pending) + "".join(lines[end:])
    return header, blocks, footer


def _snapshot(cwd: Path, validate: bool) -> tuple[int, dict, str]:
    code = ("import film_profiles as f, pickle, sys;"
            + ("f.validate_all();" if validate else "")
            + "sys.stdout.buffer.write(pickle.dumps("
              "{p.name: repr(p) for p in f.FILM_PROFILES}))")
    r = subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True)
    if r.returncode != 0:
        return r.returncode, {}, r.stderr.decode()[-900:]
    return 0, pickle.loads(r.stdout), ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--file", type=Path, default=Path("film_profiles.py"))
    ap.add_argument("--check", action="store_true",
                    help="report current order and exit without writing")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    path: Path = args.file
    if not path.is_file():
        print(f"[ERROR] not found: {path}", file=sys.stderr)
        return 2
    cwd = path.parent if str(path.parent) else Path(".")

    original = path.read_text(encoding="utf-8")
    header, blocks, footer = split_blocks(original)
    names = [n for n, _ in blocks]
    ordered = sorted(names)

    dupes = [n for n, c in collections.Counter(names).items() if c > 1]
    if dupes:
        print(f"[ERROR] duplicate stock names: {dupes}", file=sys.stderr)
        return 2

    print(f"[INFO] {path.name}: {len(blocks)} FilmProfile blocks")
    if names == ordered:
        print("[INFO] source literal is already alphabetical")
        return 0

    misplaced = sum(1 for a, b in zip(names, ordered) if a != b)
    print(f"[INFO] {misplaced} block(s) out of alphabetical position")
    print(f"[INFO] first in source: {names[0]}   first sorted: {ordered[0]}")
    if args.check:
        print("[INFO] --check given, nothing written")
        return 1

    rebuilt = header + "".join(dict(blocks)[n] for n in ordered) + footer

    _, b2, _ = split_blocks(rebuilt)
    assert len(b2) == len(blocks), "block count changed"
    assert {n for n, _ in b2} == set(names), "stock name set changed"
    assert (collections.Counter("".join(b for _, b in blocks))
            == collections.Counter("".join(b for _, b in b2))), \
        "block content is not a permutation of the original"

    rc, was, err = _snapshot(cwd, validate=False)
    if rc:
        print(f"[ERROR] original module does not import:\n{err}", file=sys.stderr)
        return 2

    if not args.no_backup:
        backup = path.with_suffix(f".py.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
        print(f"[INFO] backup -> {backup.name}")

    path.write_text(rebuilt, encoding="utf-8")

    rc, now, err = _snapshot(cwd, validate=True)

    def restore(msg: str) -> int:
        path.write_text(original, encoding="utf-8")
        print(f"[ERROR] {msg}\n[ERROR] original restored, nothing changed",
              file=sys.stderr)
        return 2

    if rc:
        return restore("rewritten module failed to import or validate:\n" + err)
    if now != was:
        changed = [k for k in was if was.get(k) != now.get(k)]
        return restore(f"profile contents changed for {changed[:5]}")

    print(f"[OK] {len(ordered)} profiles sorted, contents byte-identical, "
          f"validate_all() passes")
    print("[NEXT] regenerate the C++ tables:  python cpp_codegen.py -o .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
