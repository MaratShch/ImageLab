"""Inventory the plot pages in PDF/PROFILES, and classify each vector or raster.

WHY
---
`DIGITIZATION_QUEUE.md` carried two items whose blocker was an unperformed
inventory, and two numbers that disagreed with each other:

  * "Spectral dye density. CORRECTION to the '54 vector pages' figure: that was
    the corpus-wide count ... the rest of the seam needs a per-sheet
    vector/raster check before being counted as available."
  * "MTF as a curve ... 119 vector MTF pages waiting."  -- while
    ROADMAP_2026-08-17_fidelity.md §2.3 says "~156 vector documents" for the
    same job.

Neither number was reproducible and the per-sheet check had not been done. This
script does it, so the figures in the queue are measured rather than recalled.

METHOD
------
1. `pdftotext -layout` every PDF under PROFILES below a size cut (the handful of
   200-800 MB scanned books are excluded by default: they are page images with
   no vector art, so the vector/raster question is already answered for them).
   Page numbers come from the form-feed page breaks pdftotext emits.
2. Match plot titles per page: "spectral dye density", "modulation transfer",
   "diffuse rms granularity", "characteristic curve".
3. Classify each hit page VECTOR or RASTER with `pdfimages -list`: a page
   carrying an embedded image at least MIN_PLOT_PX on its short side has its
   plot drawn as a raster; a page with no such image has it drawn in vector art.

   VALIDATED against pages whose answer was already known:
     KODAK-VISION3-50D-5203 p3   -> raster  (three ~590 px images at 220 ppi)
     KODAK-VISION3-250D-5207 p3  -> raster
     KODAK VISION Color Print Film 2383 p6 -> vector (no images at all)
   The first two match the finding recorded in the queue on 2026-08-17; the
   third matches the 2383 dye-density extraction that was completed from vector
   paths. A classifier that disagreed with those three would be wrong.

4. Map hit files to database stocks by **4-digit catalogue code only**.

WHAT THIS MAPPING CAN AND CANNOT DO -- read before trusting the per-stock list
-----------------------------------------------------------------------------
Three matching strategies were tried and two were discarded after inspection:

  * Loose token matching linked `2254_TI2651.pdf` to fifty unrelated stocks
    through words like "kodak", "color", "film". Useless.
  * Type designations (500T, 200T, 50D) are SHARED across products, so 5219
    matched a 5229/5279 sheet and 5213 matched a 5217 sheet. Discarded.
  * Product words matched the wrong speed variant -- `velvia_100_datasheet.pdf`
    to FUJI_VELVIA_50. Discarded.

Only strict 4-digit catalogue codes survive, with years (1900-2100) and ISO
speeds (1600, 3200, ...) excluded -- "1600" as a speed had matched Neopan 1600
to a Konica Centuria sheet.

Two residual limits are stated rather than hidden:

  * KODAK REUSED CATALOGUE NUMBERS. `5248` matches both EASTMAN_EXR_100T_5248
    (1989-2006) and EASTMANCOLOR_5248_1953. One of those is wrong and the code
    cannot tell which. The queue's own DO-NOT-TRACE section records this hazard
    for 5245. Treat every match as a CANDIDATE requiring a look at the sheet.
  * Stocks with no 4-digit code (most still films, all the Soviet stocks) are
    not matched at all, and are listed so they can be assigned by hand.

Run:
    python plot_inventory.py                 # full inventory + per-stock map
    python plot_inventory.py --assert        # non-zero if counts moved
    python plot_inventory.py --topic dye     # one topic
    python plot_inventory.py --csv out.csv   # the page list, for a worklist

Needs poppler's pdftotext + pdfimages on PATH. Stdlib only otherwise.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import subprocess
import sys
from pathlib import Path

TOPICS = {
    # ⚠ The narrow pattern 'spectral dye densit' UNDERCOUNTS BY ~3x and misses
    # both already-adopted extractions: Kodak titles this plot "DIFFUSE
    # SPECTRAL DENSITY" on many sheets (5285 p4) and 2383 p6 carries no
    # "spectral dye" text at all. A pattern that misses the two known-good
    # pages is self-evidently wrong. Corrected 2026-08-18: 57 -> 191 vector.
    "dye":  (r'(spectral\s+dye\s+densit|diffuse\s+spectral\s+densit'
             r'|spectral\s+absorptions?\s+of\s+the\s+dyes'
             r'|dye\s+densit\w*\s+curves?)',  "spectral dye density"),
    "mtf":  (r'modulation[-\s]?transfer',   "modulation transfer / MTF"),
    "gran": (r'diffuse\s+rms\s+granularit', "diffuse rms granularity"),
    "chr":  (r'characteristic\s+curve',     "characteristic curve"),
}

MIN_PLOT_PX = 200        # short side of an embedded image that counts as a plot
MAX_PDF_MB = 20          # above this, the file is a scanned book: skip
SPEEDS = {"1000", "1250", "1600", "3200", "6400"}

#: Measured 2026-08-18. `--assert` fails if the corpus stops reproducing these,
#: which is what turns this from a one-off into something the build can check.
EXPECTED_PAGES = {"dye": (191, 28), "mtf": (199, 37),
                  "gran": (101, 39), "chr": (294, 73)}   # (vector, raster)

#: Pages whose classification is known independently; asserted every run.
GROUND_TRUTH = [
    ("KODAK-VISION3-50D-5203-7203-technical-information.pdf", 3, "raster"),
    ("KODAK-VISION3-250D-5207-7207-technical-information.pdf", 3, "raster"),
    ("KODAK VISION Color Print Film 2383.pdf", 6, "vector"),
]


def tool_ok(name: str) -> bool:
    try:
        subprocess.run([name, "-v"], capture_output=True, timeout=20)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def catalogue_codes(text: str) -> set:
    """Strict 4-digit codes, minus years and ISO speeds. See the docstring."""
    return {c for c in re.findall(r'(?<!\d)(\d{4})(?!\d)', text)
            if not (1900 <= int(c) <= 2100) and c not in SPEEDS}


def scan_text(root: Path, topics: dict) -> dict:
    """topic -> {(relpath, page)} from the text layer of every small PDF."""
    hits = {k: set() for k in topics}
    rx = {k: re.compile(v[0], re.I) for k, v in topics.items()}
    pdfs = [p for p in sorted(root.rglob("*.pdf"))
            if p.stat().st_size <= MAX_PDF_MB * (1 << 20)]
    for i, pdf in enumerate(pdfs, 1):
        try:
            r = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                               capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"  [skip] timed out: {pdf.name}", file=sys.stderr)
            continue
        rel = str(pdf.relative_to(root))
        for page, body in enumerate(
                r.stdout.decode("utf-8", "replace").split("\f"), 1):
            for k, r_ in rx.items():
                if r_.search(body):
                    hits[k].add((rel, page))
        if i % 100 == 0:
            print(f"  ... {i}/{len(pdfs)} files", file=sys.stderr)
    return hits


def classify(root: Path, pages: set) -> dict:
    """(relpath, page) -> 'vector' | 'raster' | 'error'."""
    out = {}
    for rel, page in sorted(pages):
        try:
            r = subprocess.run(
                ["pdfimages", "-list", "-f", str(page), "-l", str(page),
                 str(root / rel)], capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            out[(rel, page)] = "error"
            continue
        biggest = 0
        for line in r.stdout.decode("utf-8", "replace").splitlines()[2:]:
            t = line.split()
            if len(t) < 5:
                continue
            try:
                w, h = int(t[3]), int(t[4])
            except ValueError:
                continue
            if min(w, h) >= MIN_PLOT_PX:
                biggest = max(biggest, w * h)
        out[(rel, page)] = "raster" if biggest else "vector"
    return out


def load_profiles():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from film_profiles import FILM_PROFILES, PRINT_STOCKS
    except Exception as exc:
        print(f"[!] cannot import film_profiles ({exc}); "
              f"per-stock mapping skipped", file=sys.stderr)
        return {}
    out = {}
    for p in list(FILM_PROFILES) + list(PRINT_STOCKS):
        blob = " ".join((p.name,) + tuple(getattr(p, "aliases", ()) or ()))
        c = catalogue_codes(blob)
        if c:
            out[p.name] = c
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../../PDF/PROFILES",
                    help="corpus root (default ../../PDF/PROFILES)")
    ap.add_argument("--topic", choices=sorted(TOPICS), action="append")
    ap.add_argument("--csv", metavar="FILE", help="write the page list here")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    help="exit non-zero if the counts or ground truth move")
    ns = ap.parse_args()

    for t in ("pdftotext", "pdfimages"):
        if not tool_ok(t):
            print(f"[!] {t} not on PATH -- install poppler-utils")
            return 1
    root = Path(ns.root).resolve()
    if not root.is_dir():
        print(f"[!] corpus not found: {root}")
        return 1
    topics = {k: TOPICS[k] for k in (ns.topic or sorted(TOPICS))}

    print(f"[i] corpus {root}")
    hits = scan_text(root, topics)
    allpages = set().union(*hits.values()) if hits else set()
    print(f"[i] {len(allpages)} candidate plot pages; classifying")
    kind = classify(root, allpages)

    bad = 0
    for fname, page, want in GROUND_TRUTH:
        got = next((v for (rel, p), v in kind.items()
                    if rel.endswith(fname) and p == page), None)
        if got is None:
            print(f"  [warn] ground-truth page absent: {fname} p{page}")
        elif got != want:
            print(f"  [FAIL] {fname} p{page}: expected {want}, got {got}")
            bad += 1
    if not bad:
        print(f"[i] classifier reproduces all "
              f"{len(GROUND_TRUTH)} independently-known pages")

    prof = load_profiles()
    rows = []
    print()
    for k in topics:
        vec = sorted(p for p in hits[k] if kind.get(p) == "vector")
        ras = sorted(p for p in hits[k] if kind.get(p) == "raster")
        label = TOPICS[k][1]
        print(f"=== {label}: {len(vec)} vector pages, {len(ras)} raster pages "
              f"({len({f for f,_ in vec})} / {len({f for f,_ in ras})} files)")
        if k in EXPECTED_PAGES:
            ev, er = EXPECTED_PAGES[k]
            if (len(vec), len(ras)) != (ev, er):
                print(f"    [!] expected {ev} vector / {er} raster "
                      f"(recorded 2026-08-18)")
                bad += 1
        stock = collections.defaultdict(set)
        for rel, page in vec:
            fc = catalogue_codes(Path(rel).name)
            for nm, pc in prof.items():
                if fc & pc:
                    stock[nm].add((rel, page))
        print(f"    {len(stock)} DB stocks matched by catalogue code "
              f"(CANDIDATES -- number reuse is possible, see the docstring)")
        for nm in sorted(stock):
            loc = sorted(stock[nm])[:2]
            print("      " + f"{nm:30s} "
                  + "; ".join(f"{Path(a).name} p{b}" for a, b in loc))
        for rel, page in vec + ras:
            rows.append(dict(topic=k, kind=kind.get((rel, page)),
                             page=page, file=rel))
        print()

    if ns.csv:
        with open(ns.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=("topic", "kind", "page", "file"))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["topic"], r["file"],
                                                    r["page"])))
        print(f"[i] {len(rows)} rows written to {ns.csv}")

    if ns.do_assert and bad:
        print(f"[FAIL] {bad} discrepancy(ies)")
        return 1
    print("[OK] inventory complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
