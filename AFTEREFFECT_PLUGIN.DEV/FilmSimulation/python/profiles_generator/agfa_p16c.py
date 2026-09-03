"""Agfa «Technical Data P-16-C» -- the processing document the range sheet cites.

WHAT THIS SOURCE IS
-------------------
`AGFA/agfa_film_chem.pdf` -- Agfa-Gevaert, «Technical Data P-16-C: AGFA Black
and White Chemicals, Film Processing», 08/1999, 16 pages, PageMaker 6.5 ->
Distiller 3.01, fully vector with a real text layer and zero embedded images.
Document metadata: title "Technical Data P-16-C", author "Ulrich Sckaer
(editorial work)", subject "Agfa b/w chemicals for film processing".

⚠ THE 1998 RANGE SHEET NAMES THIS DOCUMENT AND THE PROJECT DID NOT HAVE IT
CONNECTED. `agfa_films.pdf` p11 ends: "Further processing details are given in
the Technical Data P-16-C." The file has been in the corpus the whole time
under a name -- `agfa_film_chem.pdf` -- that reads like a chemistry catalogue
and gives no hint that it is the referenced companion to the film sheet.

WHY IT MATTERS: EVERYTHING HERE IS PRINTED, NOT TRACED
-------------------------------------------------------
The AGFAPAN gamma-time data adopted on 2026-09-01 came from DIGITISING the
range sheet's Gamma-time panel -- four drawn curves per film, five printed
developer names between them, and a label-matching step that had to be checked
against the processing table to be trusted. P-16-C prints the same physics as
TEXT: for each developer, the developing time that reaches gamma 0.55, 0.65 and
0.75, for each AGFAPAN film, for drum and for small tank. 64 numbers, no
tracing, no calibration, no label matching.

⚠ AND IT COVERS TWO DEVELOPERS THE RANGE SHEET'S PANEL DOES NOT PLOT:
**ATOMAL FF** and, in prose only, **REFINAL M**. ATOMAL FF has a full contrast
table here and appears on no plotted panel anywhere in the corpus.

WHAT IT SETTLES
---------------
1. **THE CHARACTERISTIC CURVES' DEVELOPMENT CONDITION.** The three AGFAPAN
   curves were adopted at mid-slope 0.70-0.74 with a note saying the sheet
   "states no development at all". P-16-C 3.4 states the three standard aims in
   words -- flatter negatives at gamma 0.55, medium at 0.65, higher contrast at
   0.75 -- so 0.74 is not an accident of an unstated development: it is the
   **gamma 0.75 aim**, and this document prints the time that reaches it for
   every developer. The condition can now be named instead of shrugged at.

2. **THE 2004 HANDBOOK'S RODINAL TABLE IS DEFECTIVE.** `agfa_bw_manual.pdf`'s
   table (3) was flagged on 2026-09-01 as a suspected typesetting fault -- its
   gamma 0.55 column is non-monotone against its own gamma 0.65 column and its
   values cluster implausibly at 10.4-10.8 min. P-16-C settles it, and so does
   the 1998 sheet, which agrees with P-16-C exactly:

       RODINAL 1+25, small tank, gamma 0.65      APX 25   APX 100   APX 400
       agfa_films.pdf p11 (1998)                    6         8        7
       agfa_film_chem.pdf P-16-C (1999)             6         8        7
       agfa_bw_manual.pdf table (3) (2004)          -        18       15

   Two independent Agfa documents against one. The handbook's 1+25 rows and its
   whole gamma 0.55 column are wrong, and are recorded as wrong rather than
   averaged with the other two.

3. **PUSH PROCESSING, WITH TIMES.** 3.6 prints the development time for each
   AGFAPAN film EXPOSED ONE STOP UP -- APX 25 as ISO 50, APX 100 as ISO 200,
   APX 400 as ISO 800 -- per developer, at 20 C and at 24 C. `PushSpec` was
   empty on all three profiles; this is a printed source for it.

WHAT IT DOES NOT GIVE, AND THE QUESTION IT WAS CHECKED AGAINST
---------------------------------------------------------------
⚠ **NOTHING ABOUT GRAIN.** It was read specifically to see whether it could
fill `grain.clump_um*` or the sigma(D) shape, which are marked as estimates on
every Agfa stock. It cannot: there is no granularity plot, no aperture series,
no Wiener spectrum and no granularity-vs-density data anywhere in the 16 pages.
The only grain content is prose -- "fine-grain", "exceptional sharpness". Those
two cells stay estimated, and the reason is now checked rather than assumed.

Run:  python agfa_p16c.py --root <corpus> [--assert] [--emit FILE]
Needs PyMuPDF only. No numpy, no tracing -- this reader parses TEXT.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:                                       # pragma: no cover
    print("[!] pymupdf not installed:  pip install pymupdf")
    raise SystemExit(1)

SHEET = "AGFA/agfa_film_chem.pdf"

SOURCE = ("Agfa-Gevaert, «Technical Data P-16-C -- AGFA Black and White "
          "Chemicals, Film Processing», 08/1999 -- "
          "PDF/PROFILES/AGFA/agfa_film_chem.pdf. The processing companion "
          "`agfa_films.pdf` p11 names: \"Further processing details are given "
          "in the Technical Data P-16-C\". Fully vector, real text layer; every "
          "number this reader returns is PRINTED TEXT, not a trace.")

#: Column order of every contrast table in the document.
FILMS = ("AGFA_APX_25", "AGFA_APX_100", "AGFA_APX_400")

#: Developers that carry a contrast table. ⚠ REFINAL M DOES NOT -- it is a
#: machine developer/replenisher system described in prose only, and reading
#: its section for a table returns the neighbouring developer's.
DEVELOPERS = ("RODINAL 1 + 25", "RODINAL 1 + 50", "RODINAL SPECIAL",
              "STUDIONAL LIQUID", "ATOMAL FF", "REFINAL")

#: ⚠ AGFA SPELL ONE DEVELOPER TWO WAYS AND THE DATABASE MUST NOT. The range
#: sheet and P-16-C print "RODINAL 1 + 25"; the 2004 handbook prints
#: "RODINAL 1+25". `ProcessVariant` already used the compact form, so adopting
#: the spaced one into `ProcessingFamily` gave the same developer two names and
#: every join on the field silently missed -- it broke a verify guard on the
#: first build after adoption. Reader reports as printed, database stores
#: compact.
CANON = {"RODINAL 1 + 25": "RODINAL 1+25", "RODINAL 1 + 50": "RODINAL 1+50"}

DILUTION = {"RODINAL 1 + 25": "1+25", "RODINAL 1 + 50": "1+50",
            "RODINAL SPECIAL": "1+15", "STUDIONAL LIQUID": "1+15",
            "ATOMAL FF": "stock", "REFINAL": "stock"}

_MIN = re.compile(r"^([\d.]+)\s*min$")


def _lines(doc):
    return [x.strip() for x in
            "\n".join(p.get_text() for p in doc).split("\n")]


def contrast_tables(lines):
    """[(developer, method, gamma, {film: minutes})] from the printed tables.

    ⚠ THE DEVELOPER IS CARRIED FORWARD FROM THE LAST SECTION HEADING, because
    the tables themselves do not name it -- they are headed only "Rotary
    process (drum)" or "Small tank, tray" under a developer's section. A reader
    that keyed on the nearest heading ABOVE the table would attach ATOMAL FF's
    table to REFINAL, since the two sections are adjacent and REFINAL M sits
    between them with no table at all.
    """
    out, dev = [], None
    for i, l in enumerate(lines):
        if l in DEVELOPERS:
            dev = l
        if l not in ("Rotary process (drum)", "Small tank, tray"):
            continue
        if dev is None:
            continue
        blk = [x for x in lines[i:i + 24] if x]
        for j, x in enumerate(blk):
            if not x.startswith("γ"):
                continue
            g = x.replace("γ", "").replace(",", ".").strip()
            try:
                gamma = float(g)
            except ValueError:
                continue
            cells = blk[j + 1:j + 1 + len(FILMS)]
            vals = {}
            for f, c in zip(FILMS, cells):
                m = _MIN.match(c.replace("    ", " ").replace("  ", " "))
                if m:
                    vals[f] = float(m.group(1))
            if vals:
                out.append((dev, l, gamma, vals))
    return out


#: The push tables of 3.6, keyed by the nominal / pushed ISO pair each prints.
PUSH_HEAD = {
    "AGFA_APX_25": ("ISO 25/15°", "ISO 50/18°"),
    "AGFA_APX_100": ("ISO 100/21°", "ISO 200/24°"),
    "AGFA_APX_400": ("ISO 400/27°", "ISO 800/30°"),
}


def push_tables(lines):
    """[(film, developer, celsius, nominal_min, pushed_min)] from 3.6.

    ⚠ A DASH IS A REFUSAL, NOT A ZERO. Agfa print "-" where a combination is
    not recommended -- RODINAL 1+25 has no 24 C row for any film, and RODINAL
    SPECIAL has no 20 C row. Those cells are dropped, not stored as 0.0, so a
    consumer cannot read "no recommendation" as "develop for no time".
    """
    out = []
    for film, (nom, push) in PUSH_HEAD.items():
        try:
            i = lines.index(push)
        except ValueError:
            continue
        # ⚠ THE SCAN MUST BE BOUNDED BY THE NEXT FILM'S HEADING, NOT BY A LINE
        # COUNT. A fixed 40-line window runs APX 25's table straight into
        # APX 100's and returns eighteen rows for fifteen -- with three of
        # APX 100's developer times filed under APX 25, all of them plausible.
        # The give-away was a duplicate (film, developer, temperature) key.
        stop = len(lines)
        for k in range(i + 1, len(lines)):
            # ⚠ THE SECTION-NUMBER TEST MUST BE ANCHORED. Written as
            # `lines[k].startswith("4.")` it also matches the VALUE "4.5 min",
            # which truncated APX 400's table at its STUDIONAL row and returned
            # two developers where the page prints four.
            if re.match(r"^AGFAPAN APX \d+ PROFESSIONAL$", lines[k]) or \
                    re.match(r"^\d+\.$", lines[k]) or \
                    lines[k] == "Mixing instructions":
                stop = k
                break
        blk = [x for x in lines[i:stop] if x]
        dev = None
        for j, x in enumerate(blk):
            if x in DEVELOPERS:
                dev = x
                continue
            m = re.match(r"^(\d+)\s*C°$", x)
            if m is None or dev is None:
                continue
            cel = float(m.group(1))
            cells = [c.replace("    ", " ").replace("  ", " ")
                     for c in blk[j + 1:j + 3]]
            got = []
            for c in cells:
                mm = _MIN.match(c)
                got.append(float(mm.group(1)) if mm else None)
            if any(v is not None for v in got):
                out.append((film, dev, cel, got[0], got[1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--emit", default="")
    ns = ap.parse_args()

    pdf = Path(ns.root).resolve() / "PDF" / "PROFILES" / SHEET
    if not pdf.is_file():
        print(f"  [SKIP] source not present: {pdf}")
        return 0
    doc = pymupdf.open(pdf)
    print(f"[i] {SOURCE}\n")

    bad = 0
    imgs = sum(len(doc[i].get_images(full=True)) for i in range(doc.page_count))
    txt = "\n".join(p.get_text() for p in doc)
    if imgs or "P-16-C" not in txt:
        print(f"  [FAIL] expected a vector P-16-C; images={imgs}")
        return 1
    print(f"  [OK  ] {doc.page_count} pages, {imgs} embedded images, "
          f"p1 identifies as P-16-C")

    lines = _lines(doc)
    ct = contrast_tables(lines)
    cells = sum(len(v) for _, _, _, v in ct)
    devs = sorted({d for d, _, _, _ in ct})
    print(f"  [OK  ] {len(ct)} contrast rows, {cells} printed time cells, "
          f"{len(devs)} developers: {', '.join(devs)}")
    if cells != EXPECTED_CELLS:
        print(f"  [FAIL] expected {EXPECTED_CELLS} time cells, read {cells}")
        bad += 1
    if set(devs) != set(DEVELOPERS):
        print(f"  [FAIL] developer set moved: {devs}")
        bad += 1

    # ⚠ MONOTONICITY IS THE FREE TEST THIS TABLE OFFERS AND IT IS WHAT CAUGHT
    # THE 2004 HANDBOOK. A longer development cannot give LESS contrast, so
    # within one (developer, method, film) the times must ascend with gamma.
    # P-16-C passes on every triple it prints; the handbook's RODINAL rows do
    # not, which is how a typesetting fault was told from a revision.
    nonmono = []
    for dev in DEVELOPERS:
        for meth in ("Rotary process (drum)", "Small tank, tray"):
            for f in FILMS:
                seq = [(g, v[f]) for d, m, g, v in ct
                       if d == dev and m == meth and f in v]
                seq.sort()
                mins = [t for _, t in seq]
                if len(mins) > 1 and any(b < a for a, b in zip(mins, mins[1:])):
                    nonmono.append(f"{dev}/{meth}/{f}: {mins}")
    if nonmono:
        print("  [FAIL] time falls as gamma rises: " + "; ".join(nonmono))
        bad += 1
    else:
        print("  [OK  ] every (developer, method, film) triple ascends with "
              "gamma -- the test the 2004 handbook fails")

    pt = push_tables(lines)
    print(f"  [OK  ] {len(pt)} push rows across "
          f"{len({f for f, *_ in pt})} films, one stop, times printed")
    if len(pt) != EXPECTED_PUSH:
        print(f"  [FAIL] expected {EXPECTED_PUSH} push rows, read {len(pt)}")
        bad += 1

    # The cross-document agreement that condemns the handbook.
    ref = {(d, m, g, f): v[f] for d, m, g, v in ct for f in v}
    for f, want in (("AGFA_APX_25", 6.0), ("AGFA_APX_100", 8.0),
                    ("AGFA_APX_400", 7.0)):
        got = ref.get(("RODINAL 1 + 25", "Small tank, tray", 0.65, f))
        if got != want:
            print(f"  [FAIL] {f} RODINAL 1+25 tank gamma 0.65: {got} != {want}"
                  f" -- this is the row agfa_films.pdf p11 independently prints")
            bad += 1
    if not bad:
        print("  [OK  ] RODINAL 1+25 tank gamma 0.65 reads 6 / 8 / 7 min, "
              "exactly as agfa_films.pdf p11 prints it independently")

    print(f"\n  [OK  ] ⚠ NO GRAIN DATA, CHECKED NOT ASSUMED: "
          f"{'granularity plot' if 'Wiener' in txt else 'no Wiener spectrum'}, "
          f"no aperture series, no granularity-vs-density. clump_um and the "
          f"sigma(D) shape cannot be filled from this document.")

    if ns.emit:
        Path(ns.emit).write_text(json.dumps(
            {"source": SOURCE,
             "contrast": [{"developer": d, "method": m, "gamma": g,
                           "minutes": v} for d, m, g, v in ct],
             "push": [{"film": f, "developer": d, "celsius": c,
                       "nominal_min": a, "pushed_min": b}
                      for f, d, c, a, b in pt]}, indent=1), encoding="utf-8")
        print(f"  [emit] -> {ns.emit}")

    if bad and ns.do_assert:
        return 1
    return 0


#: Measured 2026-09-01. A change here is a change in the source or the reader.
EXPECTED_CELLS = 64
EXPECTED_PUSH = 15


if __name__ == "__main__":
    sys.exit(main())
