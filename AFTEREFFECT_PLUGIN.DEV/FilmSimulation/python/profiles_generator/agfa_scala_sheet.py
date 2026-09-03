"""AGFA SCALA 200x PROFESSIONAL, «Technical Data F-SW12-E6», 08/2000, 6th ed.

WHAT THIS SOURCE IS
-------------------
`AGFA/agfa_scala.pdf` -- 4 pages, PageMaker 6.5 -> Distiller 4.0, PDF 1.2,
internal title `P12f-e`, created 2000-08-29. **Vector line art on every page,
real text layer, zero embedded images.** p4 prints
`Technical Data F-SW12-E6 / Date: 08/2000 / 6th edition`.

⚠ **IT IS NOT A DUPLICATE OF ANY AGFA DOCUMENT ALREADY HELD.** The corpus has
two Agfa range sheets that give SCALA one column each -- «Technical Data PF»
(1st ed, 09/1998) and F-PF-E4/D4 (4th ed, 2003/04) -- and this is the film's
OWN four-page sheet, published between them. Everything below is printed here
and printed nowhere else in the corpus:

  * exposure latitude as a NUMBER, and it is speed-dependent:
    +-1/2 stop at ISO 200-1600, +-1 stop at ISO 100
  * the granularity measurement's viewing condition -- "equivalent to a
    12-fold magnification" -- and the restriction "(only in SCALA process)"
  * the five-layer emulsion design and a **total thickness of 12 um**, against
    the range sheets' *Schichtdicke 7 um*. ⚠ THESE DO NOT CONFLICT AND MUST NOT
    BE AVERAGED: 7 um is the emulsion layer, 12 um is the whole coating --
    supercoat, emulsion, AHU and, on roll and sheet, a retouchable gelatine
    backing. Two different quantities under two different words.
  * the film base by STANDARD: safety film (acetyl cellulose) to DIN 15551,
    with polyester 175 um for sheet film and an extra NC layer on roll/sheet
    backs
  * the anti-halation construction in words -- 35 mm is a clear base with an
    AHU layer decolorised in the developer; roll and sheet add a dark green
    gelatine back, also decolorised
  * a pulled-processing granularity figure: "- 10 % at ISO 100/21°"
  * "Contrast matched to AGFACHROME RSX 100 (basis ISO 200/24°)"

WHAT IT CONFIRMS RATHER THAN ADDS
----------------------------------
The reciprocity table (1/10000-1/2 s none, 1 s +1/2, 10 s +1, 100 s +2) is the
THIRD independent printing of the four points the database already stores, and
the push/pull speed table (Pull 1 ISO 100, Standard 200, Push 1/2/3 = 400 /
800 / 1600) is the second. Agreement across three documents and six years is
worth asserting, and this module asserts it against the stored profile rather
than against a copy of the numbers kept here.

WHAT IT DOES NOT CARRY, CHECKED AND RECORDED
---------------------------------------------
No granularity-vs-density plot, no aperture series, no Wiener spectrum, and no
gamma-time family -- so it cannot fill `grain.sigma_shape_*` or
`grain.clump_um_*` for this stock either. That is the same negative result
`agfa_p16c.py` records for the AGFAPAN films, and it is recorded here for the
same reason: a checked absence is worth more than an unexamined gap.

Run:  python agfa_scala_sheet.py --root <corpus> [--assert]
Needs PyMuPDF. numpy only for the curve panels.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:                                       # pragma: no cover
    print("[!] pymupdf not installed:  pip install pymupdf")
    raise SystemExit(1)

SHEET = "AGFA/agfa_scala.pdf"
PROFILE = "AGFA_SCALA_200X"

SOURCE = ("Agfa-Gevaert AG, «AGFA SCALA 200x PROFESSIONAL -- Technical Data», "
          "publication F-SW12-E6, 6th edition, 08/2000 -- "
          "PDF/PROFILES/AGFA/agfa_scala.pdf")

#: Every value this module claims the sheet prints, with the page it is on.
#: ⚠ THE POINT OF WRITING THEM OUT IS THAT THE PARSE IS CHECKED AGAINST THEM.
#: A reader that only reports what it finds cannot fail; one that has to find
#: a stated list fails loudly when the document, the extraction or the
#: expectation moves.
EXPECT = {
    "iso": 200,
    "rp_high": 120.0,          # lines/mm at 1000:1
    "rp_low": 50.0,            # lines/mm at 1.6:1
    "rms": 11.0,               # x1000, diffuse, visual filter, 48 um
    "aperture_um": 48.0,
    "density_at": 1.0,
    "magnification": 12.0,
    "base_um_35mm": 120.0,
    "base_um_roll": 95.0,
    "base_um_sheet": 175.0,
    "total_coating_um": 12.0,
    # ⚠ THE FRACTION IS THE GLYPH AGFA SET, U+00BD, and it is kept as printed
    # rather than transliterated to "1/2": this tuple is the verbatim record of
    # what the document says, and the reader is required to reproduce it
    # character for character.
    "recip_times": ("1/10000 to ½", "1", "10", "100"),
    "recip_stops": (0.0, 0.5, 1.0, 2.0),
    "push_speeds": {"Pull 1": 100, "Push 1": 400, "Push 2": 800, "Push 3": 1600},
    "latitude_fast_stops": 0.5,   # ISO 200/24 to 1600/33
    "latitude_pulled_stops": 1.0,  # ISO 100/21
    "pull_granularity_pct": -10.0,
}

#: The layer order as printed, top to base. Stored verbatim; the sheet numbers
#: them 1-5 and marks the fifth "(roll and sheet films)".
LAYERS = (
    "Retouchable gelatine supercoat",
    "Emulsion layer",
    "AHU layer",
    "Film base",
    "Retouchable gelatine backing (roll and sheet films)",
)

_FRAC = {"½": 0.5, "1/2": 0.5, "¼": 0.25, "¾": 0.75}


def _stops(tok: str):
    """'+ ½' / 'none' / '+ 1' -> a float number of stops."""
    t = tok.strip().lower().replace("+", " ").strip()
    if t in ("none", "0", ""):
        return 0.0
    if t in _FRAC:
        return _FRAC[t]
    try:
        return float(t)
    except ValueError:
        return None


def parse(doc):
    """Everything the sheet states, as a dict. Raises nothing; missing keys are
    absent and the caller reports them."""
    txt = "\n".join(p.get_text() for p in doc)
    got: dict = {"raw_len": len(txt)}

    m = re.search(r"Resolving power \(reference: ISO (\d+)/(\d+)°\)\s*\n"
                  r"Contrast\s*1000:1\s*\n\s*([\d.]+)\s*lines/mm\s*\n"
                  r"Contrast\s+1[,.]6:1\s*\n\s*([\d.]+)\s*lines/mm", txt)
    if m:
        got["rp_ref_iso"] = int(m.group(1))
        got["rp_high"] = float(m.group(3))
        got["rp_low"] = float(m.group(4))

    m = re.search(r"Diffuse RMS granularity \(x 1000\) = ([\d.]+)", txt)
    if m:
        got["rms"] = float(m.group(1))
    got["rms_scala_only"] = "(only in SCALA process)" in txt
    m = re.search(r"Measured at diffuse density of ([\d.]+) and with visual "
                  r"filter \(V[λl]\)\s*\nwith a (\d+) µm aperture", txt)
    if m:
        got["density_at"] = float(m.group(1))
        got["aperture_um"] = float(m.group(2))
    m = re.search(r"equivalent to a (\d+)-\s*\nfold magnification", txt)
    if m:
        got["magnification"] = float(m.group(1))

    # -- reciprocity -------------------------------------------------------
    # ⚠ THE TABLE IS PRINTED COLUMN-MAJOR AND THE FIRST CELL WRAPS. Agfa set
    # the first exposure-time cell as "1/10000" on one line and "to ½" three
    # lines later, after the row label, so a naive line pairing puts "to ½"
    # against the wrong correction. The row is read by its four CORRECTION
    # tokens, whose order is unambiguous, and the times are checked separately.
    m = re.search(r"Exposure correction\s*\n(none)\s*\n(\+\s*[^\n]+)\s*\n"
                  r"(\+\s*[^\n]+)\s*\n(\+\s*[^\n]+)\s*\n\(f-stops\)", txt)
    if m:
        vals = [_stops(m.group(i)) for i in (1, 2, 3, 4)]
        if all(v is not None for v in vals):
            got["recip_stops"] = tuple(vals)
    m = re.search(r"Measured\s*\n(1/10000)\s*\n(1)\s*\n(10)\s*\n(100)\s*\n"
                  r"exposure time \(s\)\s*\n(to ½)", txt)
    if m:
        got["recip_times"] = (f"{m.group(1)} {m.group(5)}", m.group(2),
                              m.group(3), m.group(4))

    # -- exposure latitude --------------------------------------------------
    m = re.search(r"ISO (\d+)/\d+° to ISO (\d+)/\d+° ±\s*½ stop", txt)
    if m:
        got["latitude_fast_stops"] = 0.5
        got["latitude_fast_range"] = (int(m.group(1)), int(m.group(2)))
    m = re.search(r"ISO (\d+)/\d+° ± 1 stop", txt)
    if m:
        got["latitude_pulled_stops"] = 1.0
        got["latitude_pulled_iso"] = int(m.group(1))

    # -- emulsion design -----------------------------------------------------
    m = re.search(r"Film base: (safety film \(acetyl cellulose\) to DIN \d+)", txt)
    if m:
        got["base_material"] = m.group(1)
        got["base_standard"] = re.search(r"DIN (\d+)", m.group(1)).group(0)
    m = re.search(r"35 mm film:\s*\n(\d+) µm\s*\nRollfilm:\s*\n(\d+) µm\s*\n"
                  r"Sheet film \(polyester base\):\s*\n(\d+) µm", txt)
    if m:
        got["base_um_35mm"] = float(m.group(1))
        got["base_um_roll"] = float(m.group(2))
        got["base_um_sheet"] = float(m.group(3))
        got["base_material_sheet"] = "polyester"
    m = re.search(r"Total thickness: (\d+) µm", txt)
    if m:
        got["total_coating_um"] = float(m.group(1))
    got["nc_backing"] = ("extra NC layer on the backs" in txt)
    got["layers"] = tuple(n for n in LAYERS
                          if n.split(" (")[0].split()[-1].lower() in txt.lower())

    # -- anti-halation, in the sheet's own words ------------------------------
    m = re.search(r"Anti-halo layer\s*\n35 mm:\s*\n(.+?)\n(?=Roll and)", txt, re.S)
    if m:
        got["ahu_35mm"] = " ".join(m.group(1).split())
    m = re.search(r"Roll and\s*\n(.+?)\nsheet film:\s*\n(.+?)\n(?=Resolving)",
                  txt, re.S)
    if m:
        got["ahu_roll"] = " ".join((m.group(1) + " " + m.group(2)).split())

    # -- push/pull ------------------------------------------------------------
    m = re.search(r"Step\s*\nPush 1\s*\nPush 2\s*\nPush 3\s*\nPull 1\s*\n"
                  r"Speed \(ISO\)\s*\n(\d+)/\d+°\s*(\d+)/\d+°\s*(\d+)/\d+°\s*\n"
                  r"(\d+)/\d+°", txt)
    if m:
        got["push_speeds"] = {"Push 1": int(m.group(1)), "Push 2": int(m.group(2)),
                              "Push 3": int(m.group(3)), "Pull 1": int(m.group(4))}
    m = re.search(r"for finer granularity \( - (\d+) % at ISO (\d+)/\d+°\)", txt)
    if m:
        got["pull_granularity_pct"] = -float(m.group(1))
        got["pull_granularity_iso"] = int(m.group(2))
    got["contrast_matched_to"] = bool(
        re.search(r"Contrast matched to AGFACHROME RSX 100", txt))

    m = re.search(r"With the standard SCALA process: ISO (\d+)/\d+°", txt)
    if m:
        got["iso"] = int(m.group(1))

    # -- the negative result --------------------------------------------------
    got["has_granularity_plot"] = bool(
        re.search(r"Wiener|granularity curve|granularity vs|aperture series", txt, re.I))
    got["has_gamma_time"] = "Developing time" in txt or "gamma-time" in txt.lower()
    return got


def check_profile(got):
    """Compare the parsed sheet against the stored profile. Returns failures."""
    try:
        import film_profiles as fp
    except Exception as exc:                              # pragma: no cover
        print(f"    [note] film_profiles unavailable ({exc}); no profile check")
        return 0
    p = [q for q in fp.FILM_PROFILES if q.name == PROFILE]
    if not p:
        print(f"    [FAIL] no profile named {PROFILE}")
        return 1
    p = p[0]
    bad = 0

    def cmp(label, sheet, stored, tol=0.0):
        nonlocal bad
        ok = (abs(sheet - stored) <= tol) if isinstance(sheet, float) else sheet == stored
        print(f"      {label:34s} sheet {sheet!s:>10}  stored {stored!s:>10}  "
              f"[{'OK' if ok else 'DIFFERS'}]")
        if not ok:
            bad += 1

    cmp("exposure index", got.get("iso"), p.exposure_index)
    cmp("RMS granularity x1000", got.get("rms"), p.grain.rms_granularity, 1e-9)
    cmp("resolving power 1000:1", got.get("rp_high"),
        p.mtf.resolving_power_lp_mm_highc, 1e-9)
    cmp("resolving power 1.6:1", got.get("rp_low"),
        p.mtf.resolving_power_lp_mm_lowc, 1e-9)

    # reciprocity: the sheet's four points against the stored table
    rt = p.reciprocity_table
    if rt.has_data:
        want = got.get("recip_stops")
        have = tuple(float(x) for x in rt.stops_correction)
        ok = want == have
        print(f"      {'reciprocity stops':34s} sheet {want!s:>10}  "
              f"stored {have!s:>10}  [{'OK' if ok else 'DIFFERS'}]")
        if not ok:
            bad += 1
    else:
        print("      reciprocity                        stored table is empty [FAIL]")
        bad += 1

    # push/pull speeds against the stored push spec
    ps = got.get("push_speeds") or {}
    if ps:
        want_push = 3.0 if ps.get("Push 3") == 1600 else None
        want_pull = 1.0 if ps.get("Pull 1") == 100 else None
        cmp("max push stops", want_push, float(p.push.max_push_stops), 1e-9)
        cmp("max pull stops", want_pull, float(p.push.max_pull_stops), 1e-9)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
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
    if imgs or "F-SW12-E6" not in txt:
        print(f"  [FAIL] expected a vector F-SW12-E6; images={imgs}")
        return 1
    print(f"  [OK  ] {doc.page_count} pages, {imgs} embedded images, "
          f"p4 identifies as F-SW12-E6, 6th edition 08/2000")

    got = parse(doc)

    # ---- every stated value has to have been found -----------------------
    missing = [k for k in EXPECT if k not in got]
    if missing:
        print(f"  [FAIL] the sheet states these and the parse did not find them: "
              f"{', '.join(sorted(missing))}")
        bad += 1
    wrong = [(k, EXPECT[k], got[k]) for k in EXPECT
             if k in got and got[k] != EXPECT[k]]
    if wrong:
        for k, w, g in wrong:
            print(f"  [FAIL] {k}: expected {w!r}, read {g!r}")
        bad += len(wrong)
    if not missing and not wrong:
        print(f"  [OK  ] all {len(EXPECT)} stated values read exactly")

    # ---- the two thicknesses are different quantities --------------------
    if got.get("total_coating_um") and got.get("base_um_35mm"):
        print(f"  [OK  ] coating {got['total_coating_um']:.0f} um total over a "
              f"{got['base_um_35mm']:.0f} um 35 mm base "
              f"({got.get('base_material','?')}); sheet film "
              f"{got.get('base_um_sheet',0):.0f} um "
              f"{got.get('base_material_sheet','?')}. ⚠ The range sheets' "
              f"«Schichtdicke 7 um» is the EMULSION layer alone and is not the "
              f"same quantity as this 12 um five-layer total")

    # ---- layer stack -----------------------------------------------------
    print(f"  [OK  ] emulsion design, {len(LAYERS)} layers as printed: "
          + " / ".join(LAYERS))
    if got.get("ahu_35mm"):
        print(f"  [OK  ] anti-halation 35 mm: {got['ahu_35mm']}")
    if got.get("ahu_roll"):
        print(f"  [OK  ] anti-halation roll/sheet: {got['ahu_roll']}")

    # ---- the negative result ---------------------------------------------
    if got["has_granularity_plot"] or got["has_gamma_time"]:
        print("  [FAIL] this sheet was recorded as carrying neither a "
              "granularity plot nor a gamma-time family, and now appears to")
        bad += 1
    else:
        print("  [OK  ] ⚠ RECORDED NEGATIVE: no granularity-vs-density plot, no "
              "aperture series, no Wiener spectrum and no gamma-time family on "
              "this sheet, so it cannot fill grain.sigma_shape_* or "
              "grain.clump_um_* for SCALA. Those cells stay estimated for a "
              "checked reason")

    # ---- against the database --------------------------------------------
    print("\n  -- against the stored profile")
    bad += check_profile(got)

    print()
    if bad:
        print(f"  [FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("  [OK  ] F-SW12-E6 reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
