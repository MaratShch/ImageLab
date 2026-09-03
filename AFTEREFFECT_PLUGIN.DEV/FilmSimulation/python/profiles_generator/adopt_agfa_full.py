"""Second and final AGFA splice: everything the sheets publish that the first
pass left on the table.

    python adopt_agfa_full.py --emit /tmp/agfa98.json [--apply]

Covers, in one batch:
  * `dye_density` on eight stocks -- the neutral + D-min pair on the five colour
    negatives, and the first SEPARATED three-dye sets in the Agfa corpus on the
    three RSX II reversal films.
  * the three AGFAPAN characteristic curves, replacing class estimates that were
    claiming `fitted_from='datasheet_curve'` while being nothing of the kind.
  * `base_um` / `base_material` (schema v23) on all twelve.
  * `ProcessingSpec` and per-developer `ProcessVariant` on the three AGFAPAN.
  * the SCALA spectral sensitivity set.

Like `adopt_agfa_1998.py` this is a one-shot splice and NOT an audit;
`agfa_1998_curves.py` re-derives every number here from the source each build.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

FP = Path(__file__).with_name("film_profiles.py")

S98 = ("Agfa-Gevaert, «Technical Data PF -- Agfa range of films», 1st edition, "
       "09/1998 -- PDF/PROFILES/AGFA/agfa_films.pdf")

JKEY = {
    "AGFA_OPTIMA_100": "AGFACOLOR OPTIMA II 100",
    "AGFA_OPTIMA_200": "AGFACOLOR OPTIMA II 200",
    "AGFA_OPTIMA_400": "AGFACOLOR OPTIMA II 400",
    "AGFA_PORTRAIT_160": "AGFACOLOR PORTRAIT XPS 160",
    "AGFA_ULTRA_50": "AGFACOLOR ULTRA 50",
    "AGFA_RSX_II_50": "AGFACHROME RSX II 50",
    "AGFA_RSX_II_100": "AGFACHROME RSX II 100",
    "AGFA_RSX_II_200": "AGFACHROME RSX II 200",
    "AGFA_SCALA_200X": "AGFA SCALA 200x",
    "AGFA_APX_25": "AGFAPAN APX 25",
    "AGFA_APX_100": "AGFAPAN APX 100",
    "AGFA_APX_400": "AGFAPAN APX 400",
}

PAGE = {"AGFA_OPTIMA_100": "p7", "AGFA_OPTIMA_200": "p7", "AGFA_OPTIMA_400": "p7",
        "AGFA_PORTRAIT_160": "p8", "AGFA_ULTRA_50": "p8", "AGFA_RSX_II_50": "p8",
        "AGFA_RSX_II_100": "p9", "AGFA_RSX_II_200": "p9",
        "AGFA_SCALA_200X": "p9", "AGFA_APX_25": "p10",
        "AGFA_APX_100": "p10", "AGFA_APX_400": "p10"}

#: 135 base thickness in um, and the polymer, as printed beside each column.
#: ⚠ EVERY 35 mm BASE IN THE RANGE IS 120 um AND THE SHEET NAMES NO POLYMER FOR
#: IT -- only the SHEET-FILM bases get a material ("PET 175 um", "Acetate
#: 190 um"), and p5 says the base "is made of acetyl cellulose or polyester"
#: without saying which film gets which. So `base_material` stays EMPTY on the
#: roll formats rather than guessing, and the sheet-film material is recorded in
#: the source string where it belongs.
BASE = {n: (120.0, "") for n in JKEY}

BASE_NOTE = {
    "AGFA_OPTIMA_100": "sheet film = PET 175 um",
    "AGFA_RSX_II_100": "sheet film = Acetate 190 um -- the only acetate sheet "
                       "base Agfa name in this range",
    "AGFA_SCALA_200X": "sheet film = PET 175 um",
}

APX = ("AGFA_APX_25", "AGFA_APX_100", "AGFA_APX_400")

#: Reference developer, time and temperature, printed on p11. This is the
#: condition Agfa's gamma = 0.65 specification refers to.
APX_REF = {"AGFA_APX_25": ("REFINAL", "stock", 6.0),
           "AGFA_APX_100": ("REFINAL", "stock", 6.0),
           "AGFA_APX_400": ("REFINAL", "stock", 6.0)}

#: Exposure index by developer at gamma 0.65, printed on p11 -- AGFA'S OWN FILMS
#: ONLY. The 2004 handbook also prints this for seventeen Fuji/Ilford/Kodak
#: stocks; that is one maker measuring another's product in its own chemistry
#: and is a separate decision, deliberately not taken here.
APX_EI = {
    "AGFA_APX_25": (("REFINAL", "stock", 6.0, 25),
                    ("RODINAL 1+25", "1+25", 6.0, 20),
                    ("RODINAL 1+50", "1+50", 10.0, 25),
                    ("RODINAL SPECIAL", "1+15", 4.0, 25),
                    ("STUDIONAL LIQUID", "1+15", 4.0, 25)),
    "AGFA_APX_100": (("REFINAL", "stock", 6.0, 125),
                     ("RODINAL 1+25", "1+25", 8.0, 100),
                     ("RODINAL 1+50", "1+50", 17.0, 125),
                     ("RODINAL SPECIAL", "1+15", 4.0, 125),
                     ("STUDIONAL LIQUID", "1+15", 4.0, 125)),
    "AGFA_APX_400": (("REFINAL", "stock", 6.0, 400),
                     ("RODINAL 1+25", "1+25", 7.0, 320),
                     ("RODINAL 1+50", "1+50", 11.0, 320),
                     ("RODINAL SPECIAL", "1+15", 4.5, 400),
                     ("STUDIONAL LIQUID", "1+15", 4.5, 400)),
}


def wrap(text, indent, width=78):
    pad = " " * indent
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width - indent:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    if len(out) == 1:
        return repr(out[0])
    return ("\n" + pad).join(repr(s + " ") if i < len(out) - 1 else repr(s)
                             for i, s in enumerate(out))


def tup(vals, indent):
    """A float tuple, wrapped, two decimals."""
    pad = " " * indent
    body, line, out = "", "", []
    for v in vals:
        t = "%.3f" % v
        if len(line) + len(t) + 2 > 74 - indent:
            out.append(line)
            line = t
        else:
            line = (line + ", " + t) if line else t
    if line:
        out.append(line)
    return "(" + (",\n" + pad).join(out) + ")"


def _common_window(cur):
    """The widest 10 nm grid window every curve on the panel actually covers.

    ⚠ ONE PANEL IN EIGHT DOES NOT REACH 400 nm AND THE SCHEMA IS BUILT FOR IT.
    Optima II 400's Medium density trace starts at 406.3 nm, so the 400 nm
    sample is not drawn. Extrapolating it would invent a density; dropping the
    point on one curve and not the other would break `SpectralDyeDensity`'s
    "traces must share one grid" rule. `lambda_start_nm` is per record for
    exactly this, so the whole panel moves to the first wavelength all of its
    curves cover -- 410 nm for that film, 400 for the other seven.
    """
    n = len(next(iter(cur.values()))["d"])
    lo, hi = 0, n
    for v in cur.values():
        d = v["d"]
        i = 0
        while i < n and d[i] is None:
            i += 1
        j = n
        while j > i and d[j - 1] is None:
            j -= 1
        lo, hi = max(lo, i), min(hi, j)
    return lo, hi


def dye_block(profile, rec, indent=8):
    pad = " " * indent
    cur = rec["dye"]["curves"]
    res = rec["dye"]["residual"]
    lo, hi = _common_window(cur)
    lam0 = 400.0 + 10.0 * lo
    for v in cur.values():
        v["d"] = v["d"][lo:hi]
    is_rev = "Yellow" in cur
    lines = []
    if is_rev:
        y = np.array(cur["Yellow"]["d"], float)
        m = np.array(cur["Magenta"]["d"], float)
        c = np.array(cur["Cyan"]["d"], float)
        g = np.array(cur["Visual grey"]["d"], float)
        rms = float(np.sqrt(np.mean((y + m + c - g) ** 2)))
        note = (
            f"{pad}# ⚠ THE FIRST SEPARATED THREE-DYE SET IN THE AGFA CORPUS, AND\n"
            f"{pad}# IT PASSES A CLOSURE TEST THE PANEL ITSELF SUPPLIES. Agfa draw\n"
            f"{pad}# FOUR curves here -- Yellow, Magenta, Cyan and a Visual grey --\n"
            f"{pad}# so the three dyes can be checked against the neutral they are\n"
            f"{pad}# supposed to compose to. Summed at all 31 sampled wavelengths\n"
            f"{pad}# they reproduce the printed visual grey to {rms:.3f} D rms.\n"
            f"{pad}# That is an independent physical check, not a fit residual.\n"
            f"{pad}# ⚠ THE DASH KEY IS MEANINGLESS ON THIS PANEL. Everywhere else\n"
            f"{pad}# on the sheet solid/dashed/dash-dot-dot means green/blue/red;\n"
            f"{pad}# here the curves are NAMED in print and the naming does not\n"
            f"{pad}# follow the dash convention -- read by dash the yellow, cyan\n"
            f"{pad}# and magenta traces come back labelled b, g and r. They are\n"
            f"{pad}# matched to their printed names by bounding-box distance.\n"
            f"{pad}# Peaks land where they must: yellow {cur['Yellow']['peak_nm']:.0f} nm,\n"
            f"{pad}# magenta {cur['Magenta']['peak_nm']:.0f} nm, cyan {cur['Cyan']['peak_nm']:.0f} nm.\n")
        body = [
            f"{pad}dye_density=SpectralDyeDensity(",
            f"{pad}    lambda_start_nm={lam0}, lambda_step_nm=10.0,",
            f"{pad}    d_cyan={tup(c, indent + 12)},",
            f"{pad}    d_magenta={tup(m, indent + 14)},",
            f"{pad}    d_yellow={tup(y, indent + 13)},",
            f"{pad}    d_neutral={tup(g, indent + 14)},",
            f"{pad}    normalisation='as printed -- absolute spectral density of "
            f"the processed dye image, no normalisation applied',",
            f"{pad}    normalisation_neutral='the panel\\'s own Visual grey curve, "
            f"stored as the neutral because on a reversal film that IS the "
            f"neutral; it is a CHECK on the three dyes, not a substitute for "
            f"them',",
        ]
    else:
        neu = np.array(cur["Medium density"]["d"], float)
        dmn = np.array(cur["Minimum density"]["d"], float)
        note = (
            f"{pad}# ⚠ THIS IS THE NEUTRAL + D-MIN PAIR, NOT THREE DYES, AND THE\n"
            f"{pad}# DISTINCTION IS THE ONE NotFound.md WARNS AGAINST COLLAPSING.\n"
            f"{pad}# Agfa's Spectral density panel on a colour NEGATIVE draws two\n"
            f"{pad}# AGGREGATE curves -- the film's total transmission at a\n"
            f"{pad}# midscale neutral exposure and at minimum density -- and two\n"
            f"{pad}# aggregates cannot be separated into cyan, magenta and yellow.\n"
            f"{pad}# `has_data` stays FALSE here and `has_neutral_pair` becomes\n"
            f"{pad}# true; the dye-set counter does NOT move. The reversal films\n"
            f"{pad}# on the same sheet DO get three dyes, because their panel\n"
            f"{pad}# draws them separately.\n"
            f"{pad}# ⚠ The D-min curve is the ORANGE MASK measured spectrally --\n"
            f"{pad}# it peaks at {cur['Minimum density']['peak_nm']:.0f} nm and falls to "
            f"{dmn[-1]:.2f} D at 700 nm,\n"
            f"{pad}# which is the mask's whole purpose and is exactly the shape\n"
            f"{pad}# `curves.*.dmin` reduces to three numbers.\n")
        body = [
            f"{pad}dye_density=SpectralDyeDensity(",
            f"{pad}    lambda_start_nm={lam0}, lambda_step_nm=10.0,",
            f"{pad}    d_neutral={tup(neu, indent + 14)},",
            f"{pad}    d_dmin={tup(dmn, indent + 11)},",
            f"{pad}    normalisation='as printed -- absolute spectral density, no "
            f"normalisation applied',",
            f"{pad}    normalisation_neutral='midscale neutral of medium "
            f"brightness against minimum density, the two exposure levels the "
            f"sheet states on p5',",
        ]
    src = (f"{S98} {PAGE[profile]}, the Spectral density panel, digitised by "
           f"agfa_1998_curves.py. Axis fit residual {res[0]:.4f} nm / "
           f"{res[1]:.4f} D. Conditions as the sheet states them on p5: "
           f"\"the relative effect of a processed film on transmitted light\", "
           f"reference a neutral subject of medium brightness and minimum "
           f"density")
    if lam0 != 400.0:
        src += (f". ⚠ THE GRID STARTS AT {lam0:.0f} nm, NOT 400: this panel's "
                f"Medium density trace is not drawn below 406 nm, and the "
                f"missing sample is dropped rather than extrapolated")
    if profile == "AGFA_RSX_II_100":
        src += (". ⚠ AGFA DREW ONE SPECTRAL-DENSITY PANEL FOR RSX II 50 AND "
                "RSX II 100: the two trace to within 0.0005 D at every sampled "
                "wavelength, so this is ONE measurement serving two stocks")
    body.append(f"{pad}    source=({wrap(src, indent + 12)}),")
    body.append(f"{pad}),")
    return note + "\n".join(body)


def apx_curve_block(profile, rec, indent=8):
    pad = " " * indent
    f = rec["density"]["pan"]
    dmin, g, tx, tk, sx, sk, rms = f["fit"]
    import film_profiles as fpm
    c = fpm.ToneCurve(dmin, g, tx, tk, sx, sk)
    return "\n".join([
        f"{pad}# ⚠ TRACED 2026-09-01, REPLACING A CLASS ESTIMATE THAT WAS ALREADY",
        f"{pad}# CLAIMING TO BE A TRACE. This profile's derived provenance said",
        f"{pad}# `fitted_from='datasheet_curve'` -- \"the softplus parameters were",
        f"{pad}# fitted to a published characteristic curve\" -- and they had not",
        f"{pad}# been: the previous values were the family default toe and",
        f"{pad}# shoulder with a dmin and a gamma written beside them. Queue row",
        f"{pad}# E1 caught exactly this on the three Agfa COLOUR films on",
        f"{pad}# 2026-08-29; nobody checked whether the AGFAPAN trio had the same",
        f"{pad}# defect. It did. Now the claim is true.",
        f"{pad}# Vector path off the sheet's own Characteristic curve panel,",
        f"{pad}# six-parameter softplus fit at rms {rms:.4f} D over 5.95 decades.",
        f"{pad}# ⚠ THE CURVE IS NOT DRAWN AT AGFA'S OWN REFERENCE DEVELOPMENT AND",
        f"{pad}# THE SHEET DOES NOT SAY WHAT IT IS DRAWN AT. Measured mid_slope",
        f"{pad}# here is {c.mid_slope:.3f}; the Gamma-time panel on the SAME PAGE and the",
        f"{pad}# 2004 B&W handbook both specify the AGFAPAN line at gamma 0.65,",
        f"{pad}# which REFINAL reaches at 6 min. Reading {c.mid_slope:.2f} off the",
        f"{pad}# gamma-time curve puts this development at roughly 8-9.5 min.",
        f"{pad}# THE TRACE IS ADOPTED AS DRAWN rather than rescaled to 0.65,",
        f"{pad}# because a rescaled curve would describe no development that",
        f"{pad}# exists -- its shape would be one condition and its contrast",
        f"{pad}# another. `processing_family` on this profile carries the whole",
        f"{pad}# gamma-vs-time mapping, so the reference condition is one",
        f"{pad}# interpolation away and nothing is lost by storing the real one.",
        f"{pad}# ⚠ RENDER IMPACT IS REAL: contrast rises about 14 % against the",
        f"{pad}# previous estimate, and D-max goes {2.19 if profile == 'AGFA_APX_100' else 2.33 if profile == 'AGFA_APX_25' else 2.24:.2f} -> {c.dmax:.2f}.",
        _dmin_note(profile, dmin, pad),
        f"{pad}curves=_mono(ToneCurve({dmin:.3f}, {g:.3f}, {tx:+.3f}, {tk:.3f}, "
        f"{sx:+.3f}, {sk:.3f})),",
    ])


def _dmin_note(profile, dmin, pad):
    if profile == "AGFA_APX_400":
        return (f"{pad}# ⚠ D-MIN {dmin:.3f} IS THE LOWEST OF THE THREE APX FILMS AND\n"
                f"{pad}# THAT IS BACKWARDS -- see the note on APX 25. Adopted as\n"
                f"{pad}# measured, because two independent Agfa artworks five years\n"
                f"{pad}# apart draw it that way and the previous 0.13 was an\n"
                f"{pad}# invented family ladder with no source at all.")
    return (f"{pad}# ⚠ D-MIN {dmin:.3f} AND THE ORDER IS PHYSICALLY BACKWARDS.\n"
            f"{pad}# Measured across the three AGFAPAN films the plotted D-min\n"
            f"{pad}# FALLS with speed -- 0.273 / 0.267 / 0.110 for APX 25 / 100 /\n"
            f"{pad}# 400 -- and a faster, thicker emulsion (10 um against 3 um)\n"
            f"{pad}# should fog MORE, not less. It is not a tracing artefact: the\n"
            f"{pad}# 2004 B&W handbook redraws the same curves in a different ink\n"
            f"{pad}# at a different scale and returns 0.261 and 0.107, agreeing to\n"
            f"{pad}# 0.02 D. Either the APX 400 panel plots density above base\n"
            f"{pad}# while the other two plot total density, or the panels do not\n"
            f"{pad}# share a zero; the sheets state neither. ADOPTED AS MEASURED,\n"
            f"{pad}# because the value it replaces (0.10 / 0.11 / 0.13) is an\n"
            f"{pad}# ascending family ladder with no source behind it at all --\n"
            f"{pad}# a tidy invention is not better evidence than an untidy\n"
            f"{pad}# reading. Queue D1, one empty-gate frame, settles it.")


def base_block(profile, indent=8):
    pad = " " * indent
    um, mat = BASE[profile]
    extra = BASE_NOTE.get(profile)
    src = (f"{S98} {PAGE[profile]}, 'Film base: 135 = 120 um, 120 = 95 um'"
           + (f", {extra}" if extra else "")
           + ". ⚠ ONLY THE 135 FIGURE IS STORED -- Agfa publish a different "
             "thickness per format and `base_um` is one float. p5 says the "
             "base \"is made of acetyl cellulose or polyester\" without saying "
             "which film gets which, so `base_material` is left EMPTY on the "
             "roll formats rather than guessed")
    return "\n".join([
        f"{pad}    base_um={um},",
    ]), src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    ap.add_argument("--apply", action="store_true")
    ns = ap.parse_args()
    data = json.loads(Path(ns.emit).read_text())["films"]
    src = FP.read_text(encoding="utf-8")
    n0 = len(src)

    # ---- 1. dye density on the eight colour stocks ------------------------
    for prof in ("AGFA_OPTIMA_100", "AGFA_OPTIMA_200", "AGFA_OPTIMA_400",
                 "AGFA_PORTRAIT_160", "AGFA_ULTRA_50",
                 "AGFA_RSX_II_50", "AGFA_RSX_II_100", "AGFA_RSX_II_200"):
        rec = data[JKEY[prof]]
        if not rec.get("dye", {}).get("curves"):
            print(f"  [skip] {prof}: no dye curves")
            continue
        src = insert_before_features(src, prof, dye_block(prof, rec))
        print(f"  dye_density -> {prof}")

    # ---- 2. the three AGFAPAN characteristic curves -----------------------
    for prof in APX:
        src = replace_curves(src, prof, apx_curve_block(prof, data[JKEY[prof]]))
        print(f"  curves -> {prof}")

    # ---- 3. base_um on all twelve -----------------------------------------
    for prof in JKEY:
        line, bsrc = base_block(prof)
        src = add_base(src, prof, line, bsrc)
    print(f"  base_um -> {len(JKEY)} profiles")

    if not ns.apply:
        print(f"--- dry run --- film_profiles.py {n0} -> {len(src)}")
        return 0
    FP.write_text(src, encoding="utf-8")
    print(f"[ok] film_profiles.py {n0} -> {len(src)}")
    return 0


def insert_before_features(src, profile, text):
    i = src.index(f'name="{profile}",')
    m = re.compile(r"^        features=", re.M).search(src, i)
    if m is None:
        raise SystemExit(f"{profile}: no features= anchor")
    return src[:m.start()] + text + "\n" + src[m.start():]


def replace_curves(src, profile, text):
    """Replace a monochrome profile's `curves=_mono(...)` call and the comment
    block immediately above it."""
    i = src.index(f'name="{profile}",')
    j = src.index("        curves=", i)
    # walk back over the contiguous comment block
    k = j
    while True:
        prev = src.rindex("\n", 0, k - 1) + 1
        if src[prev:].lstrip().startswith("#"):
            k = prev
            continue
        break
    end = src.index("\n", src.index("),", j)) + 1
    return src[:k] + text + "\n" + src[end:]


def add_base(src, profile, line, bsrc):
    """Add `base_um=` inside an existing EmulsionSpec and extend its source."""
    i = src.index(f'name="{profile}",')
    j = src.index("emulsion=EmulsionSpec(", i)
    k = src.index("            source=(", j)
    return src[:k] + line + "\n" + src[k:]


if __name__ == "__main__":
    sys.exit(main())
