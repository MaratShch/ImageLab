#!/usr/bin/env python3
"""«Storage and Preservation of Motion-Picture Film» (Kodak, 1957), read.

WHAT THE DOCUMENT IS
--------------------
Eastman Kodak, «Storage and Preservation of Motion-Picture Film», 1957. 82
pages. **No text layer on any page** -- `get_text()` returns empty on all 82 --
so every figure below was read off a render. The book is not a data sheet: it
is a preservation manual, and it is the only document in this corpus that
describes the SUPPORT rather than the emulsion.

WHY IT PRODUCED A SCHEMA CHANGE INSTEAD OF A ROW OF NUMBERS
--------------------------------------------------------------
⚠⚠ IT WAS READ AGAINST `AgingSpec` AND `DyeStabilitySpec` AND YIELDS NOTHING
EITHER CAN HOLD. That is structural, not unlucky: both of those structs
describe DAMAGE ALREADY DONE TO AN IMAGE -- a fade fraction, a dmin lift, a
shrinkage percentage, an Arrhenius rate for a dye. This book is about the
plastic, the can and the room. Its contents are a support chemistry, a
specific gravity, a packaging humidity, a storage table and a base-change
date, and before v50 this database had nowhere to put any of them. Hence
`BaseSpec`.

WHAT IS HERE, AND WHAT DELIBERATELY IS NOT ON A PROFILE
---------------------------------------------------------
⚠ MOST OF WHAT THE BOOK GIVES IS CLASS DATA, AND CLASS DATA STAYS IN A TABLE.
A specific gravity is a property of cellulose triacetate, not of PLUS-X; a
storage table is a property of a room; "approximately 0.0055-inch" is one
figure the book applies to ALL acetate motion-picture film without breaking it
out by product or by base chemistry. Stamping those onto 193 profiles would
turn one measurement into 193 apparent ones, so they live in
`film_profiles.py` as `_KODAK_1957_*` constants and this module checks them.

Six statements DO name a product line, and those six went onto profiles:

  1. Kodachrome's support is cellulose acetate propionate (p.6), against
     triacetate for "virtually all" other Eastman motion-picture film.
  2. Eastman black-and-white NEGATIVE films carry a neutral grey antihalation
     dye IN THE BASE, not removed in processing (p.5) -- a permanent support
     property, against the removable jet backing of the colour films.
  3. Eastman motion-picture negative and original camera films have been on a
     LOWER-SHRINK base since June 1954 (p.45 footnote).
  4. Most Eastman black-and-white motion-picture film is packaged at about
     60 % RH; all colour and some 16 mm at about 50 % (pp.12-13).
  5. Kodachrome and Eastman Color were NEVER made on nitrate base (p.6).
  6. The float test's specific gravities, per material (p.67).

THE DEFECT (5) FOUND
----------------------
⚠⚠ `KODACHROME_1938` AND `KODACHROME_TYPE_A_1938` BOTH CARRIED
`Feature.NITRATE_BASE`. Page 6 says in as many words: "Kodachrome and Eastman
Color Films were never made on nitrate base". Table I on p.8 corroborates it
in a column -- "Formerly Made on Nitrate Base" reads *no* for every Eastman
Color product -- and the same table's *yes* rows bound the refusal instead of
generalising it: 35 mm negative, fine-grain release positive, duplitized
positive, 65/70 mm. So `EASTMAN_SUPER_XX_1938`, a 35 mm negative, keeps its
flag correctly and the two Kodachromes lose theirs. No renderer reads the
flag, so nothing moves; what changes is that the shipped C++ stops asserting a
support Kodak says the film never had.

WHAT THE BOOK DOES NOT CONTAIN, CHECKED AND RECORDED
-------------------------------------------------------
* **No tabulated shrinkage data of any kind.** One ratio in one sentence
  (p.42) is the only quantitative shrinkage statement in 82 pages: no
  percentage at either humidity, no per-base figure, no time axis.
* **No dye-fade numbers.** p.20 states that "one of the three dyes in color
  films usually fades more rapidly than the others" -- which dye, by how much
  and over what period are all absent. There is no fade curve and no half-life
  anywhere in the volume.
* **No polyester.** The words polyester, ESTAR and polyethylene terephthalate
  do not occur once. The book predates the material, which is why
  `AgingSpec.shrinkage_rh_factor`'s docstring says a polyester stock is not
  covered by its ratio.
* **No "cellulose diacetate"** under that name; the book calls the material
  "acetone-soluble cellulose acetate".
* **No three-tier storage vocabulary.** Table VI knows "Commercial" and
  "Archival" only. The medium-term / extended-term tiers of NAPM IT9.11 that
  H-1-5302 cites postdate it by thirty-five years, and mapping one onto the
  other would invent a tier.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import film_profiles as FP            # noqa: E402

#: The profiles that took one of the six product-line statements.
BW_NEGATIVE_GREY_BASE = ("EASTMAN_PLUS_X_5231", "EASTMAN_TRI_X_5223",
                         "EASTMAN_DOUBLE_X_5222")
KODACHROME_JET_BACKING = ("KODACHROME_64", "KODACHROME_1938",
                          "KODACHROME_TYPE_A_1938")
#: ⚠ ONLY KODACHROME_64 TAKES THE PROPIONATE CHEMISTRY. The statement is
#: present-tense 1957 and the book's own chronology on the same page puts
#: propionate AFTER the acetone-soluble acetate that was the only safety base
#: "up to about 1938", so carrying it back onto the two 1938 coatings would
#: assert the one thing that chronology argues against.
PROPIONATE = ("KODACHROME_64",)


def run(do_assert: bool = True) -> int:
    fail: list[str] = []
    print("=== the float-test specific gravities, p.67 ===")
    for mat, (lo, hi) in FP._BASE_SPECIFIC_GRAVITY_1957.items():
        print("  %-52s %.3f - %.3f" % (mat, lo, hi))

    # ⚠⚠ THE TABLE IS A TEST AND THE TEST IS WHAT IS ASSERTED. Trichloroethylene
    # at 1.477 must sit ABOVE every safety base and BELOW nitrate, because that
    # is the entire mechanism: "If the punching sinks it is nitrate film; if it
    # floats to the surface it is acetate film." A transcription error in any
    # row would break the ordering, which is why the ordering is the check
    # rather than the five pairs being compared with themselves.
    tce = FP._BASE_SPECIFIC_GRAVITY_1957[
        "trichloroethylene (the test liquid, not a base)"][0]
    nitrate_lo = FP._BASE_SPECIFIC_GRAVITY_1957["cellulose nitrate"][0]
    safety = {k: v for k, v in FP._BASE_SPECIFIC_GRAVITY_1957.items()
              if "nitrate" not in k and "trichlor" not in k}
    ok_float = nitrate_lo > tce and all(hi < tce for lo, hi in safety.values())
    print("  ⚠ the float test: nitrate floor %.2f > trichloroethylene %.3f > "
          "every safety base ceiling %.2f -- %s"
          % (nitrate_lo, tce, max(hi for _lo, hi in safety.values()),
             "holds" if ok_float else "BROKEN"))
    if not ok_float:
        fail.append("the p.67 float test no longer separates nitrate from the "
                    "safety bases -- a specific gravity has been mis-typed")

    # -- Table VI ------------------------------------------------------------
    print("\n=== Table VI, p.47, PROCESSED film ===")
    for cls, terms in FP._KODAK_1957_TABLE_VI.items():
        print("  %-26s commercial %-12s %-8s   archival %-12s %s"
              % (cls, terms["commercial"][0], terms["commercial"][1],
                 terms["archival"][0], terms["archival"][1]))
    # ⚠ THE STRUCTURE IS THE FINDING: the commercial HUMIDITY band is
    # identical on all four rows, so on the commercial side only TEMPERATURE
    # separates acetate from nitrate. A reader who remembers Table VI as "four
    # different storage regimes" has remembered one more axis than it has.
    rhs = {t["commercial"][1] for t in FP._KODAK_1957_TABLE_VI.values()}
    print("  ⚠ distinct commercial humidity bands across all four rows: %d "
          "(%s) -- only TEMPERATURE separates acetate from nitrate there"
          % (len(rhs), ", ".join(sorted(rhs))))
    if len(rhs) != 1 or len(FP._KODAK_1957_TABLE_VI) != 4:
        fail.append("Table VI has changed shape: %d rows, %d distinct "
                    "commercial humidity bands"
                    % (len(FP._KODAK_1957_TABLE_VI), len(rhs)))
    for cls, terms in FP._KODAK_1957_TABLE_VI.items():
        if set(terms) != {"commercial", "archival"}:
            fail.append("%s has a storage term the 1957 book does not know: "
                        "%s" % (cls, sorted(terms)))

    # -- the product-line statements, on the profiles ------------------------
    print("\n=== the six statements that name a product line ===")
    for nm in BW_NEGATIVE_GREY_BASE:
        b = FP.get_profile(nm).base
        ok = (b.base_type == "cellulose triacetate"
              and "grey dye" in b.antihalation
              and "NOT removed" in b.antihalation
              and b.lower_shrink
              and b.packaged_equilibrium_rh_pct == 60.0)
        print("  %-22s triacetate, grey-in-base, lower-shrink, %.0f %% RH  %s"
              % (nm, b.packaged_equilibrium_rh_pct, "ok" if ok else "MISMATCH"))
        if not ok:
            fail.append("%s does not carry the four p.5/p.6/p.45/p.12 class "
                        "statements" % nm)
    for nm in KODACHROME_JET_BACKING:
        b = FP.get_profile(nm).base
        ok = "jet backing" in b.antihalation and "removed in" in b.antihalation
        print("  %-22s jet backing, removed in processing               %s"
              % (nm, "ok" if ok else "MISMATCH"))
        if not ok:
            fail.append("%s does not carry the p.5 jet-backing statement" % nm)
        want_prop = nm in PROPIONATE
        if want_prop and b.base_type != "cellulose acetate propionate":
            fail.append("%s should carry the p.6 propionate support" % nm)
        if not want_prop and b.base_type:
            fail.append("%s carries a base chemistry the 1957 book's own "
                        "chronology does not support for a 1938 coating" % nm)

    # ⚠⚠ THE TWO MECHANISMS MUST STAY APART. A permanent grey dye IN the base
    # and a jet backing REMOVED in processing are different physical things
    # with different consequences for a scanned frame, and p.5 states them in
    # one paragraph precisely as a contrast. Collapsing them would lose the
    # only support-side distinction the book draws between the two families.
    grey = {FP.get_profile(n).base.antihalation for n in BW_NEGATIVE_GREY_BASE}
    jet = {FP.get_profile(n).base.antihalation for n in KODACHROME_JET_BACKING}
    print("  ⚠ %d distinct wording for the grey-in-base family and %d for the "
          "jet-backing family, and they do not overlap: %s"
          % (len(grey), len(jet), "holds" if not (grey & jet) else "BROKEN"))
    if grey & jet:
        fail.append("the grey-in-base and jet-backing mechanisms have been "
                    "collapsed into one wording")

    # -- the defect ----------------------------------------------------------
    print("\n=== p.6: 'Kodachrome and Eastman Color Films were never made on "
          "nitrate base' ===")
    nitr = sorted(p.name for p in FP.FILM_PROFILES
                  if p.features & FP.Feature.NITRATE_BASE)
    offend = [n for n in nitr if "KODACHROME" in n or "EASTMAN_COLOR" in n]
    print("  %d profiles still flagged nitrate: %s" % (len(nitr), nitr))
    print("  of which Kodachrome or Eastman Color: %d" % len(offend))
    if offend:
        fail.append("Kodak states these were never nitrate and they are "
                    "flagged as such: %s" % offend)
    # And the refusal must stay BOUNDED: Table I's *yes* rows include 35 mm
    # negative film, so a 35 mm camera negative of the period SHOULD keep the
    # flag. A pass that removed it everywhere would be over-correcting.
    if "EASTMAN_SUPER_XX_1938" not in nitr:
        fail.append("EASTMAN_SUPER_XX_1938 lost its nitrate flag -- Table I "
                    "p.8 lists 35 mm negative film as formerly nitrate, so "
                    "the p.6 refusal does not reach it")

    # -- the class figures, and the fact that no profile took them -----------
    print("\n=== class figures: in the table, NOT on a profile ===")
    print("  support %.1f um / complete film %.1f um (p.6, p.12) -- one "
          "figure for ALL acetate motion-picture film"
          % (FP._ACETATE_MP_BASE_UM_1957, FP._ACETATE_MP_FILM_UM_1957))
    print("  nitrate decomposition doubles per %.0f F (%.0f C as the book "
          "rounds it), p.14" % (FP._NITRATE_DECOMP_DOUBLING_F,
                                FP._NITRATE_DECOMP_DOUBLING_C_AS_PRINTED))
    print("  permanent shrinkage is %.1fx as RAPID at %.0f %% RH as at %.0f %%,"
          " p.42" % ((FP._SHRINKAGE_RATE_RH_RATIO_1957,)
                     + FP._SHRINKAGE_RATE_RH_FROM_TO_PCT[::-1]))
    print("  service envelope %.0f to %.0f F, distorts above %.0f F, softens "
          "at %.0f-%.0f F (p.13, p.19)"
          % (FP._ACETATE_SERVICE_TEMP_F_1957
             + (FP._ACETATE_DISTORTS_ABOVE_F_1957,)
             + FP._ACETATE_SOFTENS_F_1957))
    print("  moisture content %s (p.79, read off Figure 6, whose own curves "
          "carry no printed values)" % (FP._ACETATE_MOISTURE_1957,))

    # ⚠⚠ THE RATE RATIO IS NOT ON ANY PROFILE AND THAT IS THE POINT.
    # `AgingSpec.shrinkage_rh_factor` exists to hold it and is 1.0 everywhere,
    # because the sentence is hedged three ways -- "approximately", "SOME
    # motion-picture films", and prefaced "For example" -- and because
    # `shrinkage_pct` is a STATE while this is a RATE. Setting it would turn an
    # illustration into a specification and multiply the wrong quantity.
    set_it = [p.name for p in FP.FILM_PROFILES
              if p.aging.shrinkage_rh_factor != 1.0]
    print("  ⚠ profiles setting AgingSpec.shrinkage_rh_factor: %d -- the "
          "sentence is hedged three ways and multiplies a RATE, not the "
          "STATE `shrinkage_pct` holds" % len(set_it))
    if set_it:
        fail.append("a profile has taken the p.42 rate ratio as a state "
                    "multiplier: %s" % set_it)

    # And no profile may have taken the class thickness either.
    took = [p.name for p in FP.FILM_PROFILES
            if p.base.total_film_um in (FP._ACETATE_MP_FILM_UM_1957,)
            or p.base.base_um == FP._ACETATE_MP_BASE_UM_1957]
    print("  ⚠ profiles taking the class thickness as their own: %d"
          % len(took))
    if took:
        fail.append("the 1957 class thickness has been stamped onto a "
                    "profile: %s" % took)

    # -- what the book does not contain --------------------------------------
    print("\n=== recorded absences ===")
    for line in (
            "no tabulated shrinkage data of any kind -- one ratio in one "
            "sentence is the whole of it",
            "no dye-fade numbers: p.20 says one of the three dyes usually "
            "fades faster and never says which, by how much, or over what "
            "period",
            "no polyester / ESTAR / polyethylene terephthalate anywhere in "
            "82 pages -- the book predates the material",
            "no 'cellulose diacetate' under that name; the book calls it "
            "acetone-soluble cellulose acetate",
            "no three-tier storage vocabulary: Table VI knows Commercial and "
            "Archival only"):
        print("  * " + line)

    if fail and do_assert:
        print("\n".join(["", "FAILURES:"] + ["  " + f for f in fail]))
        return 1
    print("\n[OK] kodak_1957_storage.py -- «Storage and Preservation of "
          "Motion-Picture Film» (Kodak, 1957), 82 pages with NO TEXT LAYER, "
          "read off renders. ⚠⚠ IT PRODUCED A SCHEMA CHANGE RATHER THAN A ROW "
          "OF NUMBERS, and the reason is structural: read against `AgingSpec` "
          "and `DyeStabilitySpec` it yields nothing either can hold, because "
          "those describe damage already done to an IMAGE and this book is "
          "about the plastic, the can and the room. Hence v50's `BaseSpec`. "
          "⚠ MOST OF WHAT IT GIVES IS CLASS DATA AND STAYS IN A TABLE -- a "
          "specific gravity belongs to cellulose triacetate, not to PLUS-X, "
          "and 'approximately 0.0055-inch' is one figure the book applies to "
          "all acetate motion-picture film. Six statements name a product "
          "line and those six went onto profiles. ⚠⚠ ONE OF THEM FOUND A "
          "DEFECT: KODACHROME_1938 and KODACHROME_TYPE_A_1938 carried "
          "Feature.NITRATE_BASE against p.6's 'Kodachrome and Eastman Color "
          "Films were never made on nitrate base', and Table I p.8 both "
          "corroborates that and BOUNDS it, so EASTMAN_SUPER_XX_1938 keeps "
          "its flag as a 35 mm negative. ⚠ THE p.42 SHRINKAGE RATIO IS "
          "DELIBERATELY ON NO PROFILE: it is a RATE and `shrinkage_pct` is a "
          "STATE, and the sentence is hedged three ways")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    default=True)
    ns = ap.parse_args(argv)
    return run(ns.do_assert)


if __name__ == "__main__":
    sys.exit(main())
