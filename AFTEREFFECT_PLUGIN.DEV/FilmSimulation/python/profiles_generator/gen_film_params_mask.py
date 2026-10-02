#!/usr/bin/env python3
"""Emit `film_params_mask.hpp` -- one uint64_t control-availability mask per film.

⚠ THIS FILE AND `gen_film_control_matrix.py` SHARE ONE SET OF PREDICATES AND
THAT IS THE WHOLE DESIGN. The Markdown matrix is the human-readable form and
this header is the machine-readable one; they are two renderings of the same
computation, imported from the same module, so they cannot disagree about a
single cell. A second implementation of the predicates here -- even a careful
one -- would be a second thing to keep in step, which is the defect this
arrangement exists to prevent.

  bit 0   = the SECOND column of the matrix (`filmFormat`)
  bit N   = matrix column N + 2
  last    = `seed`, Render / Master Seed

`filmProfile` (matrix column 1) has NO BIT. It is always available -- it is the
control that selects the film, so a host that greyed it could never reach any
other row -- and giving it a bit would spend one on a constant.

  1  the control is available on that stock: some stage will act on it
  0  the control is unavailable: either the figure is missing from the database
     or the engine has no reader (matrix `?`), or the control is physically
     inapplicable to that film (matrix `O`)

⚠ `O` AND `?` BOTH PRODUCE A ZERO AND THE HEADER CANNOT TELL THEM APART. That is
deliberate: a host needs one question answered -- draw the control or grey it --
and the distinction between a closed question and an open one is a CORPUS
question, not a UI one. It is preserved in `doc/FilmControlMatrix.md`, which is
generated from the same predicates in the same order.

Row order is `film_names.txt`, which is the database order, so row N is
enumerator N of `film::eFILM_PROFILE` and the generated `static_assert` pins the
length against `eTOTAL_FILMS_PROFILES`.

Usage:
    python3 gen_film_params_mask.py [--out PATH] [--check]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import film_profiles as FP  # noqa: E402
from cpp_codegen import COPYRIGHT_NOTICE  # noqa: E402

DEFAULT_OUT = HERE / "film_params_mask.hpp"

#: ⚠ IMPORTED BY PATH, NOT BY NAME, because the module's file name is a verb
#: and importing it for its tables rather than running it is exactly what a
#: second consumer should do. Nothing in it runs at import time except the
#: three derived counts.
_spec = importlib.util.spec_from_file_location(
    "_gen_film_control_matrix", HERE / "gen_film_control_matrix.py")
_matrix = importlib.util.module_from_spec(_spec)
sys.modules["_gen_film_control_matrix"] = _matrix
_spec.loader.exec_module(_matrix)

#: The one control that carries no bit. Asserted below rather than assumed: if
#: the panel ever stops opening with the film selector, this generator must
#: stop rather than silently shift every bit by one.
EXCLUDED_FIELD = "filmProfile"

BITS_PER_GROUP = 8          # digit-separator grouping in the emitted literals


def columns() -> list[tuple[str, str, object]]:
    """(panel group, label, C++ field, predicate) in panel order, film stock first."""
    return [(g, label, field, pred)
            for g, cols in _matrix.GROUPS
            for (label, field, pred, _o, _q) in cols]


def bit_columns() -> list[tuple[str, str, object]]:
    """The columns that get a bit, in bit order, bit 0 first."""
    cols = columns()
    if cols[0][2] != EXCLUDED_FIELD:
        raise RuntimeError(
            f"the first panel control is {cols[0][2]!r}, not {EXCLUDED_FIELD!r}. "
            "Every bit in film_params_mask.hpp is numbered from that "
            "assumption, so this generator refuses rather than renumber the "
            "whole header silently.")
    out = [c for c in cols[1:]]
    if any(c[2] == EXCLUDED_FIELD for c in out):
        raise RuntimeError("filmProfile appears twice in the panel order")
    if len(out) > 64:
        raise RuntimeError(
            f"{len(out)} controls carry a bit and a uint64_t holds 64. The "
            "mask type has to widen, or the panel has to shed a control; "
            "either way this is not a decision a generator may take.")
    if out[-1][2] != "seed":
        raise RuntimeError(
            f"the last panel control is {out[-1][2]!r}, not 'seed'. The header "
            "documents its highest bit as Render / Master Seed.")
    return out


def masks(profiles, cols) -> list[int]:
    """One mask per profile, LSB = the first bit column."""
    out = []
    for p in profiles:
        v = 0
        for i, (_g, _label, _field, pred) in enumerate(cols):
            mark = pred(p)
            if mark not in (_matrix.V, _matrix.O, _matrix.Q):
                raise RuntimeError(f"predicate for {_field} returned {mark!r}")
            if mark == _matrix.V:
                v |= 1 << i
        out.append(v)
    return out


def _literal(value: int) -> str:
    """A full 64-digit binary literal, grouped in bytes with C++14 separators.

    ⚠ FULL WIDTH AND GROUPED, ON PURPOSE. A literal trimmed to the number of
    controls would change width the day a control is added, and every reader
    who had counted digits once would be wrong. 64 digits is what the type
    holds; the `'` separators every eight make a bit position countable without
    counting, and they are C++14 digit separators, which this project compiles
    with.
    """
    bits = format(value, "064b")
    groups = [bits[i:i + BITS_PER_GROUP]
              for i in range(0, 64, BITS_PER_GROUP)]
    return "0b" + "'".join(groups)


def render(profiles, names) -> str:
    cols = bit_columns()
    vals = masks(profiles, cols)
    n = len(profiles)
    width = max(len(x) for x in names)

    L: list[str] = []
    w = L.append

    w(COPYRIGHT_NOTICE.rstrip("\n"))
    w(f"// GENERATED by gen_film_params_mask.py -- DO NOT EDIT BY HAND.")
    w(f"// film_profiles schema version {FP.SCHEMA_VERSION}; {n} profiles.")
    w("//")
    w("// PER-FILM CONTROL AVAILABILITY, AS A COMPILE-TIME BITMASK.")
    w("//")
    w("// One uint64_t per film in the database, in DATABASE ORDER: element N is")
    w("// film::eFILM_PROFILE value N and line N+1 of film_names.txt. The array is")
    w("// never sorted, grouped or filtered -- an index into it is an enumerator.")
    w("//")
    w("// Bit 0 is the LEAST significant bit and is `Film Format`. Bits then follow")
    w("// the Effect Control Panel's own order, group by group, as the mockup draws")
    w(f"// it; the highest assigned bit is {len(cols) - 1}, `Render / Master Seed`.")
    w("//")
    w("//   1  the control IS available for that film -- a stage will act on it")
    w("//   0  the control is NOT available: either the database has no figure and")
    w("//      the engine no reader, or the control is physically inapplicable to")
    w("//      that film (a monochrome stock has no white balance; a slide has no")
    w("//      print stage; a stock with no reseau plate has no mosaic)")
    w("//")
    w("// ⚠ A ZERO DOES NOT SAY WHICH OF THE TWO IT IS, and that is deliberate: a")
    w("// host needs one answer -- draw the control or grey it. The distinction")
    w("// between a gap somebody could close and a property the film does not have")
    w("// is a corpus question and is kept in doc/FilmControlMatrix.md, which is")
    w("// generated from the SAME predicates in the SAME order as this file.")
    w("//")
    w("// ⚠ `Film Stock` (filmProfile) HAS NO BIT. It is always available -- it is")
    w("// the control that selects the film -- so a bit for it would be a constant.")
    w("// Every other panel control has exactly one bit and no bit is shared.")
    w("//")
    w("// Availability is decided by the same predicate the engine uses, so a 1 is")
    w("// a control some stage acts on rather than a control the API would accept.")
    w("//")
    w("// ⚠ THIS FILE IS DERIVED AND IS REGENERATED ON EVERY BUILD. It is not source.")
    w("// New film data, a new control, or a changed control definition changes it,")
    w("// and the only correct way to update it is to regenerate -- never to edit a")
    w("// mask by hand and never to keep an old one because it was already shipped.")
    w("//")
    w("// BIT ASSIGNMENT")
    for i, (g, label, field, _p) in enumerate(cols):
        w(f"//   bit {i:2d}  {g} — {label}  ({field})")
    w("")
    w("#pragma once")
    w("")
    w("#include <array>")
    w("#include <cstdint>")
    w("")
    w('#include "film_enum.hpp"')
    w("")
    w("namespace film {")
    w("")
    w("/// Bit index of each control in kFilmControlAvailability.")
    w("///")
    w("/// Use `(kFilmControlAvailability[i] >> eCTRL_BIT_GRAIN) & 1u` rather than a")
    w("/// literal shift, so a control added to the panel cannot silently change the")
    w("/// meaning of a number written into a caller.")
    w("enum eFILM_CONTROL_BIT : std::int32_t")
    w("{")
    for i, (_g, _label, field, _p) in enumerate(cols):
        nm = field.split(".")[-1]
        sym = "eCTRL_BIT_" + "".join(
            ("_" + ch if ch.isupper() else ch.upper()) for ch in nm).lstrip("_")
        w(f"    {sym:<34} = {i},")
    w(f"    eCTRL_BIT_TOTAL_CONTROLS           = {len(cols)}")
    w("};")
    w("")
    w("/// Control availability for every film in the database, in database order.")
    w("///")
    w("/// Binary, most significant bit first, grouped in bytes: the RIGHTMOST digit")
    w("/// is bit 0 (Film Format) and the leftmost is bit 63. Bits")
    w(f"/// {len(cols)}..63 are unassigned and are always zero.")
    w(f"constexpr std::array<std::uint64_t, {n}> kFilmControlAvailability = {{{{")
    for k, (nm, v) in enumerate(zip(names, vals)):
        comma = "," if k + 1 < n else " "
        w(f"    {_literal(v)}{comma} // [{k:3d}] {nm}")
    w("}};")
    w("")
    w("static_assert(kFilmControlAvailability.size()")
    w("                  == static_cast<std::size_t>(")
    w("                         eFILM_PROFILE::eTOTAL_FILMS_PROFILES),")
    w('              "one control mask per film profile, in database order");')
    w("")
    w("/// @return true when @p control is available for film @p profile.")
    w("constexpr bool FilmControlAvailable(const eFILM_PROFILE profile,")
    w("                                    const eFILM_CONTROL_BIT control) noexcept")
    w("{")
    w("    return ((kFilmControlAvailability[")
    w("                 static_cast<std::size_t>(profile)]")
    w("             >> static_cast<std::uint64_t>(control))")
    w("            & 1ull) != 0ull;")
    w("}")
    w("")
    w("}  // namespace film")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true",
                    help="write nothing; fail if the file on disk is stale")
    ns = ap.parse_args()

    profiles = list(FP.FILM_PROFILES)
    names = _matrix.read_names()
    if len(names) != len(profiles):
        raise SystemExit(
            f"[FAIL] film_names.txt has {len(names)} names and the database "
            f"{len(profiles)} profiles")

    text = render(profiles, names)
    out = Path(ns.out)
    if ns.check:
        if not out.is_file():
            print(f"[FAIL] {out} is missing")
            return 1
        if out.read_text(encoding="utf-8") != text:
            print(f"[FAIL] {out} is stale -- regenerate with "
                  f"`python3 gen_film_params_mask.py`")
            return 1
        print(f"[OK] {out.name} matches the live database and control definitions")
        return 0
    out.write_text(text, encoding="utf-8")
    print(f"[OK] wrote {out} ({out.stat().st_size} bytes, "
          f"{len(profiles)} masks, {len(bit_columns())} bits each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
