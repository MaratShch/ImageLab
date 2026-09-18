#!/usr/bin/env python3
"""Emit the Python mirror of AlgoControlEnums.hpp.

The C++ header is the single authoritative definition of every enumerated
control value and every numeric control range. This script parses it and writes
``algo_control_enums.py`` so the Python reference pipeline uses the same
enumerators, the same integer values and the same ranges as the two C++ builds.

Deriving one from the other rather than maintaining two copies is the point.
A hand-kept Python mirror drifts the first time an enumerator is inserted
rather than appended, and the failure is silent: the reference renders a
different stock from the one the engines render, and every parity audit
compares two correct implementations of two different things.

Usage:
    python3 gen_control_enums.py [--check]

``--check`` regenerates in memory and fails if the file on disk differs, which
is what the build gate runs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_HEADER = Path("/root/work/tst/AlgoControlEnums.hpp")
DEFAULT_OUT = HERE / "algo_control_enums.py"


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return "\n".join(
        re.sub(r"//.*$", "", line)
        for line in src.splitlines()
        if not line.lstrip().startswith("//")
    )


def parse_enum(src: str, name: str) -> list[tuple[str, int]]:
    """Enumerator names and their values, honouring explicit ``= n``."""
    m = re.search(r"enum\s+class\s+" + name + r"\s*:\s*int32_t\s*\{(.*?)\}", src, re.S)
    if m is None:
        raise SystemExit(f"enum class {name} not found")
    out: list[tuple[str, int]] = []
    nxt = 0
    for token in m.group(1).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            ident, val = (t.strip() for t in token.split("=", 1))
            nxt = int(val, 0)
        else:
            ident = token
        out.append((ident, nxt))
        nxt += 1
    return out


def parse_key_table(src: str, name: str) -> list[str]:
    m = re.search(name + r"\[\]\s*=\s*\{(.*?)\}", src, re.S)
    if m is None:
        raise SystemExit(f"key table {name} not found")
    return re.findall(r'"([^"]*)"', m.group(1))


def parse_label_table(src: str, name: str) -> list[str]:
    m = re.search(name + r"\[\]\s*=\s*((?:\s*\"[^\"]*\")+)\s*;", src, re.S)
    if m is None:
        raise SystemExit(f"label table {name} not found")
    return "".join(re.findall(r'"([^"]*)"', m.group(1))).split("|")


def parse_constants(src: str) -> list[tuple[str, str, str]]:
    """(identifier, C++ type, literal) for every scalar constexpr."""
    out = []
    for m in re.finditer(
        r"constexpr\s+(double|int32_t|bool)\s+(\w+)\s*=\s*([^;]+);", src
    ):
        ctype, ident, lit = m.group(1), m.group(2), m.group(3).strip()
        out.append((ident, ctype, lit))
    return out


_CAST_RE = re.compile(r"static_cast<int32_t>\s*\(\s*(\w+)::(\w+)\s*\)")


def _py_literal(ctype: str, lit: str, known: dict[str, str],
                enums: dict[str, dict[str, int]] | None = None) -> str:
    lit = lit.strip()
    if lit in known:                       # e.g. ExposureTimeSDef = ExposureTimeSOff
        return lit
    # \u26a0 A COUNT WRITTEN AS static_cast<int32_t>(Enum::TOTAL_X) IS RESOLVED
    # RATHER THAN COPIED. ProcessVariantCtrlCount is declared that way on
    # purpose -- it is the compiler, not the author, that counts the
    # enumerators -- and a mirror that emitted the cast as a string would turn
    # the one fact this header guarantees into two.
    m = _CAST_RE.fullmatch(lit)
    if m is not None and enums is not None:
        vals = enums.get(m.group(1))
        if vals is not None and m.group(2) in vals:
            return str(vals[m.group(2)])
    if ctype == "bool":
        return "True" if lit == "true" else "False"
    if ctype == "int32_t":
        return str(int(lit, 0))
    return repr(float(lit))


def render(header: Path) -> str:
    raw = header.read_text(encoding="utf-8")
    src = _strip_comments(raw)

    fmt = parse_enum(src, "FilmFormatCtrl")
    prn = parse_enum(src, "PrintStockCtrl")
    fmt_keys = parse_key_table(src, "FilmFormatCtrlKey")
    prn_keys = parse_key_table(src, "PrintStockCtrlKey")
    fmt_lbls = parse_label_table(src, "FilmFormatCtrlStr")
    prn_lbls = parse_label_table(src, "PrintStockCtrlStr")
    pvr = parse_enum(src, "ProcessVariantCtrl")
    pvr_keys = parse_key_table(src, "ProcessVariantCtrlKey")
    pvr_lbls = parse_label_table(src, "ProcessVariantCtrlStr")

    # The TOTAL sentinels are a C++ counting idiom and are not selectable.
    fmt_sel = [(n, v) for n, v in fmt if not n.endswith("TOTAL_FORMATS")]
    prn_sel = [(n, v) for n, v in prn if not n.endswith("PRINT_STOCK_TOTAL")]
    # \u26a0 ONLY THE COUNT IS DROPPED HERE SINCE THE 2026-09-18e REBASE. It
    # used to drop eAS_SHIPPED too, on the grounds that the sentinel had no key
    # and no list-box entry. It is no longer a sentinel: it is enumerator ZERO,
    # it IS the first list-box entry, and it carries an EMPTY key -- so it takes
    # part in the index alignment like every other value and dropping it would
    # shift the other twenty-one by one. TOTAL_PROCESSES is still a COUNT and
    # still has neither a key nor a label.
    pvr_sel = [(n, v) for n, v in pvr if n != "TOTAL_PROCESSES"]
    for what, sel, keys, lbls in (
        ("film format", fmt_sel, fmt_keys, fmt_lbls),
        ("print stock", prn_sel, prn_keys, prn_lbls),
        ("process variant", pvr_sel, pvr_keys, pvr_lbls),
    ):
        if not (len(sel) == len(keys) == len(lbls)):
            raise SystemExit(
                f"{what}: {len(sel)} enumerators, {len(keys)} keys, "
                f"{len(lbls)} labels -- the three tables must be index aligned"
            )

    consts = parse_constants(src)
    known = {i: t for i, t, _ in consts}
    enums = {"FilmFormatCtrl": dict(fmt), "PrintStockCtrl": dict(prn),
             "ProcessVariantCtrl": dict(pvr)}

    L: list[str] = []
    w = L.append
    w('"""Control enumerations and numeric ranges for the Python reference.')
    w("")
    w("GENERATED FILE -- DO NOT EDIT.")
    w("")
    w(f"Emitted by gen_control_enums.py from {header.name}, which is the single")
    w("authoritative definition shared by the scalar build, the AVX2 build and")
    w("this reference. Edit the header and regenerate; a local edit here will be")
    w("overwritten and the build gate will fail before that happens.")
    w('"""')
    w("")
    w("from __future__ import annotations")
    w("")
    w("from enum import IntEnum")
    w("")
    w("")
    w("class FilmFormatCtrl(IntEnum):")
    w('    """Gate geometry. Values match FORMAT_GEOM keys through KEY below."""')
    w("")
    for n, v in fmt:
        w(f"    {n} = {v}")
    w("")
    w("    @property")
    w("    def key(self) -> str:")
    w('        """The FORMAT_GEOM key, or "" for a value with no geometry."""')
    w("        return FILM_FORMAT_KEY.get(int(self), \"\")")
    w("")
    w("    @property")
    w("    def label(self) -> str:")
    w("        return FILM_FORMAT_LABEL.get(int(self), \"\")")
    w("")
    w("")
    w("class PrintStockCtrl(IntEnum):")
    w('    """Positive stock. eSTOCKS_OWN is a sentinel, not a stock."""')
    w("")
    for n, v in prn:
        w(f"    {n} = {v}")
    w("")
    w("    @property")
    w("    def key(self) -> str:")
    w('        """The PRINT_STOCKS name, or "" for the sentinel."""')
    w("        return PRINT_STOCK_KEY.get(int(self), \"\")")
    w("")
    w("    @property")
    w("    def label(self) -> str:")
    w("        return PRINT_STOCK_LABEL.get(int(self), \"\")")
    w("")
    w("")
    w("class ProcessVariantCtrl(IntEnum):")
    w('    """A DEVELOPMENT, globally. eAS_SHIPPED is the absence of a')
    w('    selection and TOTAL_PROCESSES is a count; neither is selectable."""')
    w("")
    for n, v in pvr:
        w(f"    {n} = {v}")
    w("")
    w("    @property")
    w("    def key(self) -> str:")
    w('        """The film::ProcessVariant.variant_id, or "" for the sentinel."""')
    w("        return PROCESS_VARIANT_KEY.get(int(self), \"\")")
    w("")
    w("    @property")
    w("    def label(self) -> str:")
    w("        return PROCESS_VARIANT_LABEL.get(int(self), \"As shipped\")")
    w("")
    w("")
    w("#: dupeStock draws on the same catalogue as printStock.")
    w("DupeStockCtrl = PrintStockCtrl")
    w("")
    w("FILM_FORMAT_KEY: dict[int, str] = {")
    for (n, v), k in zip(fmt_sel, fmt_keys):
        w(f"    {v}: {k!r},")
    w("}")
    w("")
    w("FILM_FORMAT_LABEL: dict[int, str] = {")
    for (n, v), k in zip(fmt_sel, fmt_lbls):
        w(f"    {v}: {k!r},")
    w("}")
    w("")
    w("PRINT_STOCK_KEY: dict[int, str] = {")
    for (n, v), k in zip(prn_sel, prn_keys):
        w(f"    {v}: {k!r},")
    w("}")
    w("")
    w("PRINT_STOCK_LABEL: dict[int, str] = {")
    for (n, v), k in zip(prn_sel, prn_lbls):
        w(f"    {v}: {k!r},")
    w("}")
    w("")
    w("PROCESS_VARIANT_KEY: dict[int, str] = {")
    for (n, v), k in zip(pvr_sel, pvr_keys):
        w(f"    {v}: {k!r},")
    w("}")
    w("")
    w("PROCESS_VARIANT_LABEL: dict[int, str] = {")
    for (n, v), k in zip(pvr_sel, pvr_lbls):
        w(f"    {v}: {k!r},")
    w("}")
    w("")
    w("")
    w("# ---------------------------------------------------------------------------")
    w("# Numeric control metadata")
    w("# ---------------------------------------------------------------------------")
    w("# Transcribed by the header from the per-field documentation in")
    w("# AlgoControl.hpp. Most maxima are ADVISORY -- the engine does not clamp")
    w("# them -- so they describe where the model is meaningful, not where it is")
    w("# guarded. See the header for which bounds are enforced and at which stage.")
    w("")
    for ident, ctype, lit in consts:
        w(f"{ident} = {_py_literal(ctype, lit, known, enums)}")
    w("")
    w("")
    w("def film_format_key(value) -> str:")
    w('    """Resolve a control value to a FORMAT_GEOM key.')
    w("")
    w("    Accepts the enumerator, its integer value, or a bare key string, so")
    w("    existing callers that pass a string keep working. An unrecognised")
    w('    value yields "", which every caller already treats as "use the')
    w('    stock\'s own default" -- the same degradation the engine applies.')
    w('    """')
    w("    if isinstance(value, str):")
    w("        return value")
    w("    try:")
    w("        return FILM_FORMAT_KEY.get(int(value), \"\")")
    w("    except (TypeError, ValueError):")
    w("        return \"\"")
    w("")
    w("")
    w("def process_variant_key(value) -> str:")
    w('    """Resolve a control value to a film::ProcessVariant.variant_id.')
    w("")
    w("    Accepts the enumerator, its integer value, or a bare key string.")
    w('    An unrecognised value yields "", which every caller treats as "as')
    w('    shipped" -- the same degradation both engines apply, and')
    w("    deliberately not a clamp into range.")
    w('    """')
    w("    if isinstance(value, str):")
    w("        return value")
    w("    try:")
    w("        return PROCESS_VARIANT_KEY.get(int(value), \"\")")
    w("    except (TypeError, ValueError):")
    w("        return \"\"")
    w("")
    w("")
    w("def print_stock_key(value) -> str:")
    w('    """Resolve a control value to a PRINT_STOCKS name. See above."""')
    w("    if isinstance(value, str):")
    w("        return value")
    w("    try:")
    w("        return PRINT_STOCK_KEY.get(int(value), \"\")")
    w("    except (TypeError, ValueError):")
    w("        return \"\"")
    w("")
    return "\n".join(L)


def _unused_process_variant_key(value) -> str:      # pragma: no cover
    """Placeholder kept out of the emitted file; see process_variant_key."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="fail if the file on disk is not what would be written")
    ns = ap.parse_args(argv)

    text = render(ns.header)

    if ns.check:
        have = ns.out.read_text(encoding="utf-8") if ns.out.is_file() else ""
        if have != text:
            print(f"[FAIL] {ns.out.name} is stale against {ns.header.name}")
            return 1
        print(f"[OK] {ns.out.name} matches {ns.header.name}")
        return 0

    ns.out.write_text(text, encoding="utf-8")
    print(f"[OK] wrote {ns.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
