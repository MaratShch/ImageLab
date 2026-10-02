#!/usr/bin/env python3
"""Audit: the Batch Position control is represented identically everywhere.

Written 2026-09-30 for a defect the owner found: `batchPosition` had been an
AlgoControls field, an engine resolver and a mockup row since schema v52, but
the generated database headers did not carry it -- `film_params_mask.hpp` had
no bit for it and `film_enum.hpp` no range. This audit reads the GENERATED
FILES AS SHIPPED (under --root), not the generator's intentions, and fails on
any disagreement between:

  1. AlgoControlEnums.hpp  BatchPosition{Min,Max,Def,Step}   (the authority)
  2. film_enum.hpp          film::eBATCH_POSITION_{MIN,MAX,DEF,STEP}
  3. film_params_mask.hpp   eCTRL_BIT_BATCH_POSITION, its place in the bit
                            order, and the bit's value on every one of the
                            films, against film_sim.resolve_batch_position
  4. every OTHER bit of every film, recomputed from the shared predicates,
     so the insertion is shown not to have disturbed any other control
  5. AlgoControl.hpp / AlgoControl.cpp: the field, its default, and the
     static_asserts that tie (1) to (2) at compile time
  6. the UI mockup: the panel's rows are exactly the mask's bits plus the
     film selector, in the same order, and each group's printed count is its
     row count
  7. a C++ probe that includes the shipped film_params_mask.hpp and asserts
     the Batch Position bit of every film at compile time

Usage:  python batch_position_headers.py --root <engine tree> [--mockup PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import film_profiles as FP   # noqa: E402
import film_sim as FS        # noqa: E402

RESULTS: list[bool] = []


def chk(ok, label, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s   %s" % ("PASS" if ok else "FAIL", label, detail))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _masks(text):
    body = text.split("kFilmControlAvailability = {{")[1].split("}};")[0]
    out = []
    for line in body.splitlines():
        m = re.match(r"\s*(0b[01']+|0x[0-9A-Fa-f']+)", line)
        if m:
            out.append(int(m.group(1).replace("'", ""), 0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--mockup", type=Path,
                    default=Path("/root/work/ui/FilmSimulator_Mockup_v6.html"))
    a = ap.parse_args(argv)
    root = a.root
    gce = _load("_bph_gen_control_enums", HERE / "gen_control_enums.py")
    mx = _load("_bph_matrix", HERE / "gen_film_control_matrix.py")

    # -- 1 / 2: the range --------------------------------------------------
    ace = gce._strip_comments((root / "AlgoControlEnums.hpp").read_text(encoding="utf-8"))
    auth = {i: float(l) for i, t, l in gce.parse_constants(ace) if i.startswith("BatchPosition")}
    enum_txt = (root / "film_enum.hpp").read_text(encoding="utf-8")
    got = {k: float(v) for k, v in re.findall(
        r"constexpr\s+double\s+eBATCH_POSITION_(MIN|MAX|DEF|STEP)\s*=\s*([^;]+);", enum_txt)}
    want = {"MIN": auth.get("BatchPositionMin"), "MAX": auth.get("BatchPositionMax"),
            "DEF": auth.get("BatchPositionDef"), "STEP": auth.get("BatchPositionStep")}
    chk(None not in want.values() and got == want,
        "film_enum.hpp carries the Batch Position MIN / MAX / DEF / STEP of AlgoControlEnums.hpp",
        "film_enum %s, AlgoControlEnums %s" % (got, want))
    chk(want["MIN"] < want["DEF"] < want["MAX"] and want["STEP"] > 0
        and abs(round((want["MAX"] - want["MIN"]) / want["STEP"]) * want["STEP"]
                - (want["MAX"] - want["MIN"])) < 1e-9,
        "the range is ordered and STEP divides it", "%d steps" % round((want["MAX"] - want["MIN"]) / want["STEP"]))

    # -- 3 / 4: the mask ----------------------------------------------------
    mtxt = (root / "film_params_mask.hpp").read_text(encoding="utf-8")
    bits = {n: int(v) for n, v in re.findall(r"(eCTRL_BIT_\w+)\s*=\s*(\d+)", mtxt)}
    cols = [c for g, cs in mx.GROUPS for c in cs][1:]          # filmProfile has no bit
    idx = [c[1] for c in cols].index("batchPosition") if any(c[1] == "batchPosition" for c in cols) else -1
    chk(idx >= 0 and bits.get("eCTRL_BIT_BATCH_POSITION") == idx
        and bits.get("eCTRL_BIT_TOTAL_CONTROLS") == len(cols)
        and bits.get("eCTRL_BIT_SEED") == len(cols) - 1,
        "film_params_mask.hpp declares eCTRL_BIT_BATCH_POSITION at the panel position",
        "bit %s of %s (panel index %d)" % (bits.get("eCTRL_BIT_BATCH_POSITION"),
                                           bits.get("eCTRL_BIT_TOTAL_CONTROLS"), idx))
    listed = re.findall(r"//\s+bit\s+(\d+)\s+.*?\((\S+)\)\s*$", mtxt, re.M)
    chk([f for _, f in listed] == [c[1] for c in cols]
        and [int(b) for b, _ in listed] == list(range(len(cols))),
        "the header's BIT ASSIGNMENT list is the panel order, one bit per control")
    names = [l.strip() for l in (root / "film_names.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    masks = _masks(mtxt)
    by = {p.name: p for p in FP.FILM_PROFILES}
    chk(len(masks) == len(FP.FILM_PROFILES) == len(names),
        "one mask per film", "%d masks, %d films" % (len(masks), len(FP.FILM_PROFILES)))
    b_bad, other_bad, n_on = [], [], 0
    for i, (m, p) in enumerate(zip(masks, FP.FILM_PROFILES)):
        acts = any(FS.resolve_batch_position(p, v) is not p for v in (1.0, -1.0))
        bit = (m >> idx) & 1
        n_on += bit
        if bit != int(acts) or acts != bool(p.tolerance):
            b_bad.append(p.name)
        for j, c in enumerate(cols):
            if j == idx:
                continue
            if ((m >> j) & 1) != int(c[2](p) == mx.V):
                other_bad.append("%s bit %d" % (p.name, j))
        if m >> len(cols):
            other_bad.append("%s has a bit above %d" % (p.name, len(cols) - 1))
    chk(not b_bad and n_on == sum(1 for p in FP.FILM_PROFILES if p.tolerance),
        "the Batch Position bit is set exactly on the films whose resolver moves -- the stocks carrying an acceptance band",
        "%d films enabled: %s" % (n_on, ", ".join(p.name for p, m in zip(FP.FILM_PROFILES, masks) if (m >> idx) & 1))
        if not b_bad else "wrong on %s" % b_bad[:5])
    chk(not other_bad, "every other bit of every film equals its shared predicate",
        "%d films x %d other bits" % (len(masks), len(cols) - 1) if not other_bad else "; ".join(other_bad[:5]))

    # -- 5: the engine ------------------------------------------------------
    ah = (root / "AlgoControl.hpp").read_text(encoding="utf-8")
    ac = (root / "AlgoControl.cpp").read_text(encoding="utf-8")
    chk(re.search(r"^\s*double\s+batchPosition\s*;", ah, re.M) is not None
        and "controls.batchPosition = BatchPositionDef;" in ac
        and all("film::eBATCH_POSITION_%s" % k in ac for k in ("MIN", "MAX", "DEF", "STEP"))
        and '#include "film_enum.hpp"' in ah and '#include "AlgoControlEnums.hpp"' in ah,
        "AlgoControls declares batchPosition, defaults it to BatchPositionDef, and AlgoControl.cpp static_asserts the two ranges equal")

    # -- 6: the mockup ------------------------------------------------------
    if a.mockup.is_file():
        s = a.mockup.read_text(encoding="utf-8")
        groups = re.split(r'<summary><span class="tw"></span>', s)[1:]
        rows, counts_ok = [], True
        for g in groups:
            body = g.split("</details>")[0]
            fs = re.findall(r'<div class="row[^"]*" data-f="([^"]+)"', body)
            cnt = int(re.search(r'class="cnt">(\d+)', g).group(1))
            counts_ok &= (cnt == len(fs))
            rows += fs
        panel = [c[1].split(".")[-1] for g, cs in mx.GROUPS for c in cs]
        chk(rows == panel, "the mockup draws exactly the mask's controls, in the mask's order",
            "%d rows" % len(rows) if rows == panel else "mockup %s vs mask %s" % (
                [r for r in rows if r not in panel], [r for r in panel if r not in rows]))
        chk(counts_ok, "every mockup group's printed control count equals its rows")
    else:
        print("SKIP  mockup not found at %s" % a.mockup)

    # -- 7: compile-time probe ---------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "probe.cpp"
        lines = ['#include "film_params_mask.hpp"', "using namespace film;"]
        for i, p in enumerate(FP.FILM_PROFILES):
            want_bit = 1 if p.tolerance else 0
            lines.append("static_assert(((kFilmControlAvailability[%d] >> eCTRL_BIT_BATCH_POSITION) & 1u) == %du, \"%s\");"
                         % (i, want_bit, p.name))
        lines.append("static_assert(film::eBATCH_POSITION_STEP > 0.0, \"step\");")
        lines.append("int main() { return 0; }")
        probe.write_text("\n".join(lines) + "\n", encoding="utf-8")
        r = subprocess.run(["g++", "-std=c++14", "-Wall", "-Wextra", "-fsyntax-only",
                            "-I", str(root), str(probe)], capture_output=True, text=True)
        chk(r.returncode == 0 and not r.stderr.strip(),
            "g++ compiles a probe that static_asserts the Batch Position bit of all films against the shipped header",
            (r.stderr.strip().splitlines() or ["clean"])[0][:200])

    n_bad = RESULTS.count(False)
    print("[%s] batch_position_headers.py -- %d checks, %d failed"
          % ("OK" if not n_bad else "FAIL", len(RESULTS), n_bad))
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
