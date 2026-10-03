#!/usr/bin/env python3
"""Which database fields each engine actually reads (2026-10-04).

Walks the Python schema (every dataclass reachable from FilmProfile and
PrintStock), then looks for each leaf field in the three consumers:

  * Python  -- film_sim.py, plus the film_profiles.py accessor methods that
               film_sim calls (two hops: `g.rms_rgb()` counts as reading
               rms_r/g/b because the method body does);
  * Scalar  -- every hand-written C++ file in the engine root (the generated
               database files are excluded), plus the inline methods of
               film_profiles.hpp those files call (two hops as well);
  * AVX2    -- the root files that have no AVX2/ twin, plus AVX2/*.cpp.

Comments and string literals are stripped before searching, and a field
counts as read only through a member access (`.name`, `->name`) or, in
Python, `getattr(obj, "name")`. The audit is name-based: a field whose name
collides with an unrelated member elsewhere is reported with its hit files so
the collision can be seen (`field_coverage.py` of 2026-09 mislabelled such
cases; this file lists where the name was found).

Classification:
  USED       read by Python and by both C++ engines
  CPP_ONLY   read by C++ (either engine) but not by Python
  PY_ONLY    read by Python but by neither engine
  DESC       text / provenance: name, description, source, note, ...
  INERT      read by nobody

Usage: python3 field_usage.py [--root ENGINE_DIR] [--out doc/DB_FIELD_USAGE.md] [--assert]
--assert fails when a field read by one engine is not read by the other
(Scalar vs AVX2 must consume the same database), which is the owner's rule.
"""
from __future__ import annotations

import argparse
import dataclasses
import re
import sys
import typing
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DESC_TYPES = (str,)
DESC_NAMES = {"name", "aliases", "description", "era", "source", "note", "notes",
              "sources", "fitted_from", "last_reviewed", "referred", "speed_criterion",
              "mask_encoding", "designation", "criterion", "report", "conditions",
              "param", "confidence", "status", "resolving_optic",
              "resolving_target_contrast", "agitation", "alternative_regime",
              "combined_source", "measurement_mode", "magenta_coupler_class",
              "printing_matrix_source", "printing_matrix_measured", "habit",
              "sensitization", "base_material", "base_type", "antihalation",
              "developer", "dilution", "vessel", "edition", "film_format",
              "reference_developer", "reference_dilution", "reference_edition",
              "grain_reference_developer", "variant_id", "process", "print_stock",
              "default_print", "default_format", "pattern", "filter", "model",
              "category", "ageing_drift_quantity", "test_object_contrast", "order",
              "normalisation", "normalisation_neutral", "measured_through"}

# Presence / validation helpers: they test that a field is populated, they do
# not feed a pixel. Reads through them do not count as consumption.
PRESENCE_HELPERS = {"validate", "validate_all", "renders", "has_data", "has_neutral_pair",
                    "has_density", "has_rate_law", "hasData", "hasNeutralPair"}

# Python schema name -> the C++ member names the generator emits it under.
CPP_ALIASES = {
    "gamma_lo_rgb": ["gamma_lo_r", "gamma_lo_g", "gamma_lo_b"],
    "gamma_hi_rgb": ["gamma_hi_r", "gamma_hi_g", "gamma_hi_b"],
    "dmin_min_rgb": ["dmin_min_r", "dmin_min_g", "dmin_min_b"],
    "dmin_max_rgb": ["dmin_max_r", "dmin_max_g", "dmin_max_b"],
    "cut_on_nm": ["taking_filter_cut_on_nm"],
    "sigma_shape_points": ["sigma_pts_n", "sigma_pts_d", "sigma_pts_s"],
    "log_h": ["meas_x"], "density": ["meas_d"],
}

# Known readings of the name search that need a human note in the report.
NOTES = {
    "dye_impurity.ratios.lo": "name collision: the C++ hits are the `lo` member of the Callier / curve-LUT structs, not this field; INERT in fact",
    "dye_impurity.ratios.hi": "name collision: the C++ hits are the `hi` member of the Callier LUT, not this field; INERT in fact",
    "spectral.log_s_c": "the only C++ reader, AlgoSpectralFourthLayerPeakNm, is called by nothing; INERT in fact (queue P96s)",
    "PrintStock.spectral.log_s_c": "see spectral.log_s_c",
    "halation.radius_scale_r": "Python-only by design: all 222 stocks ship 1.0, the C++ engines build one shared kernel (radii_are_shared)",
    "halation.radius_scale_g": "see radius_scale_r", "halation.radius_scale_b": "see radius_scale_r",
    "taking_filter.transmission": "Python-only carrier for a MEASURED filter curve; no stock stores one, so both engines apply the same ideal long-pass at cut_on_nm",
    "spectral.measured_through.transmission": "provenance of the plotted curve; a measured curve would be read by Python only; none stored",
    "PrintStock.spectral.measured_through.transmission": "see spectral.measured_through.transmission",
    "temporal.weave_amp_x_um": "C++-only: film_sim.py has no motion model (stages 9b, 15, 16 have no Python reference)",
    "temporal.weave_amp_y_um": "see weave_amp_x_um", "temporal.weave_hz_corner": "see weave_amp_x_um",
    "temporal.dirt_events_per_frame": "see weave_amp_x_um", "temporal.scratch_persistence_frames": "see weave_amp_x_um",
}

# Accessor-ish helpers in film_profiles that are keys, not pixels.
KEYLIKE = {"variant_id", "developer", "dilution", "vessel", "edition", "film_format",
           "print_stock", "default_print", "default_format"}


def strip_cpp(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    src = re.sub(r'"(?:\\.|[^"\\\n])*"', '""', src)
    return src


def strip_py(src: str) -> str:
    # Docstrings and comments go; short string literals STAY, because
    # `getattr(obj, "field")` is a read and must remain visible.
    src = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', '""', src)
    src = re.sub(r"#[^\n]*", " ", src)
    return src


def walk_schema(root_cls, seen=None, prefix=""):
    """Yield (path, type, is_descriptive, owner_class_name) for every leaf field."""
    seen = seen or set()
    for f in dataclasses.fields(root_cls):
        t = f.type
        origin = typing.get_origin(t)
        args = typing.get_args(t)
        inner = None
        if dataclasses.is_dataclass(t):
            inner = t
        elif origin in (tuple, list) and args and dataclasses.is_dataclass(args[0]):
            inner = args[0]
        elif origin is typing.Union and args:
            for a in args:
                if dataclasses.is_dataclass(a):
                    inner = a
        if inner is not None and inner.__name__ not in seen:
            yield from walk_schema(inner, seen | {inner.__name__}, prefix + f.name + ".")
            continue
        desc = (f.name in DESC_NAMES) or (t in DESC_TYPES) or (t == "str")
        yield prefix + f.name, getattr(t, "__name__", str(t)), desc, root_cls.__name__


def engine_files(root: Path):
    rootfiles = [p for p in root.iterdir()
                 if p.suffix in (".cpp", ".hpp", ".h") and not p.name.startswith("film_profile")   # film_profiles*, film_profile_serial.hpp
                 and not p.name.startswith("LoadFilmDataBase") and p.name != "film_enum.hpp"
                 and p.name != "film_params_mask.hpp" and not p.name.startswith("test_")]
    avx = [p for p in (root / "AVX2").iterdir() if p.suffix in (".cpp", ".hpp")]
    avx_names = {p.name for p in avx}
    scalar = rootfiles
    avx2 = [p for p in rootfiles if p.name not in avx_names] + avx
    return scalar, avx2


def member_hits(text: str, name: str) -> int:
    n = 0
    for nm in [name] + CPP_ALIASES.get(name, []):
        n += len(re.findall(r"(?:\.|->)%s\b" % re.escape(nm), text))
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/work/proot")
    ap.add_argument("--out", default=str(HERE / "doc" / "DB_FIELD_USAGE.md"))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    a = ap.parse_args(argv)
    import film_profiles as fp

    root = Path(a.root)
    scalar_files, avx2_files = engine_files(root)
    scalar_txt = {p.name: strip_cpp(p.read_text(errors="replace")) for p in scalar_files}
    avx2_txt = {("AVX2/" + p.name if p.parent.name == "AVX2" else p.name):
                strip_cpp(p.read_text(errors="replace")) for p in avx2_files}
    hpp_txt = strip_cpp((root / "film_profiles.hpp").read_text(errors="replace"))
    py_txt = strip_py((HERE / "film_sim.py").read_text())

    # --- second hop: methods, scoped to the struct that owns the field, and
    #     transitive (a method called by a called method counts) --------------
    # C++: struct -> {method: body} from film_profiles.hpp
    cpp_struct_methods = {}
    for sm in re.finditer(r"\bstruct\s+(\w+)\s*\{", hpp_txt):
        sname = sm.group(1); start = sm.end(); depth = 1; i = start
        while i < len(hpp_txt) and depth:
            depth += {"{": 1, "}": -1}.get(hpp_txt[i], 0); i += 1
        body = hpp_txt[start:i]
        meths = {}
        for m in re.finditer(r"\b(\w+)\s*\([^;{)]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{", body):
            nm = m.group(1)
            if nm in ("if", "for", "while", "switch", "return", "sizeof", sname):
                continue
            st = m.end(); d = 1; k = st
            while k < len(body) and d:
                d += {"{": 1, "}": -1}.get(body[k], 0); k += 1
            meths[nm] = meths.get(nm, "") + body[st:k]
        cpp_struct_methods[sname] = meths

    # Free (namespace-scope) inline functions of film_profiles.hpp, e.g.
    # PrintingDensityMatrixFor: available to every owner.
    cpp_free = {}
    depth = 0; i = 0; body_free = []
    for m in re.finditer(r"\b(?:inline\s+)?[\w:<>&\s\*]+?\b(\w+)\s*\([^;{)]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{", hpp_txt):
        nm = m.group(1)
        if nm in ("if", "for", "while", "switch", "return", "sizeof"):
            continue
        # keep only functions not inside a struct: count braces before m.start()
        pre = hpp_txt[:m.start()]
        if pre.count("{") - pre.count("}") > 1:   # namespace film { ... } is depth 1
            continue
        st = m.end(); d = 1; k = st
        while k < len(hpp_txt) and d:
            d += {"{": 1, "}": -1}.get(hpp_txt[k], 0); k += 1
        cpp_free[nm] = cpp_free.get(nm, "") + hpp_txt[st:k]

    def cpp_called_closure(sname, txts):
        meths = dict(cpp_struct_methods.get(sname, {})); meths.update(cpp_free)
        meths = {k: v for k, v in meths.items() if k not in PRESENCE_HELPERS}
        called = {k for k in meths if any(re.search(r"(?:\.|->|\b)%s\s*\(" % re.escape(k), t) for t in txts.values())}
        grown = True
        while grown:
            grown = False
            for k, b in meths.items():
                if k not in called and any(re.search(r"\b%s\s*\(" % re.escape(k), meths[c]) for c in called):
                    called.add(k); grown = True
        return {k: meths[k] for k in called}

    # Python: class -> {method: body}, plus module functions, by AST
    import ast
    fp_src = (HERE / "film_profiles.py").read_text()
    tree = ast.parse(fp_src)
    py_class_methods, py_funcs = {}, {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            d = {}
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    d[sub.name] = d.get(sub.name, "") + strip_py(ast.get_source_segment(fp_src, sub) or "")
            py_class_methods[node.name] = d
        elif isinstance(node, ast.FunctionDef):
            py_funcs[node.name] = py_funcs.get(node.name, "") + strip_py(ast.get_source_segment(fp_src, node) or "")

    def py_closure(cname):
        meths = py_class_methods.get(cname, {})
        pool = dict(meths); pool.update(py_funcs)
        pool = {k: v for k, v in pool.items() if k not in PRESENCE_HELPERS and not k.startswith("_")}
        called = {k for k in pool if re.search(r"\b%s\s*\(" % re.escape(k), py_txt)
                  or (k in meths and re.search(r"\.%s\b" % re.escape(k), py_txt))}   # properties
        grown = True
        while grown:
            grown = False
            for k in pool:
                if k not in called and any(re.search(r"(?:\.|\b)%s\s*\(" % re.escape(k), pool[c])
                                           or (k in meths and re.search(r"self\.%s\b" % re.escape(k), pool[c]))
                                           for c in called):
                    called.add(k); grown = True
        return {k: pool[k] for k in called}

    py_closure_cache, cpp_sc_cache, cpp_av_cache = {}, {}, {}

    rows = []
    for path, typ, desc, owner in list(walk_schema(fp.FilmProfile)) + \
            [("PrintStock." + p, t, d, o) for p, t, d, o in walk_schema(fp.PrintStock)]:
        leaf = path.split(".")[-1]
        if owner not in py_closure_cache:
            py_closure_cache[owner] = py_closure(owner)
            cpp_sc_cache[owner] = cpp_called_closure(owner, scalar_txt)
            cpp_av_cache[owner] = cpp_called_closure(owner, avx2_txt)
        hits_py = []
        if re.search(r"(?:\.|getattr\([^,]+,\s*['\"])%s\b" % re.escape(leaf), py_txt):
            hits_py.append("film_sim.py")
        for k, body in py_closure_cache[owner].items():
            if re.search(r"(?:self\.|\.|getattr\([^,]+,\s*['\"])%s\b" % re.escape(leaf), body):
                hits_py.append("film_profiles.%s()" % k)
        hits_sc = [f for f, t in scalar_txt.items() if member_hits(t, leaf)]
        hits_av = [f for f, t in avx2_txt.items() if member_hits(t, leaf)]
        for k, body in cpp_sc_cache[owner].items():
            if any(re.search(r"\b%s\b" % re.escape(nm), body) for nm in [leaf] + CPP_ALIASES.get(leaf, [])):
                hits_sc.append("film_profiles.hpp::%s::%s()" % (owner, k))
        for k, body in cpp_av_cache[owner].items():
            if any(re.search(r"\b%s\b" % re.escape(nm), body) for nm in [leaf] + CPP_ALIASES.get(leaf, [])):
                hits_av.append("film_profiles.hpp::%s::%s()" % (owner, k))
        def render_reads(h):
            return [x for x in h if not any(x.endswith("%s()" % p) or x.endswith("::%s()" % p) or x == "film_profiles.%s()" % p for p in PRESENCE_HELPERS)]
        rp, rs, ra = render_reads(hits_py), render_reads(hits_sc), render_reads(hits_av)
        hits_py, hits_sc, hits_av = rp, rs, ra
        if desc:
            status = "DESC"
        elif hits_py and hits_sc and hits_av:
            status = "USED"
        elif (hits_sc or hits_av) and not hits_py:
            status = "CPP_ONLY"
        elif hits_py and not (hits_sc or hits_av):
            status = "PY_ONLY"
        elif hits_py and (hits_sc or hits_av):
            status = "USED"
        else:
            status = "INERT"
        rows.append((path, typ, status, sorted(set(hits_py)), sorted(set(hits_sc)), sorted(set(hits_av))))

    counts = {}
    for r in rows:
        counts[r[2]] = counts.get(r[2], 0) + 1
    mismatch = [r for r in rows if r[2] != "DESC" and (bool(r[4]) != bool(r[5]))]

    out = []
    out.append("# DB_FIELD_USAGE.md — which engine reads which database field (generated)\n")
    out.append("Generated by `field_usage.py` against `%s` and `film_sim.py`. Name-based member-access search, "
               "comments and strings stripped, two hops through accessor methods. "
               "Counts: %s. Fields: %d.\n" % (root, ", ".join("%s %d" % kv for kv in sorted(counts.items())), len(rows)))
    out.append("\n**Scalar / AVX2 consumption mismatch (must be empty):** %s\n"
               % (", ".join(r[0] for r in mismatch) if mismatch else "none"))
    out.append("\nStatus: USED = read by Python and by both C++ engines; CPP_ONLY / PY_ONLY = read by one side only "
               "(each such field carries a note); DESC = descriptive or provenance text; INERT = read by no engine "
               "(stored for completeness, audit or a future consumer). Reads through presence-only helpers "
               "(`validate`, `has_data`, ...) are not counted.\n")
    one_side = [r for r in rows if r[2] in ("CPP_ONLY", "PY_ONLY")]
    out.append("\n## One-engine fields (%d)\n\n| field | status | note |\n|---|---|---|\n" % len(one_side))
    for path, typ, status, hp, hs, ha in one_side:
        out.append("| `%s` | %s | %s |\n" % (path, status, NOTES.get(path, "")))
    out.append("\n## All fields\n\n| field | type | status | Python readers | Scalar readers | AVX2 readers |\n|---|---|---|---|---|---|\n")
    for path, typ, status, hp, hs, ha in rows:
        def fmt(h):
            return "<br>".join(h[:4]) + (" (+%d)" % (len(h) - 4) if len(h) > 4 else "") if h else "—"
        out.append("| `%s` | %s | %s | %s | %s | %s |\n" % (path, typ, status, fmt(hp), fmt(hs), fmt(ha)))
    Path(a.out).write_text("".join(out), encoding="utf-8")
    print("[%s] field_usage: %s; mismatch %d -> %s"
          % ("OK" if not mismatch else "FAIL", ", ".join("%s %d" % kv for kv in sorted(counts.items())),
             len(mismatch), a.out))
    return 1 if (mismatch and a.do_assert) else 0


if __name__ == "__main__":
    sys.exit(main())
