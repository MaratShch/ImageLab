#!/usr/bin/env python3
"""serial_codegen.py -- film_profile_serial.hpp: the serialization /
deserialization API of ONE film::FilmProfile, GENERATED from the one struct
definition in film_profiles.hpp (2026-10-05).

Owner rules (2026-10-05):
  * the de-/serialization header is generated as part of the film database
    and ships in the database archive, like film_profiles.hpp;
  * it defines no new data structure: it serializes film::FilmProfile and
    deserializes INTO film::FilmProfile, naming members through macro tables
    this module derives from film_profiles.hpp (add a member to the schema,
    regenerate, and it is in the buffer);
  * every numeric / bool / enum / array / vector / record member is carried,
    whether or not an engine reads it today (missing data and planned stages
    included); a std::string / const char* only when an engine or film_sim.py
    reads it as a key; the provenance subtrees never.

This module does three things:
  1. parses film_profiles.hpp (struct bodies, member declarations, enums,
     aliases, constants) and walks the tree from FilmProfile;
  2. decides the byte layout (fixed block offsets at natural alignment,
     sections in a fixed order, record layouts, string table) and computes the
     exact serialized size of every profile from the Python database, so the
     header can carry kMaxSerializedFilmProfileSize as a constexpr;
  3. emits film_profile_serial.hpp = the layout tables as X-macros + the API
     (a fixed C++ text, API_TEMPLATE below, that expands the macros).

The engine tree's test_film_serial.cpp expands the same macros to compare
a round-tripped FilmProfile with the original, field by field.

Usage: python3 serial_codegen.py [--root ENGINE_ROOT] [--out film_profile_serial.hpp] [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT_STRUCT = "FilmProfile"
OUT_NAME = "film_profile_serial.hpp"
FORMAT_VERSION = 1
MAX_STRING_BYTES = 1024
MAX_DISTINCT_STRINGS = 4096

#: Sub-trees that are provenance by definition (every member, numeric or not).
INFO_SUBTREES = {"provenance", "param_sources"}

SCALAR_KINDS = {            # C++ spelling -> kind
    "float": "F32", "double": "F64",
    "int": "I32", "int32_t": "I32", "std::int32_t": "I32",
    "uint32_t": "U32", "std::uint32_t": "U32",
    "uint8_t": "U8", "std::uint8_t": "U8",
    "bool": "BOOL",
}
KIND_SIZE = {"F32": 4, "F64": 8, "I32": 4, "U32": 4, "U8": 1, "BOOL": 1}


# ---------------------------------------------------------------------------
#  Parsing film_profiles.hpp
# ---------------------------------------------------------------------------
def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    return src


def _skip_braces(text: str, i: int) -> int:
    depth = 0
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


@dataclass
class Member:
    name: str
    ctype: str
    count = 1


@dataclass
class Struct:
    name: str
    members: list = field(default_factory=list)


@dataclass
class Enum:
    name: str
    underlying: str
    values: dict
    flags: bool = False


def _eval_enumerator(expr: str, known: dict) -> int:
    e = re.sub(r"(\d)[uUlL]+\b", r"\1", expr.strip())
    for k, v in known.items():
        e = re.sub(r"\b%s\b" % re.escape(k), str(v), e)
    if not re.match(r"^[\d\s<>|&()+\-*xXa-fA-F]+$", e):
        raise ValueError("enumerator expression not understood: %r" % expr)
    return int(eval(e, {"__builtins__": {}}, {}))


def parse_header(text: str):
    t = _strip_comments(text)
    enums = {}
    for m in re.finditer(r"\benum\s+class\s+(\w+)\s*(?::\s*([\w:]+))?\s*\{", t):
        body = t[m.end():_skip_braces(t, m.end() - 1) - 1]
        flags = "<<" in body
        vals, nxt = {}, 0
        for item in body.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                nm, ex = item.split("=", 1)
                v = _eval_enumerator(ex, vals)
            else:
                nm, v = item, nxt
            vals[nm.strip()] = v
            nxt = v + 1
        enums[m.group(1)] = Enum(m.group(1), m.group(2) or "int", vals, flags)
    aliases = {}
    for m in re.finditer(r"\busing\s+(\w+)\s*=\s*([^;]+);", t):
        aliases[m.group(1)] = re.sub(r"\s+", " ", m.group(2).strip())
    constants = {m.group(1): int(m.group(2)) for m in
                 re.finditer(r"constexpr\s+(?:int32_t|int|std::int32_t|std::size_t|size_t|unsigned)\s+(\w+)\s*=\s*(\d+)", t)}
    structs = {}
    for m in re.finditer(r"\bstruct\s+(\w+)\s*\{", t):
        start = m.end() - 1
        end = _skip_braces(t, start)
        structs[m.group(1)] = Struct(m.group(1), _parse_members(t[start + 1:end - 1]))
    return structs, enums, aliases, constants


def _parse_members(body: str) -> list:
    members, i, stmt = [], 0, ""
    while i < len(body):
        c = body[i]
        if c == "{":
            j = _skip_braces(body, i)
            stmt += " {} "
            i = j
            continue
        if c == ";":
            _parse_statement(stmt, members)
            stmt = ""
        else:
            stmt += c
        i += 1
    return members


def _parse_statement(stmt: str, members: list) -> None:
    s = re.sub(r"\s+", " ", stmt).strip()
    if not s or "(" in s:
        return
    if re.match(r"^(static|using|typedef|enum|struct|class|friend|template)\b", s):
        return
    s = re.sub(r"\s*=\s*[^,]+", "", s)
    s = re.sub(r"\s*\{\}\s*", "", s)
    m = re.match(r"^((?:const\s+)?[\w:]+(?:\s*<[^;]*?>)?(?:\s*\*)?)\s+(.+)$", s)
    if not m:
        raise ValueError("cannot parse member declaration: %r" % s)
    ctype = re.sub(r"\s+", " ", m.group(1)).replace(" *", "*").strip()
    for decl in _split_top(m.group(2)):
        decl = decl.strip()
        am = re.match(r"^(\w+)\s*(?:\[\s*(\w+)\s*\])?$", decl)
        if not am:
            raise ValueError("cannot parse declarator: %r in %r" % (decl, s))
        mem = Member(am.group(1), ctype)
        if am.group(2):
            mem.count = int(am.group(2)) if am.group(2).isdigit() else am.group(2)
        members.append(mem)


def _split_top(s: str) -> list:
    out, depth, cur = [], 0, ""
    for c in s:
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    out.append(cur)
    return out


# ---------------------------------------------------------------------------
#  Schema tree
# ---------------------------------------------------------------------------
@dataclass
class Leaf:
    path: str
    kind: str          # F32 F64 I32 U32 U8 BOOL | STRING | CSTRING | VECTOR | VSTRING | MEAS | RECORDS | EXCLUDED
    count: int = 1
    elem: str = ""     # VECTOR: element kind; RECORDS: struct name
    ctype: str = ""
    offset: int = -1
    enum: str = ""     # enum type name for enum scalars
    note: str = ""


class Model:
    def __init__(self, structs, enums, aliases, constants):
        self.structs, self.enums, self.aliases, self.constants = structs, enums, aliases, constants

    def resolve(self, ctype: str) -> str:
        seen = set()
        while ctype in self.aliases and ctype not in seen:
            seen.add(ctype)
            ctype = self.aliases[ctype]
        return ctype

    def scalar_kind(self, ctype: str):
        c = self.resolve(ctype)
        if c in SCALAR_KINDS:
            return SCALAR_KINDS[c], ""
        if c in self.enums:
            k = SCALAR_KINDS.get(self.enums[c].underlying)
            if k is None:
                raise ValueError("enum %s: unsupported underlying %s" % (c, self.enums[c].underlying))
            return k, c
        return None, ""

    def array_of(self, ctype: str):
        c = self.resolve(ctype)
        m = re.match(r"^std::array\s*<\s*(.+)\s*,\s*(\w+)\s*>$", c)
        if not m:
            return None
        inner, n = m.group(1).strip(), m.group(2)
        n = int(n) if n.isdigit() else self.constants[n]
        sub = self.array_of(inner)
        return (sub[0], sub[1] * n) if sub else (inner, n)

    def vector_of(self, ctype: str):
        m = re.match(r"^std::vector\s*<\s*(.+)\s*>$", self.resolve(ctype))
        return m.group(1).strip() if m else None


def walk(model: Model, struct_name: str, prefix: str, leaves: list, records: dict, in_record: bool) -> None:
    st = model.structs[struct_name]
    names = {m.name for m in st.members}
    if {"meas_x", "meas_d", "meas_m", "meas_n"} <= names:      # ToneCurve
        for m in st.members:
            if m.name in ("meas_x", "meas_d", "meas_m"):
                continue
            _walk_member(model, m, prefix, leaves, records, in_record)
        leaves.append(Leaf(prefix.rstrip("."), "MEAS", ctype="const float* x3 + int meas_n",
                           note="measured table: x[n], d[n], m[n] as f32"))
        return
    for m in st.members:
        _walk_member(model, m, prefix, leaves, records, in_record)


def _walk_member(model, m, prefix, leaves, records, in_record):
    path = prefix + m.name
    if path.split(".")[0] in INFO_SUBTREES:
        leaves.append(Leaf(path, "EXCLUDED", ctype=m.ctype, note="provenance subtree"))
        return
    ctype = m.ctype
    count = m.count if isinstance(m.count, int) else model.constants[m.count]
    if model.resolve(ctype) == "const char*":
        leaves.append(Leaf(path, "CSTRING", ctype="const char*"))
        return
    k, en = model.scalar_kind(ctype)
    if k:
        leaves.append(Leaf(path, k, count, ctype=ctype + ("[%d]" % count if count != 1 else ""), enum=en))
        return
    arr = model.array_of(ctype)
    if arr:
        ek, en = model.scalar_kind(arr[0])
        if not ek:
            raise ValueError("array of non-scalar: %s %s" % (path, ctype))
        leaves.append(Leaf(path, ek, arr[1] * count, ctype=ctype, enum=en))
        return
    if model.resolve(ctype) == "std::string":
        leaves.append(Leaf(path, "STRING", ctype="std::string"))
        return
    vec = model.vector_of(ctype)
    if vec is not None:
        ek, en = model.scalar_kind(vec)
        if ek:
            if ek == "BOOL":
                raise ValueError("vector<bool> is not supported: %s" % path)
            leaves.append(Leaf(path, "VECTOR", elem=ek, ctype=ctype))
            return
        if model.resolve(vec) == "std::string":
            leaves.append(Leaf(path, "VSTRING", ctype=ctype))
            return
        if vec in model.structs:
            if in_record:
                raise ValueError("vector<struct> inside a record is not supported: %s" % path)
            rec_leaves = []
            walk(model, vec, "", rec_leaves, records, True)
            records[vec] = rec_leaves
            leaves.append(Leaf(path, "RECORDS", elem=vec, ctype=ctype))
            return
        raise ValueError("vector of unsupported type: %s %s" % (path, ctype))
    if ctype in model.structs:
        walk(model, ctype, path + ".", leaves, records, in_record)
        return
    raise ValueError("unsupported member type: %s %s" % (path, ctype))


# ---------------------------------------------------------------------------
#  Which strings are keys: read by an engine or by film_sim.py
# ---------------------------------------------------------------------------
def string_readers(root: Path):
    import field_usage as fu
    scalar_files, avx2_files = fu.engine_files(root)
    cpp = "\n".join(fu.strip_cpp(p.read_text(errors="replace")) for p in set(scalar_files + avx2_files))
    hpp = fu.strip_cpp((root / "film_profiles.hpp").read_text(errors="replace"))
    called = set(re.findall(r"(?:\.|->)(\w+)\s*\(", cpp))
    bodies = []
    for m in re.finditer(r"\b(\w+)\s*\([^;{)]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{", hpp):
        if m.group(1) in called and m.group(1) not in fu.PRESENCE_HELPERS:
            st = m.end(); d = 1; k = st
            while k < len(hpp) and d:
                d += {"{": 1, "}": -1}.get(hpp[k], 0); k += 1
            bodies.append(hpp[st:k])
    cpp_all = cpp + "\n" + "\n".join(bodies)
    py = fu.strip_py((HERE / "film_sim.py").read_text())

    def readers(leaf: str) -> list:
        out = []
        if re.search(r"(?:\.|->)%s\b(?!\s*\()" % re.escape(leaf), cpp_all):
            out.append("C++")
        if re.search(r"(?:\.|getattr\([^,]+,\s*['\"])%s\b(?!\s*\()" % re.escape(leaf), py):
            out.append("Python")
        return out
    return readers


# ---------------------------------------------------------------------------
#  Layout
# ---------------------------------------------------------------------------
def _align(n, a):
    return (n + a - 1) // a * a


def layout_fixed(leaves: list) -> int:
    off = 0
    for lf in leaves:
        if lf.kind in KIND_SIZE:
            sz = KIND_SIZE[lf.kind]
            off = _align(off, sz)
            lf.offset = off
            off += sz * lf.count
    return _align(off, 8)


def record_layout(rec_leaves: list) -> dict:
    size = layout_fixed(rec_leaves)
    off = size
    strings, vectors, meas = [], [], []
    for lf in rec_leaves:
        if lf.kind in ("STRING", "CSTRING"):
            strings.append((lf.path, off, lf.kind)); off += 4
    for lf in rec_leaves:
        if lf.kind == "VECTOR":
            vectors.append((lf.path, lf.elem, off, off + 4)); off += 8
        elif lf.kind == "VSTRING":
            raise ValueError("vector<string> inside a record is not supported: %s" % lf.path)
    meas_n_off = {lf.path: lf.offset for lf in rec_leaves if lf.path.endswith(".meas_n") or lf.path == "meas_n"}
    for lf in rec_leaves:
        if lf.kind == "MEAS":
            npath = (lf.path + ".meas_n") if lf.path else "meas_n"
            meas.append((lf.path, off, npath, meas_n_off[npath])); off += 4
    return {"size": _align(off, 8), "strings": strings, "vectors": vectors, "meas": meas}


def _ident(path: str) -> str:
    return path.replace(".", "__")


def _sec_name(path: str) -> str:
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[._]", path) if w)


def enum_bounds(model: Model, name: str):
    e = model.enums[name]
    vals = list(e.values.values())
    if e.flags:
        mask = 0
        for v in vals:
            mask |= v
        return 0, 0, mask
    return min(vals), max(vals), 0


# ---------------------------------------------------------------------------
#  Exact size of a profile, from the Python database
# ---------------------------------------------------------------------------
def _py_get(obj, path: str):
    """Attribute chain; None when an optional sub-object is absent (the C++
    emitter then writes that struct's defaults: empty vectors, empty strings)."""
    for part in path.split("."):
        if not part:
            continue
        if obj is None:
            return None
        if not hasattr(obj, part):
            raise AttributeError("Python schema has no %r (needed for sizing %r)" % (part, path))
        obj = getattr(obj, part)
    return obj


def _py_len(v) -> int:
    return 0 if v is None else len(v)


def _py_meas_n(curve) -> int:
    m = getattr(curve, "measured", None) if curve is not None else None
    return len(m.log_h) if (m is not None and m.has_data) else 0


def _py_str(v) -> bytes:
    return ("" if v is None else str(v)).encode("utf-8")


def profile_size(p, plan: dict) -> int:
    """Mirror of the C++ walk: header + fixed + directory, then the sections."""
    strings = {b""}
    n = plan["data_offset"]
    for sec in plan["sections"]:
        kind, path, elem = sec["kind"], sec["path"], sec["elem"]
        if kind == "MEAS":
            n += _align(12 * _py_meas_n(_py_get(p, path)), 8)
        elif kind == "ARRAY":
            n += _align(KIND_SIZE[elem] * _py_len(_py_get(p, path)), 8)
        elif kind == "IDX" and path:
            items = _py_get(p, path) or []
            for s in items:
                strings.add(_py_str(s))
            n += _align(4 * len(items), 8)
        elif kind == "IDX":               # named strings
            for sp in plan["named"]:
                strings.add(_py_str(_py_get(p, sp)))
            n += _align(4 * len(plan["named"]), 8)
        elif kind == "RECORDS":
            recs = _py_get(p, path) or []
            n += sec["size"] * len(recs)
            for sp in sec["strings"]:
                for r in recs:
                    strings.add(_py_str(_py_get(r, sp)))
        elif kind == "POOL":
            recs = _py_get(p, sec["recpath"]) or []
            if sec["meas"]:
                n += _align(4 * sum(3 * _py_meas_n(_py_get(r, sec["sub"])) for r in recs), 8)
            else:
                n += _align(KIND_SIZE[elem] * sum(_py_len(_py_get(r, sec["sub"])) for r in recs), 8)
        elif kind == "STRINGS":
            n += _align(8 * len(strings), 8) + _align(sum(len(s) + 1 for s in strings), 8)
    return n


# ---------------------------------------------------------------------------
#  Emission
# ---------------------------------------------------------------------------
def build_plan(root: Path, hpp_path: Path = None):
    hpp_path = hpp_path or (root / "film_profiles.hpp")
    text = hpp_path.read_text(encoding="utf-8", errors="replace")
    structs, enums, aliases, constants = parse_header(text)
    model = Model(structs, enums, aliases, constants)
    leaves, records = [], {}
    walk(model, ROOT_STRUCT, "", leaves, records, False)

    readers = string_readers(root)
    excluded, kept_strings = [], []

    def select(lst, where):
        out = []
        for lf in lst:
            if lf.kind == "EXCLUDED":
                excluded.append((where + lf.path, lf.note)); continue
            if lf.kind in ("STRING", "CSTRING", "VSTRING"):
                r = readers(lf.path.split(".")[-1])
                if not r:
                    excluded.append((where + lf.path, "text, no reader")); continue
                kept_strings.append((where + lf.path, r))
            out.append(lf)
        return out

    leaves = select(leaves, "")
    for rname in list(records):
        records[rname] = select(records[rname], rname + "::")
    leaves = [lf for lf in leaves if not (lf.kind == "RECORDS" and not records[lf.elem])]

    fixed_size = layout_fixed(leaves)
    rec_layouts = {r: record_layout(records[r]) for r in records}
    meas_n_off = {lf.path: lf.offset for lf in leaves if lf.path.endswith(".meas_n")}

    sections = []
    for lf in leaves:
        if lf.kind == "MEAS":
            sections.append({"name": _sec_name(lf.path) + "Meas", "kind": "MEAS", "esize": 12, "path": lf.path,
                             "elem": "F32", "meas_n_off": meas_n_off[lf.path + ".meas_n"]})
    for lf in leaves:
        if lf.kind == "VECTOR":
            sections.append({"name": _sec_name(lf.path), "kind": "ARRAY", "esize": KIND_SIZE[lf.elem], "path": lf.path, "elem": lf.elem})
    for lf in leaves:
        if lf.kind == "VSTRING":
            sections.append({"name": _sec_name(lf.path), "kind": "IDX", "esize": 4, "path": lf.path, "elem": "U32"})
    named = [(lf.path, lf.kind) for lf in leaves if lf.kind in ("STRING", "CSTRING")]
    sections.append({"name": "NamedStrings", "kind": "IDX", "esize": 4, "path": "", "elem": "U32"})
    for lf in leaves:
        if lf.kind == "RECORDS":
            rl = rec_layouts[lf.elem]
            sections.append({"name": _sec_name(lf.path), "kind": "RECORDS", "esize": rl["size"], "path": lf.path,
                             "elem": lf.elem, "size": rl["size"], "strings": [s[0] for s in rl["strings"]]})
            for (vp, ek, _, _) in rl["vectors"]:
                sections.append({"name": _sec_name(lf.path) + "_" + _sec_name(vp), "kind": "POOL", "esize": KIND_SIZE[ek],
                                 "path": lf.path + "[]." + vp, "elem": ek, "recpath": lf.path, "sub": vp, "meas": False})
            for (mp, _, _, _) in rl["meas"]:
                sections.append({"name": _sec_name(lf.path) + "_" + _sec_name(mp) + "Meas", "kind": "POOL", "esize": 4,
                                 "path": lf.path + "[]." + mp, "elem": "F32", "recpath": lf.path, "sub": mp, "meas": True})
    sections.append({"name": "StringTable", "kind": "STRINGS", "esize": 0, "path": "", "elem": ""})

    directory_offset = 32 + fixed_size
    data_offset = _align(directory_offset + 8 * len(sections), 8)
    plan = {"model": model, "leaves": leaves, "records": records, "rec_layouts": rec_layouts,
            "sections": sections, "named": [p for p, _ in named], "named_kinds": named,
            "fixed_size": fixed_size, "directory_offset": directory_offset, "data_offset": data_offset,
            "excluded": excluded, "kept_strings": kept_strings}
    return plan


def generate(root: Path, hpp_path: Path = None):
    import film_profiles as fp
    import cpp_codegen
    plan = build_plan(root, hpp_path)
    model, leaves, records, rec_layouts, sections = (plan["model"], plan["leaves"], plan["records"],
                                                     plan["rec_layouts"], plan["sections"])
    named = plan["named_kinds"]
    db = list(fp.FILM_PROFILES)
    sizes = sorted(((profile_size(p, plan), p.name) for p in db), reverse=True)
    max_size, max_name = sizes[0]
    schema = int(fp.SCHEMA_VERSION)

    L = []
    w = L.append
    w(cpp_codegen.COPYRIGHT_NOTICE.rstrip("\n"))
    w("// film_profile_serial.hpp -- GENERATED by serial_codegen.py from film_profiles.hpp")
    w("// (schema v%d, %d profiles). DO NOT EDIT BY HAND; regenerate with the database." % (schema, len(db)))
    w("//")
    w("// Serialization of ONE film::FilmProfile into a contiguous little-endian byte")
    w("// buffer, and deserialization back INTO a film::FilmProfile. No data structure")
    w("// is defined here beyond the API's own enumerations and constants: every field")
    w("// is named through the macro tables below, which serial_codegen.py derives from")
    w("// the struct definitions in film_profiles.hpp, so the database header remains")
    w("// the one definition of the data.")
    w("//")
    w("// WHAT IS CARRIED (owner rules 2026-10-05): every numeric / bool / enum / array /")
    w("// vector / record member of FilmProfile, read by an engine today or not; a")
    w("// std::string or const char* only when an engine or film_sim.py reads it (a key);")
    w("// never the provenance subtrees (%s). Lists at the end." % ", ".join(sorted(INFO_SUBTREES)))
    w("//")
    w("// NUMERIC TYPES are stored as the database holds them (float -> f32, double -> f64,")
    w("// int -> i32, bool -> u8 0/1, enum -> underlying integer); no conversion, so a")
    w("// round trip is bit-exact. BYTE ORDER little-endian, fixed. ALIGNMENT: the reader")
    w("// assumes none (memcpy loads); the layout keeps natural alignment relative to the")
    w("// buffer start, every section at a multiple of 8.")
    w("//")
    w("// BUFFER LAYOUT (offsets from the buffer start)")
    w("//   0                    32 B   header: u32 magic 0x52455346 'FSER', u32 format_version %d," % FORMAT_VERSION)
    w("//                               u32 schema_version %d, u32 total_bytes, u32 fixed_size %d," % (schema, plan["fixed_size"]))
    w("//                               u32 section_count %d, u32 data_offset %d, u32 reserved 0" % (len(sections), plan["data_offset"]))
    w("//   32                 %4d B   FIXED BLOCK: the scalar / array members, table below" % plan["fixed_size"])
    w("//   %-4d               %4d B   DIRECTORY: %d x {u32 offset, u32 count}, section order below"
      % (plan["directory_offset"], 8 * len(sections), len(sections)))
    w("//   %-4d            variable   DATA: the sections in directory order, each 8-aligned" % plan["data_offset"])
    w("//   total_bytes                 end (a multiple of 8)")
    w("//")
    w("// SECTION ENCODINGS")
    w("//   MEAS     f32 x[n], d[n], m[n] of a measured ToneCurve table; n = that curve's meas_n")
    w("//            in the fixed block (checked); 12n bytes, padded to 8")
    w("//   ARRAY    n elements of the vector's scalar type, padded to 8")
    w("//   IDX      n x u32 string-table indices, padded to 8 (vector<string>, named strings)")
    w("//   RECORDS  n fixed-size records (layouts below; their internal padding is zero)")
    w("//   POOL     the nested vectors / measured tables of ALL records of one list,")
    w("//            concatenated in record order; each record holds {first, count} or")
    w("//            {first} (count = 3 * its meas_n) into the pool")
    w("//   STRINGS  the string table, last: u32 pos[N], u32 len[N], pad to 8, then the")
    w("//            UTF-8 bytes of every string, each followed by one NUL (so const char*")
    w("//            members can point at it), padded to 8. pos is from the buffer start,")
    w("//            strings are contiguous in index order, index 0 is always \"\".")
    w("//   Padding at the end of every section is 0xFF bytes (checked by the reader; 0xFF")
    w("//   is never a valid 4-byte element, so a count edited to swallow or release")
    w("//   padding is rejected). Padding inside the fixed block and records is zero.")
    w("//")
    w("// FIXED BLOCK (%d bytes): offset / size / kind / member of film::FilmProfile" % plan["fixed_size"])
    for lf in leaves:
        if lf.kind in KIND_SIZE:
            w("//   %5d  %4d  %-4s  %s%s%s" % (lf.offset, KIND_SIZE[lf.kind] * lf.count, lf.kind, lf.path,
                                               ("  [%d]" % lf.count) if lf.count != 1 else "",
                                               ("  (%s)" % lf.enum) if lf.enum else ""))
    w("//")
    w("// SECTIONS (directory order): # / name / kind / element bytes / source member")
    for i, s in enumerate(sections):
        w("//   %3d  %-44s %-8s %3d  %s" % (i, s["name"], s["kind"], s["esize"], s["path"]))
    w("//")
    w("// NAMED STRINGS (section NamedStrings; u32 index into StringTable each):")
    for i, (p, k) in enumerate(named):
        w("//   %3d  %s%s" % (i, p, "  (const char*)" if k == "CSTRING" else ""))
    for lf in leaves:
        if lf.kind != "RECORDS":
            continue
        rname, rl = lf.elem, rec_layouts[lf.elem]
        w("//")
        w("// RECORD film::%s (%s), %d bytes: offset / size / kind / member" % (rname, lf.path, rl["size"]))
        for r in records[rname]:
            if r.kind in KIND_SIZE:
                w("//   %5d  %4d  %-4s  %s%s%s" % (r.offset, KIND_SIZE[r.kind] * r.count, r.kind, r.path,
                                                   ("  [%d]" % r.count) if r.count != 1 else "",
                                                   ("  (%s)" % r.enum) if r.enum else ""))
        for p, o, k in rl["strings"]:
            w("//   %5d     4  IDX   %s  (string index%s)" % (o, p, ", const char*" if k == "CSTRING" else ""))
        for p, ek, o1, o2 in rl["vectors"]:
            w("//   %5d     8  POOL  %s  (u32 first, u32 count)" % (o1, p))
        for p, o, pn, _ in rl["meas"]:
            w("//   %5d     4  POOL  %s  (u32 first; count = 3 * %s)" % (o, p, pn))
    w("//")
    w("// MAXIMUM SIZE: kMaxSerializedFilmProfileSize = %d = the exact serialized size of" % max_size)
    w("// the largest profile of this database (%s), computed by serial_codegen.py" % max_name)
    w("// from the same layout over the Python database; no margin. test_film_serial.cpp")
    w("// serializes every profile in C++ and fails unless the largest equals it.")
    w("// Smallest %d (%s), median %d." % (sizes[-1][0], sizes[-1][1], sizes[len(sizes) // 2][0]))
    w("//")
    w("// EXCLUDED members (not in the buffer; a round trip leaves them default):")
    for p, why in plan["excluded"]:
        w("//   %-58s %s" % (p, why))
    w("//")
    w("// STRING members carried, and who reads them:")
    for p, r in plan["kept_strings"]:
        w("//   %-58s %s" % (p, ", ".join(r)))
    w("")
    w("#ifndef FILM_PROFILE_SERIAL_HPP")
    w("#define FILM_PROFILE_SERIAL_HPP")
    w("")
    w("#define FILM_SERIAL_SCHEMA_VERSION %d" % schema)
    w("#define FILM_SERIAL_FORMAT_VERSION %d" % FORMAT_VERSION)
    w("#define FILM_SERIAL_FIXED_SIZE %d" % plan["fixed_size"])
    w("#define FILM_SERIAL_SECTION_COUNT %d" % len(sections))
    w("#define FILM_SERIAL_NAMED_STRING_COUNT %d" % len(named))
    w("#define FILM_SERIAL_MAX_SIZE %d" % max_size)
    w("")
    w("// Sections, directory order: S(name, kind, element bytes)")
    w("#define FILM_SERIAL_SECTIONS(S) \\")
    for s in sections:
        w("    S(%s, %s, %d) \\" % (s["name"], s["kind"], s["esize"]))
    w("")
    w("// Fixed block: X(member path, kind, byte offset in the fixed block, element count)")
    w("#define FILM_SERIAL_FIXED(X) \\")
    for lf in leaves:
        if lf.kind in KIND_SIZE:
            w("    X(%s, %s, %d, %d) \\" % (lf.path, lf.kind, lf.offset, lf.count))
    w("")
    w("// Enum members of the fixed block: X(member path, kind, offset, lo, hi, flag mask); flags when mask != 0")
    w("#define FILM_SERIAL_FIXED_ENUMS(X) \\")
    for lf in leaves:
        if lf.enum and lf.count == 1:
            lo, hi, mask = enum_bounds(model, lf.enum)
            w("    X(%s, %s, %d, %d, %d, 0x%Xu) \\" % (lf.path, lf.kind, lf.offset, lo, hi, mask))
    w("")
    w("// Named strings: X(member path, index in NamedStrings, STRING | CSTRING)")
    w("#define FILM_SERIAL_STRINGS(X) \\")
    for i, (p, k) in enumerate(named):
        w("    X(%s, %d, %s) \\" % (p, i, k))
    w("")
    w("// Vectors of scalars: X(member path, element kind, section)")
    w("#define FILM_SERIAL_VECTORS(X) \\")
    for lf in leaves:
        if lf.kind == "VECTOR":
            w("    X(%s, %s, %s) \\" % (lf.path, lf.elem, _sec_name(lf.path)))
    w("")
    w("// Vectors of strings: X(member path, section)")
    w("#define FILM_SERIAL_VSTRINGS(X) \\")
    for lf in leaves:
        if lf.kind == "VSTRING":
            w("    X(%s, %s) \\" % (lf.path, _sec_name(lf.path)))
    w("")
    w("// Measured curve tables: X(ToneCurve member path, section, fixed-block offset of its meas_n)")
    w("#define FILM_SERIAL_MEAS(X) \\")
    for s in sections:
        if s["kind"] == "MEAS":
            w("    X(%s, %s, %d) \\" % (s["path"], s["name"], s["meas_n_off"]))
    w("")
    w("// Record lists: R(member path, record type, section, macro prefix)")
    w("#define FILM_SERIAL_RECORDS(R) \\")
    for lf in leaves:
        if lf.kind == "RECORDS":
            w("    R(%s, film::%s, %s, FILM_SERIAL_REC_%s) \\" % (lf.path, lf.elem, _sec_name(lf.path), lf.elem))
    w("")
    for lf in leaves:
        if lf.kind != "RECORDS":
            continue
        rname, rl, pre = lf.elem, rec_layouts[lf.elem], "FILM_SERIAL_REC_" + lf.elem
        w("// ---- record film::%s (%s) ----" % (rname, lf.path))
        w("#define %s_SIZE %d" % (pre, rl["size"]))
        w("#define %s_FIXED(X) \\" % pre)
        for r in records[rname]:
            if r.kind in KIND_SIZE:
                w("    X(%s, %s, %d, %d) \\" % (r.path, r.kind, r.offset, r.count))
        w("")
        w("#define %s_ENUMS(X) \\" % pre)
        for r in records[rname]:
            if r.enum and r.count == 1:
                lo, hi, mask = enum_bounds(model, r.enum)
                w("    X(%s, %s, %d, %d, %d, 0x%Xu) \\" % (r.path, r.kind, r.offset, lo, hi, mask))
        w("")
        w("#define %s_STRINGS(X) \\" % pre)
        for p, o, k in rl["strings"]:
            w("    X(%s, %d, %s) \\" % (p, o, k))
        w("")
        w("#define %s_VECTORS(X) \\" % pre)
        for p, ek, o1, o2 in rl["vectors"]:
            w("    X(%s, %s, %s, %d, %d) \\" % (p, ek, _sec_name(lf.path) + "_" + _sec_name(p), o1, o2))
        w("")
        w("#define %s_MEAS(X) \\" % pre)
        for p, o, pn, on in rl["meas"]:
            w("    X(%s, %s, %d, %d) \\" % (p, _sec_name(lf.path) + "_" + _sec_name(p) + "Meas", o, on))
        w("")
    w("// Fixed-block byte offsets (from the start of the fixed block, i.e. buffer offset 32 + value)")
    w("// for readers that have no film::FilmProfile, e.g. device code:")
    w("namespace film { namespace serial { namespace off {")
    for lf in leaves:
        if lf.kind in KIND_SIZE:
            w("constexpr unsigned %s = %d;" % (_ident(lf.path), lf.offset))
    w("}}}  // namespace film::serial::off")
    w(API_TEMPLATE)
    w("#endif  // FILM_PROFILE_SERIAL_HPP")
    w("")
    stats = {"fixed_size": plan["fixed_size"], "sections": len(sections), "named": len(named),
             "scalars": sum(1 for lf in leaves if lf.kind in KIND_SIZE),
             "vectors": sum(1 for lf in leaves if lf.kind == "VECTOR"),
             "vstrings": sum(1 for lf in leaves if lf.kind == "VSTRING"),
             "meas": sum(1 for lf in leaves if lf.kind == "MEAS"),
             "records": {r: (len(records[r]), rec_layouts[r]["size"]) for r in records},
             "excluded": plan["excluded"], "kept_strings": plan["kept_strings"],
             "max": (max_size, max_name), "min": sizes[-1], "median": sizes[len(sizes) // 2][0]}
    return "\n".join(L), stats


# ---------------------------------------------------------------------------
#  The API: fixed C++ text that expands the tables above.
# ---------------------------------------------------------------------------
API_TEMPLATE = r'''
// ===========================================================================
//  API
//
//    host:   std::size_t SerializeFilmProfile(const film::FilmProfile&, std::uint8_t* buf, std::size_t cap) noexcept;
//            std::size_t SerializedFilmProfileSize(const film::FilmProfile&) noexcept;
//            bool DeserializeFilmProfile(const std::uint8_t* buf, std::size_t size, film::FilmProfile& out) noexcept;
//    any:    bool ValidateFilmProfileBuffer(const std::uint8_t* buf, std::size_t size) noexcept;   (FILM_SERIAL_HD)
//            std::uint32_t SectionOffset(buf, Section), SectionCount(buf, Section)
//            bool StringAt(buf, index, pos&, len&)
//            constexpr std::size_t kMaxSerializedFilmProfileSize
//
//  SerializeFilmProfile writes [buf, buf + n) and returns n (a multiple of 8),
//  or 0 when cap is too small, the host is big-endian, a string exceeds
//  kMaxStringBytes or the profile has more than kMaxDistinctStrings distinct
//  strings. It never writes past cap and stores no pointer. No heap.
//
//  DeserializeFilmProfile validates the buffer completely (ValidateFilmProfileBuffer)
//  and then assigns every carried member of `out`; members not carried are
//  left as `out` had them (assign a fresh film::FilmProfile first to get
//  defaults). ⚠ POINTER MEMBERS POINT INTO THE BUFFER: the measured curve
//  tables (ToneCurve::meas_x/meas_d/meas_m) and const char* members refer to
//  bytes of `buf`, exactly as the compiled database's point at its static
//  arrays, so `buf` must stay alive and unchanged while `out` is used, and must
//  be 4-byte aligned (checked; false otherwise). std::string and std::vector
//  members are copied (host-side allocation).
//
//  GPU: the reader functions are plain functions over the byte buffer, marked
//  FILM_SERIAL_HD (__host__ __device__ under NVCC); they do not need
//  film::FilmProfile. Define FILM_SERIAL_READER_ONLY to compile them without
//  film_profiles.hpp. Scalars are read at kHeaderSize + the FIXED BLOCK offsets
//  listed above (also as constants film::serial::off::<path with underscores>).
// ===========================================================================

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
  #define FILM_SERIAL_HD __host__ __device__
#else
  #define FILM_SERIAL_HD
#endif

#if !defined(FILM_SERIAL_READER_ONLY)
  #include "film_profiles.hpp"
  #include <string>
  #include <vector>
#endif

namespace film {
namespace serial {

constexpr std::uint32_t kMagic               = 0x52455346u;
constexpr std::uint32_t kFormatVersion       = FILM_SERIAL_FORMAT_VERSION;
constexpr std::uint32_t kSerialSchemaVersion = FILM_SERIAL_SCHEMA_VERSION;   // == film::kSchemaVersion
constexpr std::size_t   kHeaderSize          = 32;
constexpr std::size_t   kFixedSize           = FILM_SERIAL_FIXED_SIZE;
constexpr std::uint32_t kSectionCount        = FILM_SERIAL_SECTION_COUNT;
constexpr std::uint32_t kNamedStringCount    = FILM_SERIAL_NAMED_STRING_COUNT;
constexpr std::size_t   kDirectoryOffset     = kHeaderSize + kFixedSize;
constexpr std::size_t   kDataOffset          = ((kDirectoryOffset + 8u * kSectionCount) + 7u) & ~std::size_t(7);
constexpr std::uint32_t kMaxStringBytes      = 1024u;
constexpr std::uint32_t kMaxDistinctStrings  = 4096u;
constexpr std::size_t   kMaxSerializedFilmProfileSize = FILM_SERIAL_MAX_SIZE;

enum class Section : std::uint32_t
{
#define FILM_SERIAL_S_ENUM(name, kind, esize) name,
    FILM_SERIAL_SECTIONS(FILM_SERIAL_S_ENUM)
#undef FILM_SERIAL_S_ENUM
    Count_
};
static_assert(static_cast<std::uint32_t>(Section::Count_) == kSectionCount, "section table");

enum SectionKind : std::uint32_t { MEAS, ARRAY, IDX, RECORDS, POOL, STRINGS };
enum Kind : std::uint32_t { F32, F64, I32, U32, U8, BOOL, STRING, CSTRING };


FILM_SERIAL_HD inline std::size_t   Align8  (std::size_t n) noexcept { return (n + 7u) & ~std::size_t(7); }
FILM_SERIAL_HD inline std::uint32_t LoadU32 (const std::uint8_t* p) noexcept { std::uint32_t v; std::memcpy(&v, p, 4); return v; }
FILM_SERIAL_HD inline std::int32_t  LoadI32 (const std::uint8_t* p) noexcept { std::int32_t  v; std::memcpy(&v, p, 4); return v; }
FILM_SERIAL_HD inline float         LoadF32 (const std::uint8_t* p) noexcept { float  v; std::memcpy(&v, p, 4); return v; }
FILM_SERIAL_HD inline double        LoadF64 (const std::uint8_t* p) noexcept { double v; std::memcpy(&v, p, 8); return v; }

FILM_SERIAL_HD constexpr std::uint32_t KindSize (Kind k) noexcept
{ return (k == F64) ? 8u : (k == U8 || k == BOOL) ? 1u : (k == STRING || k == CSTRING) ? 0u : 4u; }

FILM_SERIAL_HD constexpr SectionKind SectionKindOf (Section s) noexcept
{
    return
#define FILM_SERIAL_S_KIND(name, kind, esize) (s == Section::name) ? kind :
    FILM_SERIAL_SECTIONS(FILM_SERIAL_S_KIND)
#undef FILM_SERIAL_S_KIND
    STRINGS;
}
FILM_SERIAL_HD constexpr std::uint32_t SectionElemSize (Section s) noexcept
{
    return
#define FILM_SERIAL_S_ESIZE(name, kind, esize) (s == Section::name) ? esize##u :
    FILM_SERIAL_SECTIONS(FILM_SERIAL_S_ESIZE)
#undef FILM_SERIAL_S_ESIZE
    0u;
}

// ---- raw readers over a VALIDATED buffer (host or device) ---------------------------------------
FILM_SERIAL_HD inline std::uint32_t SectionOffset (const std::uint8_t* buf, Section s) noexcept
{ return LoadU32(buf + kDirectoryOffset + 8u * static_cast<std::uint32_t>(s)); }
FILM_SERIAL_HD inline std::uint32_t SectionCount (const std::uint8_t* buf, Section s) noexcept
{ return LoadU32(buf + kDirectoryOffset + 8u * static_cast<std::uint32_t>(s) + 4u); }
FILM_SERIAL_HD inline std::uint32_t SectionEnd (const std::uint8_t* buf, Section s) noexcept
{
    const std::uint32_t i = static_cast<std::uint32_t>(s) + 1u;
    return (i < kSectionCount) ? LoadU32(buf + kDirectoryOffset + 8u * i) : LoadU32(buf + 12);
}
/// String `index` of the table: byte position (from the buffer start) and length; the bytes
/// are followed by a NUL. False when index is out of range.
FILM_SERIAL_HD inline bool StringAt (const std::uint8_t* buf, std::uint32_t index,
                                     std::uint32_t& pos, std::uint32_t& len) noexcept
{
    const std::uint32_t n = SectionCount(buf, Section::StringTable);
    if (index >= n) return false;
    const std::uint8_t* t = buf + SectionOffset(buf, Section::StringTable);
    pos = LoadU32(t + 4u * index); len = LoadU32(t + 4u * (n + index));
    return true;
}
/// The i-th string index of an IDX section (vector<string> members, NamedStrings).
FILM_SERIAL_HD inline std::uint32_t IndexAt (const std::uint8_t* buf, Section s, std::uint32_t i) noexcept
{ return LoadU32(buf + SectionOffset(buf, s) + 4u * i); }

// ---- validation ---------------------------------------------------------------------------------
namespace detail
{
    FILM_SERIAL_HD inline bool enumOk (const std::uint8_t* p, Kind k, std::int64_t lo, std::int64_t hi, std::uint32_t mask) noexcept
    {
        std::int64_t v = 0;
        if (k == U8 || k == BOOL) v = p[0];
        else if (k == I32) v = LoadI32(p);
        else v = static_cast<std::int64_t>(LoadU32(p));
        if (mask) return (static_cast<std::uint64_t>(v) & ~static_cast<std::uint64_t>(mask)) == 0u;
        return v >= lo && v <= hi;
    }
}

/// True when [buf, buf + size) is a complete, internally consistent buffer of this
/// format and schema. Reads nothing outside it, whatever the content.
FILM_SERIAL_HD inline bool ValidateFilmProfileBuffer (const std::uint8_t* buf, std::size_t size) noexcept
{
    if (buf == nullptr || size < kDataOffset || size > 0xFFFFFFFFu || (size & 7u) != 0u) return false;
    if (LoadU32(buf + 0)  != kMagic || LoadU32(buf + 4) != kFormatVersion || LoadU32(buf + 8) != kSerialSchemaVersion) return false;
    if (LoadU32(buf + 12) != static_cast<std::uint32_t>(size)) return false;
    if (LoadU32(buf + 16) != static_cast<std::uint32_t>(kFixedSize) || LoadU32(buf + 20) != kSectionCount) return false;
    if (LoadU32(buf + 24) != static_cast<std::uint32_t>(kDataOffset) || LoadU32(buf + 28) != 0u) return false;
    const std::uint8_t* fx = buf + kHeaderSize;

    // fixed block: booleans 0/1, enumerations in range, measured counts >= 0
#define FILM_SERIAL_X_BOOL(path, kind, o, n) if (kind == BOOL) { for (unsigned bi = 0; bi < n; bi++) { if (fx[o + bi] > 1u) { return false; } } }
    FILM_SERIAL_FIXED(FILM_SERIAL_X_BOOL)
#undef FILM_SERIAL_X_BOOL
#define FILM_SERIAL_X_ENUM(path, kind, o, lo, hi, mask) if (!detail::enumOk(fx + o, kind, lo, hi, mask)) { return false; }
    FILM_SERIAL_FIXED_ENUMS(FILM_SERIAL_X_ENUM)
#undef FILM_SERIAL_X_ENUM

    // directory: non-decreasing, 8-aligned, inside the buffer, first == kDataOffset
    std::uint32_t prev = static_cast<std::uint32_t>(kDataOffset);
    for (std::uint32_t i = 0; i < kSectionCount; i++)
    {
        const std::uint32_t o = LoadU32(buf + kDirectoryOffset + 8u * i);
        if (o < prev || (o & 7u) != 0u || o > size || (i == 0u && o != kDataOffset)) return false;
        prev = o;
    }
    const std::uint32_t nStr = SectionCount(buf, Section::StringTable);
    if (nStr == 0u || nStr > kMaxDistinctStrings) return false;

    // every section's declared content fills it exactly
    for (std::uint32_t i = 0; i < kSectionCount; i++)
    {
        const Section s = static_cast<Section>(i);
        const std::uint64_t avail = static_cast<std::uint64_t>(SectionEnd(buf, s)) - SectionOffset(buf, s);
        const std::uint64_t cnt = SectionCount(buf, s);
        const SectionKind k = SectionKindOf(s);
        std::uint64_t need;
        if (k == STRINGS) { need = (cnt * 8u + 7u) & ~std::uint64_t(7); if (need > avail) return false; continue; }
        need = cnt * SectionElemSize(s);
        if (k != RECORDS)
        {
            const std::uint64_t padded = (need + 7u) & ~std::uint64_t(7);
            if (padded != avail) return false;
            for (std::uint64_t z = need; z < padded; z++) if (buf[SectionOffset(buf, s) + z] != 0xFFu) return false;
        }
        else if (need != avail) return false;
    }
    if (SectionCount(buf, Section::NamedStrings) != kNamedStringCount) return false;

    // string table: contiguous NUL-terminated strings in index order, index 0 == ""
    {
        const std::uint32_t tOff = SectionOffset(buf, Section::StringTable), tEnd = SectionEnd(buf, Section::StringTable);
        const std::uint32_t blob = static_cast<std::uint32_t>(tOff + Align8(8u * static_cast<std::size_t>(nStr)));
        for (std::uint32_t z = tOff + 8u * nStr; z < blob; z++) if (buf[z] != 0xFFu) return false;
        std::uint64_t at = blob;
        for (std::uint32_t i = 0; i < nStr; i++)
        {
            const std::uint32_t pos = LoadU32(buf + tOff + 4u * i), len = LoadU32(buf + tOff + 4u * (nStr + i));
            if (len > kMaxStringBytes || pos != at || at + len + 1u > tEnd) return false;
            if (buf[pos + len] != 0u) return false;
            at += len + 1u;
        }
        if (LoadU32(buf + tOff + 4u * nStr) != 0u) return false;
        if (Align8(static_cast<std::size_t>(at - blob)) != tEnd - blob) return false;
        for (std::uint64_t z = at; z < tEnd; z++) if (buf[z] != 0xFFu) return false;
    }
    // index sections
    for (std::uint32_t i = 0; i < kSectionCount; i++)
    {
        const Section s = static_cast<Section>(i);
        if (SectionKindOf(s) != IDX) continue;
        for (std::uint32_t j = 0; j < SectionCount(buf, s); j++) if (IndexAt(buf, s, j) >= nStr) return false;
    }
    // measured tables of the fixed block: count == meas_n
#define FILM_SERIAL_X_MEAS(path, sec, nOff) if (LoadI32(fx + nOff) < 0 || static_cast<std::uint32_t>(LoadI32(fx + nOff)) != SectionCount(buf, Section::sec)) { return false; }
    FILM_SERIAL_MEAS(FILM_SERIAL_X_MEAS)
#undef FILM_SERIAL_X_MEAS
    // records: booleans, enumerations, string indices, pool references
#define FILM_SERIAL_R_BOOL(path, kind, o, n) if (kind == BOOL) { for (unsigned bi = 0; bi < n; bi++) { if (rb[o + bi] > 1u) { return false; } } }
#define FILM_SERIAL_R_ENUM(path, kind, o, lo, hi, mask) if (!detail::enumOk(rb + o, kind, lo, hi, mask)) { return false; }
#define FILM_SERIAL_R_STR(path, o, kind) if (LoadU32(rb + o) >= nStr) { return false; }
#define FILM_SERIAL_R_VEC(path, kind, pool, oFirst, oCount) \
        { const std::uint64_t f = LoadU32(rb + oFirst), c = LoadU32(rb + oCount); \
          if (f != used_##pool || f + c > SectionCount(buf, Section::pool)) { return false; } \
          used_##pool += c; }
#define FILM_SERIAL_R_MEAS(path, pool, oFirst, oN) \
        { const std::int32_t mn = LoadI32(rb + oN); if (mn < 0) { return false; } \
          const std::uint64_t f = LoadU32(rb + oFirst), c = 3u * static_cast<std::uint64_t>(mn); \
          if (f != used_##pool || f + c > SectionCount(buf, Section::pool)) { return false; } \
          used_##pool += c; }
#define FILM_SERIAL_R_VEC_DECL(path, kind, pool, oFirst, oCount) std::uint64_t used_##pool = 0;
#define FILM_SERIAL_R_MEAS_DECL(path, pool, oFirst, oN) std::uint64_t used_##pool = 0;
#define FILM_SERIAL_R_VEC_END(path, kind, pool, oFirst, oCount) if (used_##pool != SectionCount(buf, Section::pool)) { return false; }
#define FILM_SERIAL_R_MEAS_END(path, pool, oFirst, oN) if (used_##pool != SectionCount(buf, Section::pool)) { return false; }
#define FILM_SERIAL_R(path, Rec, sec, PRE) \
    { \
        PRE##_VECTORS(FILM_SERIAL_R_VEC_DECL) PRE##_MEAS(FILM_SERIAL_R_MEAS_DECL) \
        const std::uint32_t n = SectionCount(buf, Section::sec); \
        const std::uint8_t* rb = buf + SectionOffset(buf, Section::sec); \
        for (std::uint32_t i = 0; i < n; i++, rb += PRE##_SIZE) \
        { \
            PRE##_FIXED(FILM_SERIAL_R_BOOL) PRE##_ENUMS(FILM_SERIAL_R_ENUM) PRE##_STRINGS(FILM_SERIAL_R_STR) \
            PRE##_VECTORS(FILM_SERIAL_R_VEC) PRE##_MEAS(FILM_SERIAL_R_MEAS) \
        } \
        PRE##_VECTORS(FILM_SERIAL_R_VEC_END) PRE##_MEAS(FILM_SERIAL_R_MEAS_END) \
    }
    FILM_SERIAL_RECORDS(FILM_SERIAL_R)
#undef FILM_SERIAL_R
#undef FILM_SERIAL_R_MEAS_END
#undef FILM_SERIAL_R_VEC_END
#undef FILM_SERIAL_R_MEAS_DECL
#undef FILM_SERIAL_R_VEC_DECL
#undef FILM_SERIAL_R_MEAS
#undef FILM_SERIAL_R_VEC
#undef FILM_SERIAL_R_STR
#undef FILM_SERIAL_R_ENUM
#undef FILM_SERIAL_R_BOOL
    return true;
}

#if !defined(FILM_SERIAL_READER_ONLY)

namespace detail
{
    // Field access by kind. Put/Get copy the object's own bytes (the database type IS the
    // storage type: float/double/int/enum); BOOL is normalised to 0/1 on the way out.
    template<Kind K> struct Field
    {
        template<unsigned N, class T> static void put (std::uint8_t* d, const T& v) noexcept
        { static_assert(sizeof(T) == KindSize(K) * N, "member size vs serialized kind"); std::memcpy(d, &v, sizeof(T)); }
        template<unsigned N, class T> static void get (T& v, const std::uint8_t* s) noexcept
        { static_assert(sizeof(T) == KindSize(K) * N, "member size vs serialized kind"); std::memcpy(&v, s, sizeof(T)); }
    };
    template<> struct Field<BOOL>
    {
        template<unsigned N> static void put (std::uint8_t* d, const bool& v) noexcept { static_assert(N == 1, "bool arrays"); d[0] = v ? 1u : 0u; }
        template<unsigned N> static void get (bool& v, const std::uint8_t* s) noexcept { static_assert(N == 1, "bool arrays"); v = s[0] != 0u; }
    };

    inline const char* strData (const std::string& s) noexcept { return s.data(); }
    inline std::size_t strLen  (const std::string& s) noexcept { return s.size(); }
    inline const char* strData (const char* s) noexcept { return s ? s : ""; }
    inline std::size_t strLen  (const char* s) noexcept { return s ? std::strlen(s) : 0u; }
    inline void strSet (std::string& dst, const std::uint8_t* p, std::uint32_t n) { dst.assign(reinterpret_cast<const char*>(p), n); }
    inline void strSet (const char*& dst, const std::uint8_t* p, std::uint32_t) noexcept { dst = reinterpret_cast<const char*>(p); }

    inline std::int32_t measCount (const film::ToneCurve& c) noexcept
    { return (c.meas_n > 0 && c.meas_x && c.meas_d && c.meas_m) ? c.meas_n : 0; }

    /// Bounded sequential writer; with buf == nullptr it only counts.
    struct Writer
    {
        std::uint8_t* buf; std::size_t cap; std::size_t pos; bool ok;
        void raw (const void* p, std::size_t n) noexcept
        { if (!ok) return; if (n > cap - pos) { ok = false; return; } if (buf && n) std::memcpy(buf + pos, p, n); pos += n; }
        void zero (std::size_t n) noexcept
        { if (!ok) return; if (n > cap - pos) { ok = false; return; } if (buf && n) std::memset(buf + pos, 0, n); pos += n; }
        void fill (std::size_t n, int v) noexcept
        { if (!ok) return; if (n > cap - pos) { ok = false; return; } if (buf && n) std::memset(buf + pos, v, n); pos += n; }
        void u32 (std::uint32_t v) noexcept { raw(&v, 4); }
        /// Section padding is 0xFF, never a valid 4-byte element of any section (index
        /// >= table size; a float NaN pattern the database never stores), so a count that
        /// has been changed to swallow or release padding is rejected.
        void padTo8 (void) noexcept { fill(Align8(pos) - pos, 0xFF); }
        std::uint8_t* at (std::size_t p) noexcept { return buf ? buf + p : nullptr; }
    };

    /// String interning without heap allocation (~48 KB of automatic storage).
    struct Interner
    {
        static constexpr std::uint32_t kSlots = 2u * kMaxDistinctStrings;
        const char*   data[kMaxDistinctStrings];
        std::uint32_t len [kMaxDistinctStrings];
        std::uint16_t slot[kSlots];
        std::uint32_t n;
        bool          ok;
        void init (void) noexcept { std::memset(slot, 0, sizeof(slot)); n = 0; ok = true; intern("", 0); }
        std::uint32_t intern (const char* s, std::size_t m) noexcept
        {
            if (m > kMaxStringBytes) { ok = false; return 0u; }
            std::uint32_t h = 2166136261u;
            for (std::size_t i = 0; i < m; i++) { h ^= static_cast<unsigned char>(s[i]); h *= 16777619u; }
            std::uint32_t i = h & (kSlots - 1u);
            for (;;)
            {
                const std::uint16_t e = slot[i];
                if (e == 0u) break;
                if (len[e - 1u] == m && (m == 0u || 0 == std::memcmp(data[e - 1u], s, m))) return e - 1u;
                i = (i + 1u) & (kSlots - 1u);
            }
            if (n >= kMaxDistinctStrings) { ok = false; return 0u; }
            data[n] = s; len[n] = static_cast<std::uint32_t>(m); slot[i] = static_cast<std::uint16_t>(n + 1u);
            return n++;
        }
        template<class S> std::uint32_t intern (const S& s) noexcept { return intern(strData(s), strLen(s)); }
    };
    static_assert(kMaxDistinctStrings <= 32768u, "Interner slot type");

    inline bool hostIsLittleEndian (void) noexcept
    { const std::uint32_t one = 1u; std::uint8_t b; std::memcpy(&b, &one, 1); return b == 1u; }

    template<Kind K, class V> inline void putVector (Writer& w, const V& v) noexcept
    { for (std::size_t i = 0; i < v.size(); i++) { typename V::value_type x = v[i]; w.raw(&x, KindSize(K)); } }
    template<Kind K, class V> inline void getVector (V& v, const std::uint8_t* s, std::uint32_t n)
    { v.resize(n); if (n) std::memcpy(v.data(), s, static_cast<std::size_t>(n) * KindSize(K)); }
    inline void putMeas (Writer& w, const film::ToneCurve& c) noexcept
    {
        const std::int32_t n = measCount(c);
        if (n) { w.raw(c.meas_x, 4u * n); w.raw(c.meas_d, 4u * n); w.raw(c.meas_m, 4u * n); }
    }
    inline void getMeas (film::ToneCurve& c, const std::uint8_t* s, std::int32_t n) noexcept
    {
        c.meas_n = n;
        c.meas_x = n ? reinterpret_cast<const float*>(s) : nullptr;
        c.meas_d = n ? reinterpret_cast<const float*>(s) + n : nullptr;
        c.meas_m = n ? reinterpret_cast<const float*>(s) + 2 * n : nullptr;
    }

    /// The one walk shared by SerializeFilmProfile (buf) and SerializedFilmProfileSize (nullptr).
    inline std::size_t serializeWalk (const film::FilmProfile& p, std::uint8_t* buffer, std::size_t capacity) noexcept
    {
        Writer w; w.buf = buffer; w.cap = capacity; w.pos = 0; w.ok = true;
        Interner in; in.init();
        w.u32(kMagic); w.u32(kFormatVersion); w.u32(kSerialSchemaVersion); w.u32(0u);
        w.u32(static_cast<std::uint32_t>(kFixedSize)); w.u32(kSectionCount);
        w.u32(static_cast<std::uint32_t>(kDataOffset)); w.u32(0u);
        // fixed block: zero it, then place every member at its offset
        const std::size_t fixedPos = w.pos;
        w.zero(kFixedSize);
        if (!w.ok) return 0u;
        if (buffer)
        {
            std::uint8_t* fx = buffer + fixedPos;
#define FILM_SERIAL_X_PUT(path, kind, o, n) Field<kind>::put<n>(fx + o, p.path);
            FILM_SERIAL_FIXED(FILM_SERIAL_X_PUT)
#undef FILM_SERIAL_X_PUT
        }
        w.zero(kDataOffset - w.pos);   // directory placeholder + alignment
        if (!w.ok) return 0u;

        std::uint32_t offs[kSectionCount]; std::uint32_t cnts[kSectionCount]; std::uint32_t sec = 0;
        auto begin = [&](std::size_t count) { offs[sec] = static_cast<std::uint32_t>(w.pos); cnts[sec] = static_cast<std::uint32_t>(count); sec++; };

        // sections, in directory order (the same order the macros list them)
#define FILM_SERIAL_X_MEAS(path, sname, nOff) begin(static_cast<std::size_t>(measCount(p.path))); putMeas(w, p.path); w.padTo8();
        FILM_SERIAL_MEAS(FILM_SERIAL_X_MEAS)
#undef FILM_SERIAL_X_MEAS
#define FILM_SERIAL_X_VEC(path, kind, sname) begin(p.path.size()); putVector<kind>(w, p.path); w.padTo8();
        FILM_SERIAL_VECTORS(FILM_SERIAL_X_VEC)
#undef FILM_SERIAL_X_VEC
#define FILM_SERIAL_X_VSTR(path, sname) begin(p.path.size()); for (std::size_t i = 0; i < p.path.size(); i++) w.u32(in.intern(p.path[i])); w.padTo8();
        FILM_SERIAL_VSTRINGS(FILM_SERIAL_X_VSTR)
#undef FILM_SERIAL_X_VSTR
        begin(kNamedStringCount);
#define FILM_SERIAL_X_STR(path, idx, kind) w.u32(in.intern(p.path));
        FILM_SERIAL_STRINGS(FILM_SERIAL_X_STR)
#undef FILM_SERIAL_X_STR
        w.padTo8();
#define FILM_SERIAL_R_PUT(path, kind, o, n) Field<kind>::put<n>(rb + o, r.path);
#define FILM_SERIAL_R_STR(path, o, kind) { const std::uint32_t ix = in.intern(r.path); std::memcpy(rb + o, &ix, 4); }
#define FILM_SERIAL_R_VEC(path, kind, pool, oFirst, oCount) \
            { const std::uint32_t f = static_cast<std::uint32_t>(used_##pool), c = static_cast<std::uint32_t>(r.path.size()); \
              std::memcpy(rb + oFirst, &f, 4); std::memcpy(rb + oCount, &c, 4); used_##pool += c; }
#define FILM_SERIAL_R_MEAS(path, pool, oFirst, oN) \
            { const std::uint32_t f = static_cast<std::uint32_t>(used_##pool); std::memcpy(rb + oFirst, &f, 4); \
              used_##pool += 3u * static_cast<std::size_t>(measCount(r.path)); }
#define FILM_SERIAL_R_VEC_DECL(path, kind, pool, oFirst, oCount) std::size_t used_##pool = 0;
#define FILM_SERIAL_R_MEAS_DECL(path, pool, oFirst, oN) std::size_t used_##pool = 0;
#define FILM_SERIAL_R_VEC_POOL(path, kind, pool, oFirst, oCount) \
            begin(used_##pool); for (std::size_t i = 0; i < recs.size(); i++) putVector<kind>(w, recs[i].path); w.padTo8();
#define FILM_SERIAL_R_MEAS_POOL(path, pool, oFirst, oN) \
            begin(used_##pool); for (std::size_t i = 0; i < recs.size(); i++) putMeas(w, recs[i].path); w.padTo8();
#define FILM_SERIAL_R(path, Rec, sname, PRE) \
        { \
            const auto& recs = p.path; \
            PRE##_VECTORS(FILM_SERIAL_R_VEC_DECL) PRE##_MEAS(FILM_SERIAL_R_MEAS_DECL) \
            begin(recs.size()); \
            const std::size_t recPos = w.pos; \
            w.zero(static_cast<std::size_t>(PRE##_SIZE) * recs.size()); \
            for (std::size_t i = 0; i < recs.size(); i++) \
            { \
                const Rec& r = recs[i]; \
                std::uint8_t scratch[PRE##_SIZE]; \
                std::uint8_t* rb = w.ok && buffer ? buffer + recPos + static_cast<std::size_t>(PRE##_SIZE) * i : scratch; \
                PRE##_FIXED(FILM_SERIAL_R_PUT) PRE##_STRINGS(FILM_SERIAL_R_STR) \
                PRE##_VECTORS(FILM_SERIAL_R_VEC) PRE##_MEAS(FILM_SERIAL_R_MEAS) \
            } \
            PRE##_VECTORS(FILM_SERIAL_R_VEC_POOL) PRE##_MEAS(FILM_SERIAL_R_MEAS_POOL) \
        }
        FILM_SERIAL_RECORDS(FILM_SERIAL_R)
#undef FILM_SERIAL_R
#undef FILM_SERIAL_R_MEAS_POOL
#undef FILM_SERIAL_R_VEC_POOL
#undef FILM_SERIAL_R_MEAS_DECL
#undef FILM_SERIAL_R_VEC_DECL
#undef FILM_SERIAL_R_MEAS
#undef FILM_SERIAL_R_VEC
#undef FILM_SERIAL_R_STR
#undef FILM_SERIAL_R_PUT

        // string table, last
        if (!in.ok) return 0u;
        begin(in.n);
        {
            std::size_t pos = w.pos + Align8(8u * static_cast<std::size_t>(in.n));
            for (std::uint32_t i = 0; i < in.n; i++) { w.u32(static_cast<std::uint32_t>(pos)); pos += in.len[i] + 1u; }
            for (std::uint32_t i = 0; i < in.n; i++) w.u32(in.len[i]);
            w.padTo8();
            for (std::uint32_t i = 0; i < in.n; i++) { w.raw(in.data[i], in.len[i]); w.zero(1); }
            w.padTo8();
        }
        if (sec != kSectionCount) w.ok = false;
        if (!w.ok || w.pos > 0xFFFFFFFFu) return 0u;
        if (buffer)
        {
            for (std::uint32_t i = 0; i < kSectionCount; i++)
            {
                std::memcpy(buffer + kDirectoryOffset + 8u * i,      &offs[i], 4);
                std::memcpy(buffer + kDirectoryOffset + 8u * i + 4u, &cnts[i], 4);
            }
            const std::uint32_t total = static_cast<std::uint32_t>(w.pos);
            std::memcpy(buffer + 12, &total, 4);
        }
        return w.pos;
    }
}  // namespace detail

inline std::size_t SerializeFilmProfile (const film::FilmProfile& profile, std::uint8_t* buffer, std::size_t capacity) noexcept
{
    if (buffer == nullptr || !detail::hostIsLittleEndian()) return 0u;
    return detail::serializeWalk(profile, buffer, capacity);
}

inline std::size_t SerializedFilmProfileSize (const film::FilmProfile& profile) noexcept
{
    return detail::serializeWalk(profile, nullptr, ~std::size_t(0));
}

/// Validates the buffer, then assigns every carried member of `out`. See the API note
/// above about pointer members referring into `buf` (4-byte aligned, kept alive).
inline bool DeserializeFilmProfile (const std::uint8_t* buf, std::size_t size, film::FilmProfile& out)
{
    if (!ValidateFilmProfileBuffer(buf, size)) return false;
    if ((reinterpret_cast<std::uintptr_t>(buf) & 3u) != 0u) return false;
    const std::uint8_t* fx = buf + kHeaderSize;
#define FILM_SERIAL_X_GET(path, kind, o, n) detail::Field<kind>::get<n>(out.path, fx + o);
    FILM_SERIAL_FIXED(FILM_SERIAL_X_GET)
#undef FILM_SERIAL_X_GET
#define FILM_SERIAL_X_MEAS(path, sname, nOff) detail::getMeas(out.path, buf + SectionOffset(buf, Section::sname), LoadI32(fx + nOff));
    FILM_SERIAL_MEAS(FILM_SERIAL_X_MEAS)
#undef FILM_SERIAL_X_MEAS
#define FILM_SERIAL_X_VEC(path, kind, sname) detail::getVector<kind>(out.path, buf + SectionOffset(buf, Section::sname), SectionCount(buf, Section::sname));
    FILM_SERIAL_VECTORS(FILM_SERIAL_X_VEC)
#undef FILM_SERIAL_X_VEC
    std::uint32_t sp = 0, sl = 0;
#define FILM_SERIAL_X_VSTR(path, sname) \
    out.path.resize(SectionCount(buf, Section::sname)); \
    for (std::uint32_t i = 0; i < SectionCount(buf, Section::sname); i++) { StringAt(buf, IndexAt(buf, Section::sname, i), sp, sl); detail::strSet(out.path[i], buf + sp, sl); }
    FILM_SERIAL_VSTRINGS(FILM_SERIAL_X_VSTR)
#undef FILM_SERIAL_X_VSTR
#define FILM_SERIAL_X_STR(path, idx, kind) StringAt(buf, IndexAt(buf, Section::NamedStrings, idx), sp, sl); detail::strSet(out.path, buf + sp, sl);
    FILM_SERIAL_STRINGS(FILM_SERIAL_X_STR)
#undef FILM_SERIAL_X_STR
#define FILM_SERIAL_R_GET(path, kind, o, n) detail::Field<kind>::get<n>(r.path, rb + o);
#define FILM_SERIAL_R_STR(path, o, kind) StringAt(buf, LoadU32(rb + o), sp, sl); detail::strSet(r.path, buf + sp, sl);
#define FILM_SERIAL_R_VEC(path, kind, pool, oFirst, oCount) \
        detail::getVector<kind>(r.path, buf + SectionOffset(buf, Section::pool) + static_cast<std::size_t>(KindSize(kind)) * LoadU32(rb + oFirst), LoadU32(rb + oCount));
#define FILM_SERIAL_R_MEAS(path, pool, oFirst, oN) \
        detail::getMeas(r.path, buf + SectionOffset(buf, Section::pool) + 4u * static_cast<std::size_t>(LoadU32(rb + oFirst)), LoadI32(rb + oN));
#define FILM_SERIAL_R(path, Rec, sname, PRE) \
    { \
        const std::uint32_t n = SectionCount(buf, Section::sname); \
        out.path.resize(n); \
        const std::uint8_t* rb = buf + SectionOffset(buf, Section::sname); \
        for (std::uint32_t i = 0; i < n; i++, rb += PRE##_SIZE) \
        { \
            Rec& r = out.path[i]; \
            PRE##_FIXED(FILM_SERIAL_R_GET) PRE##_STRINGS(FILM_SERIAL_R_STR) \
            PRE##_VECTORS(FILM_SERIAL_R_VEC) PRE##_MEAS(FILM_SERIAL_R_MEAS) \
        } \
    }
    FILM_SERIAL_RECORDS(FILM_SERIAL_R)
#undef FILM_SERIAL_R
#undef FILM_SERIAL_R_MEAS
#undef FILM_SERIAL_R_VEC
#undef FILM_SERIAL_R_STR
#undef FILM_SERIAL_R_GET
    (void)sp; (void)sl;
    return true;
}

#endif  // !FILM_SERIAL_READER_ONLY

}  // namespace serial
}  // namespace film
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="engine root: the engine sources that are scanned for string readers (default: here)")
    ap.add_argument("--hpp", default=None, help="the film_profiles.hpp to derive the layout from (default: <root>/film_profiles.hpp)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="fail if the file on disk differs")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve() if a.root else HERE
    out = Path(a.out) if a.out else HERE / OUT_NAME
    text, st = generate(root, Path(a.hpp).resolve() if a.hpp else None)
    if a.check:
        if not out.is_file() or out.read_text(encoding="utf-8") != text:
            print("[!] %s is stale -- regenerate (serial_codegen.py)" % out)
            return 1
    else:
        out.write_text(text, encoding="utf-8")
    recs = ", ".join("%s %d/%d B" % (r, n, s) for r, (n, s) in st["records"].items())
    print("[OK] film_profile_serial.hpp: fixed %d B (%d scalar/array members), %d vectors, %d vector<string>, "
          "%d measured tables, %d key strings, %d sections; records (members/bytes): %s; excluded %d "
          "(%d unread text, %d provenance); max %d B (%s), min %d B (%s), median %d B"
          % (st["fixed_size"], st["scalars"], st["vectors"], st["vstrings"], st["meas"], st["named"], st["sections"],
             recs, len(st["excluded"]), sum(1 for _, x in st["excluded"] if x.startswith("text")),
             sum(1 for _, x in st["excluded"] if x.startswith("provenance")),
             st["max"][0], st["max"][1], st["min"][0], st["min"][1], st["median"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
