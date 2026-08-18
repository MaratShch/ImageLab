"""Build and audit driver for the film-profile generator.

WHY THIS EXISTS
---------------
Before this file there was no entry point for the generator chain. `run.cmd`
contained one line -- `python film_sim.py Lady.png -p all` -- which renders an
image and regenerates nothing. The regeneration sequence existed only as a
loose command list in `doc/README.md`, and the extraction scripts existed only
as prose references. Every consequence of that has actually happened at least
once in this project:

  * the project-root and profile_generator copies of the generated C++ drifted a
    whole generation apart, and nothing noticed;
  * `film_names.txt` has two competing generators and the documented order runs
    the DEPRECATED one last, silently replacing the file the effect panel loads;
  * `sigma_shape_toe/mid/dmax` sat in the schema for weeks, populated and
    validated, while no renderer read it;
  * a delivery was reported as applied when the files had not in fact changed.

None of those is a coding mistake. They are all the same missing thing: no
single command that regenerates everything in the right order and fails loudly
when an invariant breaks. That is what this is.

    python build.py                 # full regeneration + audit
    python build.py --check         # READ-ONLY: audits, and reports drift
    python build.py --only verify   # one stage
    python build.py --skip compile  # everything but one stage
    python build.py --list          # stages and what each needs

Stdlib only, no third-party imports at module level, Windows and POSIX. The
stages themselves need numpy/Pillow (and PyMuPDF for the PDF audits); a stage
whose dependency or input is missing SKIPS with a reason and does not fail the
build.

STAGE ORDER IS NOT COSMETIC
---------------------------
  audit    re-derive adopted numbers from the source documents. First, because
           if the stored data no longer matches its own sources there is no
           point regenerating anything from it.
  verify   the check suite. Second, for the same reason.
  codegen  film_profiles.{hpp,cpp}, film_enum.hpp, film_names.txt. Must run
           before `sync`, and `film_names.txt` must be written by THIS step --
           see the note on gen_film_names.py below.
  sync     copy the four generated artefacts to the project root and assert
           both copies are byte-identical. Closes the drift trap.
  docs     FilmActiveProfiles.md, FilmCurves.md -- regenerated from the data,
           so they cannot describe a database that no longer exists.
  compile  g++ -std=c++14 -Wall -Wextra on the generated table. Gated on the
           compiler's own exit code AND an empty stderr, not on a piped head:
           that mistake once reported "clean" while a string literal was broken.

DELIBERATELY NOT RUN: gen_film_names.py
---------------------------------------
`doc/README.md` line 548 records it as "Deprecated. Superseded by
cpp_codegen.write_film_names(), which derives order from the emitted .cpp
instead of from FILM_PROFILES." It is nonetheless still listed in the README's
own command block, AFTER cpp_codegen.py -- so following the documented sequence
overwrites `film_names.txt` with a different set of display names (19 of 154
differ: "KODAK T-MAX 100" against "KODAK TMAX 100", and so on). The owner's
in-service file is cpp_codegen's version. This driver never invokes the
deprecated script and asserts afterwards that the file still matches what
cpp_codegen produces.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

def _default_root() -> Path:
    """Project root, which holds the second copy of the generated C++.

    On the owner's layout HERE is <root>\\PYTHON\\profile_generator, so the root
    is two levels up. Overridable with --root (or FILMSIM_ROOT) so the driver
    can be exercised against a staged copy of the corpus.
    """
    env = os.environ.get("FILMSIM_ROOT")
    return Path(env).resolve() if env else HERE.parent.parent

ROOT = _default_root()

#: The machine-generated artefacts, and the two places each must exist.
#: 2026-08-18: the profile table is split across 16 data-slot TUs plus the
#: explicit-initialisation API (LoadFilmDataBase), because a single 676 KB
#: init-list function was beyond VS2015 SP3 / ICC -- see cpp_codegen.py's
#: "Split emission" banner. film_names.txt is unchanged in format and order.
GENERATED = ("film_profiles.hpp", "film_profiles.cpp",
             "film_enum.hpp", "film_names.txt",
             "film_profiles_detail.hpp",
             "LoadFilmDataBase.h", "LoadFilmDataBase.cpp") + tuple(
    f"film_profiles_data_{i:02d}.cpp" for i in range(1, 17))

#: Audit scripts: re-derive adopted values from the original documents and exit
#: non-zero if they stop reproducing. ONE LINE PER SCRIPT -- this table is the
#: thing that was missing, and it is why a new extraction script now becomes
#: part of the build instead of an orphan.
#:   script, argv, the input it needs, what it guards
def audits(root: Path):
    return (
        ("vision3_granularity.py",
         ["--pdfdir", str(root / "PDF" / "PROFILES" / "KODAK")],
         root / "PDF" / "PROFILES" / "KODAK",
         "the four VISION3 sigma(D) triples, traced from the Kodak TI sheets"),
        ("mees_granularity.py",
         ["--root", str(root)],
         root / "PDF" / "PROFILES" / "RETRO"
              / "THE THEORY OF THE Photographic PROCESS.pdf",
         "B&W silver-negative sigma(D) shape, Mees Fig. 302"),
        ("dye_density.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK",
         "the 8 adopted spectral dye density sets, re-derived from the "
         "sheets' vector paths (5285 and 2383 are the validation pair)"),
        ("plot_inventory.py",
         ["--root", str(root / "PDF" / "PROFILES"), "--assert"],
         root / "PDF" / "PROFILES",
         "the corpus plot inventory: 191 vector dye-density pages (57 under the narrow title pattern), 199 MTF, "
         "101 granularity, 294 characteristic-curve, and the classifier's "
         "three known-answer pages"),
    )

#: verify.py exits 1 whenever ANY check fails, and two checks fail BY DESIGN --
#: the owner's instruction was explicit: "don't try to fix them". So the exit
#: code alone is a useless gate. Compare the FAIL SET against this baseline
#: instead: a new failure fails the build, and a baseline entry that starts
#: passing also reports, so the baseline is shrunk deliberately rather than by
#: accident.
VERIFY_BASELINE = {
    "saturation hierarchy is ordered clean -> impure dyes",
    "neighbour pairs couple harder than the far red-blue pair",
}


class Result:
    __slots__ = ("name", "state", "detail")

    def __init__(self, name, state, detail=""):
        self.name, self.state, self.detail = name, state, detail

    def __repr__(self):
        return f"{self.state:4} {self.name}  {self.detail}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


#: cpp_codegen stamps a UTC generation time into the three C++ files (never into
#: film_names.txt -- that file is required to stay pure data). So a byte compare
#: of a fresh run against the file on disk ALWAYS differs, on the stamp alone.
#: Reporting that as drift would make --check warn every single time, and a gate
#: that always warns is a gate that gets ignored. Compare content instead, with
#: the stamp lines removed, so a real difference is the only thing that shows.
_STAMP = ("// generated:", "Generated by cpp_codegen.py")


def content_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if any(m in line for m in _STAMP):
                continue
            h.update(line.encode("utf-8", "replace"))
    return h.hexdigest()


def run(argv, cwd=HERE, timeout=3600):
    """Run a child process, capturing both streams. Never uses a shell."""
    try:
        p = subprocess.run(argv, cwd=str(cwd), timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        return None, "", f"{exc}"
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout}s"
    return p.returncode, p.stdout.decode("utf-8", "replace"), \
        p.stderr.decode("utf-8", "replace")


def py(*args):
    """This interpreter, not whatever 'python' happens to resolve to."""
    return [sys.executable, *args]


# ---------------------------------------------------------------- stages -----

def stage_audit(opts) -> list:
    """Re-derive adopted numbers from the source documents."""
    out = []
    for script, argv, needs, guards in audits(opts.root):
        if not (HERE / script).is_file():
            out.append(Result(script, "SKIP", "script not present"))
            continue
        if not Path(needs).exists():
            out.append(Result(script, "SKIP",
                              f"source not present: {Path(needs).name}"))
            continue
        rc, so, se = run(py(script, *argv))
        if rc is None:
            out.append(Result(script, "SKIP", se))
        elif rc == 0:
            tail = [ln for ln in so.splitlines() if ln.startswith("[OK]")]
            out.append(Result(script, "OK",
                              tail[-1] if tail else f"reproduces {guards}"))
        else:
            bad = [ln for ln in (so + se).splitlines()
                   if ln.startswith(("[!]", "[FAIL]"))]
            out.append(Result(script, "FAIL",
                              "; ".join(bad[:3]) or f"exit {rc}"))
    if not out:
        out.append(Result("audit", "SKIP", "no audit scripts registered"))
    return out


def stage_verify(opts) -> list:
    """Run verify.py and compare the FAIL set against VERIFY_BASELINE."""
    rc, so, se = run(py("verify.py"))
    if rc is None:
        return [Result("verify.py", "FAIL", se)]
    fails = {ln[len("FAIL"):].strip().split("   ")[0].strip()
             for ln in so.splitlines() if ln.startswith("FAIL")}
    npass = sum(1 for ln in so.splitlines() if ln.startswith("PASS"))
    res = [Result("verify.py", "OK" if fails == VERIFY_BASELINE else "FAIL",
                  f"{npass} PASS / {len(fails)} FAIL")]
    for extra in sorted(fails - VERIFY_BASELINE):
        res.append(Result("  NEW FAILURE", "FAIL", extra))
    for gone in sorted(VERIFY_BASELINE - fails):
        res.append(Result("  baseline entry now PASSES", "WARN",
                          f"{gone}  -- remove it from VERIFY_BASELINE"))
    if not so.strip():
        res.append(Result("  verify.py produced no output", "FAIL",
                          se.splitlines()[-1] if se.strip() else f"exit {rc}"))
    return res


def stage_codegen(opts) -> list:
    """Regenerate the C++ tables; assert film_names.txt is cpp_codegen's."""
    if opts.check:
        tmp = Path(tempfile.mkdtemp(prefix="fs_codegen_"))
        try:
            rc, so, se = run(py("cpp_codegen.py", "-o", str(tmp)))
            if rc != 0:
                return [Result("cpp_codegen.py", "FAIL", se.strip()[-200:])]
            res = [Result("cpp_codegen.py", "OK", "generated into a temp dir")]
            for name in GENERATED:
                fresh, live = tmp / name, HERE / name
                if not live.is_file():
                    res.append(Result(f"  {name}", "FAIL", "missing on disk"))
                elif content_sha256(fresh) == content_sha256(live):
                    same_bytes = sha256(fresh) == sha256(live)
                    res.append(Result(f"  {name}", "OK", "up to date"
                                      if same_bytes
                                      else "up to date (differs only in the "
                                           "generation timestamp)"))
                else:
                    res.append(Result(f"  {name}", "WARN",
                                      "CONTENT DIFFERS from a fresh run -- "
                                      "regenerate (run without --check)"))
            return res
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # Snapshot film_names.txt BEFORE regenerating. Checking it afterwards is
    # vacuous: cpp_codegen has already rewritten it, so the check could never
    # fail and gave a false all-clear. What is worth knowing is whether the file
    # on disk was NOT cpp_codegen's before this run -- i.e. whether something
    # (the deprecated gen_film_names.py) had clobbered the list the effect panel
    # loads. That can only be seen from the pre-run state.
    names = HERE / "film_names.txt"
    before = sha256(names) if names.is_file() else None

    rc, so, se = run(py("cpp_codegen.py", "-o", "."))
    if rc != 0:
        return [Result("cpp_codegen.py", "FAIL", se.strip()[-200:])]
    res = [Result("cpp_codegen.py", "OK",
                  f"wrote {len(GENERATED)} artefacts")]
    missing = [n for n in GENERATED if not (HERE / n).is_file()]
    if missing:
        res.append(Result("  artefacts", "FAIL", f"not written: {missing}"))

    after = sha256(names) if names.is_file() else None
    if before is None:
        res.append(Result("  film_names.txt", "OK", "created"))
    elif before == after:
        res.append(Result("  film_names.txt owner", "OK",
                          "was already cpp_codegen's output"))
    else:
        res.append(Result("  film_names.txt owner", "WARN",
                          "the file on disk was NOT cpp_codegen's output and "
                          "has been CORRECTED -- the deprecated "
                          "gen_film_names.py had probably been run over it"))
    return res


def stage_sync(opts) -> list:
    """Keep the project-root copy of the generated C++ identical."""
    res = []
    for name in GENERATED:
        src, dst = HERE / name, opts.root / name
        if not src.is_file():
            res.append(Result(f"{name}", "FAIL", "not generated"))
            continue
        if not dst.exists():
            if opts.check:
                res.append(Result(f"{name}", "WARN",
                                  f"absent at {opts.root} -- would be created"))
                continue
            shutil.copy2(src, dst)
            res.append(Result(f"{name}", "OK", f"created at {opts.root}"))
            continue
        if sha256(src) == sha256(dst):
            res.append(Result(f"{name}", "OK", "both copies identical"))
        elif content_sha256(src) == content_sha256(dst):
            res.append(Result(f"{name}", "OK",
                              "same content, differs only in the timestamp"))
        elif opts.check:
            res.append(Result(f"{name}", "WARN",
                              "root copy DIFFERS -- run without --check"))
        else:
            shutil.copy2(src, dst)
            res.append(Result(f"{name}", "OK", "root copy refreshed"))
    return res


def stage_docs(opts) -> list:
    """Regenerate the reports that describe the database."""
    res = []
    if opts.check:
        tmp = Path(tempfile.mkdtemp(prefix="fs_docs_"))
        try:
            rc, so, se = run(py("gen_active_profiles.py",
                                "-o", str(tmp / "FilmActiveProfiles.md")))
            live = HERE / "doc" / "FilmActiveProfiles.md"
            if rc != 0:
                res.append(Result("gen_active_profiles.py", "FAIL",
                                  se.strip()[-200:]))
            elif not live.is_file():
                res.append(Result("FilmActiveProfiles.md", "FAIL", "missing"))
            else:
                same = sha256(tmp / "FilmActiveProfiles.md") == sha256(live)
                res.append(Result("FilmActiveProfiles.md",
                                  "OK" if same else "WARN",
                                  "up to date" if same else "stale"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        res.append(Result("gen_film_curves_md.py", "SKIP",
                          "no -o option, cannot run read-only"))
        return res

    for script, argv, produced in (
            ("gen_active_profiles.py", [], "doc/FilmActiveProfiles.md"),
            ("gen_film_curves_md.py", [], "doc/FilmCurves.md")):
        rc, so, se = run(py(script, *argv))
        if rc == 0:
            res.append(Result(script, "OK", produced))
        else:
            res.append(Result(script, "FAIL", (se or so).strip()[-200:]))
    return res


def stage_compile(opts) -> list:
    """Compile the generated table. Gate on exit code AND empty stderr."""
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        return [Result("compile", "SKIP",
                       "no g++/clang++ on PATH (set CXX to override)")]
    sources = [n for n in GENERATED if n.endswith(".cpp")]
    missing = [n for n in sources if not (HERE / n).is_file()]
    if missing:
        return [Result("compile", "SKIP", f"not generated: {missing[:3]}")]
    tmp = Path(tempfile.mkdtemp(prefix="fs_cc_"))
    try:
        res = []
        for n in sources:
            rc, so, se = run([cxx, "-std=c++14", "-Wall", "-Wextra",
                              "-c", str(HERE / n),
                              "-o", str(tmp / (n[:-4] + ".o"))], cwd=HERE)
            if rc is None:
                return [Result("compile", "SKIP", se)]
            noise = len(se.encode()) + len(so.encode())
            if rc != 0 or noise != 0:
                first = (se or so).strip().splitlines()
                res.append(Result(f"  {n}", "FAIL",
                                  f"exit {rc}, {noise} bytes: "
                                  f"{first[0][:140] if first else ''}"))
        if res:
            return res
        return [Result("compile", "OK",
                       f"{Path(cxx).name} -std=c++14 -Wall -Wextra on all "
                       f"{len(sources)} TUs: exit 0 AND zero bytes of output")]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


STAGES = (
    ("audit",   stage_audit,   "re-derive adopted numbers from source documents"),
    ("verify",  stage_verify,  "verify.py, FAIL set compared to the baseline"),
    ("codegen", stage_codegen, "regenerate the four C++ artefacts"),
    ("sync",    stage_sync,    "keep the project-root C++ copy identical"),
    ("docs",    stage_docs,    "regenerate FilmActiveProfiles.md, FilmCurves.md"),
    ("compile", stage_compile, "g++ -std=c++14 -Wall -Wextra, gated strictly"),
)


def main() -> int:
    names = [n for n, _, _ in STAGES]
    ap = argparse.ArgumentParser(
        description="Regenerate and audit the film-profile database.",
        epilog="Stages run in the listed order; the order is load-bearing.")
    ap.add_argument("--only", action="append", metavar="STAGE", choices=names,
                    help="run only this stage (repeatable)")
    ap.add_argument("--skip", action="append", metavar="STAGE", choices=names,
                    help="skip this stage (repeatable)")
    ap.add_argument("--check", action="store_true",
                    help="READ-ONLY: audit and report drift, write nothing")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="project root holding PDF/ and the second C++ copy "
                         "(default: two levels above this script)")
    opts = ap.parse_args()
    opts.root = Path(opts.root).resolve()

    if opts.list:
        print("stages, in order:\n")
        for n, _, why in STAGES:
            print(f"  {n:8} {why}")
        print("\naudit scripts registered:\n")
        for script, _, needs, guards in audits(opts.root):
            have = "present" if Path(needs).exists() else "SOURCE MISSING"
            print(f"  {script:26} {guards}\n"
                  f"  {'':26} needs {Path(needs).name}  [{have}]")
        return 0

    chosen = [s for s in STAGES if (not opts.only or s[0] in opts.only)
              and (not opts.skip or s[0] not in opts.skip)]

    mode = "CHECK (read-only)" if opts.check else "BUILD"
    print(f"film-profile generator -- {mode}")
    print(f"  generator : {HERE}")
    print(f"  root      : {opts.root}")
    print(f"  python    : {sys.version.split()[0]}")

    failed = warned = 0
    for name, fn, why in chosen:
        print(f"\n=== {name} -- {why}")
        try:
            results = fn(opts)
        except Exception as exc:                       # a stage must not abort
            results = [Result(name, "FAIL", f"{type(exc).__name__}: {exc}")]
        for r in results:
            print(f"  [{r.state:4}] {r.name}"
                  + (f"  --  {r.detail}" if r.detail else ""))
            failed += r.state == "FAIL"
            warned += r.state == "WARN"

    print("\n" + "-" * 70)
    if failed:
        print(f"BUILD FAILED  --  {failed} failure(s), {warned} warning(s)")
        return 1
    print(f"OK  --  0 failures, {warned} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
