"""Python vs C++ scalar vs C++ AVX2 over the WHOLE render chain, every stock.

WHY THIS EXISTS
---------------
On 2026-09-27 two defects shipped in both C++ engines that no audit could see:

  1. Stage 13 re-timed the print against a mid grey that the Callier stage had
     not corrected (film_sim does correct it, after `neutral_mid_density`).
     Every silver stock rendered with mid grey near 0.8 display-linear instead
     of 0.2 -- FERRANIA_P30 came out almost white.
  2. A black-and-white negative was printed through the scan stock's three
     colour curves and dye matrix, where film_sim prints it through a neutral
     print (green curve on all three channels, identity matrix). A faint cast
     on a greyscale image, up to 0.0155 display-linear per channel.

Every stage was right in isolation. `cpp_parity` compares the LAWS one by one,
`stage_parity` compares scalar against AVX2, and the per-stage twin audits each
cover one stage. Nothing rendered a frame through Python AND through the plugin
and compared the pictures, so an error in how the stages were WIRED together
was invisible. This audit is that comparison.

HOW IT KEEPS THE TWO FLAVOURS APART
-----------------------------------
This file contains no C++. The C++ side is `test_chain_dump.cpp`, a harness
that lives in the ENGINE tree beside the plugin sources (and, being test_*,
never ships inside an engine archive). This file only compiles that harness
against the engine, hands it a frame as raw float32, and reads back raw float32.
Python and C++ exchange data, not code.

WHAT IS COMPARED
----------------
A 24-patch ColorChecker (the published sRGB values) on a dark card, 480 x 336,
rendered through EVERY stock in the database at default controls with the four
stochastic stages off on both sides -- negative grain, misregistration, the
coating field and print grain -- because film_sim draws from numpy and the
engine from a counter-based generator, and the two can never agree sample for
sample. Everything deterministic stays on: halation, flare, vignette, reseau,
Callier, the print chain, the silver tone.

For each stock and each engine the MEDIAN of a 24 x 24 window at the centre of
every patch is taken on all three channels, and the audit fails if any of the
72 numbers disagrees by more than the tolerance below. Patch interiors only:
the two implementations evaluate the emulsion MTF differently (frequency
domain in Python, separable kernels in C++), so edges are expected to differ
and are not what this audit is for.

MEASURED 2026-09-28, 200 stocks:
  Python vs scalar   worst 2.65e-04 (ORWOCOLOR_NC3), none above 1e-3
  Python vs AVX2     worst 2.65e-04
  scalar vs AVX2     worst 5.96e-05
Against the engines as they stood BEFORE the two fixes, the same run fails 65
stocks -- every black-and-white negative -- by up to 0.50. The tolerance sits
7.5x above the worst correct pairing and 250x below the defect it was written
for, and 7x below the smaller of the two (the neutral-print cast).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HARNESS = "test_chain_dump.cpp"

#: Display-linear tolerance on a patch median, Python against either engine.
TOL_PY_CPP = 2.0e-3
#: Display-linear tolerance on a patch median, scalar against AVX2.
TOL_SCALAR_AVX2 = 5.0e-4

#: The published ColorChecker 24 sRGB values, row by row.
COLORCHECKER = (
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67),
    (133, 128, 177), (103, 189, 170), (214, 126, 44), (80, 91, 166),
    (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46),
    (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31),
    (187, 86, 149), (8, 133, 161), (243, 243, 242), (200, 200, 200),
    (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52))
PATCH_NAMES = (
    "dark skin", "light skin", "blue sky", "foliage", "blue flower",
    "bluish green", "orange", "purplish blue", "moderate red", "purple",
    "yellow green", "orange yellow", "blue", "green", "red", "yellow",
    "magenta", "cyan", "white", "neutral 8", "neutral 6.5", "neutral 5",
    "neutral 3.5", "black")

WIDTH, HEIGHT = 480, 336
PATCH, GAP, CARD = 64, 14, 30
HALF_WINDOW = 12


def make_frame():
    """-> (scene-linear frame, patch centres)."""
    import film_sim as fs
    x0 = (WIDTH - (6 * PATCH + 5 * GAP)) // 2
    y0 = (HEIGHT - (4 * PATCH + 3 * GAP)) // 2
    img = np.full((HEIGHT, WIDTH, 3), CARD / 255.0, np.float32)
    centres = []
    for k, rgb in enumerate(COLORCHECKER):
        r, q = divmod(k, 6)
        x = x0 + q * (PATCH + GAP)
        y = y0 + r * (PATCH + GAP)
        img[y:y + PATCH, x:x + PATCH] = np.array(rgb, np.float32) / 255.0
        centres.append((y + PATCH // 2, x + PATCH // 2))
    return fs.srgb_to_linear(img).astype(np.float32), centres


def patch_medians(img, centres):
    w = HALF_WINDOW
    return np.array([[np.median(img[y - w:y + w, x - w:x + w, k])
                      for k in range(3)] for y, x in centres])


def build(root: Path, tmp: Path):
    """-> {"scalar": exe, "avx2": exe}. The database is compiled once."""
    import stage_parity as sp
    from interimage_parity import stage_avx2_tree
    cxx = sp._cxx()
    base = ["-std=c++14", "-O1", "-DALGO_PROFILE_STAGES=0"]
    obj = tmp / "obj"
    obj.mkdir(parents=True, exist_ok=True)
    db = ([root / "film_profiles.cpp", root / "LoadFilmDataBase.cpp"]
          + sorted(root.glob("film_profiles_data_*.cpp")))
    db_obj = [obj / ("db_" + p.stem + ".o") for p in db]
    sp._compile_all([([cxx] + base + ["-I", str(root), "-c", str(s),
                                      "-o", str(o)], "database " + s.name)
                     for s, o in zip(db, db_obj)], False)
    exes = {}
    for tag in ("scalar", "avx2"):
        tu_dir, extra, inc = root, [], ["-I", str(root)]
        if tag == "avx2":
            tu_dir = stage_avx2_tree(root, tmp)
            if tu_dir is None:
                raise SystemExit("[!] %s carries no AVX2 twins" % root)
            extra = ["-mavx2", "-mfma"]
            inc = ["-I", str(tu_dir), "-I", str(root)]
        harness = tu_dir / HARNESS
        if not harness.is_file():
            raise SystemExit("[!] %s is not in %s -- this audit needs it and "
                             "will not pretend it passed without it"
                             % (HARNESS, tu_dir))
        src = sp._engine_tus(tu_dir) + [harness]
        eobj = [obj / ("%s_%s.o" % (tag, p.stem)) for p in src]
        sp._compile_all([([cxx] + base + extra + inc + ["-c", str(s),
                                                        "-o", str(o)],
                          "%s %s" % (tag, s.name))
                         for s, o in zip(src, eobj)], False)
        exe = tmp / ("chain_dump_" + tag)
        r = subprocess.run([cxx] + extra + ["-o", str(exe)]
                           + [str(o) for o in eobj + db_obj],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("[!] %s did not link:\n  %s"
                             % (tag, (r.stderr or r.stdout).strip()[:1200]))
        exes[tag] = exe
    return exes


def run_engine(exe: Path, frame_path: Path, out_path: Path, n: int, names):
    r = subprocess.run([str(exe), str(frame_path), str(WIDTH), str(HEIGHT),
                        str(out_path)] + [str(i) for i in range(n)],
                       capture_output=True, text=True)
    lines = r.stdout.split("\n")
    if r.returncode != 0 or "END" not in lines:
        raise SystemExit("[!] %s failed: %s" % (exe.name,
                                                (r.stdout + r.stderr)[-400:]))
    seen = [ln.split(None, 2)[2] for ln in lines if ln.startswith("STOCK ")]
    if seen != list(names):
        raise SystemExit("[!] %s rendered the stocks in a different order or "
                         "under different names than film_profiles -- the "
                         "database index is not what Python thinks it is"
                         % exe.name)
    data = np.fromfile(out_path, dtype=np.float32)
    if data.size != n * 3 * WIDTH * HEIGHT:
        raise SystemExit("[!] %s wrote %d floats, expected %d"
                         % (exe.name, data.size, n * 3 * WIDTH * HEIGHT))
    return data.reshape(n, 3, HEIGHT, WIDTH).transpose(0, 2, 3, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("FILMSIM_ENGINE",
                                                     "/root/work/tst"),
                    help="engine tree holding AlgorithmMain.cpp, the AVX2/ "
                         "twins and test_chain_dump.cpp")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    if not (root / "AlgorithmMain.cpp").is_file():
        print("  [SKIP] no engine tree at %s" % root)
        return 0

    import film_sim as fs
    import film_profiles as fp

    profiles = list(fp.FILM_PROFILES)
    names = [p.name for p in profiles]
    n = len(profiles)
    frame, centres = make_frame()

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="chain_parity_") as td:
        tmp = Path(td)
        exes = build(root, tmp)
        frame_path = tmp / "frame.raw"
        frame.transpose(2, 0, 1).tofile(frame_path)
        cpp = {tag: run_engine(exe, frame_path, tmp / ("out_" + tag), n, names)
               for tag, exe in exes.items()}

        settings = fs.RenderSettings(grain_scale=0.0, misreg_scale=0.0,
                                     coating_scale=0.0, print_grain=False)
        bad = []
        worst = {"py_sc": (0.0, ""), "py_avx": (0.0, ""), "sc_avx": (0.0, "")}
        for i, p in enumerate(profiles):
            m_py = patch_medians(fs.simulate(frame, p, settings), centres)
            m_sc = patch_medians(cpp["scalar"][i], centres)
            m_vx = patch_medians(cpp["avx2"][i], centres)
            for key, a, b, tol in (("py_sc", m_py, m_sc, TOL_PY_CPP),
                                   ("py_avx", m_py, m_vx, TOL_PY_CPP),
                                   ("sc_avx", m_sc, m_vx, TOL_SCALAR_AVX2)):
                d = np.abs(a - b)
                k = int(d.argmax())
                if d.flat[k] > worst[key][0]:
                    worst[key] = (float(d.flat[k]), p.name)
                if d.flat[k] > tol:
                    patch, ch = divmod(k, 3)
                    bad.append("%s %s: %s, %s channel = %.4f vs %.4f "
                               "(|d| %.2e > %.1e)"
                               % (p.name, key, PATCH_NAMES[patch],
                                  "RGB"[ch], a.flat[k], b.flat[k],
                                  d.flat[k], tol))

    print("CHAIN PARITY -- %d stocks, 24 patches x 3 channels, Python vs "
          "scalar vs AVX2, stochastic stages off (%.0f s)" % (n, time.time() - t0))
    for key, label in (("py_sc", "Python vs scalar"),
                       ("py_avx", "Python vs AVX2  "),
                       ("sc_avx", "scalar vs AVX2  ")):
        print("  %s worst %.2e  (%s)" % (label, worst[key][0], worst[key][1]))
    if bad:
        for line in bad[:40]:
            print("  [FAIL] " + line)
        if len(bad) > 40:
            print("  [FAIL] ... and %d more" % (len(bad) - 40))
        return 1 if ns.assert_ else 0
    print("\n[OK] chain_parity.py -- all %d stocks render the same picture in "
          "Python, the scalar engine and the AVX2 engine: every patch median "
          "within %.0e (Python vs C++) and %.0e (scalar vs AVX2)"
          % (n, TOL_PY_CPP, TOL_SCALAR_AVX2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
