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
72 numbers disagrees by more than the tolerance below.

⚠ AND, SINCE 2026-10-06, THE WHOLE FRAME. Until that date the comparison was
patch interiors only, because the engines approximated every frequency-domain
transfer with separable spatial kernels and edges were expected to differ -- by
up to 0.10 display-linear (AGFA_RSX_II_200), and the scalar and AVX2 engines
disagreed with EACH OTHER by 0.077 at edges. With the owner's FFT the engines
apply film_sim's transfers exactly (AlgoFrequency.hpp), so every pixel is now
compared: the maximum absolute difference over the full 480 x 336 x 3 frame.

MEASURED 2026-10-06, 222 stocks, after the FFT integration:
  patch medians  Python vs scalar 3.40e-06, Python vs AVX2 1.19e-05,
                 scalar vs AVX2 1.24e-05
  full frame     Python vs scalar 9.06e-06, Python vs AVX2 1.73e-05,
                 scalar vs AVX2 1.75e-05
  (before it, same run: patch 1.92e-04, full frame 1.02e-01 / 1.12e-01 / 7.69e-02)
The tolerances below were tightened to 1e-4 for both measures and all three
pairings: 5.7x above the worst correct pairing, and 1000x below the edge
disagreement the separable kernels left.

MEASURED 2026-09-28, 200 stocks (separable kernels, patch medians only):
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
import engine_env  # noqa: F401 -- owner FFT include path for engine compiles (2026-10-06)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HARNESS = "test_chain_dump.cpp"

#: Display-linear tolerance on a patch median, Python against either engine.
#: 2.0e-3 until 2026-10-06; see the module docstring for the measurement.
TOL_PY_CPP = 1.0e-4
#: Display-linear tolerance on a patch median, scalar against AVX2 (5.0e-4 before).
TOL_SCALAR_AVX2 = 1.0e-4
#: Display-linear tolerance on the MAXIMUM over the whole frame, any pairing
#: (new 2026-10-06, possible only now that all three implementations apply the
#: same frequency-domain transfers).
TOL_FRAME = 1.0e-4

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

#: ⚠ A SECOND, HIGHER-RESOLUTION PASS ON A FEW STOCKS (2026-10-06). At 480 px
#: wide a 35 mm frame is 19 px/mm, where every DIR coupler EDGE term (9-13 um,
#: 0.17-0.25 px) sits under the 0.25 px gate and the reseau mosaic is disabled
#: -- so the 480 px pass never executes either. The first run of the FFT
#: engines passed it while the AVX2 stage 9 had lost its floor on the
#: both-terms path (0.093 display-linear at 1440 px on FUJI_VELVIA_50). These
#: stocks are rendered again at HIRES_W x HIRES_H (64 px/mm on 35 mm: edge
#: terms on, Dufay pitch 3.2 px) and held to the same full-frame tolerance.
HIRES_W, HIRES_H = 1600, 1120
HIRES_STOCKS = ("KODAK_PORTRA_400", "FUJI_VELVIA_50", "KODAK_VISION3_500T_5219",
                "KODACHROME_64", "DUFAYCOLOR_1937", "KODAK_TECHNICAL_PAN")
PATCH, GAP, CARD = 64, 14, 30
HALF_WINDOW = 12


def make_frame(width: int = None, height: int = None):
    """-> (scene-linear frame, patch centres)."""
    import film_sim as fs
    width = WIDTH if width is None else width
    height = HEIGHT if height is None else height
    x0 = (width - (6 * PATCH + 5 * GAP)) // 2
    y0 = (height - (4 * PATCH + 3 * GAP)) // 2
    img = np.full((height, width, 3), CARD / 255.0, np.float32)
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


def run_engine(exe: Path, frame_path: Path, out_path: Path, n: int, names,
               width: int = None, height: int = None, indices=None):
    width = WIDTH if width is None else width
    height = HEIGHT if height is None else height
    indices = list(range(n)) if indices is None else list(indices)
    r = subprocess.run([str(exe), str(frame_path), str(width), str(height),
                        str(out_path)] + [str(i) for i in indices],
                       capture_output=True, text=True)
    lines = r.stdout.split("\n")
    if r.returncode != 0 or "END" not in lines:
        raise SystemExit("[!] %s failed: %s" % (exe.name,
                                                (r.stdout + r.stderr)[-400:]))
    seen = [ln.split(None, 2)[2] for ln in lines if ln.startswith("STOCK ")]
    if seen != [names[i] for i in indices]:
        raise SystemExit("[!] %s rendered the stocks in a different order or "
                         "under different names than film_profiles -- the "
                         "database index is not what Python thinks it is"
                         % exe.name)
    data = np.fromfile(out_path, dtype=np.float32)
    m = len(indices)
    if data.size != m * 3 * width * height:
        raise SystemExit("[!] %s wrote %d floats, expected %d"
                         % (exe.name, data.size, m * 3 * width * height))
    return data.reshape(m, 3, height, width).transpose(0, 2, 3, 1)


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

        missing = [h for h in HIRES_STOCKS if h not in names]
        if missing:
            raise SystemExit("[!] HIRES_STOCKS names not in the database: %s"
                             % ", ".join(missing))
        hi_idx = [names.index(h) for h in HIRES_STOCKS]
        hi_frame, _ = make_frame(HIRES_W, HIRES_H)
        hi_path = tmp / "frame_hi.raw"
        hi_frame.transpose(2, 0, 1).tofile(hi_path)
        cpp_hi = {tag: run_engine(exe, hi_path, tmp / ("hi_" + tag), n, names,
                                  HIRES_W, HIRES_H, hi_idx)
                  for tag, exe in exes.items()}

        settings = fs.RenderSettings(grain_scale=0.0, misreg_scale=0.0,
                                     coating_scale=0.0, print_grain=False)
        bad = []
        worst = {"py_sc": (0.0, ""), "py_avx": (0.0, ""), "sc_avx": (0.0, "")}
        wframe = {"py_sc": (0.0, ""), "py_avx": (0.0, ""), "sc_avx": (0.0, "")}
        for i, p in enumerate(profiles):
            img_py = fs.simulate(frame, p, settings)
            m_py = patch_medians(img_py, centres)
            m_sc = patch_medians(cpp["scalar"][i], centres)
            m_vx = patch_medians(cpp["avx2"][i], centres)
            for key, a, b in (("py_sc", img_py, cpp["scalar"][i]),
                              ("py_avx", img_py, cpp["avx2"][i]),
                              ("sc_avx", cpp["scalar"][i], cpp["avx2"][i])):
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                k = int(d.argmax())
                if d.flat[k] > wframe[key][0]:
                    wframe[key] = (float(d.flat[k]), p.name)
                if d.flat[k] > TOL_FRAME:
                    y, rem = divmod(k, WIDTH * 3)
                    x, ch = divmod(rem, 3)
                    bad.append("%s %s: full frame (%d,%d) %s = %.5f vs %.5f "
                               "(|d| %.2e > %.1e)"
                               % (p.name, key, x, y, "RGB"[ch], a.flat[k],
                                  b.flat[k], d.flat[k], TOL_FRAME))
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

        whi = {"py_sc": (0.0, ""), "py_avx": (0.0, ""), "sc_avx": (0.0, "")}
        for j, i in enumerate(hi_idx):
            p = profiles[i]
            img_py = fs.simulate(hi_frame, p, settings)
            for key, a, b in (("py_sc", img_py, cpp_hi["scalar"][j]),
                              ("py_avx", img_py, cpp_hi["avx2"][j]),
                              ("sc_avx", cpp_hi["scalar"][j], cpp_hi["avx2"][j])):
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                k = int(d.argmax())
                if d.flat[k] > whi[key][0]:
                    whi[key] = (float(d.flat[k]), p.name)
                if d.flat[k] > TOL_FRAME:
                    y, rem = divmod(k, HIRES_W * 3)
                    x, ch = divmod(rem, 3)
                    bad.append("%s %s: %dx%d frame (%d,%d) %s = %.5f vs %.5f "
                               "(|d| %.2e > %.1e)"
                               % (p.name, key, HIRES_W, HIRES_H, x, y, "RGB"[ch],
                                  a.flat[k], b.flat[k], d.flat[k], TOL_FRAME))

    print("CHAIN PARITY -- %d stocks, 24 patches x 3 channels, Python vs "
          "scalar vs AVX2, stochastic stages off (%.0f s)" % (n, time.time() - t0))
    for key, label in (("py_sc", "Python vs scalar"),
                       ("py_avx", "Python vs AVX2  "),
                       ("sc_avx", "scalar vs AVX2  ")):
        print("  %s patch worst %.2e (%s)   full-frame worst %.2e (%s)"
              % (label, worst[key][0], worst[key][1],
                 wframe[key][0], wframe[key][1]))
    print("  hi-res pass, %d stocks at %d x %d:" % (len(HIRES_STOCKS), HIRES_W, HIRES_H))
    for key, label in (("py_sc", "Python vs scalar"),
                       ("py_avx", "Python vs AVX2  "),
                       ("sc_avx", "scalar vs AVX2  ")):
        print("  %s full-frame worst %.2e (%s)" % (label, whi[key][0], whi[key][1]))
    if bad:
        for line in bad[:40]:
            print("  [FAIL] " + line)
        if len(bad) > 40:
            print("  [FAIL] ... and %d more" % (len(bad) - 40))
        return 1 if ns.assert_ else 0
    print("\n[OK] chain_parity.py -- all %d stocks render the same picture in "
          "Python, the scalar engine and the AVX2 engine: every patch median "
          "within %.0e (Python vs C++) and %.0e (scalar vs AVX2), every pixel "
          "of the full frame within %.0e"
          % (n, TOL_PY_CPP, TOL_SCALAR_AVX2, TOL_FRAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
