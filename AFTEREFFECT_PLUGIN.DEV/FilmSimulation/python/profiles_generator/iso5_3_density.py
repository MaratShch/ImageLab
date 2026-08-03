"""ISO 5-3 spectral products for visual, Type 1 and Type 2 density.

SOURCES, verbatim from the document headers:
  Table 2 -- ISO 5-3:1995(E), "Photography -- Density measurements -- Part 3:
  Spectral conditions", "log10 spectral products for ISO visual, type 1 and
  type 2 densities (Normalized to 5,000 peak)".
  Tables 3 and 4 -- ANSI/ISO 5-3-1995, ANSI/NAPM IT2.18-1996 (the US national
  adoption, which carries the same tables in full):
    "Table 3 -- Status A -- log10 spectral products Pi_A (Normalized to 5,000
     peak)"
    "Table 4 -- Status M -- log10 spectral products Pi_M (Normalized to 5,000
     peak)"

Transcribed from a rendered page image, not from the PDF text layer: the text
layer interleaved the wavelength and value columns, which would have silently
mis-paired every row. Note the document uses the European decimal comma, so
its "4,957" is 4.957 here.

WHAT IS HERE
  Table 2: visual, Type 1 (printing: diazo and vesicular), Type 2 (printing:
  silver halide), 340-770 nm.
  Table 3: Status A blue/green/red -- reflection and transparency colour
  densitometry, "defined to match closely the responses historically used in
  evaluating transparency films".
  Table 4: Status M blue/green/red -- "defined to match closely the responses
  historically used in evaluating colour negative films".

  Status M and A were recovered from ANSI/NAPM IT2.18-1996 after the ISO 5-3
  copy available here turned out to be a standards.iteh.ai PREVIEW that stops
  mid-sentence immediately after naming Table 4.

OUT-OF-RANGE BEHAVIOUR (Tables 3 and 4)
  Unlike Table 2, the Status tables do not print "< 1,000" outside their
  tabulated range. They print a SLOPE and an arrow, meaning the response
  continues linearly in log10 beyond the last tabulated value. Those slopes
  are recorded in _SLOPES and applied by ``weights()``, because truncating to
  zero instead would silently narrow every channel's skirt:
    Status M  blue  +0.250/nm below, -0.220/nm above
              green +0.106/nm below, -0.120/nm above
              red   +0.260/nm below, -0.040/nm above
    Status A  blue  +0.380/nm below, -0.140/nm above
              green +0.220/nm below, -0.170/nm above
              red   +0.270/nm below, -0.040/nm above

CONVENTION
  Values are log10 of the spectral product Pi = S * s, where S is the relative
  spectral power of the influx and s the relative spectral response of the
  receiver (ISO 5-3 clause 4.3). They are normalised so the peak is exactly
  5.000, i.e. linear products are 10**(log10 Pi - 5.0) with a peak of 1.0.
  Entries printed as "< 1,000" mean the product is below 10**1.0 relative to
  the same normalisation -- effectively zero, and stored here as None.

USE
  density = -log10( SUM_lambda  Pi(lambda) * T(lambda) / SUM_lambda Pi(lambda) )
  for a sample of spectral transmittance T. `weights()` returns the normalised
  linear Pi ready for that sum.
"""
from __future__ import annotations

#: Wavelength grid, nm.
LAMBDAS: tuple[int, ...] = tuple(range(340, 780, 10))

#: log10 spectral products, Table 2. None = printed "< 1,000" (below floor).
#: Order matches LAMBDAS exactly; verified length below.
VISUAL: tuple[float | None, ...] = (
    None, None, None, None, None, None,          # 340-390
    None,                                        # 400  "< 1,000"
    1.322, 1.914, 2.447, 2.811,                  # 410-440
    3.090, 3.346, 3.582, 3.818, 4.041,           # 450-490
    4.276, 4.513, 4.702, 4.825, 4.905,           # 500-540
    4.957, 4.989, 5.000, 4.989, 4.956,           # 550-590
    4.902, 4.827, 4.731, 4.593, 4.433,           # 600-640
    4.238, 4.013, 3.749, 3.490, 3.188,           # 650-690
    2.901, 2.622, 2.334, 2.041, 1.732,           # 700-740
    1.431, 1.146, None,                          # 750-770
)

TYPE1: tuple[float | None, ...] = (
    None, None,                                  # 340-350
    None,                                        # 360  "< 1,000"
    1.640, 2.860, 4.460,                         # 370-390
    5.000, 4.460, 2.860, 1.640,                  # 400-430
    None,                                        # 440  "< 1,000"
) + (None,) * 33                                 # 450-770 blank in the table

TYPE2: tuple[float | None, ...] = (
    None,                                        # 340  "< 1,000"
    2.708, 4.280, 4.583, 4.760, 4.851,           # 350-390
    4.916, 4.956, 4.988, 5.000, 4.990,           # 400-440
    4.951, 4.864, 4.743, 4.582, 4.351,           # 450-490
    3.993, 3.402, 2.805, 2.211,                  # 500-530
    None,                                        # 540  "< 1,000"
) + (None,) * 23                                 # 550-770 blank in the table



#: Table 4 -- Status M, log10 spectral products. None = outside the tabulated
#: range; ``weights()`` extends those with the printed slopes rather than
#: truncating to zero.
STATUS_M_B: tuple[float | None, ...] = (
    None, None, None, None, None, None,          # 340-390
    None, 2.103, 4.111, 4.632, 4.871,            # 400-440  (400 = slope start)
    5.000, 4.955, 4.743, 4.343, 3.743,           # 450-490
    2.990, 1.852, None, None, None,              # 500-540
    None, None, None, None, None,                # 550-590
    None, None, None, None, None,                # 600-640
    None, None, None, None, None,                # 650-690
    None, None, None, None, None,                # 700-740
    None, None, None,                            # 750-770
)
STATUS_M_G: tuple[float | None, ...] = (
    None, None, None, None, None, None,          # 340-390
    None, None, None, None, None,                # 400-440
    None, None, 1.152, 2.207, 3.156,             # 450-490
    3.804, 4.272, 4.626, 4.872, 5.000,           # 500-540
    4.995, 4.818, 4.458, 3.915, 3.172,           # 550-590
    2.239, 1.070, None, None, None,              # 600-640
    None, None, None, None, None,                # 650-690
    None, None, None, None, None,                # 700-740
    None, None, None,                            # 750-770
)
STATUS_M_R: tuple[float | None, ...] = (
    None, None, None, None, None, None,          # 340-390
    None, None, None, None, None,                # 400-440
    None, None, None, None, None,                # 450-490
    None, None, None, None, None,                # 500-540
    None, None, None, None, None,                # 550-590
    None, None, 2.109, 4.479, 5.000,             # 600-640
    4.899, 4.578, 4.252, 3.875, 3.491,           # 650-690
    3.099, 2.687, 2.269, 1.859, 1.449,           # 700-740
    1.054, 0.654, 0.254,                         # 750-770
)

#: Table 3 -- Status A, log10 spectral products.
STATUS_A_B: tuple[float | None, ...] = (
    None, None, None, None, None, None,          # 340-390
    None, None, 3.602, 4.819, 5.000,             # 400-440
    4.912, 4.620, 4.040, 2.989, 1.566,           # 450-490
    0.165, None, None, None, None,               # 500-540
    None, None, None, None, None,                # 550-590
    None, None, None, None, None,                # 600-640
    None, None, None, None, None,                # 650-690
    None, None, None, None, None,                # 700-740
    None, None, None,                            # 750-770
)
STATUS_A_G: tuple[float | None, ...] = (
    None, None, None, None, None, None,          # 340-390
    None, None, None, None, None,                # 400-440
    None, None, None, None, None,                # 450-490
    1.650, 3.822, 4.782, 5.000, 4.906,           # 500-540
    4.644, 4.221, 3.609, 2.766, 1.579,           # 550-590
    None, None, None, None, None,                # 600-640
    None, None, None, None, None,                # 650-690
    None, None, None, None, None,                # 700-740
    None, None, None,                            # 750-770
)
STATUS_A_R: tuple[float | None, ...] = (
    None, None, None, None, None, None,          # 340-390
    None, None, None, None, None,                # 400-440
    None, None, None, None, None,                # 450-490
    None, None, None, None, None,                # 500-540
    None, None, None, None, None,                # 550-590
    2.568, 4.638, 5.000, 4.871, 4.604,           # 600-640
    4.286, 3.900, 3.551, 3.165, 2.776,           # 650-690
    2.383, 1.970, 1.551, 1.141, 0.741,           # 700-740
    0.341, None, None,                           # 750-770
)

#: Printed extrapolation slopes, log10 units per nm: (below_range, above_range).
_SLOPES = {
    "status_m_b": (0.250, -0.220), "status_m_g": (0.106, -0.120),
    "status_m_r": (0.260, -0.040),
    "status_a_b": (0.380, -0.140), "status_a_g": (0.220, -0.170),
    "status_a_r": (0.270, -0.040),
}

_TABLES = {
    "visual": VISUAL, "type1": TYPE1, "type2": TYPE2,
    "status_m_b": STATUS_M_B, "status_m_g": STATUS_M_G, "status_m_r": STATUS_M_R,
    "status_a_b": STATUS_A_B, "status_a_g": STATUS_A_G, "status_a_r": STATUS_A_R,
}


def weights(kind: str = "visual") -> list[float]:
    """Linear spectral products, peak-normalised to 1.0.

    For the Status tables the printed slopes extend the response past the
    tabulated range; Table 2's "< 1,000" entries are genuinely floor and stay
    at zero. Values are clamped at 1e-6 relative so the skirts terminate.
    """
    tab = _TABLES[kind]
    log = [None if v is None else v for v in tab]
    if kind in _SLOPES:
        lo_s, hi_s = _SLOPES[kind]
        idx = [i for i, v in enumerate(log) if v is not None]
        first, last = idx[0], idx[-1]
        for i in range(first - 1, -1, -1):
            log[i] = log[i + 1] - lo_s * 10.0
        for i in range(last + 1, len(log)):
            log[i] = log[i - 1] + hi_s * 10.0
    out = []
    for v in log:
        if v is None:
            out.append(0.0)
        else:
            lin = 10.0 ** (v - 5.0)
            out.append(lin if lin > 1e-6 else 0.0)
    return out


def density(transmittance, kind: str = "visual") -> float:
    """ISO density of a sample given its spectral transmittance on LAMBDAS."""
    import math
    w = weights(kind)
    num = sum(wi * t for wi, t in zip(w, transmittance))
    den = sum(w)
    if den <= 0.0 or num <= 0.0:
        return float("inf")
    return -math.log10(num / den)


def _self_check() -> None:
    for name, tab in _TABLES.items():
        assert len(tab) == len(LAMBDAS), (name, len(tab), len(LAMBDAS))
        vals = [v for v in tab if v is not None]
        assert abs(max(vals) - 5.000) < 1e-9, (name, max(vals))
    # peak positions stated by the tables
    assert LAMBDAS[VISUAL.index(5.000)] == 570
    assert LAMBDAS[TYPE1.index(5.000)] == 400
    assert LAMBDAS[TYPE2.index(5.000)] == 430
    assert LAMBDAS[STATUS_M_B.index(5.000)] == 450
    assert LAMBDAS[STATUS_M_G.index(5.000)] == 540
    assert LAMBDAS[STATUS_M_R.index(5.000)] == 640
    assert LAMBDAS[STATUS_A_B.index(5.000)] == 440
    assert LAMBDAS[STATUS_A_G.index(5.000)] == 530
    assert LAMBDAS[STATUS_A_R.index(5.000)] == 620
    # Status M red peaks longer than Status A red -- the documented reason the
    # two exist: M matches colour NEGATIVE responses, A matches transparency
    assert LAMBDAS[STATUS_M_R.index(5.000)] > LAMBDAS[STATUS_A_R.index(5.000)]
    # a spectrally non-selective sample must read density 0 in every metric
    flat = [1.0] * len(LAMBDAS)
    for k in _TABLES:
        assert abs(density(flat, k)) < 1e-12, k
    # and a uniform 10% transmitter must read exactly 1.0
    grey = [0.1] * len(LAMBDAS)
    for k in _TABLES:
        assert abs(density(grey, k) - 1.0) < 1e-12, k


if __name__ == "__main__":
    _self_check()
    print("ISO 5-3 / IT2.18 spectral products loaded and self-checked.")
    print("  grid: %d points, %d-%d nm at 10 nm" % (len(LAMBDAS), LAMBDAS[0], LAMBDAS[-1]))
    for k in _TABLES:
        w = weights(k)
        nz = [i for i, v in enumerate(w) if v > 0]
        print("  %-6s peak at %d nm, nonzero %d-%d nm, %d active points"
              % (k, LAMBDAS[w.index(max(w))], LAMBDAS[nz[0]], LAMBDAS[nz[-1]], len(nz)))
