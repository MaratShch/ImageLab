"""
Physically-parameterised film stock profiles.

Every value here is a *physical* quantity, not a pixel-space fudge factor.
That is the whole point: the simulation must look identical at 1080p, 4K and
8K, which is only possible if grain size, halation radius and MTF cutoff are
expressed in micrometres / cycles-per-millimetre and converted to pixels at
render time from the scan resolution.

Units used throughout
---------------------
* Densities            : optical density (log10 of opacity), dimensionless
* logE                 : log10 of relative exposure, 0.0 == 18% mid grey
* Grain size           : micrometres (um) of the mean developed clump
* RMS granularity      : sigma(density) * 1000, measured through a 48 um
                         circular aperture at density 1.0 (industry convention)
* MTF f50              : cycles/mm at which modulation transfer falls to 50%
* Halation radii       : micrometres on the negative
* Misregistration      : micrometres on the negative
* Colour temperature   : kelvin the stock is balanced for

Negative vs reversal
--------------------
``StockKind.NEGATIVE`` stocks record an inverted image and must be printed;
``StockKind.REVERSAL`` stocks (Kodachrome, Ektachrome, Velvia, Tri-X Reversal)
*are* the positive, so the renderer skips the print stage entirely for them.
Reversal curves are expressed against **negated** log exposure, because on a
slide more light means less density -- see ``ToneCurve`` for the consequence.
Reversal stocks have high gamma, very high Dmax and famously little latitude,
all of which falls out of the curve parameters rather than being faked.

!! CALIBRATION HONESTY NOTE !!
-----------------------------
Values tagged ``# EST`` are plausible engineering estimates, not transcriptions
from manufacturer datasheets. They produce a convincing result but they are not
authoritative, and the older and more obscure the stock the rougher the
estimate. To make this a true emulation rather than a good-looking
approximation, replace them with digitised datasheet data:

* Kodak publishes D-logE curves, MTF curves, spectral sensitivity and RMS
  granularity for every current VISION3 stock in its Technical Data sheets.
* Fujifilm published equivalents for ETERNA / SUPER F while the stocks shipped.
* ORWO and Svema data survives mostly in scanned GDR/USSR technical handbooks.
* Kodachrome and Technicolor parameters here are reconstructions from published
  descriptions and surviving prints, not measurements. Treat them as artistic
  targets, not physics.

Digitise the curves (WebPlotDigitizer or similar), fit the parameters below to
them, and the "can a colourist tell?" answer changes from "probably" to "no".

Schema version 2 (2026-07, post domain review)
----------------------------------------------
Adds temporal behaviour (gate weave / flicker / dirt, DM-13), reciprocity
failure (DM-07), aging hooks (DM-01), provenance metadata (DM-19),
calibration metadata (density metric, speed criterion, mask encoding,
Callier coefficient -- DM-06/DM-14/DM-18), per-channel grain RMS and a
grain-sigma shape (DM-09/DM-10), MTF tail and resolving-power fields
(DM-11), printer lights on print stocks (DM-15) and full format geometry
via ``FORMAT_GEOM`` (DM-17). Every new field has an inert default chosen to
reproduce version-1 behaviour, so version-1 consumers keep working
unchanged.

!! FEATURE FLAGS ARE DERIVED AND NON-AUTHORITATIVE (DM-20) !!
``FilmProfile.features`` is a convenience summary of the numeric fields for
UI and tooling. A renderer must key its behaviour on the numeric fields
themselves (halation gains, coupler strengths, reseau presence, ...), never
on the flags. ``validate_all`` only *warns* about flag/numeric
disagreements; it does not fail on them.

Requires Python 3.12+. Pure stdlib.
"""

import re
import warnings
from dataclasses import dataclass, field, replace
from enum import Enum, Flag, auto

__all__ = [
    "Feature",
    "StockKind",
    "ToneCurve",
    "RGBCurves",
    "GrainSpec",
    "MTFSpec",
    "HalationSpec",
    "CouplerSpec",
    "ReseauSpec",
    "TemporalSpec",
    "ReciprocitySpec",
    "AgingSpec",
    "Provenance",
    "FilmProfile",
    "PrintStock",
    "FILM_PROFILES",
    "PRINT_STOCKS",
    "FORMATS",
    "FORMAT_GEOM",
    "IDENTITY3",
    "SCHEMA_VERSION",
    "get_profile",
    "get_print_stock",
    "profile_names",
    "validate_all",
]

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

IDENTITY3: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

#: Data-model schema version. Bumped to 2 by the 2026-07 domain review; see
#: the module docstring for what was added. Mirrored into the generated C++.
SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class StockKind(Enum):
    """Whether the developed film is a negative or already a positive.

    Plain ``Enum`` rather than ``StrEnum`` so the module also imports on 3.10;
    only identity comparison is used, so the two behave identically here.
    """

    NEGATIVE = "negative"
    REVERSAL = "reversal"


class Feature(Flag):
    """Optional per-stock behaviour.

    A real Flag enum, not substring matching on a string. The original code did
    ``"HALATION" in profile["features"]``, which also matched
    ``"HAS_BLUE_HALATION"`` -- so the tungsten stock silently took the red
    halation branch. Flags make that class of bug impossible.
    """

    NONE = 0
    HALATION = auto()             # weak or absent anti-halation layer
    STRONG_DIR_COUPLERS = auto()  # modern stock, pronounced inter-image effect
    UNEVEN_EMULSION = auto()      # loose QC: slow large-scale sensitivity drift
    ORTHO_RESPONSE = auto()       # reduced red sensitivity
    NO_REMJET = auto()            # remjet removed: extreme halation
    THREE_STRIP = auto()          # beam-splitter camera, three separate records
    TABULAR_GRAIN = auto()        # T-grain crystals rather than cubic
    MOSAIC_RESEAU = auto()        # additive colour via a physical filter grid
    NITRATE_BASE = auto()         # cellulose nitrate support, pre-1951


# ---------------------------------------------------------------------------
# Characteristic curve (D-logE / Hurter-Driffield)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ToneCurve:
    """One channel of a characteristic curve.

    Built as a difference of two softplus ramps::

        D(x) = dmin + gamma * ( sp(x - toe_x, toe_k)
                              - sp(x - shoulder_x, shoulder_k) )

    which gives, in order: a flat base+fog region, a smooth toe, a straight
    line of slope ``gamma``, a smooth shoulder, and a flat Dmax. That is
    exactly the topology of a real H&D curve, with only five free parameters
    and monotonicity in practice as long as ``shoulder_k <= 1.4 * toe_k``.
    (``validate`` still only rejects above 2x, which is the analytic bound for
    the sign of the second derivative. Measured on the actual transfer, ratios
    above roughly 1.4 can produce a reversal of order 1e-6 near the shoulder
    asymptote -- harmless visually, but verify.py checks for it, so keep new
    stocks at or below 1.4x.)

    For **negative** stocks ``x`` is log exposure, so density rises with light.
    For **reversal** stocks ``x`` is *negated* log exposure, so density falls
    with light as it must on a slide. One consequence worth remembering when
    editing reversal numbers: ``toe_x`` controls the **highlight** end and
    ``shoulder_x`` the **shadow** end, the opposite of a negative.

    Attributes:
        dmin: Base + fog density (density at the clear end).
        gamma: Slope of the straight-line section, density per decade of
            exposure. Colour negative 0.50-0.70, B&W negative 0.55-0.90,
            print stock 1.6-3.0, colour reversal 1.6-2.1.
        toe_x: Position of the toe, in the curve's own x units.
        toe_k: Toe softness. Larger = more gradual.
        shoulder_x: Position of the shoulder.
        shoulder_k: Shoulder softness. Larger = more gradual.
    """

    dmin: float
    gamma: float
    toe_x: float
    toe_k: float
    shoulder_x: float
    shoulder_k: float

    @property
    def dmax(self) -> float:
        """Asymptotic maximum density."""
        return self.dmin + self.gamma * (self.shoulder_x - self.toe_x)

    @property
    def latitude_stops(self) -> float:
        """Exposure range between toe and shoulder, in stops."""
        return (self.shoulder_x - self.toe_x) * 3.321928  # decades -> stops

    def validate(self, label: str = "") -> None:
        """Raise if the parameters would produce a non-monotonic curve."""
        if self.shoulder_x <= self.toe_x:
            raise ValueError(f"{label}: shoulder_x must exceed toe_x")
        if self.toe_k <= 0 or self.shoulder_k <= 0:
            raise ValueError(f"{label}: softness constants must be > 0")
        if self.shoulder_k > 2.0 * self.toe_k:
            raise ValueError(
                f"{label}: shoulder_k > 2*toe_k can make the curve "
                "non-monotonic (the bright end would reverse)"
            )
        if self.gamma <= 0:
            raise ValueError(f"{label}: gamma must be > 0")


@dataclass(frozen=True, slots=True)
class RGBCurves:
    """Three characteristic curves, one per dye layer.

    The *differences* between these three curves are what produce a stock's
    colour signature. Where the curves diverge in the toe you get a shadow
    colour cast; where they diverge in the shoulder you get a highlight cast.
    Cinematographers call this crossover, and it is the real mechanism behind
    "Fuji cyan shadows" or "Kodak warm highlights". Tinting the output image is
    a cosmetic imitation of it; diverging the curves *is* it.
    """

    r: ToneCurve
    g: ToneCurve
    b: ToneCurve

    def validate(self, label: str = "") -> None:
        self.r.validate(f"{label}.r")
        self.g.validate(f"{label}.g")
        self.b.validate(f"{label}.b")

    def as_tuple(self) -> tuple[ToneCurve, ToneCurve, ToneCurve]:
        return (self.r, self.g, self.b)


# ---------------------------------------------------------------------------
# Grain
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GrainSpec:
    """Silver-halide grain description.

    Grain is *not* additive noise of fixed amplitude. Developed crystals are
    discrete and their count per unit area follows Poisson statistics, so the
    density standard deviation grows roughly as sqrt(density). That is modelled
    at render time; this struct describes the spatial and statistical character
    of the emulsion.

    Attributes:
        rms_granularity: sigma(D) * 1000 through a 48 um aperture at D = 1.0.
            The renderer calibrates the noise field to hit this number exactly,
            which is what makes the result scan-resolution independent.
        clump_um_r/g/b: Mean developed clump diameter per layer, micrometres.
            In colour negative the fast blue-sensitive layer usually has the
            coarsest grain; the red layer sits at the bottom of the stack.
        clump_gain: Extra low-frequency energy = clumpiness. Classic cubic
            crystals cluster strongly (0.8-1.6); modern tabular "T-grain"
            crystals lie flat and pack evenly (0.1-0.4). This one number is
            most of the difference between HP5's velvety look and VISION3's
            even sand, independent of grain size.
        fog_grain: Grain floor present even at zero exposure, as a fraction of
            the D = 1.0 sigma. Never zero on real film -- perfectly clean
            blacks are one of the loudest digital tells.
        anisotropy: Ratio of vertical to horizontal grain correlation length.
            1.0 = isotropic. Slightly non-unity models emulsion coating flow.

    Schema-v2 extensions (DM-09/DM-10). All defaults reproduce v1 behaviour:

        rms_r/g/b: Per-channel RMS granularity, same metric as
            ``rms_granularity``. 0.0 means "use rms_granularity for every
            channel" (the v1 behaviour). Where populated for colour negative,
            the blue record is noisiest (top of the stack, fastest layer) and
            red slightly noisier than green -- tier-2 estimates, b ~1.3x and
            r ~1.1x of the green value.
        sigma_shape_toe/mid/dmax: Grain-sigma multipliers at D = dmin (toe),
            D = 1.0 (mid) and D = dmax. The triple (0, 1, 0) -- the default --
            means "legacy sqrt(D - dmin) law"; anything else describes a
            piecewise sigma(D) profile through those three anchors. Negatives
            are monotone (~0.4 / 1.0 / 1.2); reversal stocks turn over past
            mid-scale (~0.7 / 1.0 / 0.5) because a slide's densest regions
            received the least exposure. Tier-3 estimates.
        size_sigma_log: Log-normal dispersion of developed grain size. 0.35
            is typical of conventional cubic emulsions; fast pushed stocks
            run ~0.55, modern T-grain ~0.25. Tier-3.
        cluster_um: Correlation length of grain *clusters* (super-clumps),
            micrometres. 0.0 = disabled (v1 behaviour); ``clump_gain`` above
            remains the primary clustering control.
        dye_cloud_um: Developed dye-cloud diameter for chromogenic stocks,
            micrometres. 0.0 for B&W silver images; ~1.5-2.5 for colour
            stocks per tier-2 practice.
    """

    rms_granularity: float
    clump_um_r: float
    clump_um_g: float
    clump_um_b: float
    clump_gain: float
    fog_grain: float = 0.18
    anisotropy: float = 1.0
    # -- schema v2 (DM-09/DM-10); defaults are inert, see class docstring ----
    rms_r: float = 0.0
    rms_g: float = 0.0
    rms_b: float = 0.0
    sigma_shape_toe: float = 0.0
    sigma_shape_mid: float = 1.0
    sigma_shape_dmax: float = 0.0
    size_sigma_log: float = 0.35
    cluster_um: float = 0.0
    dye_cloud_um: float = 0.0

    def clumps(self) -> tuple[float, float, float]:
        return (self.clump_um_r, self.clump_um_g, self.clump_um_b)

    def rms_rgb(self) -> tuple[float, float, float]:
        """Per-channel RMS, falling back to the pooled figure (v1 rule)."""
        return (
            self.rms_r or self.rms_granularity,
            self.rms_g or self.rms_granularity,
            self.rms_b or self.rms_granularity,
        )


# ---------------------------------------------------------------------------
# Modulation transfer (sharpness)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MTFSpec:
    """Per-layer modulation transfer function.

    Modelled as ``MTF(f) = exp(-ln2 * (f / f50)**2)`` so that MTF(f50) = 0.5
    exactly. A real MTF curve is not Gaussian -- it usually shows adjacency
    overshoot above 100% at low frequency from development effects -- but the
    Gaussian is a fair fit through the mid band and cheap in the frequency
    domain. Replace with a digitised curve for datasheet accuracy.

    The per-channel ordering matters and is physical: in colour negative the
    blue-sensitive layer is on top of the stack, green in the middle, red at
    the bottom. Light reaching the red layer has already been scattered by two
    layers of gelatin, so **red is the softest channel and blue the sharpest**.
    That channel-dependent softness is a strong film signature and is missing
    from almost every naive grain filter.

    Attributes:
        f50_r/g/b: Cycles per millimetre at 50% modulation, per layer.
        adjacency: Strength of development adjacency overshoot (0 disables).
        adjacency_um: Spatial scale of that overshoot, micrometres.

    Schema-v2 extensions (DM-11), defaults reproduce v1 behaviour:

        resolving_power_lp_mm_lowc: Datasheet resolving power, line pairs per
            millimetre, at the low-contrast 1.6:1 test-object contrast. 0.0 =
            not published / not verified -- deliberately left 0 rather than
            invented for most stocks.
        resolving_power_lp_mm_highc: Same, at 1000:1 TOC. 0.0 = unknown.
        mtf_tail_a: Weight of the Gaussian core in a core+tail MTF model,
            ``MTF(f) = a*exp(-ln2*(f/f50)^2) + (1-a)*exp(-(f/f50)^p)``.
            1.0 = pure Gaussian, i.e. the legacy v1 model.
        mtf_tail_f_exp: Exponent ``p`` of the tail term. 0.0 = unused
            (only meaningful when mtf_tail_a < 1.0).
    """

    f50_r: float
    f50_g: float
    f50_b: float
    adjacency: float = 0.0
    adjacency_um: float = 25.0
    # -- schema v2 (DM-11); defaults are inert, see class docstring ----------
    resolving_power_lp_mm_lowc: float = 0.0
    resolving_power_lp_mm_highc: float = 0.0
    mtf_tail_a: float = 1.0
    mtf_tail_f_exp: float = 0.0

    def f50s(self) -> tuple[float, float, float]:
        return (self.f50_r, self.f50_g, self.f50_b)


# ---------------------------------------------------------------------------
# Halation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HalationSpec:
    """Light that passed through the emulsion, bounced off the base, came back.

    Three things the usual single-Gaussian-on-the-red-channel hack gets wrong:

    1. The radial profile has a long tail. A single Gaussian gives a tight
       halo; real halation has a faint wide bloom that extends far past it.
       Modelled here as a weighted sum of three Gaussians spanning more than an
       order of magnitude in radius.
    2. It is wavelength dependent but not red-only. Red penetrates the emulsion
       deepest so it dominates, but green and blue contribute measurably.
    3. It is an *exposure* phenomenon. It must be added to linear-light
       exposure before the characteristic curve, not to output pixel values
       after everything. Adding it at the end is why bolted-on halation reads
       as a Photoshop glow.

    Attributes:
        radii_um: Three Gaussian sigmas in micrometres, small to large.
        weights: Relative energy of each Gaussian. Normalised internally.
        gain_r/g/b: Per-channel coupling strength.
        threshold_stops: Exposure above mid grey, in stops, where halation
            becomes significant. Soft knee, not a hard cut.
    """

    radii_um: tuple[float, float, float] = (12.0, 60.0, 320.0)
    weights: tuple[float, float, float] = (0.62, 0.28, 0.10)
    gain_r: float = 0.0
    gain_g: float = 0.0
    gain_b: float = 0.0
    threshold_stops: float = 1.6

    def gains(self) -> tuple[float, float, float]:
        return (self.gain_r, self.gain_g, self.gain_b)

    @property
    def active(self) -> bool:
        return max(self.gains()) > 0.0


# ---------------------------------------------------------------------------
# DIR couplers / inter-image effects
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CouplerSpec:
    """Development-Inhibitor-Releasing coupler behaviour.

    Modern colour negative releases inhibitors as it develops. Heavy
    development in one layer suppresses development in its neighbours, which
    (a) increases saturation without increasing gamma and (b) creates a local
    adjacency edge effect. This is a large part of why VISION3 looks different
    from 1980s EXR stock beyond grain size, and no amount of grain tuning
    substitutes for it.

    Kodachrome deliberately has none: its dyes were formed in the processing
    machine, not in the film, so ``strength`` is zero for it.

    Attributes:
        strength: Cross-layer inhibition amount, 0 disables.
        radius_um: Diffusion distance of the inhibitor in the emulsion.
        edge_strength: Short-range component producing the adjacency edge lift.
        edge_um: Spatial scale of the edge component.
    """

    strength: float = 0.0
    radius_um: float = 55.0
    edge_strength: float = 0.0
    edge_um: float = 12.0

    @property
    def active(self) -> bool:
        return self.strength > 0.0 or self.edge_strength > 0.0


# ---------------------------------------------------------------------------
# Additive colour via a physical filter grid (reseau)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ReseauSpec:
    """Geometric colour filter grid ruled onto the film base.

    Dufaycolor and its relatives are *additive* colour: instead of three dye
    layers, the base carries a microscopic grid of colour filters, and a single
    panchromatic emulsion behind it records whatever each filter passes. On
    projection the positive is viewed back through the same grid in register,
    which reassembles the colour.

    Three consequences the layered-stock model cannot express, which is why this
    needs its own code path rather than a parameter:

    1. Colour resolution is limited by the grid pitch, not the emulsion. The
       whole image is one B&W record, so there is exactly one grain field and no
       inter-layer effects of any kind.
    2. Each filter absorbs roughly two thirds of the light, costing about two
       stops of speed. That is why these stocks are rated so slow.
    3. The grid itself remains faintly visible as texture, and it beats against
       the sampling grid if the render is too small -- real scans of Dufaycolor
       moire for exactly this reason.

    Attributes:
        lines_per_mm: Grid pitch on the film. Dufaycolor's cine reseau was
            around 20 lines/mm; finer grids resolve better but transmit less.
        filter_matrix: What each filter passes, row-major, rows in R/G/B filter
            order and columns in R/G/B light order. So ``filter_matrix[0][1]``
            is how much green light the red filter lets through.

            The off-diagonal terms are the whole reason additive colour looks
            pastel. Ruled colour filters are dyed gelatin with broad, heavily
            overlapping passbands -- the red filter passes a good deal of green,
            the green filter passes both neighbours. Model them as pure (a single
            transmission per filter, diagonal only) and the process comes out
            more saturated than Kodachrome, which is precisely backwards. The
            desaturation is not an artefact to add afterwards; it falls out of
            the filters not being pure.
        pattern: Geometry. ``"dufay"`` gives Dufaycolor's arrangement --
            continuous red lines with blue and green squares chequered between
            them, each colour taking about a third of the area.
        reconstruction_pitches: Radius of the reconstruction blur, in grid
            pitches. Smaller leaves more visible grid texture; larger gives
            cleaner colour and a softer image, which is exactly the tradeoff a
            projectionist could not escape.
    """

    lines_per_mm: float = 20.0
    filter_matrix: Matrix3 = (
        (0.62, 0.14, 0.03),
        (0.16, 0.55, 0.14),
        (0.05, 0.20, 0.52),
    )
    pattern: str = "dufay"
    reconstruction_pitches: float = 0.62

    def pitch_um(self) -> float:
        """Grid cell size in micrometres."""
        return 1000.0 / self.lines_per_mm

    def neutral_gain(self) -> float:
        """Mean row sum: what a cell records from an equal-energy neutral.

        Used only to renormalise the record so that a neutral grey comes out of
        the grid unchanged, which keeps the anchor solve (which cannot see the
        mask) valid. Not the same number as :meth:`mean_throughput` -- see there.
        """
        return sum(sum(row) for row in self.filter_matrix) / 3.0

    def mean_throughput(self) -> float:
        """Fraction of incident *white* light reaching the emulsion.

        Each of the three bands carries a third of white light, so a cell passes
        ``row_sum / 3`` of it and the grid average is the sum of all nine entries
        over nine. Around 0.27 here, i.e. about 1.9 stops surrendered to the
        filters, which is why additive stocks were rated so slow.

        Distinct from :meth:`neutral_gain`, which is three times larger. Using
        one where the other belongs either reports the wrong speed penalty or
        breaks the neutral anchor.
        """
        return sum(sum(row) for row in self.filter_matrix) / 9.0

    def validate(self, label: str = "") -> None:
        if self.lines_per_mm <= 0:
            raise ValueError(f"{label}: lines_per_mm must be > 0")
        if self.pattern != "dufay":
            raise ValueError(f"{label}: unknown reseau pattern {self.pattern!r}")
        for r, row in enumerate(self.filter_matrix):
            if min(row) < 0.0 or max(row) > 1.0:
                raise ValueError(
                    f"{label}: filter_matrix row {r} must lie in [0, 1]"
                )
            if sum(row) <= 0.0:
                raise ValueError(f"{label}: filter_matrix row {r} passes no light")
        for c in range(3):
            if self.filter_matrix[c][c] != max(self.filter_matrix[c]):
                raise ValueError(
                    f"{label}: filter {c} must pass its own colour most strongly"
                )
        if self.reconstruction_pitches <= 0:
            raise ValueError(f"{label}: reconstruction_pitches must be > 0")


# ---------------------------------------------------------------------------
# Temporal behaviour (schema v2, DM-13)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TemporalSpec:
    """Per-stock temporal behaviour for moving-image rendering.

    ALL VALUES ARE TIER-3 ESTIMATES: era-typical figures for the camera /
    printer / projector chain a stock normally passed through, not
    measurements of a specific emulsion. A still render ignores this struct
    entirely.

    Attributes:
        weave_amp_x_um: RMS horizontal gate-weave amplitude, micrometres on
            the negative. Silent-era chains ~20, 1950s ~10, modern ~3.
        weave_amp_y_um: RMS vertical gate-weave amplitude, micrometres.
        weave_hz_corner: Corner frequency of the weave power spectrum, Hz.
            Weave is a random walk with most energy below ~1 Hz.
        flicker_pct: RMS frame-to-frame exposure flicker, percent. Early
            processing/printing 4-6%, 1950s ~1.5%, modern ~0.2%.
        flicker_hz: Corner frequency of the flicker spectrum, Hz.
        grain_frame_correlation: 0.0 = a fresh grain field every frame (the
            physical truth for camera negative and the v1 behaviour);
            > 0 models grain that persists across frames (dupe chains).
        dirt_events_per_frame: Expected visible dirt/dust events per frame.
        scratch_persistence_frames: Mean lifetime of a running scratch,
            frames.
        fps_native: Frame rate the stock was normally shot/projected at.
    """

    weave_amp_x_um: float = 0.0
    weave_amp_y_um: float = 0.0
    weave_hz_corner: float = 0.0
    flicker_pct: float = 0.0
    flicker_hz: float = 0.0
    grain_frame_correlation: float = 0.0
    dirt_events_per_frame: float = 0.0
    scratch_persistence_frames: float = 0.0
    fps_native: float = 24.0


# ---------------------------------------------------------------------------
# Reciprocity failure (schema v2, DM-07)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ReciprocitySpec:
    """Schwarzschild reciprocity-failure description.

    Effective exposure for time t beyond onset: ``E_eff = I * t**p`` with
    p <= 1.0 per channel. p = 1.0 (the default) means no failure -- the v1
    behaviour, correct for normal cine exposure times. Values are tier-2/3
    estimates except where a datasheet documents them (Acros famously needs
    no correction out to 120 s).

    Attributes:
        schwarzschild_p_r/g/b: Schwarzschild exponent per channel. 1.0 = no
            failure. Typical B&W ~0.95; colour reversal ~0.92-0.94 with a
            slight channel spread (the origin of long-exposure colour casts).
        onset_s: Exposure time, seconds, below which no correction applies.
    """

    schwarzschild_p_r: float = 1.0
    schwarzschild_p_g: float = 1.0
    schwarzschild_p_b: float = 1.0
    onset_s: float = 1.0


# ---------------------------------------------------------------------------
# Aging / storage damage hooks (schema v2, DM-01)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AgingSpec:
    """Storage-age damage model. ALL ZEROS = FRESH STOCK.

    Pure data hooks for now: every profile ships fresh (all zeros) and the
    reference renderer does not yet consume this struct. It exists so aged
    looks become data rather than post hacks when renderer support lands.

    Attributes:
        dye_fade_c/m/y: Fractional fade of the cyan/magenta/yellow dye, 0-1.
            Magenta usually survives longest, cyan goes first on Eastman
            stock of the 1950s-70s -- hence pink archival prints.
        base_yellowing_d: Density of base yellowing (blue-density lift).
        dmin_lift: Uniform fog growth, density.
        shrinkage_pct: Base shrinkage, percent. Drives registration and
            focus problems in real transports.
        scratch_rate_base_per_m: Base-side scratches per metre of run length.
        scratch_rate_emulsion_per_m: Emulsion-side scratches per metre.
        dust_area_ppm: Dust coverage, parts per million of frame area.
        mottle_amplitude: Amplitude of storage mottle (fraction of density).
        mottle_scale_mm: Spatial scale of that mottle, millimetres.
    """

    dye_fade_c: float = 0.0
    dye_fade_m: float = 0.0
    dye_fade_y: float = 0.0
    base_yellowing_d: float = 0.0
    dmin_lift: float = 0.0
    shrinkage_pct: float = 0.0
    scratch_rate_base_per_m: float = 0.0
    scratch_rate_emulsion_per_m: float = 0.0
    dust_area_ppm: float = 0.0
    mottle_amplitude: float = 0.0
    mottle_scale_mm: float = 0.0


# ---------------------------------------------------------------------------
# Provenance (schema v2, DM-19)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a profile's numbers come from, machine-readable.

    Mirrors the human-readable [T1]/[T2]/[T3] tags in the descriptions (which
    are kept untouched); this struct is the queryable form.

    Attributes:
        tier: 1 = datasheet-grounded, 2 = partially grounded,
            3 = reconstruction. See the confidence-tier note in the database.
        sources: Full citations backing the numbers, formatted
            "Document title, Publisher, year" (year omitted when the
            document does not state one), e.g. "KODACHROME 25/64/200
            Films, Kodak publication E-55, Eastman Kodak Company, 2009".
            When no official manufacturer document is on file, carries the
            explicit ``_NO_DATASHEET`` placeholder instead of being empty.
        fitted_from: One of "datasheet_curve", "secondary_sources",
            "analogy".
        last_reviewed: ISO date of the last data review.
    """

    tier: int = 3
    sources: tuple[str, ...] = ()
    fitted_from: str = "analogy"
    last_reviewed: str = "2026-07-30"


# ---------------------------------------------------------------------------
# Film profile
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FilmProfile:
    """Complete description of one film stock.

    Attributes:
        name: Stable identifier.
        aliases: Alternative lookup keys -- catalogue numbers, common names.
            Matching ignores case, spaces, hyphens and underscores, so
            "vision3 500t", "5219" and "KODAK_VISION3_500T_5219" all resolve.
        description: Human-readable character summary.
        era: Production years, for documentation.
        kind: Negative or reversal. Reversal stocks skip the print stage.
        is_monochrome: Single silver image instead of three dye layers.
        exposure_index: ISO/EI rating.
        balance_kelvin: Colour temperature the stock is balanced for. Feeding a
            5500 K daylight image to a 3200 K tungsten stock should produce the
            strong blue cast you would really get without an 85 filter -- this
            field is what makes that happen instead of being hand-tinted.
        curves: Per-layer characteristic curves.
        grain: Grain statistics and geometry.
        mtf: Per-layer sharpness.
        halation: Base-reflection glow.
        couplers: Inter-layer development effects.
        taking_matrix: 3x3 applied to *linear exposure*, modelling the spectral
            sensitivity of each record. Identity for ordinary integral tripack
            stocks; strongly off-diagonal for a Technicolor beam-splitter
            camera, whose broad overlapping taking filters are the origin of
            its famous reds.
        dye_matrix: 3x3 applied to the *density* vector, modelling impure dye
            absorption spectra plus scanner channel crosstalk. Row-major.
            Negative off-diagonals sharpen colour separation (Kodachrome,
            Technicolor imbibition dyes); positive off-diagonals muddy it
            (ORWOcolor). This replaces the crude "add cyan to shadows" hack.
        spectral_weights: Scene RGB -> single-layer exposure weights for
            monochrome stocks. NOT Rec.601 luma -- panchromatic film has its
            own spectral sensitivity, which is why B&W film renders red lips
            dark and blue skies bright. Using video luma throws that away.
        base_tint: Residual cast of the film base after printer lights
            neutralise the mask.
        misregistration_um: RMS channel registration error on the negative.
            Micrometres, so it scales with resolution like everything else.
            Tiny for an integral tripack, large and characteristic for
            three-strip Technicolor.
        default_print: Print stock used when the caller does not choose one.
            Ignored for reversal stocks.
        reseau: Additive colour filter grid, or None for an ordinary stock.
            Selects a separate code path -- see :class:`ReseauSpec`.
        default_flare: Veiling flare fraction of the *lens* typically used with
            this stock, not a property of the emulsion. Uncoated pre-1940
            lenses scattered 6-14% of the light entering them into a broad haze
            across the whole frame; anti-reflection coating dropped that below
            1%. It is stored here because era of stock and era of glass go
            together in practice, and it is overridable per render. Without it,
            a 1930s emulsion still renders with modern black levels, which is
            most of why period profiles otherwise disappoint.
        features: Optional behaviour flags. DERIVED AND NON-AUTHORITATIVE
            (DM-20): a convenience summary of the numeric fields. Renderers
            must key on the numeric fields, never on these flags.

    Schema-v2 fields (populated by the decoration pass at the bottom of this
    module; every default is inert):

        temporal: Gate weave / flicker / dirt for motion rendering (DM-13).
        reciprocity: Schwarzschild reciprocity failure (DM-07).
        aging: Storage-damage hooks, all zeros = fresh (DM-01).
        provenance: Machine-readable source/confidence metadata (DM-19).
        trim: Static reversal exposure trim in the mapping
            ``x = -log10(E) - trim``. The reference renderer has no static
            constant -- ``solve_anchors`` in film_sim.py solves per-channel
            log-exposure trims at render time -- so this stays 0.0 and the
            per-render anchor solve refines it. Documented for C++ ports.
        density_metric: Densitometry the curve/granularity numbers are
            expressed in: "status_m" (colour negative), "status_a" (colour
            reversal and print stocks), "visual_iso" (B&W silver).
        referred: "scan" when default_print is the SCAN_DI transform (all
            current camera stocks except Technicolor), "print" when the
            profile is calibrated against a real print stock.
        speed_point_x: Curve-abscissa position of metered mid-grey, in the
            curve's own x units. 0.0 = mid grey sits at logE = 0 (the v1
            convention baked into the renderer).
        speed_criterion: How exposure_index was assigned: "iso6" (B&W),
            "iso5800" (colour negative), "iso2240" (colour reversal),
            "manufacturer_ei" (historic stocks predating the standards).
        mask_encoding: How the orange coupler mask is encoded in the curve
            data: "dmin_ladder" when the per-channel dmin values ARE the mask
            (r << g << b, e.g. 0.20/0.62/1.02), "neutral_dmin" when dmin is
            near-neutral and the mask lives in base_tint/dye_matrix, "none"
            for B&W and reversal. Documents the split found in the datasheet
            audit; does not change render output.
        callier_q: Callier coefficient -- ratio of specular to diffuse
            density seen by a condenser system. 1.0 for dye images (colour),
            ~1.3 for B&W silver negatives, ~1.25 for B&W reversal.
    """

    name: str
    description: str
    era: str
    exposure_index: int
    balance_kelvin: int
    curves: RGBCurves
    grain: GrainSpec
    mtf: MTFSpec
    kind: StockKind = StockKind.NEGATIVE
    is_monochrome: bool = False
    aliases: tuple[str, ...] = ()
    halation: HalationSpec = field(default_factory=HalationSpec)
    couplers: CouplerSpec = field(default_factory=CouplerSpec)
    taking_matrix: Matrix3 = IDENTITY3
    dye_matrix: Matrix3 = IDENTITY3
    spectral_weights: tuple[float, float, float] = (0.30, 0.59, 0.11)
    base_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    misregistration_um: float = 6.0
    default_print: str = "SCAN_DI"
    #: Gauge this stock was actually sold on. The renderer uses it whenever the
    #: caller does not pass --format explicitly, which is what makes `-p all`
    #: meaningful: an 8 mm home-movie stock rendered at Super 35 shows 35 mm
    #: grain and 35 mm detail, i.e. nothing like 8 mm. Every spatial quantity in
    #: this database is in micrometres or cycles/mm, so the gauge is the single
    #: number that converts emulsion physics into pixels.
    default_format: str = "super35"
    reseau: ReseauSpec | None = None
    #: Image-tone of the developed silver, for monochrome stocks only.
    #: Positive = warm / brown-black, negative = cool / blue-black, 0 = neutral.
    #: Physically this is particle-size dependent: fine silver scatters short
    #: wavelengths and reads warm, coarse filamentary silver reads neutral to
    #: blue. It is strongest at LOW density (highlights) and fades as density
    #: rises, which is why an untoned fine-grain print still looks faintly
    #: sepia in its light tones.
    #:
    #: NOTE this is deliberately NOT base_tint. base_tint is compensated by the
    #: printer-light anchor solve (a real printer neutralises the film base), so
    #: it cannot produce a visible cast -- measured, it produces R-B of exactly
    #: 0.0000. silver_tone is applied after the anchor solve and survives.
    #:
    #: Calibrated against user-supplied Tasma FN64 scans: two of three frames
    #: showed a mean R-G of +8.6 and +15.6 / 255 in their bright regions.
    silver_tone: float = 0.0
    default_flare: float = 0.0
    features: Feature = Feature.NONE
    # -- schema v2 (see class docstring). Inert defaults; the decoration pass
    # at the bottom of this module fills in per-stock values. ----------------
    temporal: TemporalSpec = field(default_factory=TemporalSpec)
    reciprocity: ReciprocitySpec = field(default_factory=ReciprocitySpec)
    aging: AgingSpec = field(default_factory=AgingSpec)
    provenance: Provenance = field(default_factory=Provenance)
    #: Static reversal trim; 0.0 because the per-render anchor solve refines it.
    trim: float = 0.0
    density_metric: str = "status_m"
    referred: str = "scan"
    #: Curve-abscissa position of metered mid-grey (v1 convention: 0.0).
    speed_point_x: float = 0.0
    speed_criterion: str = "manufacturer_ei"
    mask_encoding: str = "none"
    callier_q: float = 1.0

    @property
    def is_reversal(self) -> bool:
        return self.kind is StockKind.REVERSAL

    @property
    def has_reseau(self) -> bool:
        return self.reseau is not None

    def validate(self) -> None:
        self.curves.validate(self.name)
        if self.reseau is not None:
            self.reseau.validate(self.name)
            if self.is_monochrome:
                raise ValueError(
                    f"{self.name}: a reseau stock records colour through the "
                    "grid, so it must not also be flagged monochrome"
                )
        if not 0.0 <= self.default_flare < 1.0:
            raise ValueError(f"{self.name}: default_flare must be in [0, 1)")
        if self.grain.rms_granularity <= 0:
            raise ValueError(f"{self.name}: rms_granularity must be > 0")
        if min(self.grain.clumps()) <= 0:
            raise ValueError(f"{self.name}: clump sizes must be > 0")
        if min(self.mtf.f50s()) <= 0:
            raise ValueError(f"{self.name}: MTF f50 must be > 0")
        w = sum(self.spectral_weights)
        if not 0.9 < w < 1.1:
            raise ValueError(
                f"{self.name}: spectral_weights should sum to ~1.0, got {w:.3f}"
            )
        if self.misregistration_um < 0:
            raise ValueError(f"{self.name}: misregistration_um must be >= 0")

    def with_overrides(self, **kw) -> "FilmProfile":
        """Return a copy with fields replaced. Useful for parameter sweeps."""
        return replace(self, **kw)


@dataclass(frozen=True, slots=True)
class PrintStock:
    """Positive stock, or a digital-intermediate print-emulating transform.

    Nobody ever looks at a negative. What the eye recognises as "film" is a
    negative printed onto positive stock, or scanned and inverted through a
    print-emulating transform. The system gamma is the product of the two
    stages: negative gamma ~0.6 times print gamma ~1.75 gives ~1.05 for a
    scan-to-display look; print gamma 2.7 gives the ~1.6 of a theatrical
    release print.

    Modelling this as a second real curve rather than a hand-drawn S-curve is
    what produces correct highlight rolloff and shadow crush for free.
    """

    name: str
    description: str
    curves: RGBCurves
    mtf_f50: float = 95.0        # cycles/mm, print + scanner optics combined  # EST
    grain_rms: float = 3.0       # print stock adds its own, finer grain  # EST
    grain_clump_um: float = 5.5  # EST
    dye_matrix: Matrix3 = IDENTITY3  # print dye impurity / gamut
    # -- schema v2 (DM-15). Printer-light calibration, additive-lamp points.
    # 25.0/25.0/25.0 is the LAD-neutral mid-scale setting; one point moves
    # print log exposure by log_e_per_point (0.025 logE = 1/12 stop, the
    # industry convention). The anchor solve in the reference renderer
    # supersedes these at run time; they document the lab-side calibration.
    printer_light_r: float = 25.0
    printer_light_g: float = 25.0
    printer_light_b: float = 25.0
    log_e_per_point: float = 0.025
    #: Densitometry of the curve data. Print stocks are read in Status A.
    density_metric: str = "status_a"

    def validate(self) -> None:
        self.curves.validate(self.name)


# ---------------------------------------------------------------------------
# Film gauges. Aperture dimensions per SMPTE / manufacturer specification.
# ---------------------------------------------------------------------------
FORMAT_GEOM: dict[str, tuple[float, float, float, float]] = {
    # name: (width_mm, height_mm, anamorphic_squeeze, perf_pitch_mm)
    # perf_pitch 0.0 = not perforated / not transport-relevant (sheet, roll,
    # instant film). Squeeze > 1 means the lens squeezes horizontally and the
    # projector unsqueezes; grain and MTF are anisotropic by that factor.
    "super35": (24.89, 18.66, 1.0, 4.75),     # Super 35 full aperture
    "academy35": (21.95, 16.00, 1.0, 4.75),   # Academy aperture
    "anamorphic35": (21.95, 18.60, 2.0, 4.75),
    "techni35": (24.89, 18.66, 1.0, 4.75),    # three-strip used full aperture
    "super16": (12.52, 7.41, 1.0, 7.62),
    "16mm": (10.26, 7.49, 1.0, 7.62),
    "8mm": (4.80, 3.50, 1.0, 3.81),           # Standard 8 / Double 8
    "super8": (5.79, 4.01, 1.0, 4.23),        # smaller sprockets, bigger frame
    "ff35": (36.00, 24.00, 1.0, 4.75),        # 35 mm still full frame
    "medium645": (56.00, 41.50, 1.0, 0.0),
    "large4x5": (127.00, 101.60, 1.0, 0.0),
    "imax15": (70.41, 52.63, 1.0, 4.75),      # 15-perf horizontal 65 mm
    # Instant film image areas, so the Polaroid profiles scale correctly.
    "polaroid_sx70": (79.00, 77.00, 1.0, 0.0),  # SX-70/600 integral area
    "polaroid_pack": (95.00, 73.00, 1.0, 0.0),  # 664/667 peel-apart area
}
"""Full frame geometry (schema v2, DM-17): image width and height on the
negative in millimetres, anamorphic squeeze factor, and perforation pitch in
millimetres."""

FORMATS: dict[str, float] = {name: geom[0] for name, geom in FORMAT_GEOM.items()}
"""Image width on the negative, millimetres -- the v1 view, derived from
FORMAT_GEOM so the two can never disagree. Grain, halation, MTF and channel
registration are all scaled from this plus the render width in pixels, which is
what makes a profile resolution independent."""


# ---------------------------------------------------------------------------
# Helpers for terse curve construction
# ---------------------------------------------------------------------------
def _neg(
    dmin: float,
    gamma: float,
    toe_x: float = -1.55,
    toe_k: float = 0.30,
    shoulder_x: float = 1.75,
    shoulder_k: float = 0.42,
) -> ToneCurve:
    """Colour-negative-shaped curve with sensible defaults."""
    return ToneCurve(dmin, gamma, toe_x, toe_k, shoulder_x, shoulder_k)


def _mono(c: ToneCurve) -> RGBCurves:
    """One curve replicated across all three channels, for B&W stocks."""
    return RGBCurves(c, c, c)


def _dye(k_r: float, k_g: float | None = None, k_b: float | None = None) -> Matrix3:
    """Dye crosstalk as a pure saturation operator, with unit row sums.

    ``k`` is the fraction of each record blended toward the local mean of the
    three. Positive k desaturates (impure dyes, ORWOcolor and the 1930s stocks);
    negative k increases separation (clean dyes: Kodachrome, Velvia, the
    Technicolor imbibition set).

    Unit row sums are the point. Hand-written matrices tend to have row sums
    away from 1 -- 1.27 for a "muddy" stock, 0.92 for a "clean" one -- which
    means the matrix silently shifts neutral *density* as well as colour. Two
    unrelated effects on one knob: the anchor solve then has to undo the density
    part, and the stock's black level ends up depending on its saturation
    setting. With row sums pinned to 1, a neutral grey passes through unchanged
    and the matrix does nothing but colour, leaving dmin and gamma solely
    responsible for level.

    Pass one k for a symmetric matrix, or three for per-record character.
    """
    ks = (
        k_r,
        k_r if k_g is None else k_g,
        k_r if k_b is None else k_b,
    )
    rows = []
    for c, k in enumerate(ks):
        row = [k / 3.0, k / 3.0, k / 3.0]
        row[c] = 1.0 - 2.0 * k / 3.0
        rows.append((row[0], row[1], row[2]))
    return (rows[0], rows[1], rows[2])


def _rev(
    dmin: float,
    gamma: float,
    toe_x: float = -0.80,
    toe_k: float = 0.18,
    shoulder_x: float = 0.92,
    shoulder_k: float = 0.30,
) -> ToneCurve:
    """Reversal-shaped curve. Remember x is *negated* log exposure, so toe_x
    governs the highlight end. High gamma and short throw give slide film its
    contrast and its notoriously narrow latitude."""
    return ToneCurve(dmin, gamma, toe_x, toe_k, shoulder_x, shoulder_k)


# ===========================================================================
# THE STOCK DATABASE
# All numeric values are estimates; see the CALIBRATION HONESTY NOTE above.
# ===========================================================================
FILM_PROFILES: tuple[FilmProfile, ...] = (
    FilmProfile(
        name="AGFACOLOR_NEU_1936",
        aliases=("agfacolor", "agfacolor neu", "agfa 1936", "sovcolor"),
        description=(
            "The first modern integral tripack: three dye layers on one strip, "
            "the ancestor of every colour film since. As a 1936 product its "
            "dyes were badly impure, so it desaturates and cross-contaminates "
            "even while running high contrast -- the muted, slightly sickly "
            "palette of 1940s German colour features. Captured Agfa technology "
            "later became the Soviet Sovcolor process, hence the alias."
        ),
        era="1936-1945",
        kind=StockKind.REVERSAL,
        exposure_index=8,
        balance_kelvin=5500,
        # Reversal, so these are expressed against negated log exposure and
        # toe_x governs the highlight end. High gamma, very little latitude.
        curves=RGBCurves(
            r=_rev(0.30, 1.62, toe_x=-0.62, toe_k=0.20, shoulder_x=0.72),
            g=_rev(0.28, 1.70, toe_x=-0.66, toe_k=0.20, shoulder_x=0.70),
            b=_rev(0.33, 1.78, toe_x=-0.58, toe_k=0.20, shoulder_x=0.66),
        ),
        grain=GrainSpec(11.0, 14.0, 13.0, 17.0, clump_gain=1.25, fog_grain=0.30),
        mtf=MTFSpec(26.0, 30.0, 34.0, adjacency=0.02),
        halation=HalationSpec(
            radii_um=(22.0, 110.0, 500.0),
            gain_r=0.30, gain_g=0.16, gain_b=0.09,
            threshold_stops=1.0,
        ),
        couplers=CouplerSpec(),
        # The combination nothing else in this database has: a reversal stock
        # with strongly *positive* dye off-diagonals. Every other reversal stock
        # here has clean negative terms and gains saturation; this one bleeds
        # between records and loses it, while the steep curves keep contrast
        # high. Desaturated and contrasty at once, which is hard to fake with a
        # saturation control and falls straight out of the matrix.
        dye_matrix=_dye(0.45),
        base_tint=(0.985, 1.000, 0.945),
        misregistration_um=9.0,
        default_flare=0.09,
        default_format="ff35",
        features=Feature.HALATION | Feature.UNEVEN_EMULSION | Feature.NITRATE_BASE,
    ),
    FilmProfile(
        name="AGFA_APX_100",
        aliases=("apx100", "apx 100", "agfa apx 100"),
        description=(
            "[T1] The middle APX and the general-purpose one. Slightly lower "
            "contrast and a longer straight line than the Kodak equivalents "
            "of the same speed, which is why it was liked for portraiture."
        ),
        era="1990s-Present",
        is_monochrome=True,
        exposure_index=100,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.11, 0.620, -1.56, 0.31, 1.80, 0.42)),
        # rms 9.0: published diffuse RMS granularity, CONFIRMED 2026-07-31.
        # SOURCE PDF/PROFILES/AGFA/apx100.pdf p1: "Granularity: RMS (x 1000):
        # 9.0  (REFINAL, 6 min, 20 C)"; corroborated by agfa_films.pdf p10 and
        # Datasheet_F_PF_E4.pdf p9. Resolving power 150 lp/mm at 1000:1 (same
        # page) is recorded in _RESOLVING_POWER; Agfa publishes no 1.6:1 figure
        # for the APX films. Agfa also prints a Schwarzschild table on p1
        # (1 s -> +1 stop, 10 s -> +2, 100 s -> +3); it is deliberately NOT
        # fitted into ReciprocitySpec because a single Schwarzschild exponent
        # cannot reproduce those three points (p would have to be 0.40 at 10 s
        # and 0.55 at 100 s), so forcing one would invent data.
        grain=GrainSpec(9.0, 9.0, 9.0, 9.0, clump_gain=0.85, fog_grain=0.17),
        mtf=MTFSpec(80.0, 80.0, 80.0, adjacency=0.10, adjacency_um=14.0),
        spectral_weights=(0.28, 0.56, 0.16),
        misregistration_um=0.0,
        default_format="ff35",
        features=Feature.NONE,
    ),

    # =======================================================================
    # EXPANSION SET -- added after the original 26.
    #
    # CONFIDENCE TIERS. The original block carries a blanket "# EST" note.
    # This block is graded, because the sources behind these numbers vary
    # enormously in quality and you should be able to see which is which
    # without digging:
    #
    #   [T1] Datasheet-grounded. Published ISO speed, RMS granularity or
    #        diffuse grain number, and an MTF or resolving-power figure exist
    #        for this emulsion. Numbers below are fitted to them. Treat as
    #        trustworthy to maybe 10 %.
    #
    #   [T2] Partially grounded. Speed and general reputation are documented;
    #        grain and MTF are interpolated from siblings in the same family or
    #        from the manufacturer's other stocks of that era and speed.
    #
    #   [T3] Reconstruction. No datasheet available to me. Built from era,
    #        speed class, process type and written descriptions of the look.
    #        These are plausible, internally consistent, and NOT measurements.
    #        Do not cite them as such.
    # =======================================================================

    # ---------------------------- Agfa B&W ---------------------------------
    FilmProfile(
        name="AGFA_APX_25",
        aliases=("apx25", "apx 25", "agfa apx 25"),
        description=(
            "[T1] Agfa's slow fine-grain B&W. One of the finest-grained "
            "conventional cubic emulsions ever sold: grain is essentially "
            "below the resolution of a normal scan, so what you see instead "
            "is bite and micro-contrast. Punishing to expose -- ISO 25 means "
            "a tripod indoors -- and discontinued in 2005."
        ),
        era="1990s-2005",
        is_monochrome=True,
        exposure_index=25,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.10, 0.640, -1.62, 0.30, 1.86, 0.40)),
        # rms 7.0: published diffuse RMS granularity, CONFIRMED 2026-07-31.
        # SOURCE PDF/PROFILES/AGFA/agfapanapx25.pdf p1: "Granularity: RMS
        # (x 1000): 7.0"; corroborated by agfa_films.pdf p10. Resolving power
        # 200 lp/mm at 1000:1 (same page) -> _RESOLVING_POWER.
        grain=GrainSpec(7.0, 5.0, 5.0, 5.0, clump_gain=0.55, fog_grain=0.14),
        mtf=MTFSpec(112.0, 112.0, 112.0, adjacency=0.13, adjacency_um=11.0),
        spectral_weights=(0.28, 0.56, 0.16),
        misregistration_um=0.0,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="AGFA_APX_400",
        aliases=("apx400", "apx 400", "agfa apx 400"),
        description=(
            "[T1] The fast APX. Classic clumpy cubic grain, noticeably "
            "coarser than Ilford's 400 of the same period and with a slightly "
            "softer shoulder, so highlights roll rather than block."
        ),
        era="1990s-2000s",
        is_monochrome=True,
        exposure_index=400,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.13, 0.660, -1.50, 0.29, 1.70, 0.40)),
        # rms 14.0: published diffuse RMS granularity, CONFIRMED 2026-07-31.
        # SOURCE PDF/PROFILES/AGFA/apx400.pdf p1: "Granularity: RMS (x 1000):
        # 14.0"; corroborated by agfa_films.pdf p10. Resolving power 110 lp/mm
        # at 1000:1 (same page) -> _RESOLVING_POWER.
        grain=GrainSpec(14.0, 15.0, 15.0, 15.0, clump_gain=1.25, fog_grain=0.22),
        mtf=MTFSpec(48.0, 48.0, 48.0, adjacency=0.06, adjacency_um=19.0),
        spectral_weights=(0.28, 0.56, 0.16),
        misregistration_um=0.0,
        default_format="ff35",
        features=Feature.NONE,
    ),

    # --------------------------- Agfa colour -------------------------------
    FilmProfile(
        name="AGFA_OPTIMA_100",
        aliases=("optima", "agfa optima", "optima 100", "optima ii"),
        description=(
            "[T1] Agfa's consumer colour negative. The Agfa house palette: "
            "warm, slightly restrained, with yellows and skin tones favoured "
            "over the saturated primaries Kodak and Fuji were chasing. Dye "
            "purity is a step below either, which shows as gentle desaturation "
            "rather than a colour cast."
        ),
        era="1990s-2000s",
        exposure_index=100,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.20, 0.600),
            g=_neg(0.62, 0.610),
            b=_neg(1.02, 0.620),
        ),
        # rms 4.0: published diffuse RMS granularity (x1000) for AGFACOLOR
        # OPTIMA II 100. SOURCE: Agfa "Professional Films" brochure
        # (PDF/PROFILES/AGFA/agfa_films.pdf p7): "Granularity (x 1000):
        # RMS 4.0". Corrected 2026-07-31 from the previous estimate of 7.8.
        # The per-channel rms_r/g/b values are derived from this figure by
        # _grain_v2's tier-2 stack rule (b ~1.3x, r ~1.1x of green); Agfa does
        # not publish per-layer granularity.
        grain=GrainSpec(4.0, 11.0, 12.0, 14.0, clump_gain=0.80, fog_grain=0.18),
        # f50 values remain engineering estimates: Agfa publishes sharpness
        # only as a plotted transfer-factor curve (agfa_films.pdf p7), never as
        # a numeric MTF. The published resolving power (50 lp/mm at 1.6:1,
        # 140 lp/mm at 1000:1, same page) is recorded in _RESOLVING_POWER.
        mtf=MTFSpec(62.0, 70.0, 76.0, adjacency=0.09, adjacency_um=17.0),
        couplers=CouplerSpec(0.22, 52.0, 0.10, 12.0),
        dye_matrix=_dye(0.07),
        base_tint=(1.0, 0.985, 0.955),
        misregistration_um=5.5,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="AGFA_VISTA_200",
        aliases=("vista", "agfa vista", "vista 200"),
        description=(
            "[T2] Punchier and cooler than Optima -- Agfa's answer to the "
            "supermarket-film wars, with higher saturation and a steeper "
            "green. Note the name outlived the emulsion: late AgfaPhoto-branded "
            "Vista was Fuji stock in an Agfa box. This models the real Agfa one."
        ),
        era="1990s-2000s",
        exposure_index=200,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.21, 0.615),
            g=_neg(0.64, 0.635),
            b=_neg(1.05, 0.640),
        ),
        grain=GrainSpec(9.4, 13.0, 14.0, 16.5, clump_gain=0.92, fog_grain=0.19),
        mtf=MTFSpec(56.0, 63.0, 69.0, adjacency=0.08, adjacency_um=18.0),
        couplers=CouplerSpec(0.30, 50.0, 0.13, 12.0),
        dye_matrix=_dye(-0.05),
        base_tint=(0.99, 0.995, 1.0),
        misregistration_um=5.5,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="CINESTILL_800T",
        aliases=("cinestill", "800t", "cinestill 800t"),
        description=(
            "VISION3 500T with the remjet anti-halation layer stripped off so "
            "it can run through C-41. The result is the most extreme halation "
            "in production: every streetlight grows a red corona. Useful here "
            "as a stress test of the halation model."
        ),
        era="2012-present",
        exposure_index=800,
        balance_kelvin=3200,
        curves=RGBCurves(
            r=_neg(0.22, 0.610, toe_x=-1.52, shoulder_x=1.86),
            g=_neg(0.20, 0.630, toe_x=-1.46, shoulder_x=1.78),
            b=_neg(0.19, 0.652, toe_x=-1.36, shoulder_x=1.66),
        ),
        grain=GrainSpec(8.4, 11.5, 12.5, 15.0, clump_gain=0.36, fog_grain=0.22),
        mtf=MTFSpec(40.0, 48.0, 56.0, adjacency=0.09, adjacency_um=24.0),
        # No remjet: the glow is enormous and reaches very far.
        halation=HalationSpec(
            radii_um=(20.0, 130.0, 700.0),
            weights=(0.42, 0.34, 0.24),
            gain_r=1.05, gain_g=0.30, gain_b=0.10,
            threshold_stops=0.9,
        ),
        couplers=CouplerSpec(0.12, 60.0, 0.06, 13.0),
        dye_matrix=_dye(-0.11),
        base_tint=(1.000, 0.986, 0.958),
        misregistration_um=5.0,
        default_format="ff35",
        features=Feature.HALATION | Feature.NO_REMJET | Feature.STRONG_DIR_COUPLERS,
    ),
    FilmProfile(
        name="DUFAYCOLOR_1937",
        aliases=("dufaycolor", "dufay", "reseau", "mosaic"),
        description=(
            "Additive colour with no dye layers at all: a microscopic grid of "
            "red lines and chequered blue and green squares ruled onto the base, "
            "with one panchromatic emulsion behind it. Pastel, low-saturation "
            "colour, soft, very slow, and the grid stays faintly visible as "
            "texture. RENDER THIS ONE LARGE -- the grid is a physical 20 "
            "lines/mm, so below about 2000 px wide there are not enough pixels "
            "to resolve it and the mosaic is disabled with a warning."
        ),
        era="1932-1950s",
        kind=StockKind.REVERSAL,
        exposure_index=10,
        balance_kelvin=5500,
        # One emulsion, so one curve. All the colour behaviour comes from the
        # reseau, not from these.
        curves=_mono(ToneCurve(0.30, 1.48, -0.72, 0.22, 0.94, 0.34)),
        grain=GrainSpec(12.5, 16.5, 16.5, 16.5, clump_gain=1.35, fog_grain=0.34),
        mtf=MTFSpec(30.0, 30.0, 30.0, adjacency=0.03),
        halation=HalationSpec(
            radii_um=(22.0, 105.0, 470.0),
            gain_r=0.24, gain_g=0.24, gain_b=0.24,
            threshold_stops=1.05,
        ),
        couplers=CouplerSpec(),
        reseau=ReseauSpec(
            lines_per_mm=20.0,
            # Broad, overlapping dyed-gelatin passbands. These off-diagonals are
            # what make the process pastel rather than lurid.
            filter_matrix=(
                (0.62, 0.14, 0.03),
                (0.16, 0.55, 0.14),
                (0.05, 0.20, 0.52),
            ),
            pattern="dufay",
            reconstruction_pitches=0.62,
        ),
        misregistration_um=0.0,  # one record, so nothing to misregister
        default_flare=0.11,
        features=(
            Feature.HALATION | Feature.MOSAIC_RESEAU
            | Feature.UNEVEN_EMULSION | Feature.NITRATE_BASE
        ),
    ),
    FilmProfile(
        name="EASTMAN_5247_1974",
        aliases=("5247", "eastman 5247"),
        description=(
            "The 1970s. Low saturation, heavy clustered grain, soft everywhere "
            "and prone to a warm cast. If you want the look of a film shot "
            "between 1974 and 1982, this is closer than any grain plugin."
        ),
        era="1974-1982",
        exposure_index=100,
        balance_kelvin=3200,
        curves=RGBCurves(
            r=_neg(0.30, 0.545, toe_x=-1.24, toe_k=0.40, shoulder_x=1.46),
            g=_neg(0.28, 0.560, toe_x=-1.18, toe_k=0.38, shoulder_x=1.40),
            b=_neg(0.29, 0.580, toe_x=-1.08, toe_k=0.36, shoulder_x=1.30),
        ),
        grain=GrainSpec(13.0, 18.5, 20.0, 24.0, clump_gain=1.40, fog_grain=0.30,
                        anisotropy=1.04),
        mtf=MTFSpec(24.0, 28.0, 33.0, adjacency=0.03, adjacency_um=34.0),
        halation=HalationSpec(
            radii_um=(22.0, 110.0, 520.0),
            weights=(0.50, 0.32, 0.18),
            gain_r=0.52, gain_g=0.22, gain_b=0.09,
            threshold_stops=1.0,
        ),
        couplers=CouplerSpec(0.02, 80.0),
        dye_matrix=_dye(0.22),
        base_tint=(1.000, 0.968, 0.918),
        misregistration_um=10.0,
        features=Feature.HALATION | Feature.UNEVEN_EMULSION,
    ),
    FilmProfile(
        name="EASTMAN_DOUBLE_X_5222",
        aliases=("5222", "double x", "double-x", "xx"),
        description=(
            "The B&W motion picture stock: Manhattan, Raging Bull, Schindler's "
            "List. Cubic grain with a distinctive silvery mid-tone and a very "
            "long straight line. Still in production, essentially unchanged "
            "since 1959."
        ),
        era="1959-present",
        is_monochrome=True,
        exposure_index=250,
        balance_kelvin=5500,  # EI 250 is the daylight rating (200 tungsten)
        curves=_mono(ToneCurve(0.13, 0.620, -1.70, 0.32, 2.26, 0.58)),
        # rms 14.0: published diffuse RMS (Kodak Double-X Technical Data).
        grain=GrainSpec(14.0, 12.0, 12.0, 12.0, clump_gain=1.05, fog_grain=0.22),
        mtf=MTFSpec(56.0, 56.0, 56.0, adjacency=0.09),
        spectral_weights=(0.32, 0.47, 0.21),
        misregistration_um=0.0,
        features=Feature.NONE,
    ),

    # ------------------ Eastman Ektachrome EF news reversal ----------------
    FilmProfile(
        name="EASTMAN_EKTACHROME_5239",
        aliases=("5239", "ektachrome ef", "eastman daylight 5239", "ef daylight"),
        description=(
            "[T2] Ektachrome EF Daylight, 35 mm. Fast reversal news and "
            "documentary stock, EI 160, routinely push-processed further. "
            "Lower contrast than a pictorial slide film because it had to "
            "survive being projected straight off the camera roll, and grainy "
            "in a way 1960s television flattered. Lineage note: the 5239/7239 "
            "daylight designation belongs to the VNF era, succeeding "
            "Ektachrome EF 5241/7241."
        ),
        era="1960s-1970s",
        kind=StockKind.REVERSAL,
        exposure_index=160,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_rev(0.19, 1.45, toe_x=-0.86, shoulder_x=1.04),
            g=_rev(0.20, 1.48, toe_x=-0.88, shoulder_x=1.02),
            b=_rev(0.23, 1.50, toe_x=-0.92, shoulder_x=0.98),
        ),
        grain=GrainSpec(10.4, 12.5, 13.5, 16.0, clump_gain=0.72, fog_grain=0.20),
        mtf=MTFSpec(48.0, 54.0, 60.0, adjacency=0.08, adjacency_um=18.0),
        halation=HalationSpec(gain_r=0.05, gain_g=0.018, gain_b=0.005,
                             threshold_stops=1.9),
        couplers=CouplerSpec(0.10, 50.0, 0.05, 11.0),
        dye_matrix=_dye(0.04),
        misregistration_um=5.0,
        # Carries nonzero halation gains, so the derived flag is set (DM-20).
        features=Feature.HALATION,
    ),
    FilmProfile(
        name="EASTMAN_EKTACHROME_7239",
        aliases=("7239", "eastman daylight 7239", "ef 16mm"),
        description=(
            "[T2] Ektachrome EF Daylight, 16 mm -- the same emulsion as 5239 "
            "on 16 mm base. The emulsion numbers here are deliberately "
            "identical: the visible difference between the two is entirely "
            "magnification, which the renderer derives from the frame width "
            "you choose, not from the profile. Pick format '16mm' or 'super16' "
            "with this one and the grain will grow on its own."
        ),
        era="1960s-1970s",
        kind=StockKind.REVERSAL,
        exposure_index=160,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_rev(0.19, 1.45, toe_x=-0.86, shoulder_x=1.04),
            g=_rev(0.20, 1.48, toe_x=-0.88, shoulder_x=1.02),
            b=_rev(0.23, 1.50, toe_x=-0.92, shoulder_x=0.98),
        ),
        grain=GrainSpec(10.4, 12.5, 13.5, 16.0, clump_gain=0.72, fog_grain=0.20),
        mtf=MTFSpec(48.0, 54.0, 60.0, adjacency=0.08, adjacency_um=18.0),
        halation=HalationSpec(gain_r=0.05, gain_g=0.018, gain_b=0.005,
                             threshold_stops=1.9),
        couplers=CouplerSpec(0.10, 50.0, 0.05, 11.0),
        dye_matrix=_dye(0.04),
        misregistration_um=5.0,
        default_format="16mm",
        # Carries nonzero halation gains, so the derived flag is set (DM-20).
        features=Feature.HALATION,
    ),
    # -----------------------------------------------------------------------
    # Older Eastman colour negative. Cubic crystals, no coupler sophistication.
    # -----------------------------------------------------------------------
    FilmProfile(
        name="EASTMAN_EXR_500T_5296",
        aliases=("5296", "exr 500t", "exr"),
        description=(
            "Early-90s blockbuster stock. Classic cubic crystals: large and "
            "strongly clustered. Weak anti-halation, softer than anything "
            "modern, and it clips highlights far sooner than VISION3."
        ),
        era="1989-1996",
        exposure_index=500,
        balance_kelvin=3200,
        curves=RGBCurves(
            r=_neg(0.26, 0.560, toe_x=-1.40, toe_k=0.36, shoulder_x=1.62),
            g=_neg(0.24, 0.585, toe_x=-1.32, toe_k=0.34, shoulder_x=1.55),
            b=_neg(0.23, 0.615, toe_x=-1.20, toe_k=0.32, shoulder_x=1.44),
        ),
        grain=GrainSpec(10.5, 16.0, 17.5, 21.0, clump_gain=1.15, fog_grain=0.26),
        mtf=MTFSpec(30.0, 36.0, 42.0, adjacency=0.05, adjacency_um=30.0),
        halation=HalationSpec(
            radii_um=(18.0, 95.0, 480.0),
            weights=(0.52, 0.32, 0.16),
            gain_r=0.46, gain_g=0.18, gain_b=0.07,
            threshold_stops=1.2,
        ),
        couplers=CouplerSpec(0.05, 70.0, 0.02),
        dye_matrix=_dye(-0.06),
        base_tint=(1.000, 0.975, 0.935),
        misregistration_um=8.0,
        features=Feature.HALATION,
    ),
    # =======================================================================
    # 1930s-1940s stocks.
    #
    # !! READ THIS BEFORE TRUSTING ANY NUMBER BELOW !!
    #
    # Everything in this block is a *reconstruction*, not an estimate. For the
    # modern stocks above, the numbers are engineering guesses anchored to
    # published datasheets I could reason about. Here there are no datasheets I
    # can consult: the figures are inferred from how surviving footage looks,
    # from the physics of the emulsion technology of the period, and from
    # internal consistency with the rest of the database. Treat them as
    # artistic targets. Super-XX is the firmest of the five because it stayed
    # in production for decades; Agfacolor Neu and the Soviet stock are the
    # softest, and Dufaycolor's reseau pitch is the only figure there I would
    # defend within a factor of two.
    #
    # Period characteristics these share, and which are the real content:
    #   * high base fog (dmin 0.25-0.38 vs 0.12-0.22 modern) -- weak blacks
    #   * low Dmax and short shoulder -- highlights clip early
    #   * coarse, strongly clustered cubic grain
    #   * soft: f50 of 20-35 c/mm against 44-98 for modern stock
    #   * no DIR couplers at all: the chemistry did not exist yet
    #   * heavy halation: anti-halation backing was primitive or absent
    #   * large default_flare, because the lenses were uncoated
    # =======================================================================
    FilmProfile(
        name="EASTMAN_ORTHO_1930",
        aliases=("ortho", "orthochromatic", "1930 ortho", "eastman ortho"),
        description=(
            "Orthochromatic black and white negative: sensitive to blue and "
            "green, effectively blind to red. Red renders as black and a blue "
            "sky renders as blank white. This single property is the most "
            "recognisable cue of pre-1930s cinema, and the reason period makeup "
            "was so extreme -- ordinary red lipstick photographed black, so "
            "actors wore yellow and green greasepaint instead."
        ),
        era="1920s-early 1930s",
        is_monochrome=True,
        exposure_index=25,
        balance_kelvin=3400,  # carbon arc / early incandescent
        curves=_mono(ToneCurve(0.32, 0.700, -1.06, 0.26, 1.44, 0.40)),
        grain=GrainSpec(13.5, 17.0, 17.0, 17.0, clump_gain=1.45, fog_grain=0.38,
                        anisotropy=1.05),
        mtf=MTFSpec(28.0, 28.0, 28.0, adjacency=0.02),
        halation=HalationSpec(
            radii_um=(24.0, 120.0, 560.0),
            weights=(0.46, 0.34, 0.20),
            gain_r=0.30, gain_g=0.30, gain_b=0.30,
            threshold_stops=1.0,
        ),
        couplers=CouplerSpec(),  # no coupler chemistry existed
        # The whole point of this profile. Red sensitivity is not merely low, it
        # is nearly absent; the residual 2% stands in for slight far-red leakage.
        spectral_weights=(0.02, 0.45, 0.53),
        misregistration_um=0.0,
        default_flare=0.13,
        features=(
            Feature.HALATION | Feature.ORTHO_RESPONSE
            | Feature.UNEVEN_EMULSION | Feature.NITRATE_BASE
        ),
    ),
    FilmProfile(
        name="EASTMAN_PLUS_X_5231",
        aliases=("plus-x", "plusx", "5231", "plus x 5231", "eastman plus-x"),
        description=(
            "[T1] Eastman Plus-X negative 5231, EI 80 daylight / 64 tungsten. "
            "The fine-grain B&W cine negative of the era and the aspirational "
            "choice for Indian studios that could afford imported Eastman over "
            "cheaper European stock. Noticeably finer and sharper than the "
            "Gevaert equivalent, with a long straight line that grades easily. "
            "If you want a period Indian B&W look that came off a well-funded "
            "production rather than a poverty-row one, this is the one."
        ),
        era="1950s-2000s",
        is_monochrome=True,
        exposure_index=80,
        balance_kelvin=5500,  # EI 80 is the daylight rating (64 tungsten)
        curves=_mono(ToneCurve(0.12, 0.680, -1.48, 0.30, 1.74, 0.42)),
        # rms 10.0: published diffuse RMS (Kodak 5231 Technical Data).
        grain=GrainSpec(10.0, 11.0, 11.0, 11.0, clump_gain=1.00, fog_grain=0.20),
        mtf=MTFSpec(60.0, 60.0, 60.0, adjacency=0.08, adjacency_um=16.0),
        spectral_weights=(0.27, 0.54, 0.19),
        misregistration_um=0.0,
        features=Feature.NONE,
    ),
    FilmProfile(
        name="EASTMAN_SUPER_XX_1938",
        aliases=("super xx", "superxx", "super-xx", "1201", "1938"),
        description=(
            "The fast panchromatic negative that made 1940s Hollywood look the "
            "way it does -- deep-focus photography and film noir were shot on "
            "this. Fast for its day, so coarse and clustered grain, soft by "
            "modern standards, with a long straight line that holds shadow "
            "detail far better than its contemporaries."
        ),
        era="1938-1950s",
        is_monochrome=True,
        exposure_index=100,
        balance_kelvin=5500,  # EI 100 is the daylight rating
        curves=_mono(ToneCurve(0.28, 0.610, -1.52, 0.34, 1.92, 0.52)),
        grain=GrainSpec(12.0, 16.0, 16.0, 16.0, clump_gain=1.30, fog_grain=0.32),
        mtf=MTFSpec(35.0, 35.0, 35.0, adjacency=0.04),
        halation=HalationSpec(
            radii_um=(20.0, 100.0, 460.0),
            gain_r=0.26, gain_g=0.26, gain_b=0.26,
            threshold_stops=1.1,
        ),
        couplers=CouplerSpec(),
        # Panchromatic but, like most emulsions of the period, still weaker in
        # red than a modern film and rather hot in blue.
        spectral_weights=(0.24, 0.46, 0.30),
        misregistration_um=0.0,
        default_flare=0.10,
        features=Feature.HALATION | Feature.NITRATE_BASE,
    ),

    # ----------------------------- 8 mm gauges ------------------------------
    # "8mm" is a gauge, not an emulsion. These two are representative home-movie
    # reversal stocks -- what was actually loaded in a Standard 8 or Super 8
    # cartridge. The overwhelming visual signature of 8 mm is magnification:
    # a 4.80 mm wide frame blown up to HD is a 400x area enlargement, so grain
    # and softness that would be invisible on 35 mm dominate the image. That
    # comes from the format width, so render these with format '8mm' or
    # 'super8' or the whole point is lost. MEASURED: at 1024 px wide a 16 um
    # clump spans 0.66 px on super35 but 3.41 px on 8mm. The DEFAULT FORMAT IS
    # super35, so `film_sim.py img -p '8mm bw'` with no -f renders 8 mm grain at
    # 35 mm scale and looks wrong. Always pass -f 8mm or -f super8.
    # Emulsion grain is deliberately FINE here (EI 40 reversal, finer than the
    # EI 200 Tri-X Reversal in this set). Do not coarsen it to 'make 8 mm look
    # like 8 mm' -- that double-counts the magnification the renderer already
    # applies.
    FilmProfile(
        name="EIGHT_MM_BW",
        aliases=("8mm bw", "8mm b&w", "8mm mono", "eight mm bw", "regular8 bw"),
        description=(
            "[T3] Representative 8 mm B&W reversal home-movie stock, EI 40, of "
            "the Plus-X / Tri-X reversal class. Reversal because home movies "
            "were projected directly -- there was no print stage and no "
            "negative to grade. Moderate contrast, chunky cubic grain, and a "
            "resolution that was never the limiting factor next to the camera "
            "lenses of the day."
        ),
        era="1930s-1980s",
        kind=StockKind.REVERSAL,
        is_monochrome=True,
        exposure_index=40,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.14, 1.45, -0.62, 0.22, 0.73, 0.30)),
        # Grain raised because a REVERSAL stock gets no print stage, and the
        # print stage is where a negative's grain is multiplied by the print
        # gamma (~1.75). Measured: from grain field to final output a negative
        # KEEPS 1.24x of its grain amplitude, a reversal stock keeps 0.63x.
        # That asymmetry is physically right -- reversal film is the viewed
        # image, there is nothing downstream to amplify it -- but it means a
        # reversal emulsion needs a genuinely higher RMS to read as grainy.
        # Justified by amateur emulsion quality and by reversal processing,
        # which develops the unexposed silver and yields coarser apparent
        # grain than negative processing of the same crystals.
        grain=GrainSpec(19.0, 17.0, 17.0, 17.0, clump_gain=1.45, fog_grain=0.26,
                        anisotropy=1.06),
        mtf=MTFSpec(44.0, 44.0, 44.0, adjacency=0.05, adjacency_um=19.0),
        spectral_weights=(0.27, 0.54, 0.19),
        misregistration_um=0.0,
        default_format="8mm",
        features=Feature.UNEVEN_EMULSION,
    ),
    FilmProfile(
        name="EIGHT_MM_COLOR",
        aliases=("8mm color", "8mm colour", "eight mm color", "super8 color"),
        description=(
            "[T3] Representative 8 mm colour reversal home-movie stock, EI 40, "
            "of the Kodachrome II / Ektachrome 160 cartridge class. Warm, "
            "saturated, contrasty -- reversal film projected in a dark room "
            "could afford contrast that a print chain could not. The nostalgic "
            "'home movie' look is this emulsion plus enormous magnification "
            "plus, usually, decades of dye fade."
        ),
        era="1930s-1980s",
        kind=StockKind.REVERSAL,
        exposure_index=40,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_rev(0.17, 1.66, toe_x=-0.78, shoulder_x=0.94),
            g=_rev(0.18, 1.68, toe_x=-0.80, shoulder_x=0.92),
            b=_rev(0.21, 1.64, toe_x=-0.84, shoulder_x=0.90),
        ),
        # Same reversal reasoning as EIGHT_MM_BW: no print stage to amplify
        # the grain, so the emulsion number has to carry it.
        grain=GrainSpec(12.0, 10.5, 11.5, 13.0, clump_gain=0.70, fog_grain=0.22,
                        anisotropy=1.06),
        mtf=MTFSpec(50.0, 56.0, 62.0, adjacency=0.08, adjacency_um=17.0),
        halation=HalationSpec(gain_r=0.05, gain_g=0.018, gain_b=0.005,
                             threshold_stops=2.0),
        couplers=CouplerSpec(0.08, 50.0, 0.05, 11.0),
        dye_matrix=_dye(-0.10),
        base_tint=(1.0, 0.99, 0.965),
        misregistration_um=5.0,
        default_format="8mm",
        features=Feature.UNEVEN_EMULSION,
    ),
    FilmProfile(
        name="EKTACHROME_160T",
        aliases=("ektachrome 2", "ektachrome 160", "e160t", "ektachrome 160t"),
        description=(
            "[T1] Tungsten Ektachrome, EI 160, your 'Ektachrome 2'. Balanced "
            "for 3200 K lamps, so shot in daylight without correction it goes "
            "strongly blue. Faster and correspondingly grainier than the 64, "
            "and the stock behind a great deal of 1970s interior and stage "
            "photography."
        ),
        era="1970s-2000s",
        kind=StockKind.REVERSAL,
        exposure_index=160,
        balance_kelvin=3200,
        curves=RGBCurves(
            r=_rev(0.17, 1.62, toe_x=-0.82, shoulder_x=0.98),
            g=_rev(0.18, 1.65, toe_x=-0.84, shoulder_x=0.96),
            b=_rev(0.20, 1.66, toe_x=-0.88, shoulder_x=0.88),
        ),
        # rms 13.0: published diffuse RMS granularity, CONFIRMED 2026-07-31.
        # SOURCE PDF/PROFILES/KODAK/e144-Ektachrome_160T_EPT.pdf p4: "Diffuse
        # rms Granularity*  13 (very fine)". Same sheet: EI 160 tungsten,
        # Process E-6, densitometry Status A.
        grain=GrainSpec(13.0, 9.5, 10.5, 12.5, clump_gain=0.52, fog_grain=0.17),
        mtf=MTFSpec(58.0, 65.0, 72.0, adjacency=0.10, adjacency_um=16.0),
        halation=HalationSpec(gain_r=0.05, gain_g=0.017, gain_b=0.005,
                             threshold_stops=2.0),
        couplers=CouplerSpec(0.09, 50.0, 0.05, 11.0),
        dye_matrix=_dye(-0.14),
        misregistration_um=4.0,
        default_format="ff35",
        features=Feature.NONE,
    ),

    # -------------------------- Ektachrome stills --------------------------
    FilmProfile(
        name="EKTACHROME_64",
        aliases=("ektachrome 1", "e64", "ektachrome 64", "ektachrome64"),
        description=(
            "[T1] Daylight Ektachrome, EI 64. Interpreted here as your "
            "'Ektachrome 1'. Cooler and more neutral than Kodachrome, with "
            "cleaner greens and less of the red bias, and processed in ordinary "
            "E-6 rather than Kodachrome's proprietary line."
        ),
        era="1970s-2012",
        kind=StockKind.REVERSAL,
        exposure_index=64,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_rev(0.15, 1.70, toe_x=-0.78, shoulder_x=0.94),
            g=_rev(0.16, 1.74, toe_x=-0.79, shoulder_x=0.92),
            b=_rev(0.17, 1.76, toe_x=-0.80, shoulder_x=0.89),
        ),
        # rms 11.0: published diffuse RMS granularity, CONFIRMED 2026-07-31.
        # SOURCE PDF/PROFILES/KODAK/e8-Ektachrome_64_EPR.pdf p5: "Diffuse rms
        # Granularity* 11 (very fine)". Same sheet: EI 64, Process E-6,
        # densitometry Status A, and a numeric reciprocity/CC-filter table.
        # Kodak publishes no resolving-power figure for this stock.
        grain=GrainSpec(11.0, 6.0, 6.5, 7.5, clump_gain=0.34, fog_grain=0.13),
        mtf=MTFSpec(72.0, 80.0, 88.0, adjacency=0.12, adjacency_um=14.0),
        halation=HalationSpec(gain_r=0.045, gain_g=0.015, gain_b=0.004,
                             threshold_stops=2.1),
        couplers=CouplerSpec(0.08, 48.0, 0.05, 10.0),
        dye_matrix=_dye(-0.18),
        misregistration_um=3.5,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="FERRANIA_P30",
        aliases=("p30", "ferrania", "ferrania p30", "neorealism"),
        description=(
            "[T2] Ferrania P30, Italy, EI 80. High silver content, contrasty, "
            "and fine-grained for its speed -- the cine stock behind Italian "
            "neorealism and, because Ferrania undercut Kodak across southern "
            "Europe and Latin America, behind a great deal of Argentine, "
            "Brazilian and Mexican production too. A note on your question "
            "about South America: no country there manufactured raw film at "
            "scale in 1940-1980. Its studios shot on imports, and Ferrania was "
            "one of the most common. So this is an Italian stock, honestly "
            "labelled, that gets you closest to that cinema. High gamma and "
            "deep Dmax are the signature -- P30 blacks are genuinely black."
        ),
        era="1960s / revived 2017",
        is_monochrome=True,
        exposure_index=80,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.11, 0.880, -1.40, 0.28, 1.72, 0.38)),
        grain=GrainSpec(7.4, 10.0, 10.0, 10.0, clump_gain=1.15, fog_grain=0.18),
        mtf=MTFSpec(66.0, 66.0, 66.0, adjacency=0.09, adjacency_um=15.0),
        spectral_weights=(0.27, 0.55, 0.18),
        misregistration_um=0.0,
        features=Feature.NONE,
    ),
    FilmProfile(
        name="FOMAPAN_400_ACTION",
        aliases=("fomapan", "fomapan 400", "fomapan 400 action"),
        description=(
            "Czech B&W, nominally 400 but metering closer to 250. Older-style "
            "emulsion: coarser and more clustered than HP5, softer, and with a "
            "steeper curve that gives it noticeably more bite."
        ),
        era="1990s-present",
        is_monochrome=True,
        exposure_index=400,
        balance_kelvin=5500,
        # ISO 400/27 deg confirmed against the datasheet
        # (PDF/PROFILES/FOMACOLOR/fomapan-400.pdf p1).
        curves=_mono(ToneCurve(0.14, 0.690, -1.44, 0.30, 1.98, 0.50)),
        # rms 17.5: published diffuse RMS granularity. SOURCE:
        # PDF/PROFILES/FOMACOLOR/fomapan-400.pdf p1: "RMS = 17.5 (Microphen at
        # 20 oC, developed to [gamma] = 0.6 (measured at D = 1.0)". The stated
        # conditions match this project's metric definition (sigma(D)*1000 at
        # D = 1.0), so the figure is adopted verbatim. Corrected 2026-07-31
        # from the previous estimate of 11.5 -- a 52 % under-statement.
        grain=GrainSpec(17.5, 15.5, 15.5, 15.5, clump_gain=1.30, fog_grain=0.28,
                        anisotropy=1.03),
        # f50 stays an estimate: Foma publishes no MTF whatsoever. The single
        # published resolving-power figure (90 lines/mm, same page) is recorded
        # in _RESOLVING_POWER.
        mtf=MTFSpec(42.0, 42.0, 42.0, adjacency=0.05),
        spectral_weights=(0.30, 0.48, 0.22),
        misregistration_um=0.0,
        default_format="ff35",
        features=Feature.UNEVEN_EMULSION,
    ),
    # -----------------------------------------------------------------------
    # Fuji
    # -----------------------------------------------------------------------
    FilmProfile(
        name="FUJICOLOR_SUPER_F500_8572",
        aliases=("8572", "super f500", "super f-500", "f500"),
        description=(
            "Classic Fuji look. Curve crossover in the toe pushes shadows cyan "
            "and keeps highlights cool; finer grain than the Kodak equivalent "
            "of the period."
        ),
        era="1999-2007",  # audit: Super F-500 8572 production window
        exposure_index=500,
        balance_kelvin=3200,
        # The cyan shadow signature lives here: the blue and green toes start
        # earlier than red, so cyan leads in the low exposure region. No
        # output tinting is involved anywhere.
        curves=RGBCurves(
            r=_neg(0.23, 0.590, toe_x=-1.38, toe_k=0.34, shoulder_x=1.78),
            g=_neg(0.22, 0.615, toe_x=-1.56, toe_k=0.30, shoulder_x=1.80),
            b=_neg(0.21, 0.640, toe_x=-1.66, toe_k=0.28, shoulder_x=1.76),
        ),
        grain=GrainSpec(7.4, 10.0, 9.2, 12.6, clump_gain=0.62, fog_grain=0.22),
        mtf=MTFSpec(46.0, 56.0, 62.0, adjacency=0.08, adjacency_um=20.0),
        halation=HalationSpec(
            radii_um=(12.0, 62.0, 300.0),
            gain_r=0.18, gain_g=0.09, gain_b=0.05,
            threshold_stops=1.7,
        ),
        couplers=CouplerSpec(0.11, 52.0, 0.06, 12.0),
        dye_matrix=_dye(-0.13, -0.14, -0.09),
        base_tint=(0.985, 1.000, 0.985),
        misregistration_um=5.5,
        features=Feature.HALATION | Feature.STRONG_DIR_COUPLERS,
    ),
    FilmProfile(
        name="FUJI_ETERNA_VIVID_500T_8547",
        aliases=("8547", "eterna vivid", "eterna vivid 500t"),
        description=(
            "Fuji's answer to VISION3: higher saturation than Super F, tighter "
            "grain, and a cooler, more separated palette. The last generation "
            "of Fuji motion picture negative."
        ),
        era="2009-2013",  # audit: Eterna Vivid 500T shipped 2009
        exposure_index=500,
        balance_kelvin=3200,
        curves=RGBCurves(
            r=_neg(0.22, 0.615, toe_x=-1.48, shoulder_x=1.84),
            g=_neg(0.21, 0.638, toe_x=-1.58, shoulder_x=1.86),
            b=_neg(0.20, 0.660, toe_x=-1.64, shoulder_x=1.80),
        ),
        grain=GrainSpec(6.8, 9.6, 9.0, 11.8, clump_gain=0.34, fog_grain=0.20),
        mtf=MTFSpec(50.0, 58.0, 66.0, adjacency=0.10, adjacency_um=19.0),
        halation=HalationSpec(
            radii_um=(11.0, 56.0, 270.0),
            gain_r=0.16, gain_g=0.075, gain_b=0.04,
            threshold_stops=1.8,
        ),
        couplers=CouplerSpec(0.16, 50.0, 0.09, 11.0),
        dye_matrix=_dye(-0.20),
        base_tint=(0.988, 1.000, 0.980),
        misregistration_um=4.8,
        features=Feature.HALATION | Feature.STRONG_DIR_COUPLERS | Feature.TABULAR_GRAIN,
    ),

    # ------------------------------ Fuji -----------------------------------
    FilmProfile(
        name="FUJI_F125_8530",
        aliases=("f125", "8530", "fuji f125", "f-125", "fuji f125 8530"),
        description=(
            "[T2] Fujicolor F-125 tungsten negative, 35 mm. Fuji's fine-grain "
            "studio stock of the period: tighter grain than the Eastman 100T "
            "equivalents and the characteristic Fuji green-cyan lean in the "
            "shadows, which comes from where the three curves cross rather "
            "than from any tint."
        ),
        era="1980s-1990s",  # audit: F-125 is the pre-Super-F generation
        exposure_index=125,
        balance_kelvin=3200,
        curves=RGBCurves(
            r=_neg(0.20, 0.575),
            g=_neg(0.60, 0.585),
            b=_neg(0.98, 0.600),
        ),
        grain=GrainSpec(5.4, 8.0, 9.0, 11.0, clump_gain=0.42, fog_grain=0.16),
        mtf=MTFSpec(70.0, 78.0, 84.0, adjacency=0.13, adjacency_um=13.0),
        halation=HalationSpec(gain_r=0.035, gain_g=0.012, gain_b=0.003,
                             threshold_stops=2.0),
        couplers=CouplerSpec(0.34, 54.0, 0.16, 11.0),
        dye_matrix=_dye(-0.08),
        base_tint=(0.98, 1.0, 0.99),
        misregistration_um=4.5,
        features=Feature.STRONG_DIR_COUPLERS,
    ),
    FilmProfile(
        name="FUJI_F125_8630",
        aliases=("8630", "fuji f125 8630", "f125 16mm"),
        description=(
            "[T2] Fujicolor F-125, 16 mm base. Same emulsion as 8530 -- see "
            "the note on 7239 about why the numbers are identical and the "
            "gauge does the work."
        ),
        era="1980s-1990s",  # audit: F-125 is the pre-Super-F generation
        exposure_index=125,
        balance_kelvin=3200,
        curves=RGBCurves(
            r=_neg(0.20, 0.575),
            g=_neg(0.60, 0.585),
            b=_neg(0.98, 0.600),
        ),
        grain=GrainSpec(5.4, 8.0, 9.0, 11.0, clump_gain=0.42, fog_grain=0.16),
        mtf=MTFSpec(70.0, 78.0, 84.0, adjacency=0.13, adjacency_um=13.0),
        halation=HalationSpec(gain_r=0.035, gain_g=0.012, gain_b=0.003,
                             threshold_stops=2.0),
        couplers=CouplerSpec(0.34, 54.0, 0.16, 11.0),
        dye_matrix=_dye(-0.08),
        base_tint=(0.98, 1.0, 0.99),
        misregistration_um=4.5,
        default_format="16mm",
        features=Feature.STRONG_DIR_COUPLERS,
    ),
    FilmProfile(
        name="FUJI_NEOPAN_1600",
        aliases=("neopan 1600", "neopan1600", "fuji neopan 1600 lomo"),
        description=(
            "[T1] Fast Fuji B&W, EI 1600. Big open grain and a deliberately "
            "flat curve so that the shadows it was designed to reach do not "
            "crush. Available-light and concert film; often the grain is the "
            "point rather than a compromise."
        ),
        era="1990s-2010s",
        is_monochrome=True,
        exposure_index=1600,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.17, 0.610, -1.44, 0.28, 1.62, 0.40)),
        grain=GrainSpec(17.2, 22.0, 22.0, 22.0, clump_gain=1.50, fog_grain=0.30),
        mtf=MTFSpec(34.0, 34.0, 34.0, adjacency=0.04, adjacency_um=22.0),
        spectral_weights=(0.27, 0.55, 0.18),
        misregistration_um=0.0,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="FUJI_NEOPAN_ACROS_100",
        aliases=("acros", "neopan acros", "acros 100", "neopan 100 acros"),
        description=(
            "[T1] Fuji's fine-grain B&W, built on flat tabular crystals rather "
            "than cubic ones. That is why clump_gain is so low: tabular grains "
            "lie flat and pack evenly instead of clustering, so Acros looks "
            "smooth rather than velvety at the same grain size. Also famous "
            "for near-absent reciprocity failure -- not modelled here, since "
            "this pipeline has no exposure-time axis."
        ),
        era="2000-Present",
        is_monochrome=True,
        exposure_index=100,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.10, 0.600, -1.58, 0.32, 1.84, 0.44)),
        # rms 7.0: published diffuse RMS granularity, CONFIRMED 2026-07-31.
        # SOURCE PDF/PROFILES/FUJI/NeopanAcros100.pdf p4: "10.   DIFFUSE RMS
        # GRANULARITY VALUE ··············· 7" (Microfine, 48 um aperture, 12X,
        # density 1.0 above minimum). Resolving power 60 lines/mm at 1.6:1 and
        # 200 lines/mm at 1000:1, same page -> _RESOLVING_POWER. The onset_s=120
        # reciprocity entry for this stock is likewise datasheet-backed, not an
        # estimate. (The sheet is 6 pages; an earlier draft of this comment said
        # p7, which does not exist.)
        grain=GrainSpec(7.0, 7.0, 7.0, 7.0, clump_gain=0.22, fog_grain=0.13),
        mtf=MTFSpec(104.0, 104.0, 104.0, adjacency=0.14, adjacency_um=11.0),
        spectral_weights=(0.27, 0.55, 0.18),
        misregistration_um=0.0,
        default_format="ff35",
        features=Feature.TABULAR_GRAIN,
    ),
    FilmProfile(
        name="FUJI_PROVIA_400X",
        aliases=("provia", "provia 400x", "400x", "rxp"),
        description=(
            "[T1] Fuji's fast professional reversal. Remarkably fine-grained "
            "for EI 400 -- roughly the granularity a 100-speed slide film had "
            "a generation earlier -- with Fuji's cool, clean colour and "
            "well-behaved neutrals. Discontinued 2013."
        ),
        era="2000s-2013",
        kind=StockKind.REVERSAL,
        exposure_index=400,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_rev(0.16, 1.55, toe_x=-0.84, shoulder_x=1.00),
            g=_rev(0.17, 1.58, toe_x=-0.86, shoulder_x=0.98),
            b=_rev(0.18, 1.60, toe_x=-0.88, shoulder_x=0.96),
        ),
        # rms 11.0: published diffuse RMS granularity, CONFIRMED 2026-07-31.
        # SOURCE PDF/PROFILES/FUJI/Provia_400X_PIB_1007.pdf p6: "18. DIFFUSE
        # RMS GRANULARITY VALUE .... 11", aperture 48 um, sample density 1.0
        # above D-min. Resolving power 55 lp/mm at 1.6:1 and 135 lp/mm at
        # 1000:1 (same page) -> _RESOLVING_POWER. The sheet also documents
        # push latitude of -1/2 stop (EI 280) to +2 stops (EI 1600) and
        # densitometry via Fuji FAD-30S (Status A).
        grain=GrainSpec(11.0, 12.0, 13.0, 15.0, clump_gain=0.44, fog_grain=0.18),
        mtf=MTFSpec(60.0, 68.0, 74.0, adjacency=0.11, adjacency_um=15.0),
        halation=HalationSpec(gain_r=0.04, gain_g=0.014, gain_b=0.004,
                             threshold_stops=2.1),
        couplers=CouplerSpec(0.10, 48.0, 0.06, 10.0),
        dye_matrix=_dye(-0.16),
        misregistration_um=4.0,
        default_format="ff35",
        # Sigma Crystal (tabular) emulsion -- flag was missing (audit DM-20).
        features=Feature.TABULAR_GRAIN,
    ),
    FilmProfile(
        name="FUJI_SENSIA_100",
        aliases=("sensia", "sensia 100", "sensia ii"),
        description=(
            "[T1] Fuji's consumer slide film -- essentially a de-tuned Provia "
            "sold in a cheaper box. Slightly softer, slightly grainier, "
            "slightly warmer than the professional line, and less careful about "
            "neutral rendering. The holiday-slide look of the 1990s."
        ),
        era="1990s-2000s",
        kind=StockKind.REVERSAL,
        exposure_index=100,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_rev(0.16, 1.62, toe_x=-0.80, shoulder_x=0.96),
            g=_rev(0.17, 1.66, toe_x=-0.82, shoulder_x=0.94),
            b=_rev(0.19, 1.68, toe_x=-0.86, shoulder_x=0.90),
        ),
        # rms 10.0: published diffuse RMS granularity, CONFIRMED 2026-07-31.
        # SOURCE PDF/PROFILES/FUJI/sensia_100_datasheet.pdf p4: "13. DIFFUSE
        # RMS GRANULARITY VALUE ............ 10". Resolving power 55 lp/mm at
        # 1.6:1 and 135 lp/mm at 1000:1 (same page) -> _RESOLVING_POWER. The
        # sheet also gives the base: cellulose triacetate, 127 um, 135 only.
        grain=GrainSpec(10.0, 9.0, 9.5, 11.0, clump_gain=0.40, fog_grain=0.15),
        mtf=MTFSpec(64.0, 72.0, 78.0, adjacency=0.11, adjacency_um=14.0),
        halation=HalationSpec(gain_r=0.042, gain_g=0.015, gain_b=0.004,
                             threshold_stops=2.05),
        couplers=CouplerSpec(0.09, 48.0, 0.05, 10.0),
        dye_matrix=_dye(-0.13),
        base_tint=(1.0, 0.995, 0.985),
        misregistration_um=4.0,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="FUJI_VELVIA_50",
        aliases=("velvia", "velvia 50", "rvp50"),
        description=(
            "The most saturated colour film ever sold, and the reason a "
            "generation of landscape photographs look the way they do. "
            "Extremely fine grain, brutal contrast, about five usable stops. "
            "Use with format ff35 or medium645."
        ),
        era="1990-present",
        kind=StockKind.REVERSAL,
        exposure_index=50,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_rev(0.13, 2.00, toe_x=-0.66, toe_k=0.16, shoulder_x=0.80),
            g=_rev(0.13, 2.06, toe_x=-0.68, toe_k=0.16, shoulder_x=0.78),
            b=_rev(0.14, 2.14, toe_x=-0.72, toe_k=0.16, shoulder_x=0.76),
        ),
        # rms 9.0: published diffuse RMS granularity, CONFIRMED 2026-07-31.
        # SOURCE PDF/PROFILES/FUJI/velvia_50_datasheet.pdf p7: "17. DIFFUSE RMS
        # GRANULARITY VALUE ......9", measured through a 48 um aperture at
        # "Sample Density: 1.0 above minimum density" -- the same metric this
        # field defines. Resolving power 80 lp/mm at 1.6:1 and 160 lp/mm at
        # 1000:1 (same page) already recorded in _RESOLVING_POWER.
        # NOTE: AF3-0221E2Velvia50PIB.pdf is the same document, not a second
        # independent source.
        grain=GrainSpec(9.0, 3.6, 3.8, 4.4, clump_gain=0.18, fog_grain=0.12),
        mtf=MTFSpec(88.0, 98.0, 108.0, adjacency=0.15, adjacency_um=12.0),
        halation=HalationSpec(
            radii_um=(7.0, 36.0, 160.0),
            gain_r=0.05, gain_g=0.02, gain_b=0.006,
            threshold_stops=2.3,
        ),
        couplers=CouplerSpec(0.20, 44.0, 0.10, 9.0),
        # Aggressive negative off-diagonals: this is where Velvia's colour
        # comes from, not from a saturation slider.
        dye_matrix=_dye(-0.42),
        misregistration_um=3.0,
        default_format="ff35",
        features=Feature.STRONG_DIR_COUPLERS,
    ),

    # -------------------- Indian cinema, 1940-1960 --------------------------
    # India had a large film industry long before it had a raw-stock industry.
    # Through the 1940s and 1950s studios in Bombay, Madras and Calcutta shot
    # on imported negative -- Gevaert from Belgium, Eastman from Rochester,
    # Agfa and later ORWO. Domestic manufacture began only in 1960, when
    # Hindustan Photo Films opened at Ootacamund and began producing "Indu"
    # branded stock under Bell & Howell licence; that is just outside the
    # window you asked about, so the three entries here are the imports that
    # were actually threaded through Indian cameras in that period.
    FilmProfile(
        name="GEVACOLOR_1952",
        aliases=("gevacolor", "geva", "gevacolor 1952"),
        description=(
            "[T3] Gevaert's subtractive colour process, Belgium. The colour "
            "stock of early Indian colour cinema: Aan (1952), the first Indian "
            "feature in full colour, and Mother India (1957) were both shot on "
            "Gevacolor. Very slow, coarse, and with markedly impure dyes -- the "
            "positive dye_matrix here is doing real work, desaturating "
            "everything toward a muddy pastel. Gevacolor prints also faded "
            "notoriously fast, which is why surviving material looks pinker "
            "than the stock ever did new. Fade is not modelled; this is the "
            "emulsion as shot."
        ),
        era="1948-1960s",
        exposure_index=16,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.26, 0.590, toe_x=-1.42, shoulder_x=1.52),
            g=_neg(0.66, 0.605, toe_x=-1.44, shoulder_x=1.50),
            b=_neg(1.06, 0.575, toe_x=-1.48, shoulder_x=1.46),
        ),
        grain=GrainSpec(14.2, 17.0, 18.0, 21.0, clump_gain=1.50, fog_grain=0.30,
                        anisotropy=1.10),
        mtf=MTFSpec(30.0, 33.0, 36.0, adjacency=0.03, adjacency_um=24.0),
        halation=HalationSpec(
            radii_um=(16.0, 80.0, 380.0),
            gain_r=0.10, gain_g=0.04, gain_b=0.012,
            threshold_stops=1.3,
        ),
        couplers=CouplerSpec(0.10, 60.0, 0.04, 14.0),
        dye_matrix=_dye(0.20),
        base_tint=(1.0, 0.965, 0.905),
        misregistration_um=11.0,
        default_flare=0.030,
        features=Feature.UNEVEN_EMULSION | Feature.HALATION,
    ),
    FilmProfile(
        name="GEVAERT_PANCHRO_1950",
        aliases=("gevaert", "gevaert panchro", "panchro 1950", "geva bw"),
        description=(
            "[T3] Gevaert panchromatic B&W cine negative, around EI 32. The "
            "workhorse behind a great deal of Indian B&W production in the "
            "1940s and 1950s, and of European production too. Panchromatic, so "
            "unlike the true ortho stocks it does register red -- but weakly, "
            "which is why skin in period Indian films often reads darker and "
            "more sculptural than a modern panchromatic stock would render it."
        ),
        era="1940s-1960s",
        is_monochrome=True,
        exposure_index=32,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.16, 0.750, -1.34, 0.27, 1.58, 0.38)),
        grain=GrainSpec(11.0, 14.0, 14.0, 14.0, clump_gain=1.40, fog_grain=0.28,
                        anisotropy=1.08),
        mtf=MTFSpec(38.0, 38.0, 38.0, adjacency=0.04, adjacency_um=22.0),
        spectral_weights=(0.24, 0.53, 0.23),
        misregistration_um=0.0,
        default_flare=0.022,
        features=Feature.UNEVEN_EMULSION,
    ),
    FilmProfile(
        name="ILFORD_DELTA_3200",
        aliases=("delta 3200", "delta3200", "delta"),
        description=(
            "Extreme available-light B&W, true speed nearer 1000. Tabular "
            "crystals so the grain is enormous but oddly even rather than "
            "clumpy -- a useful demonstration that grain size and grain "
            "character are independent parameters."
        ),
        era="1998-present",
        is_monochrome=True,
        # exposure_index 3200 is the MARKETED meter setting, kept because that
        # is how the stock is used and named. Harman's own datasheet states the
        # measured speed is lower -- it gives ISO 1000/31 deg, measured in
        # ILFORD ID-11, while explaining that the 3200 in the product name is a
        # recommended meter setting rather than an ISO speed. SOURCE
        # PDF/PROFILES/ILFORD/Delta-3200_201811.pdf p1, same substance in the
        # 2002 edition Delta_3200-200209.pdf p1. (Paraphrase, not a quotation:
        # the sheet spreads this over two sentences.) The description above
        # already says "true speed nearer 1000"; this is the citation for it.
        exposure_index=3200,
        balance_kelvin=5500,
        # Curve stays an estimate. Harman prints a characteristic curve as an
        # IMAGE only and publishes no numeric gamma, D-min or D-max.
        curves=_mono(ToneCurve(0.18, 0.600, -1.56, 0.36, 2.10, 0.60)),
        # !! rms 16.0 IS NOT A PUBLISHED FIGURE. Harman/ILFORD publish no
        # diffuse RMS granularity, no resolving power and no MTF for any of
        # their films -- verified across all 18 ILFORD datasheets on file (20
        # ILFORD PDFs, of which one is a Kodak-equivalence table and one a
        # processing chart) plus both KENTMERE sheets. This value and the f50s
        # below are engineering estimates; the provenance tier was corrected
        # from 1 to 2 on 2026-07-31 to reflect that.
        grain=GrainSpec(16.0, 22.0, 22.0, 22.0, clump_gain=0.45, fog_grain=0.34),
        mtf=MTFSpec(30.0, 30.0, 30.0, adjacency=0.06),
        spectral_weights=(0.33, 0.46, 0.21),
        misregistration_um=0.0,
        default_format="ff35",
        features=Feature.TABULAR_GRAIN,
    ),

    # ------------------ Britain, France, Italy / Latin America -------------
    FilmProfile(
        name="ILFORD_HP3",
        aliases=("hp3", "hp-3", "ilford hp3", "hypersensitive panchromatic"),
        description=(
            "[T2] Ilford Hypersensitive Panchromatic 3 -- the British B&W "
            "standard for two decades, in press cameras, on newsreel cameras, "
            "and in the hands of most of Fleet Street. Two generations of "
            "emulsion technology before the HP5 in this database: coarser, "
            "softer, lower in contrast, and with a much longer gentle toe. "
            "Rated ISO 400 in later packaging though earlier ratings were lower "
            "under the old ASA system."
        ),
        era="1941-1965",
        is_monochrome=True,
        exposure_index=400,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.16, 0.720, -1.46, 0.34, 1.66, 0.44)),
        grain=GrainSpec(13.6, 18.0, 18.0, 18.0, clump_gain=1.50, fog_grain=0.26,
                        anisotropy=1.06),
        mtf=MTFSpec(38.0, 38.0, 38.0, adjacency=0.04, adjacency_um=22.0),
        spectral_weights=(0.26, 0.54, 0.20),
        misregistration_um=0.0,
        default_flare=0.014,
        default_format="ff35",
        features=Feature.UNEVEN_EMULSION,
    ),
    # -----------------------------------------------------------------------
    # Black and white negative
    # -----------------------------------------------------------------------
    FilmProfile(
        name="ILFORD_HP5_PLUS_400",
        aliases=("hp5", "hp5 plus", "hp5+", "ilford hp5"),
        description=(
            "Photojournalism B&W. Cubic crystals, velvety clustered grain, and "
            "a long straight line with a shoulder so gentle it almost never "
            "clips. Pushes to 1600 without complaint."
        ),
        era="1989-present",
        is_monochrome=True,
        # ISO 400/27 deg confirmed. SOURCE
        # PDF/PROFILES/ILFORD/HP5-Plus_201811.pdf p1: "ISO 400/27, BLACK AND
        # WHITE PROFESSIONAL FILM". The same sheet documents the usable range
        # as EI 400/27 to EI 3200/36 and notes that this EI range "is based on
        # a practical evaluation of film speed and is not based on foot speed,
        # as is the ISO standard" -- so 400 here is the true ISO speed, not a
        # marketing number (contrast with DELTA 3200 above).
        exposure_index=400,
        balance_kelvin=5500,
        # Curve stays an estimate. The datasheet's characteristic curve is an
        # IMAGE with no numeric gamma / D-min / D-max; its caption does pin the
        # processing it represents: "developed in ILFORD ILFOTEC HC (1+31)
        # stock for 6 1/2 minutes at 20C/68F with intermittent agitation"
        # (HP5-Plus_201811.pdf p5) -- the condition any future digitisation of
        # this curve must be labelled with.
        curves=_mono(ToneCurve(0.12, 0.640, -1.62, 0.34, 2.30, 0.60)),
        # !! rms 9.0 IS NOT A PUBLISHED FIGURE, and is implausibly fine for a
        # cubic-grain ISO 400 emulsion (Agfa's published figure for APX 400 is
        # 14.0). Harman/ILFORD publish no granularity, resolving power or MTF
        # for any film. Left unchanged because no documented replacement
        # exists, but the provenance tier was corrected from 1 to 2 on
        # 2026-07-31 and this value must not be cited as a datasheet number.
        grain=GrainSpec(9.0, 13.0, 13.0, 13.0, clump_gain=1.00, fog_grain=0.24),
        mtf=MTFSpec(52.0, 52.0, 52.0, adjacency=0.08),
        # Panchromatic, hotter in blue and red than video luma. This is why
        # B&W film darkens red lips and lightens a blue sky.
        spectral_weights=(0.34, 0.46, 0.20),
        misregistration_um=0.0,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="ILFORD_HPS",
        aliases=("hps", "ilford hps", "nouvelle vague", "hps 800"),
        description=(
            "[T2] Ilford HPS, EI 800 -- for a decade the fastest B&W film "
            "generally available anywhere. British emulsion, but its fame is "
            "French: Raoul Coutard shot Breathless in 1960 on HPS 35 mm still "
            "stock, bulk-spliced into hundred-foot rolls, because nothing else "
            "was fast enough to shoot Paris interiors and streets with "
            "available light. Push-processed beyond box speed on top of that. "
            "The result -- enormous open grain, a flat curve, grey rather than "
            "black blacks, heavy base fog -- became the visual signature of the "
            "Nouvelle Vague and, second-hand, of every low-budget film that "
            "wanted to look urgent. The high dmin and fog_grain here are "
            "deliberate: clean shadows would be wrong."
        ),
        era="1954-1960s",
        is_monochrome=True,
        exposure_index=800,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.21, 0.620, -1.38, 0.32, 1.54, 0.42)),
        grain=GrainSpec(19.0, 26.0, 26.0, 26.0, clump_gain=1.65, fog_grain=0.40,
                        anisotropy=1.08),
        mtf=MTFSpec(26.0, 26.0, 26.0, adjacency=0.02, adjacency_um=26.0),
        spectral_weights=(0.26, 0.54, 0.20),
        misregistration_um=0.0,
        default_flare=0.020,
        default_format="ff35",
        features=Feature.UNEVEN_EMULSION,
    ),
    # -----------------------------------------------------------------------
    # Colour reversal. No print stage: the film IS the positive.
    # -----------------------------------------------------------------------
    FilmProfile(
        name="KODACHROME_64",
        aliases=("kodachrome", "kodachrome 64", "k64", "kr64"),
        description=(
            "The one everybody means when they say film. Dyes were formed in "
            "the processing machine rather than in the emulsion, which is why "
            "it has no DIR couplers, exceptional sharpness, deep blacks and "
            "those unmistakable saturated reds. Punishing latitude."
        ),
        era="1974-2009",
        kind=StockKind.REVERSAL,
        exposure_index=64,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_rev(0.16, 1.72, toe_x=-0.74, shoulder_x=0.94),
            g=_rev(0.17, 1.78, toe_x=-0.78, shoulder_x=0.92),
            b=_rev(0.19, 1.86, toe_x=-0.84, shoulder_x=0.88),
        ),
        # rms 10.0: published diffuse RMS granularity (Kodak E-55). The old 2.2
        # was on a different metric; the header convention is diffuse RMS.
        # CONFIRMED 2026-07-31 against a second Kodak publication:
        # PDF/PROFILES/KODAK/e88-2009_06.pdf p4, under the heading "KODACHROME
        # 64 Film", prints "Diffuse rms Granularity: 10". (p5 of the same sheet
        # gives 16 for KODACHROME 200 -- do not cross-wire the two.) Process
        # K-14, densitometry Status A, EI 64; push processing explicitly not
        # recommended for the 64 emulsion.
        grain=GrainSpec(10.0, 3.8, 4.0, 4.6, clump_gain=0.30, fog_grain=0.12),
        mtf=MTFSpec(86.0, 96.0, 104.0, adjacency=0.14, adjacency_um=13.0),
        halation=HalationSpec(
            radii_um=(8.0, 40.0, 180.0),
            gain_r=0.06, gain_g=0.02, gain_b=0.006,
            threshold_stops=2.2,
        ),
        # Dyes were added in processing, so there is no inter-image effect.
        couplers=CouplerSpec(0.0, 0.0, 0.06, 9.0),
        # Very clean dye separation: strong negative off-diagonals.
        dye_matrix=_dye(-0.30),
        base_tint=(1.0, 1.0, 1.0),
        misregistration_um=3.0,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="KODAK_EKTACHROME_100D_5285",
        aliases=("5285", "ektachrome", "ektachrome 100d", "100d"),
        description=(
            "E-6 colour reversal for motion picture. Cooler and more neutral "
            "than Kodachrome, slightly softer, with a cleaner highlight "
            "shoulder. Currently the only reversal motion stock in production."
        ),
        era="2005-present",
        kind=StockKind.REVERSAL,
        exposure_index=100,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_rev(0.14, 1.62, toe_x=-0.80, shoulder_x=1.00),
            g=_rev(0.14, 1.66, toe_x=-0.82, shoulder_x=0.98),
            b=_rev(0.15, 1.72, toe_x=-0.86, shoulder_x=0.95),
        ),
        grain=GrainSpec(3.0, 5.0, 5.4, 6.2, clump_gain=0.24, fog_grain=0.14),
        mtf=MTFSpec(74.0, 82.0, 90.0, adjacency=0.12, adjacency_um=15.0),
        halation=HalationSpec(
            radii_um=(9.0, 46.0, 210.0),
            gain_r=0.08, gain_g=0.03, gain_b=0.01,
            threshold_stops=2.0,
        ),
        couplers=CouplerSpec(0.09, 46.0, 0.06, 10.0),
        dye_matrix=_dye(-0.22),
        misregistration_um=3.5,
        features=Feature.STRONG_DIR_COUPLERS,
    ),
    # -----------------------------------------------------------------------
    # Still colour negative
    # -----------------------------------------------------------------------
    FilmProfile(
        name="KODAK_PORTRA_400",
        aliases=("portra", "portra 400"),
        description=(
            "Modern still colour negative. Enormous latitude, gentle toe, and "
            "the flattering warm skin rendering it was designed around. Use "
            "with format ff35 or medium645."
        ),
        era="1998-present",
        exposure_index=400,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.21, 0.560, toe_x=-1.86, toe_k=0.34, shoulder_x=2.24),
            g=_neg(0.20, 0.578, toe_x=-1.80, toe_k=0.32, shoulder_x=2.18),
            b=_neg(0.20, 0.596, toe_x=-1.70, toe_k=0.30, shoulder_x=2.08),
        ),
        # !! rms 4.0 IS NOT A KODAK FIGURE. Kodak stopped publishing diffuse
        # rms granularity for colour negative and publishes Print Grain Index
        # instead: PGI 37 / 59 / 89 for 135 format at 4.4X / 8.8X / 17.8X
        # magnification -- SOURCE PDF/PROFILES/KODAK/e4050_portra_400-2016.pdf
        # p3. The same sheet states PGI "replaces rms granularity and has a
        # different scale which cannot be compared to rms granularity", and
        # Kodak publication E-58 (Kodak_Print-Grain-Index_E-58.pdf) publishes
        # NO conversion factor. So PGI 37 cannot be turned into an rms number
        # without inventing one; 4.0 is left as the engineering estimate it is.
        # Also documented on p3: densitometry is Status M.
        grain=GrainSpec(4.0, 6.6, 7.2, 8.6, clump_gain=0.22, fog_grain=0.17),
        # f50 stays an estimate: Portra's MTF is a plotted curve, and Kodak
        # publishes no resolving-power figure for this stock at all.
        mtf=MTFSpec(66.0, 74.0, 84.0, adjacency=0.12, adjacency_um=17.0),
        halation=HalationSpec(
            radii_um=(10.0, 50.0, 240.0),
            gain_r=0.12, gain_g=0.045, gain_b=0.016,
            threshold_stops=2.0,
        ),
        couplers=CouplerSpec(0.15, 50.0, 0.08, 10.0),
        dye_matrix=_dye(-0.11),
        base_tint=(1.000, 0.992, 0.972),
        misregistration_um=4.0,
        default_format="ff35",
        features=Feature.STRONG_DIR_COUPLERS | Feature.TABULAR_GRAIN,
    ),
    # -----------------------------------------------------------------------
    # Black and white reversal
    # -----------------------------------------------------------------------
    FilmProfile(
        name="KODAK_TRI_X_REVERSAL_200",
        # The catalogue number here is worth a note: Tri-X Reversal ships as
        # 7266 in 16 mm. A "5266" designation does not correspond to a shipped
        # Tri-X reversal product (confirmed by the datasheet audit), so the
        # 5266 alias has been removed and this profile is built as the 7266
        # emulsion.
        aliases=("7266", "tri-x reversal", "tri x reversal", "trix"),
        description=(
            "B&W reversal for 16 mm news and documentary work. Contrasty, "
            "grainy, short latitude, and it produces a projectable positive "
            "straight out of the tank -- which is why every 1960s newsreel "
            "looks like this."
        ),
        era="1954-present",
        kind=StockKind.REVERSAL,
        is_monochrome=True,
        exposure_index=200,
        balance_kelvin=5500,  # EI 200 is the daylight rating (160 tungsten)
        curves=_mono(ToneCurve(0.16, 1.50, -0.86, 0.22, 1.04, 0.34)),
        grain=GrainSpec(10.0, 14.0, 14.0, 14.0, clump_gain=1.20, fog_grain=0.26),
        mtf=MTFSpec(46.0, 46.0, 46.0, adjacency=0.07),
        spectral_weights=(0.32, 0.47, 0.21),
        misregistration_um=0.0,
        default_format="16mm",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="KODAK_VISION3_200T_5213",
        aliases=("5213", "vision3 200t", "200t"),
        description=(
            "Mid-speed tungsten stock. Interiors without the grain penalty of "
            "500T; slightly warmer curve crossover than the daylight stocks."
        ),
        era="2010-present",
        exposure_index=200,
        balance_kelvin=3200,
        curves=RGBCurves(
            r=_neg(0.215, 0.592, toe_x=-1.64, shoulder_x=1.96),
            g=_neg(0.205, 0.610, toe_x=-1.58, shoulder_x=1.90),
            b=_neg(0.195, 0.632, toe_x=-1.48, shoulder_x=1.79),
        ),
        grain=GrainSpec(4.6, 7.6, 8.2, 9.8, clump_gain=0.22, fog_grain=0.19),
        mtf=MTFSpec(58.0, 66.0, 76.0, adjacency=0.11, adjacency_um=20.0),
        halation=HalationSpec(
            radii_um=(12.0, 60.0, 290.0),
            gain_r=0.19, gain_g=0.07, gain_b=0.026,
            threshold_stops=1.8,
        ),
        couplers=CouplerSpec(0.133, 56.0, 0.073, 12.0),
        dye_matrix=_dye(-0.12),
        base_tint=(1.000, 0.987, 0.959),
        misregistration_um=4.8,
        features=Feature.STRONG_DIR_COUPLERS | Feature.TABULAR_GRAIN,
    ),
    FilmProfile(
        name="KODAK_VISION3_250D_5207",
        aliases=("5207", "vision3 250d", "250d"),
        description=(
            "Mid-speed daylight workhorse. The family compromise: noticeably "
            "finer than 500T but with real speed, and the cleanest highlight "
            "rolloff of the daylight stocks."
        ),
        era="2009-present",
        exposure_index=250,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.21, 0.590, toe_x=-1.66, shoulder_x=1.98),
            g=_neg(0.20, 0.608, toe_x=-1.60, shoulder_x=1.92),
            b=_neg(0.19, 0.628, toe_x=-1.50, shoulder_x=1.82),
        ),
        grain=GrainSpec(4.2, 7.0, 7.6, 9.0, clump_gain=0.20, fog_grain=0.18),
        mtf=MTFSpec(62.0, 70.0, 80.0, adjacency=0.11, adjacency_um=19.0),
        halation=HalationSpec(
            radii_um=(11.0, 55.0, 260.0),
            gain_r=0.15, gain_g=0.055, gain_b=0.020,
            threshold_stops=1.9,
        ),
        couplers=CouplerSpec(0.135, 54.0, 0.075, 11.0),
        dye_matrix=_dye(-0.125),
        base_tint=(1.000, 0.988, 0.962),
        misregistration_um=4.5,
        features=Feature.STRONG_DIR_COUPLERS | Feature.TABULAR_GRAIN,
    ),
    FilmProfile(
        name="KODAK_VISION3_500T_5219",
        aliases=("5219", "vision3 500t", "500t"),
        description=(
            "Modern Hollywood workhorse. Coarse but clean tabular structure, "
            "tungsten balanced, strong DIR couplers, and enough highlight "
            "latitude to make blown windows recoverable."
        ),
        era="2007-present",
        exposure_index=500,
        balance_kelvin=3200,
        # Slight gamma spread plus offset toes: the crossover that gives
        # VISION3 its warm highlight and neutral shadow.
        curves=RGBCurves(
            r=_neg(0.22, 0.600, toe_x=-1.62, shoulder_x=1.90),
            g=_neg(0.20, 0.620, toe_x=-1.55, shoulder_x=1.82),
            b=_neg(0.19, 0.645, toe_x=-1.44, shoulder_x=1.70),
        ),
        grain=GrainSpec(6.6, 10.5, 11.5, 13.5, clump_gain=0.28, fog_grain=0.20),
        mtf=MTFSpec(44.0, 52.0, 60.0, adjacency=0.10, adjacency_um=22.0),
        halation=HalationSpec(
            radii_um=(14.0, 70.0, 360.0),
            weights=(0.58, 0.30, 0.12),
            gain_r=0.30, gain_g=0.11, gain_b=0.04,
            threshold_stops=1.5,
        ),
        couplers=CouplerSpec(0.13, 60.0, 0.07, 13.0),
        dye_matrix=_dye(-0.115),
        base_tint=(1.000, 0.985, 0.955),
        misregistration_um=5.0,
        features=Feature.HALATION | Feature.STRONG_DIR_COUPLERS | Feature.TABULAR_GRAIN,
    ),
    # -----------------------------------------------------------------------
    # Kodak VISION3 family. Tabular grain, strong DIR couplers, wide latitude.
    # Grain and softness scale monotonically with speed across the family,
    # which is the single most useful sanity check on the numbers below.
    # -----------------------------------------------------------------------
    FilmProfile(
        name="KODAK_VISION3_50D_5203",
        aliases=("5203", "vision3 50d", "50d"),
        description=(
            "Tack-sharp slow daylight stock. Very fine tabular grain, highest "
            "resolving power of the family, strong anti-halation backing so "
            "almost no bloom."
        ),
        era="2011-present",
        exposure_index=50,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.20, 0.585, toe_x=-1.70, shoulder_x=2.05),
            g=_neg(0.19, 0.600, toe_x=-1.64, shoulder_x=2.00),
            b=_neg(0.18, 0.615, toe_x=-1.56, shoulder_x=1.92),
        ),
        grain=GrainSpec(2.6, 4.2, 4.6, 5.4, clump_gain=0.14, fog_grain=0.16),
        mtf=MTFSpec(78.0, 88.0, 98.0, adjacency=0.12, adjacency_um=16.0),
        halation=HalationSpec(
            radii_um=(9.0, 45.0, 200.0),
            gain_r=0.07, gain_g=0.025, gain_b=0.008,
            threshold_stops=2.1,
        ),
        couplers=CouplerSpec(0.14, 48.0, 0.08, 10.0),
        dye_matrix=_dye(-0.13),
        base_tint=(1.000, 0.990, 0.968),
        misregistration_um=4.0,
        features=Feature.STRONG_DIR_COUPLERS | Feature.TABULAR_GRAIN,
    ),
    FilmProfile(
        name="LUMIERE_LUMICHROME",
        aliases=("lumiere", "lumichrome", "lumiere lumichrome"),
        description=(
            "[T3] Lumiere B&W negative, Lyon, around EI 40. The most speculative "
            "profile in this database and flagged accordingly -- I have no "
            "datasheet for it, only the general behaviour of French B&W negative "
            "of the period. Lumiere manufactured independently until Ilford "
            "absorbed the company in 1961, and their emulsions had a reputation "
            "for a soft, long-scale rendering quite unlike the contrastier "
            "German and British stocks. Treat the numbers as a plausible French "
            "period look, not as a measurement of a specific product."
        ),
        era="1940s-1961",
        is_monochrome=True,
        exposure_index=40,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.15, 0.700, -1.40, 0.34, 1.70, 0.46)),
        grain=GrainSpec(12.0, 16.0, 16.0, 16.0, clump_gain=1.50, fog_grain=0.30,
                        anisotropy=1.10),
        mtf=MTFSpec(34.0, 34.0, 34.0, adjacency=0.03, adjacency_um=24.0),
        spectral_weights=(0.24, 0.52, 0.24),
        misregistration_um=0.0,
        default_flare=0.024,
        default_format="ff35",
        features=Feature.UNEVEN_EMULSION | Feature.ORTHO_RESPONSE,
    ),
    # -----------------------------------------------------------------------
    # Eastern bloc colour
    # -----------------------------------------------------------------------
    FilmProfile(
        name="ORWOCOLOR_NC21",
        aliases=("nc21", "nc 21", "orwocolor"),
        description=(
            "East German colour negative. Low saturation from weak dye "
            "separation, coarse clustered grain, muted palette and visible "
            "coating unevenness."
        ),
        era="1970s-1990s",
        exposure_index=100,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.30, 0.520, toe_x=-1.28, toe_k=0.38, shoulder_x=1.42),
            g=_neg(0.29, 0.530, toe_x=-1.34, toe_k=0.38, shoulder_x=1.46),
            b=_neg(0.31, 0.545, toe_x=-1.22, toe_k=0.36, shoulder_x=1.38),
        ),
        # rms 18 [T2, capped]: 109-frame v2 batch (1000 px web scans),
        # green mid sigma(D) 0.0802 -- ~4.3x the FN-64 anchor at matched
        # pixel pitch. Literal fit demands rms ~92: rejected outright (no
        # emulsion is that coarse; dense-negative regions scan DARK, where
        # web-JPEG shadow noise piles onto the dense bin). 18 keeps NC21
        # clearly grainier than its western contemporaries. Shape 0.50/1.8
        # from measured 0.46/1.0/2.42 with the dense bin discounted for
        # the same shadow-noise reason.
        # Batch gamma estimates (0.92-1.10) REJECTED: colour negative
        # process chemistry pins gamma near 0.5-0.65; per-channel batch
        # statistics on colour scans are polluted by scene colours and the
        # mask. The measured CROSSOVER table (toe_r -0.17, dense_r -0.99)
        # is the mask + layer divergence showing up exactly where the
        # per-channel dmin/toe/shoulder spreads below already encode it.
        grain=GrainSpec(18.0, 14.0, 15.0, 18.0, clump_gain=1.35, fog_grain=0.30,
                        anisotropy=1.06,
                        sigma_shape_toe=0.50, sigma_shape_dmax=1.80),
        mtf=MTFSpec(26.0, 30.0, 34.0, adjacency=0.02),
        # [T2] halation re-weighted: batch measures strength r/g/b
        # 0.24/0.21/0.28 at radius ~230 um. The BLUE dominance is rejected
        # as physics (through-base halation is red-dominant; blue-strong
        # halo on an aged web batch is scatter + scan glare), but the
        # measured overall strength supports raising green and blue toward
        # red rather than the old steep 0.22/0.12/0.08 fall-off.
        halation=HalationSpec(
            radii_um=(20.0, 100.0, 420.0),
            gain_r=0.22, gain_g=0.15, gain_b=0.12,
            threshold_stops=1.3,
        ),
        couplers=CouplerSpec(0.03, 80.0),
        # Heavily impure dyes: large *positive* off-diagonals desaturate for
        # real, instead of the old blend-toward-luma trick.
        dye_matrix=_dye(0.40),
        base_tint=(0.965, 1.000, 0.950),
        misregistration_um=11.0,
        features=Feature.HALATION | Feature.UNEVEN_EMULSION,
    ),
    FilmProfile(
        name="ORWOCOLOR_NC24",
        aliases=("nc24", "nc-24", "orwocolor nc24", "orwo nc 24"),
        description=(
            "[T3] ORWO colour negative, modelled as a later and faster member "
            "of the NC family than the NC 21 already in this database. HONEST "
            "CAVEAT: I could not confirm 'NC 24' as a shipped ORWO product "
            "designation. The documented NC series runs NC 3, NC 5, NC 16, "
            "NC 19, NC 21. If you have a real speed or datasheet for NC 24, "
            "give it to me and I will refit; until then this is a family "
            "interpolation, not a product. Built as EI 160 with the ORWO house "
            "signature intact: heavy orange mask residue, very impure dyes (the "
            "largest positive dye_matrix in the set after Gevacolor), weak "
            "couplers, soft MTF and coarse grain -- but a step cleaner and "
            "faster than NC 21, as later production generally was."
        ),
        era="1980s-1990s",
        exposure_index=160,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.29, 0.545, toe_x=-1.32, toe_k=0.36, shoulder_x=1.46),
            g=_neg(0.28, 0.555, toe_x=-1.38, toe_k=0.36, shoulder_x=1.50),
            b=_neg(0.30, 0.570, toe_x=-1.26, toe_k=0.34, shoulder_x=1.42),
        ),
        grain=GrainSpec(13.0, 15.0, 16.0, 19.0, clump_gain=1.30, fog_grain=0.29,
                        anisotropy=1.08),
        mtf=MTFSpec(28.0, 32.0, 36.0, adjacency=0.02),
        couplers=CouplerSpec(0.05, 78.0),
        dye_matrix=_dye(0.34),
        base_tint=(0.970, 1.000, 0.955),
        misregistration_um=10.0,
        features=Feature.UNEVEN_EMULSION,
    ),
    FilmProfile(
        name="POLAROID_664",
        aliases=("664", "polaroid 664", "type 664"),
        description=(
            "[T2] Peel-apart pack B&W, ISO 100. Sharper and more contrasty "
            "than SX-70 because the negative and receiving sheet are pressed "
            "together and then separated, rather than viewed through a stack "
            "of layers. Still a print, so still a limited Dmax."
        ),
        era="1980s-2009",
        kind=StockKind.REVERSAL,
        is_monochrome=True,
        exposure_index=100,
        balance_kelvin=5500,
        # ISO 100/DIN 21 confirmed against the datasheet
        # (PDF/PROFILES/POLAROID/664fds.pdf p2). The curve stays an estimate:
        # unlike the 667 sheet, the 664 sheet prints the DEFINITIONS of D-Max,
        # D-Min and Slope but no values for this film, so there is nothing to
        # transcribe.
        curves=_mono(ToneCurve(0.14, 1.30, -0.65, 0.24, 0.74, 0.32)),
        grain=GrainSpec(8.6, 13.0, 13.0, 13.0, clump_gain=0.85, fog_grain=0.20),
        # Published resolution 20-25 lp/mm at 1000:1 -> _RESOLVING_POWER.
        mtf=MTFSpec(40.0, 40.0, 40.0, adjacency=0.05, adjacency_um=20.0),
        spectral_weights=(0.28, 0.56, 0.16),
        misregistration_um=0.0,
        default_format="polaroid_pack",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="POLAROID_667",
        aliases=("667", "polaroid 667", "type 667"),
        description=(
            "[T2] The very fast pack B&W, ISO 3000. Coarse, flat and grey, "
            "with a Dmax among the lowest here (of the three Polaroids only "
            "SX-70 computes lower). Used where "
            "there was no light and no second chance -- oscilloscope traces, "
            "forensic work, backstage. Loved now for exactly the qualities "
            "that made it a compromise then."
        ),
        era="1980s-2009",
        kind=StockKind.REVERSAL,
        is_monochrome=True,
        exposure_index=3000,
        balance_kelvin=5500,
        # Curve fitted to the three published sensitometric numbers -- the only
        # numeric characteristic-curve data available for any stock in this
        # database. (Polaroid publishes the same D-Max/D-Min/Slope triple for
        # about a dozen other pack and sheet types, e.g. 53/553/803 and 55/85,
        # but none of those has a profile here.) SOURCE:
        # PDF/PROFILES/POLAROID/667fds.pdf p2: "At 71o F/21o C:  D-Max = 1.75
        # D-Min = .10   Slope = 1.55". Polaroid defines Slope by the 1/4-3/4
        # increment method, i.e. the average gradient of the straight-line
        # section, which is what ToneCurve.gamma is. So dmin = 0.10 and
        # gamma = 1.55 are transcriptions, and the toe/shoulder separation is
        # then FORCED by dmax: shoulder_x - toe_x = (1.75 - 0.10) / 1.55
        # = 1.0645 decades. The pair is placed symmetrically about the previous
        # mid-scale anchor (+0.035) so that matching the published densities
        # does not silently shift the stock's exposure placement -- toe_k and
        # shoulder_k are untouched. Previous values (0.16 / 1.15 / dmax 1.85)
        # were estimates; superseded 2026-07-31.
        # Two honest caveats. (1) Polaroid's Slope is the 1/4-3/4 increment
        # gradient, not the analytic slope of this model's straight-line
        # section, so forcing shoulder_x - toe_x = (Dmax - Dmin) / Slope imports
        # that approximation. (2) ToneCurve.dmax is an ASYMPTOTE, so rendered
        # peak density approaches 1.749 without reaching it. Side effect worth
        # knowing: latitude drops from 4.88 to 3.54 stops, which is what the
        # published numbers imply for an ISO 3000 instant print material.
        curves=_mono(ToneCurve(0.10, 1.55, -0.497, 0.24, 0.567, 0.32)),
        # ISO 3000/DIN 36 confirmed on 667fds.pdf p2. Polaroid publishes no
        # granularity metric for any film, so rms 19.4 remains an estimate.
        grain=GrainSpec(19.4, 26.0, 26.0, 26.0, clump_gain=1.35, fog_grain=0.34),
        # f50 stays an estimate: Polaroid prints an MTF graph with no tabulated
        # values. Published resolution 14-20 lp/mm at 1000:1 -> _RESOLVING_POWER.
        mtf=MTFSpec(26.0, 26.0, 26.0, adjacency=0.03, adjacency_um=24.0),
        spectral_weights=(0.28, 0.56, 0.16),
        misregistration_um=0.0,
        default_format="polaroid_pack",
        features=Feature.NONE,
    ),

    # ----------------------------- Polaroid --------------------------------
    # The signature of instant film is NOT grain -- it is a low Dmax. SX-70
    # tops out near 1.85 where Kodachrome reaches 3.2, so the blacks are
    # genuinely open and slightly milky no matter how you expose. Combined
    # with a soft MTF (the image-receiving layer is thick and diffuse) and dye
    # clouds rather than silver clumps, that is the whole look.
    FilmProfile(
        name="POLAROID_SX70",
        aliases=("sx70", "sx-70", "polaroid sx 70", "sx 70"),
        description=(
            "[T2] Integral instant colour, EI 150. Low Dmax gives the "
            "characteristic open, chalky blacks; the diffuse receiving layer "
            "gives the softness; dye diffusion gives a gentle mottle instead "
            "of grain. Warm, low contrast, and dependent on temperature during "
            "development in ways nothing here models."
        ),
        era="1972-2008",
        kind=StockKind.REVERSAL,
        # ISO 150/DIN 23 confirmed against Polaroid's Time-Zero Supercolor
        # SX-70 product sheet (PDF/PROFILES/POLAROID/timezfds.pdf p1). That
        # sheet is a product page only: it carries NO technical-data section,
        # so speed and the ~5 min development time are the only published
        # figures available. Every other number below remains an estimate.
        exposure_index=150,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=ToneCurve(0.21, 1.44, -0.52, 0.26, 0.60, 0.34),
            g=ToneCurve(0.22, 1.46, -0.54, 0.26, 0.59, 0.34),
            b=ToneCurve(0.25, 1.42, -0.58, 0.26, 0.56, 0.34),
        ),
        grain=GrainSpec(6.8, 17.0, 18.0, 20.0, clump_gain=0.95, fog_grain=0.26,
                        anisotropy=1.06),
        mtf=MTFSpec(18.0, 20.0, 22.0, adjacency=0.0),
        couplers=CouplerSpec(0.0, 55.0, 0.0, 12.0),
        dye_matrix=_dye(0.12),
        base_tint=(1.0, 0.975, 0.930),
        misregistration_um=9.0,
        default_format="polaroid_sx70",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="SOVIET_PANCHROM_1939",
        aliases=("panchrom", "sovkino", "shostka", "soviet 1939", "kinoplenka"),
        description=(
            "Soviet panchromatic negative of the late 1930s, as made at the "
            "Shostka film factory. Coarse, foggy, soft, weak in red, and above "
            "all inconsistent: batch-to-batch and within-roll sensitivity "
            "variation was bad enough that major productions of the period "
            "often preferred imported Agfa or Kodak stock when they could get "
            "it. That unevenness is modelled here and is most of the character."
        ),
        era="1930s-1940s",
        is_monochrome=True,
        exposure_index=45,
        balance_kelvin=3200,
        # Steeper and shorter than Super-XX: more contrast, less latitude,
        # highlights gone sooner.
        curves=_mono(ToneCurve(0.36, 0.780, -1.14, 0.26, 1.42, 0.36)),
        grain=GrainSpec(14.5, 18.5, 18.5, 18.5, clump_gain=1.60, fog_grain=0.40,
                        anisotropy=1.12),
        mtf=MTFSpec(24.0, 24.0, 24.0, adjacency=0.02),
        halation=HalationSpec(
            radii_um=(26.0, 130.0, 600.0),
            weights=(0.44, 0.34, 0.22),
            gain_r=0.34, gain_g=0.34, gain_b=0.34,
            threshold_stops=0.95,
        ),
        couplers=CouplerSpec(),
        spectral_weights=(0.20, 0.47, 0.33),
        misregistration_um=0.0,
        default_flare=0.12,
        features=(
            Feature.HALATION | Feature.UNEVEN_EMULSION
            | Feature.ORTHO_RESPONSE | Feature.NITRATE_BASE
        ),
    ),
    FilmProfile(
        name="SVEMA_FN_64",
        aliases=("svema", "fn64", "fn-64", "svema fn64"),
        description=(
            "Soviet B&W. High contrast, coarse and irregular crystals, weak "
            "red sensitivity, and visible large-scale coating unevenness from "
            "loose quality control. That unevenness reads as 'old film' far "
            "more strongly than extra grain does."
        ),
        era="1980s-1990s",
        is_monochrome=True,
        exposure_index=64,
        balance_kelvin=5500,
        # dmin 0.16 [T3]: REVERTED to the estimate. The 0.174 briefly adopted
        # here came from misreading the v1 analyzer's output (its "dmin" was a
        # raw pixel value of the DENSE end, not a base density). The owner's
        # 355-frame v2 batch (native 4416x2944, sRGB-decoded, density space)
        # reports base at 0.008-0.013 D RELATIVE to scanner white -- the DSLR
        # rig auto-exposes the base to white, so ABSOLUTE base+fog is
        # unknowable without an --empty-gate calibration frame. Estimate
        # stands until that frame exists.
        # gamma 0.83 [T2]: 509-frame batch (supersedes the 355-frame run on
        # the same rig, which gave 0.787; per-channel now 0.806/0.834/0.850,
        # green adopted). Same stated assumption: 1.90 logE interdecile
        # scene span. Still in the plausible FN-64 development range.
        curves=_mono(ToneCurve(0.16, 0.830, -1.18, 0.24, 1.52, 0.34)),
        # Grain, same batch, 56800 flat blocks at native resolution [T2]:
        #   sigma(D) toe/mid/dense = 0.021/0.028/0.037 -> shape 0.70/1.0/1.35
        #   after removing a ~0.01 scanner noise floor in quadrature
        #   (previous 0.4/1.0/1.2 was T3).
        #   grain correlation length 3.48 px at 122.7 px/mm = 28 um raw,
        #   ~23 um after deconvolving a ~2 px scanner PSF -> clump 23 um
        #   (previous 15 um was T3; the RMS calibration integral keeps the
        #   rendered amplitude pinned, so only the grain SCALE shifts).
        #   rms 11.5 kept: still [T1], and the new batch's native-res mid
        #   sigma 0.028 agrees with the 0.030 that fit produced.
        #   anisotropy 0.62 REJECTED: grain is physically isotropic; a value
        #   that far from 1 on a Bayer-demosaiced DSLR scan is the sensor
        #   pattern, not the film. 1.10 (transport smear estimate) stands.
        #   sigma shape 0.65/1.0/1.65 [T2]: 509-frame batch green channel
        #   0.0191/0.0292/0.0482; mid sigma re-confirms the rms 11.5 fit.
        grain=GrainSpec(11.5, 23.0, 23.0, 23.0, clump_gain=1.55, fog_grain=0.32,
                        anisotropy=1.10,
                        sigma_shape_toe=0.65, sigma_shape_dmax=1.65),
        mtf=MTFSpec(34.0, 34.0, 34.0, adjacency=0.03),
        spectral_weights=(0.26, 0.50, 0.24),
        misregistration_um=0.0,
        # [T2] halation: ENABLED from the 355-frame batch, 58 usable
        # highlight frames. Measured excess density next to blown highlights:
        # 0.24 D (analyzer documents a 15-20% low bias -> ~0.28 true), 1/e
        # radius 69 um. The analyzer's masked-background method removes
        # long-range lens veiling, so the short-range component is film-level
        # scatter. Gain from inverting D = gamma*log10(1 + 0.5*A*gain/E_mid)
        # at an assumed 5 stops highlight overshoot -> 0.09; the assumption
        # spans 0.04 (7 stops) to 0.19 (4 stops), hence T2. Radii: middle
        # lobe moved to the measured 69 um, weights biased onto it.
        halation=HalationSpec(radii_um=(12.0, 69.0, 320.0),
                              weights=(0.30, 0.55, 0.15),
                              gain_r=0.09, gain_g=0.09, gain_b=0.09),
        # [T2] base_tint: 509-frame batch, (0.991, 1.000, 0.991) -- green
        # fractionally strong, R and B symmetric. CONTAMINATED tier as
        # always: scanner illuminant + WB folded in.
        base_tint=(0.991, 1.000, 0.991),
        # silver_tone +0.40 [T2] -- SIGN REVERSAL; the evidence trail
        # matters: one frame said cold (-0.25); the 355-frame batch said
        # near neutral (-0.10 kept); the 509-frame batch measures
        # tone_slope_r -0.0205 / _b +0.0079: dense areas transmit MORE red
        # and LESS blue, so the image's bright regions print WARM. The
        # crow-wing cold shadows survive as the relative complement.
        # Magnitude by inverting stage 14c at w~0.9 against the measured
        # dense-end r/g transmission ratio 1.10 -> tone ~0.40.
        silver_tone=0.40,
        features=Feature.UNEVEN_EMULSION | Feature.ORTHO_RESPONSE,
    ),
    FilmProfile(
        name="SVEMA_FN_64_16MM",
        aliases=("fn64 16mm", "fn64-16", "svema 16mm"),
        description=(
            "SVEMA FN-64 in 16 mm. Same emulsion, same curve, same base -- "
            "the coating line did not know what width the roll would be slit "
            "to. Everything that differs on screen (coarser apparent grain, "
            "lower resolved detail) comes from the ~2.1x higher magnification "
            "of the smaller frame, which the pipeline derives from the format "
            "via px_per_mm. Do not retune grain or density here."
        ),
        era="1980s-1990s",
        is_monochrome=True,
        exposure_index=64,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.16, 0.830, -1.18, 0.24, 1.52, 0.34)),
        grain=GrainSpec(11.5, 23.0, 23.0, 23.0, clump_gain=1.55, fog_grain=0.32,
                        anisotropy=1.10,
                        sigma_shape_toe=0.65, sigma_shape_dmax=1.65),
        halation=HalationSpec(radii_um=(12.0, 69.0, 320.0),
                              weights=(0.30, 0.55, 0.15),
                              gain_r=0.09, gain_g=0.09, gain_b=0.09),
        mtf=MTFSpec(34.0, 34.0, 34.0, adjacency=0.03),
        spectral_weights=(0.26, 0.50, 0.24),
        misregistration_um=0.0,
        base_tint=(0.991, 1.000, 0.991),
        silver_tone=0.40,
        default_format="16mm",
        features=Feature.UNEVEN_EMULSION | Feature.ORTHO_RESPONSE,
    ),
    FilmProfile(
        name="SVEMA_FN_64_8MM",
        aliases=("fn64 8mm", "fn64-8", "svema 8mm"),
        description=(
            "SVEMA FN-64 slit for 8 mm home movie cameras. Emulsion identical "
            "to the 35 mm entry; the ~5.4x magnification of the tiny frame "
            "does all the damage, and the pipeline derives it from the "
            "format. Temporal behaviour is the exception: weave and flicker "
            "belong to the amateur camera and projector, not the emulsion, "
            "so this entry carries home-gear transport numbers."
        ),
        era="1980s-1990s",
        is_monochrome=True,
        exposure_index=64,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.16, 0.830, -1.18, 0.24, 1.52, 0.34)),
        grain=GrainSpec(11.5, 23.0, 23.0, 23.0, clump_gain=1.55, fog_grain=0.32,
                        anisotropy=1.10,
                        sigma_shape_toe=0.65, sigma_shape_dmax=1.65),
        halation=HalationSpec(radii_um=(12.0, 69.0, 320.0),
                              weights=(0.30, 0.55, 0.15),
                              gain_r=0.09, gain_g=0.09, gain_b=0.09),
        mtf=MTFSpec(34.0, 34.0, 34.0, adjacency=0.03),
        spectral_weights=(0.26, 0.50, 0.24),
        misregistration_um=0.0,
        base_tint=(0.991, 1.000, 0.991),
        silver_tone=0.40,
        default_format="8mm",
        features=Feature.UNEVEN_EMULSION | Feature.ORTHO_RESPONSE,
    ),

    # ------------------------------- USSR ----------------------------------
    FilmProfile(
        name="SVEMA_FOTO_250",
        aliases=("foto250", "foto-250", "svema foto 250", "svema fn250",
                 "fn250", "svema 250"),
        description=(
            "[T3] Svema's fast B&W, the high-speed sibling of FN-64. Note on "
            "naming: Svema's FN line was cine negative and the Foto- line was "
            "still film; both names are in circulation for the fast stock and "
            "both resolve to this entry. Compared with FN-64: about two stops "
            "faster, grain roughly 40 % coarser, resolution down by a quarter, "
            "and the coating unevenness worse rather than better -- fast Soviet "
            "emulsions were where quality control gave up first. (rms/clump "
            "ordering flagged in audit -- unverified)"
        ),
        era="1970s-1990s",
        is_monochrome=True,
        exposure_index=250,
        balance_kelvin=5500,
        # gamma 0.85 [T2]: 26-frame v2 batch estimates 0.844 -- essentially
        # the same slope as FN-64's 0.834, NOT higher. This contradicts the
        # remembered "more contrast than FN-64" (which had set 0.95); the
        # memory may reflect development practice rather than the emulsion,
        # and expired stock loses contrast besides. 26 frames is thin
        # evidence, so the adoption rounds toward the old value, not past it.
        curves=_mono(ToneCurve(0.19, 0.850, -1.24, 0.26, 1.46, 0.36)),
        # rms_granularity is [T1] -- FITTED TO MEASUREMENT, not estimated.
        # Flat-region sigma over 3 supplied scans at matched mid density gave
        # FN250 0.0502 against SVEMA_FN_64's 0.0299, a ratio of 1.68x.
        # Tuned through the FULL PIPELINE, not by scaling RMS directly: a
        # naive 11.5*1.68=19.4 only gives 1.42x, because the coarser clump
        # (21.5 um vs 15.0) spreads spectral energy differently and the
        # calibration integral compensates. Swept against rendered output at
        # matched mid density, 25.0 lands on 1.70x. Was 16.2 (1.42x).
        # clump_um and clump_gain stay [T3]: grain SIZE is not measurable from
        # those files. At 1216 px across 36 mm one pixel spans 29.6 um while
        # the clumps are ~0.7 px, so the measured correlation length is the
        # scanner/JPEG MTF, not the emulsion.
        # rms 33 [T2, capped]: the 26-frame v2 batch measures mid-bin
        # sigma(D) 0.1237 at 33.8 px/mm. Fitting the pipeline to reproduce
        # that number outright demands rms ~70 -- beyond any emulsion ever
        # coated (Delta 3200 class sits ~25-30), and the same scans hand-
        # measured earlier on their flattest regions gave 0.0502 (rms 25).
        # Verdict: the mid bin of a 26-frame busy web batch carries heavy
        # scene-texture leakage; adopted value moves decisively toward the
        # measurement but stops at the physical ceiling for a bad fast
        # Soviet emulsion. sigma shape 0.67/1.0/1.69 adopted as measured
        # (ratios are leakage-resistant: it cancels between bins).
        # Raise rms further only after a >=100-frame native-res batch.
        grain=GrainSpec(33.0, 21.5, 21.5, 21.5, clump_gain=1.70, fog_grain=0.38,
                        anisotropy=1.14,
                        sigma_shape_toe=0.67, sigma_shape_dmax=1.69),
        mtf=MTFSpec(26.0, 26.0, 26.0, adjacency=0.02, adjacency_um=26.0),
        spectral_weights=(0.25, 0.49, 0.26),
        misregistration_um=0.0,
        default_format="ff35",
        # Left neutral: all three supplied FN250 scans were stored as pure greyscale (R-G exactly 0.0), so they carry no tone information at all.
        silver_tone=0.0,
        # [T2] halation ENABLED: 26-frame batch measures 0.35 D excess next
        # to blown highlights (~0.41 bias-corrected), 1/e radius 5.9 px at
        # 33.8 px/mm = 175 um. Gain by the same inversion used for FN-64
        # (5-stop overshoot assumption, gamma 0.85) -> 0.14. Fast Soviet
        # emulsion on a thick pad: entirely plausible.
        halation=HalationSpec(radii_um=(15.0, 175.0, 500.0),
                              weights=(0.20, 0.60, 0.20),
                              gain_r=0.14, gain_g=0.14, gain_b=0.14),
        features=Feature.UNEVEN_EMULSION | Feature.ORTHO_RESPONSE,
    ),
    FilmProfile(
        name="TASMA_FN_64",
        aliases=("tasma", "fn64t", "tasma fn64", "tasma fn 64",
                 "fn65", "fn-65", "tasma fn65", "tasma fn 65"),
        description=(
            "[T3] Tasma (Kazan) B&W cine negative, EI 64 -- Svema's rival "
            "supplier to Soviet studios. Broadly the same class as FN-64 with "
            "marginally better coating consistency and slightly finer grain, "
            "which is why Tasma tended to be preferred for features when it "
            "could be got. Still unmistakably Eastern Bloc: contrasty, weak in "
            "the red, and never quite even across the frame. Designation note: "
            "both 'FN-64' and 'FN-65' circulate for this stock -- 65 matches the "
            "GOST speed step, 64 the ISO equivalent, and Lomography's community "
            "database indexes it as Tasma FN64. Renamed to FN_64 to match the "
            "commonest usage; the fn65 aliases still resolve here."
        ),
        era="1960s-1990s",
        is_monochrome=True,
        exposure_index=64,
        balance_kelvin=5500,
        # gamma 1.03 [T2]: 132-frame v2 batch, per-channel 1.022/1.029/1.031.
        # Tasma measures HIGHER contrast than Svema (0.83) -- consistent
        # with its reputation. Same 1.90 logE span assumption as always.
        curves=_mono(ToneCurve(0.15, 1.030, -1.22, 0.25, 1.50, 0.34)),
        # rms 20 [T2, capped]: 132-frame batch (1042 px web scans) measures
        # mid sigma(D) 0.0768 -- ~4x the FN-64 anchor at matched pixel
        # pitch; a literal pipeline fit demands rms ~55, past any physical
        # ceiling for a 64-speed emulsion. Same verdict as FOTO-250: web
        # batch mid-bin scene leakage. Adopted 20 = decisively coarser than
        # Svema FN-64 (11.5), stops at plausibility. corr length 1.73 px is
        # below the 2 px scan floor -> clump 16 um stands, unmeasured.
        # Shape: measured toe 0.36; the dense bin (0.90) came out BELOW
        # mid, which negatives do not do -- leakage again; dmax capped at
        # 1.0 (direction of the data, not its face value).
        grain=GrainSpec(20.0, 16.0, 16.0, 16.0, clump_gain=1.45, fog_grain=0.30,
                        anisotropy=1.08,
                        sigma_shape_toe=0.36, sigma_shape_dmax=1.00),
        mtf=MTFSpec(32.0, 32.0, 32.0, adjacency=0.03, adjacency_um=24.0),
        spectral_weights=(0.26, 0.50, 0.24),
        misregistration_um=0.0,
        # Measured: two of three user-supplied scans show a warm cast in their bright regions, R-G of +8.6 and +15.6 out of 255. Calibrated to the larger of the two.
        # silver_tone +0.30 [T2] (was 1.0, a memory-based guess): 132-frame
        # batch measures tone_slope_r -0.0156 / _b +0.0011 -- warm-dense
        # drift, SMALLER in magnitude than Svema's -0.0205. The remembered
        # brown-black Tasma look is real but gentler than the guess; the
        # measured dense-end r/g ratio 1.07 inverts to ~0.30 by the same
        # stage-14c mapping used for Svema.
        silver_tone=0.30,
        # [T2] halation ENABLED: 132-frame batch, 0.24 D excess (~0.29
        # bias-corrected), radius 8.3 px at 28.9 px/mm = 287 um -- but one
        # ring is 69 um at this scan pitch, so the radius is coarse; the
        # middle lobe lands at 120 um as a compromise between the crude
        # measurement and FN-64's well-resolved 69 um. Gain inversion at
        # gamma 1.03 -> 0.06.
        halation=HalationSpec(radii_um=(12.0, 120.0, 400.0),
                              weights=(0.25, 0.55, 0.20),
                              gain_r=0.06, gain_g=0.06, gain_b=0.06),
        features=Feature.UNEVEN_EMULSION | Feature.ORTHO_RESPONSE,
    ),
    # -----------------------------------------------------------------------
    # Technicolor three-strip. Not a stock but a whole imaging system.
    # -----------------------------------------------------------------------
    FilmProfile(
        name="TECHNICOLOR_THREE_STRIP",
        aliases=("technicolor", "three strip", "three-strip", "process 4", "ib tech"),
        description=(
            "Beam-splitter camera exposing three separate B&W negatives through "
            "broad overlapping filters, printed by imbibition dye transfer. "
            "Three things make the look, and none of them is grain: the wide "
            "taking filters (hence the famous reds), the very saturated and "
            "clean transfer dyes, and visible registration error between the "
            "three records. Very slow -- it needed enormous arc lighting."
        ),
        era="1932-1955",
        exposure_index=5,
        balance_kelvin=3400,  # carbon arc
        # Three identical B&W emulsions, so the curves match; the colour comes
        # entirely from the taking and transfer matrices.
        curves=_mono(ToneCurve(0.18, 0.660, -1.34, 0.30, 1.72, 0.44)),
        grain=GrainSpec(7.5, 11.0, 11.0, 11.0, clump_gain=1.10, fog_grain=0.24),
        mtf=MTFSpec(40.0, 40.0, 40.0, adjacency=0.06, adjacency_um=26.0),
        halation=HalationSpec(
            radii_um=(16.0, 80.0, 380.0),
            gain_r=0.24, gain_g=0.24, gain_b=0.24,  # separate B&W records
            threshold_stops=1.3,
        ),
        couplers=CouplerSpec(0.0, 0.0, 0.05, 14.0),
        # Broad, heavily overlapping taking filters. Positive off-diagonals in
        # *exposure* are the physical origin of the Technicolor red.
        taking_matrix=(
            (1.000, 0.240, 0.060),
            (0.150, 1.000, 0.150),
            (0.050, 0.190, 1.000),
        ),
        # Imbibition dyes were unusually pure: strong negative off-diagonals
        # claw the saturation back and then some.
        dye_matrix=_dye(-0.36),
        base_tint=(1.0, 1.0, 1.0),
        # The signature flaw: three physically separate strips never registered
        # perfectly, giving coloured fringing on high-contrast edges.
        misregistration_um=26.0,
        default_print="TECHNICOLOR_IB",
        default_flare=0.075,
        default_format="techni35",
        features=Feature.HALATION | Feature.THREE_STRIP | Feature.UNEVEN_EMULSION,
    ),
    # =================== 2026-07-31 additions: measured + datasheet ========
    FilmProfile(
        name="ORWO_UT18",
        aliases=("ut18", "orwo ut18", "ut-18"),
        description=(
            "[T2] ORWO's colour reversal for daylight, DIN 18. The GDR "
            "holiday-slide stock. Profile built from the owner's 78-frame "
            "scan batch of real (aged) slides: yellow-shifted extremes and "
            "strong halation are what surviving UT18 actually looks like; "
            "fresh-stock behaviour is not recoverable from aged film."
        ),
        era="1960s-1980s",
        kind=StockKind.REVERSAL,
        exposure_index=50,
        balance_kelvin=4500,
        # Aged-dye signature from the batch's crossover table (medians per
        # density bin -- robust to scene colour): blue-dense at BOTH ends
        # relative to mid (toe_b +0.45, dense_b +0.29) = yellowed highlights
        # and warm-brown shadows. Encoded as a blue dmin lift plus a
        # slightly steeper blue curve; the linear tone_slope figures were
        # NOT used (regression on colour material is scene-dominated;
        # the binned medians are the trustworthy form).
        curves=RGBCurves(
            r=_rev(0.14, 1.52, toe_x=-0.80, shoulder_x=0.96),
            g=_rev(0.13, 1.55, toe_x=-0.82, shoulder_x=0.94),
            b=_rev(0.19, 1.63, toe_x=-0.80, shoulder_x=0.90),
        ),
        # Grain [T3]: slide sigma in the batch is scan-limited (3.9 px corr
        # at ~29 px/mm is the scanner, not the film). Class estimate for a
        # 50-speed 1960s reversal from a non-Kodak line.
        grain=GrainSpec(13.0, 14.0, 15.0, 17.0, clump_gain=1.10, fog_grain=0.20),
        mtf=MTFSpec(40.0, 45.0, 50.0, adjacency=0.05),
        # [T2] halation from the batch: 0.28 D excess (bias-corrected),
        # 1/e ~180 um at 47 px/mm.
        halation=HalationSpec(radii_um=(12.0, 90.0, 320.0),
                              weights=(0.30, 0.55, 0.15),
                              gain_r=0.09, gain_g=0.08, gain_b=0.07),
        couplers=CouplerSpec(0.05, 70.0),
        base_tint=(1.0, 1.0, 1.0),
        misregistration_um=4.0,
        features=Feature.NONE,
    ),
    FilmProfile(
        name="KONICA_INFRARED_750",
        aliases=("inf750", "konica infrared", "konica ir", "infrared 750"),
        description=(
            "[T1] Konica's B&W infrared. Single thin emulsion sensitised "
            "640-820 nm with a 750 nm peak, plus the silver halide's "
            "intrinsic 400-500 nm blue response and a valley between -- "
            "shoot through orange/red filters to get the IR look. Gentler "
            "and finer-grained than the IR films with deeper reach. "
            "SOURCE PDF/PROFILES/KONICA/INF750.pdf (TDSB-701): spectral "
            "band, ISO 32 unfiltered, development matrix; the sheet prints "
            "no granularity or resolving-power numbers."
        ),
        era="1980s-2000s",
        is_monochrome=True,
        exposure_index=32,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.15, 0.720, -1.30, 0.26, 1.55, 0.35)),
        # rms 13 [T3]: not in sheet; IR emulsions run grainier than their
        # speed suggests. Weights: 640-820 nm band dominates through the
        # usual red filter; the intrinsic blue lobe stays visible.
        grain=GrainSpec(13.0, 16.0, 16.0, 16.0, clump_gain=1.20, fog_grain=0.22),
        mtf=MTFSpec(52.0, 52.0, 52.0, adjacency=0.04),
        spectral_weights=(0.55, 0.15, 0.30),
        halation=HalationSpec(radii_um=(15.0, 90.0, 350.0),
                              weights=(0.30, 0.50, 0.20),
                              gain_r=0.05, gain_g=0.04, gain_b=0.03),
        misregistration_um=0.0,
        features=Feature.NONE,
    ),
    FilmProfile(
        name="KONICA_IMPRESA_50",
        aliases=("impresa", "impresa 50", "konica 50"),
        description=(
            "[T1] Konica's slow professional colour negative -- the "
            "sharpest thing they made: 160 lp/mm at 1000:1, best in this "
            "library's Konica set. SOURCE PDF/PROFILES/KONICA/IMP50.pdf "
            "(TDSN-501): ISO triple 50/16(80B)/12(80A), resolving 63/160, "
            "reciprocity +1/2 stop at 10 s with no CC shift. RMS not "
            "printed on the sheet."
        ),
        era="1990s-2000s",
        exposure_index=50,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.20, 0.600),
            g=_neg(0.62, 0.615),
            b=_neg(1.00, 0.620),
        ),
        grain=GrainSpec(3.5, 9.0, 10.0, 12.0, clump_gain=0.55, fog_grain=0.12),
        mtf=MTFSpec(72.0, 80.0, 88.0, adjacency=0.10, adjacency_um=14.0),
        couplers=CouplerSpec(0.22, 52.0, 0.10, 11.0),
        dye_matrix=_dye(-0.03),
        base_tint=(0.99, 0.995, 1.0),
        misregistration_um=4.5,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="KONICA_VX_100",
        aliases=("vx100", "konica vx100", "vx 100"),
        description=(
            "[T1] Konica's consumer 100 -- Centuria-derived emulsion, the "
            "everyday Japanese drugstore film of the late 90s. SOURCE "
            "PDF/PROFILES/KONICA/VX100Improved.pdf: ISO 100/32(80B)/25(80A), "
            "RMS granularity 4 (48 um, Dmin+1.0), resolving 63/125, "
            "reciprocity +1 stop at 10 s."
        ),
        era="1990s-2000s",
        exposure_index=100,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.21, 0.615),
            g=_neg(0.63, 0.625),
            b=_neg(1.02, 0.630),
        ),
        grain=GrainSpec(4.0, 10.0, 11.0, 13.0, clump_gain=0.60, fog_grain=0.14),
        mtf=MTFSpec(62.0, 69.0, 76.0, adjacency=0.09, adjacency_um=16.0),
        couplers=CouplerSpec(0.24, 50.0, 0.11, 11.0),
        dye_matrix=_dye(-0.02),
        base_tint=(0.99, 0.995, 1.0),
        misregistration_um=5.0,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="KONICA_CENTURIA_SUPER_400",
        aliases=("centuria", "centuria 400", "konica 400", "centuria super"),
        description=(
            "[T1] The last generation of Konica's consumer 400 before the "
            "company left film -- warm-neutral, softer palette than Fuji's "
            "Superia, the classic 2000s Japanese snapshot look. SOURCE "
            "PDF/PROFILES/KONICA/csuper400.pdf: ISO 400/125(80B)/100(80A), "
            "RMS 4 (48 um, Dmin+1.0), resolving 50/100, reciprocity +1 "
            "stop at 10 s, DX 26-5."
        ),
        era="2000s",
        exposure_index=400,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.22, 0.620),
            g=_neg(0.65, 0.630),
            b=_neg(1.05, 0.635),
        ),
        grain=GrainSpec(4.0, 12.0, 13.0, 16.0, clump_gain=0.70, fog_grain=0.16),
        mtf=MTFSpec(52.0, 58.0, 64.0, adjacency=0.08, adjacency_um=18.0),
        couplers=CouplerSpec(0.26, 48.0, 0.12, 12.0),
        dye_matrix=_dye(-0.02),
        base_tint=(0.99, 0.995, 1.0),
        misregistration_um=5.0,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="KONICA_CENTURIA_SUPER_1600",
        aliases=("centuria 1600", "konica 1600", "cs1600"),
        description=(
            "[T1] Konica's speed king -- ISO 1600 consumer negative for "
            "night snapshots and indoor sports, visibly grainy and proud "
            "of it. SOURCE PDF/PROFILES/KONICA/csuper1600.pdf: ISO "
            "1600/520(80B)/400(80A), RMS 6 (48 um), resolving 50/100, "
            "characteristic curve drawn to log H -4 (true high speed), "
            "DX 26-4."
        ),
        era="2000s",
        exposure_index=1600,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_neg(0.24, 0.600),
            g=_neg(0.68, 0.610),
            b=_neg(1.08, 0.615),
        ),
        grain=GrainSpec(6.0, 15.0, 16.0, 20.0, clump_gain=0.85, fog_grain=0.20),
        mtf=MTFSpec(44.0, 50.0, 56.0, adjacency=0.07, adjacency_um=20.0),
        couplers=CouplerSpec(0.24, 46.0, 0.11, 12.0),
        dye_matrix=_dye(-0.02),
        base_tint=(0.99, 0.995, 1.0),
        misregistration_um=5.5,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="KONICA_CHROME_CENTURIA_100",
        aliases=("chrome centuria", "chrome centuria 100", "konica chrome"),
        description=(
            "[T1] Konica's late reversal (SRA), E-6/CRK-2. Sharper than its "
            "spec class (60/140 lp/mm) with a deep Dmax drawn to ~4.0, and "
            "unusually good long-exposure manners for a slide film: no "
            "correction out to 4 s. SOURCE PDF/PROFILES/KONICA/"
            "chrocen100.pdf: ISO 100, RMS 11 (48 um, net D 1.0), full "
            "reciprocity table to 64 s (+1 stop, CC10C)."
        ),
        era="2000s",
        kind=StockKind.REVERSAL,
        exposure_index=100,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_rev(0.15, 1.62, toe_x=-0.82, shoulder_x=1.02),
            g=_rev(0.16, 1.65, toe_x=-0.84, shoulder_x=1.00),
            b=_rev(0.17, 1.67, toe_x=-0.86, shoulder_x=0.98),
        ),
        grain=GrainSpec(11.0, 12.0, 13.0, 15.0, clump_gain=0.46, fog_grain=0.16),
        mtf=MTFSpec(58.0, 65.0, 72.0, adjacency=0.10, adjacency_um=15.0),
        halation=HalationSpec(gain_r=0.04, gain_g=0.015, gain_b=0.005,
                              threshold_stops=2.0),
        couplers=CouplerSpec(0.10, 48.0, 0.06, 10.0),
        base_tint=(1.0, 1.0, 1.0),
        misregistration_um=4.0,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="KONICA_CHROME_R100",
        aliases=("chrome r100", "konica r100", "r-100"),
        description=(
            "[T1] Konica's earlier-generation reversal. Same RMS 11 as the "
            "later Chrome Centuria but softer (50/125 lp/mm) and with the "
            "old-school reciprocity cliff: correction already needed at "
            "1 s (+1/2 stop and CC5R -- the origin of its greenish long "
            "exposures). SOURCE PDF/PROFILES/KONICA/R100.pdf: ISO "
            "100/32(80B)/25(80A), CRK-2/E-6."
        ),
        era="1980s-1990s",
        kind=StockKind.REVERSAL,
        exposure_index=100,
        balance_kelvin=5500,
        curves=RGBCurves(
            r=_rev(0.16, 1.58, toe_x=-0.80, shoulder_x=0.98),
            g=_rev(0.17, 1.60, toe_x=-0.82, shoulder_x=0.96),
            b=_rev(0.18, 1.62, toe_x=-0.84, shoulder_x=0.94),
        ),
        grain=GrainSpec(11.0, 13.0, 14.0, 16.0, clump_gain=0.50, fog_grain=0.17),
        mtf=MTFSpec(50.0, 56.0, 62.0, adjacency=0.09, adjacency_um=16.0),
        halation=HalationSpec(gain_r=0.04, gain_g=0.015, gain_b=0.005,
                              threshold_stops=2.0),
        couplers=CouplerSpec(0.10, 50.0, 0.06, 10.0),
        base_tint=(1.0, 1.0, 1.0),
        misregistration_um=4.5,
        default_format="ff35",
        features=Feature.NONE,
    ),
    FilmProfile(
        name="ROLLEI_R3",
        aliases=("r3", "rollei r3", "r-3"),
        description=(
            "[T1] Rollei's three-emulsion chameleon: one coating rated "
            "anywhere from EI 25 to 6400 by developer choice, cubic "
            "crystals, super-panchromatic to ~730 nm. On glass-clear "
            "polyester with no grey base -- which is why its highlights "
            "bloom: nothing between emulsion and air but supercoat. "
            "SOURCE PDF/PROFILES/ROLLEI/TARoR3_e.pdf: gamma 0.65 dev "
            "target, 100 lp/mm at EI 400 (300 at EI 25, 1000:1), base+fog "
            "~0.25-0.30 from the curve, full reciprocity table (60 s "
            "metered -> 350 s), filter factors, clear PET 100 um."
        ),
        era="2000s-2010s",
        is_monochrome=True,
        exposure_index=400,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.28, 0.650, -1.35, 0.28, 1.60, 0.36)),
        grain=GrainSpec(15.0, 15.0, 15.0, 15.0, clump_gain=1.05, fog_grain=0.24),
        mtf=MTFSpec(50.0, 50.0, 50.0, adjacency=0.06),
        spectral_weights=(0.32, 0.40, 0.28),
        # Clear-base halation is the R3 signature look.
        halation=HalationSpec(radii_um=(15.0, 110.0, 450.0),
                              weights=(0.30, 0.50, 0.20),
                              gain_r=0.10, gain_g=0.09, gain_b=0.08),
        misregistration_um=0.0,
        features=Feature.NONE,
    ),
    FilmProfile(
        name="ROLLEI_INFRARED_400",
        aliases=("rollei ir", "rollei infrared", "ir400", "ir 820"),
        description=(
            "[T1] Rollei Infrared 400 -- panchromatic plus IR reach to "
            "820 nm; EI 400 unfiltered, a real EI 25 behind a 715 nm "
            "filter for the Wood-effect look. Clear polyester base, and "
            "the sheet itself markets the AURA halation glow. SOURCE "
            "PDF/PROFILES/ROLLEI/Rollei_Infrared.pdf: RMS 11.0 (Refinal), "
            "160 lp/mm at 1000:1, no reciprocity correction to 1/2 s, "
            "7.5 um emulsion on 100 um clear PET."
        ),
        era="2005-present",
        is_monochrome=True,
        exposure_index=400,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.16, 0.700, -1.28, 0.26, 1.55, 0.35)),
        grain=GrainSpec(11.0, 14.0, 14.0, 14.0, clump_gain=1.00, fog_grain=0.20),
        mtf=MTFSpec(58.0, 58.0, 58.0, adjacency=0.05),
        spectral_weights=(0.52, 0.20, 0.28),
        # AURA: clear base + IR scatter. The marquee feature, not a defect.
        halation=HalationSpec(radii_um=(18.0, 120.0, 500.0),
                              weights=(0.28, 0.50, 0.22),
                              gain_r=0.12, gain_g=0.10, gain_b=0.08),
        misregistration_um=0.0,
        features=Feature.NONE,
    ),
    FilmProfile(
        name="ROLLEI_RETRO_400",
        aliases=("retro 400", "rollei retro", "retro400"),
        description=(
            "[T1] Rollei Retro 400 -- panchromatic that STOPS at 630 nm, "
            "a generation earlier than modern pan films reach: reds render "
            "dark, skies drop, skin goes pale. That short red is the whole "
            "'retro' of the name. Triacetate base (the one Rollei here NOT "
            "on clear PET). SOURCE PDF/PROFILES/ROLLEI/TARRete.pdf: "
            "380-630 nm, 110 lp/mm, 10 um layer, push to EI 800; no RMS "
            "or gamma printed."
        ),
        era="2000s-2010s",
        is_monochrome=True,
        exposure_index=400,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.18, 0.680, -1.28, 0.26, 1.52, 0.35)),
        grain=GrainSpec(17.0, 17.0, 17.0, 17.0, clump_gain=1.15, fog_grain=0.24),
        mtf=MTFSpec(46.0, 46.0, 46.0, adjacency=0.05),
        # Short red cutoff -> red-starved weights; the retro tonality.
        spectral_weights=(0.16, 0.44, 0.40),
        misregistration_um=0.0,
        features=Feature.NONE,
    ),
    FilmProfile(
        name="KENTMERE_PAN_100",
        aliases=("kentmere", "kentmere 100", "pan 100"),
        description=(
            "[T1] Harman's budget 100 -- FP4's plainer sibling from the "
            "same Mobberley plant: honest tones, a bit more grain and a "
            "bit less edge than the Ilford lines it undercuts. SOURCE "
            "PDF/PROFILES/KENTMERE/Pan-100_201901.pdf: ISO 100 in ID-11, "
            "EI 50-200, reciprocity Ta = Tm^1.26 past 1 s, 0.125 mm "
            "acetate. The sheet prints no curve numbers, RMS or lp/mm."
        ),
        era="2009-present",
        is_monochrome=True,
        exposure_index=100,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.16, 0.650, -1.30, 0.27, 1.58, 0.36)),
        grain=GrainSpec(13.0, 14.0, 14.0, 14.0, clump_gain=0.95, fog_grain=0.18),
        mtf=MTFSpec(56.0, 56.0, 56.0, adjacency=0.06),
        spectral_weights=(0.28, 0.46, 0.26),
        misregistration_um=0.0,
        features=Feature.NONE,
    ),
    FilmProfile(
        name="KENTMERE_PAN_400",
        aliases=("kentmere 400", "pan 400", "kp400"),
        description=(
            "[T1] Harman's budget 400 -- the student's HP5, grainier and "
            "cheaper, forgiving from EI 320 to 800. SOURCE PDF/PROFILES/"
            "KENTMERE/Pan-400_201901.pdf: ISO 400 in ID-11, reciprocity "
            "Ta = Tm^1.30 past 1 s, 0.125 mm acetate. No curve numbers, "
            "RMS or lp/mm printed."
        ),
        era="2009-present",
        is_monochrome=True,
        exposure_index=400,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.17, 0.660, -1.28, 0.26, 1.55, 0.35)),
        grain=GrainSpec(17.5, 17.0, 17.0, 17.0, clump_gain=1.10, fog_grain=0.22),
        mtf=MTFSpec(48.0, 48.0, 48.0, adjacency=0.05),
        spectral_weights=(0.28, 0.46, 0.26),
        misregistration_um=0.0,
        features=Feature.NONE,
    ),
)

# Presented in alphabetical order by name. The literal above is grouped by
# manufacturer and era because that is how it is maintained; this is how it is
# consumed -- by --list, by the C++ table, and by `-p all`. Sorting here rather
# than reordering the literal keeps related stocks editable side by side.
FILM_PROFILES = tuple(sorted(FILM_PROFILES, key=lambda _p: _p.name))


# ===========================================================================
# SCHEMA V2 DECORATION PASS
#
# The profile literals above stay v1-shaped; everything the 2026-07 domain
# review added is filled in here by name- and rule-based decoration, so the
# per-stock v2 values live in one reviewable place. Unless a datasheet is
# cited, everything in this block is a TIER-2/3 ESTIMATE.
# ===========================================================================
def _era_start(era: str) -> int:
    """First four-digit year in an era string, or 1950 if none parses."""
    m = re.search(r"(\d{4})", era)
    return int(m.group(1)) if m else 1950


def _temporal_for_era(era: str, kind: StockKind) -> TemporalSpec:
    """Era-typical camera/printer/projector transport behaviour (DM-13).

    ALL VALUES ARE TIER-3 ESTIMATES: silent/1930s chains weave ~20/14 um RMS
    with 4-6% flicker and heavy dirt; 1950s chains ~10/7 um and ~1.5%;
    modern (1990+) chains ~3/2 um and ~0.2% with little dirt. Reversal
    stocks get 25% more weave: originals were projected directly on
    amateur/news transports rather than printed on pin-registered gear.
    """
    y = _era_start(era)
    if y < 1946:
        t = TemporalSpec(20.0, 14.0, 0.8, 5.0, 8.0, 0.0, 3.0, 12.0,
                         18.0 if y < 1930 else 24.0)
    elif y < 1970:
        t = TemporalSpec(10.0, 7.0, 0.6, 1.5, 4.0, 0.0, 1.0, 8.0, 24.0)
    elif y < 1990:
        t = TemporalSpec(6.0, 4.0, 0.5, 0.7, 2.0, 0.0, 0.5, 6.0, 24.0)
    else:
        t = TemporalSpec(3.0, 2.0, 0.4, 0.2, 1.0, 0.0, 0.1, 4.0, 24.0)
    if kind is StockKind.REVERSAL:
        t = replace(t, weave_amp_x_um=t.weave_amp_x_um * 1.25,
                    weave_amp_y_um=t.weave_amp_y_um * 1.25)
    return t


#: Hand-tuned temporal overrides where the profile text implies more than the
#: era rule: Technicolor's three-strip camera weaved heavily, Soviet transport
#: and QC were worse than western contemporaries, VISION3-era chains are
#: nearly steady, home-movie gear ran slow and loose. Tier-3 estimates.
_TEMPORAL_OVERRIDES: dict[str, TemporalSpec] = {
    "TECHNICOLOR_THREE_STRIP": TemporalSpec(26.0, 18.0, 0.8, 3.0, 8.0, 0.0, 2.0, 12.0, 24.0),
    "SOVIET_PANCHROM_1939": TemporalSpec(28.0, 20.0, 0.9, 6.0, 8.0, 0.0, 4.0, 16.0, 24.0),
    "SVEMA_FN_64": TemporalSpec(14.0, 10.0, 0.7, 2.5, 4.0, 0.0, 1.5, 10.0, 24.0),
    "SVEMA_FN_64_16MM": TemporalSpec(16.0, 11.0, 0.8, 2.8, 4.5, 0.0, 1.8, 10.0, 24.0),
    "SVEMA_FN_64_8MM": TemporalSpec(24.0, 16.0, 1.0, 4.0, 6.0, 0.0, 2.5, 10.0, 16.0),
    "SVEMA_FOTO_250": TemporalSpec(16.0, 11.0, 0.7, 3.0, 4.0, 0.0, 2.0, 10.0, 24.0),
    "TASMA_FN_64": TemporalSpec(13.0, 9.0, 0.7, 2.2, 4.0, 0.0, 1.2, 10.0, 24.0),
    "ORWOCOLOR_NC21": TemporalSpec(10.0, 7.0, 0.6, 1.8, 4.0, 0.0, 1.0, 8.0, 24.0),
    "ORWOCOLOR_NC24": TemporalSpec(10.0, 7.0, 0.6, 1.6, 4.0, 0.0, 0.9, 8.0, 24.0),
    "KODAK_VISION3_50D_5203": TemporalSpec(2.0, 1.5, 0.4, 0.1, 1.0, 0.0, 0.05, 2.0, 24.0),
    "KODAK_VISION3_250D_5207": TemporalSpec(2.0, 1.5, 0.4, 0.1, 1.0, 0.0, 0.05, 2.0, 24.0),
    "KODAK_VISION3_200T_5213": TemporalSpec(2.0, 1.5, 0.4, 0.1, 1.0, 0.0, 0.05, 2.0, 24.0),
    "KODAK_VISION3_500T_5219": TemporalSpec(2.0, 1.5, 0.4, 0.1, 1.0, 0.0, 0.05, 2.0, 24.0),
    "EIGHT_MM_BW": TemporalSpec(24.0, 16.0, 1.0, 4.0, 6.0, 0.0, 2.5, 10.0, 16.0),
    "EIGHT_MM_COLOR": TemporalSpec(24.0, 16.0, 1.0, 4.0, 6.0, 0.0, 2.5, 10.0, 18.0),
}


def _reciprocity_for(p: FilmProfile) -> ReciprocitySpec:
    """DM-07. Non-default only where the literature supports it."""
    if p.name in _RECIPROCITY_OVERRIDES:
        # Datasheet-published behaviour wins over every heuristic below.
        return _RECIPROCITY_OVERRIDES[p.name]
    if p.name == "FUJI_NEOPAN_ACROS_100":
        # Acros' documented distinction: no correction needed out to 120 s.
        return ReciprocitySpec(1.0, 1.0, 1.0, onset_s=120.0)
    if p.is_monochrome:
        # Typical conventional B&W: p ~0.95 past ~1 s. Tier-2/3 estimate.
        return ReciprocitySpec(0.95, 0.95, 0.95, onset_s=1.0)
    if p.is_reversal:
        # Colour reversal fails faster, with a channel spread that is the
        # origin of long-exposure colour casts. Tier-2/3 estimate.
        return ReciprocitySpec(0.93, 0.92, 0.94, onset_s=1.0)
    # Colour negative: no documented figure on file -- leave at "no failure".
    return ReciprocitySpec()


#: Placeholder source for every profile without an official manufacturer
#: document on file. Emitted verbatim into the generated C++ comments, so a
#: reader of film_profiles.cpp can tell citation-backed data from estimates.
_NO_DATASHEET: tuple[str, ...] = (
    "No official manufacturer datasheet available -- values estimated "
    "from secondary/historical sources",
)

#: Datasheet-audit citations (Part 2): full "Document title, Publisher,
#: year" strings, verified against the actual documents during the datasheet
#: audit. Only names with a document we can actually point at appear here;
#: everyone else falls back to ``_NO_DATASHEET``. Years are stated only where
#: the document itself carries one -- omitted rather than guessed.
#:
#: TIER SYMMETRY (2026-07-31). The tier tags were re-checked in BOTH directions,
#: not only downward: AGFA_OPTIMA_100 and FUJI_SENSIA_100 were promoted T2 -> T1
#: because published speed, granularity and resolving power all exist for them.
#: POLAROID_664 and POLAROID_667 gained real citations but stay T2, because tier
#: 1 needs a granularity figure and Polaroid publishes none for any film.
#:
#: LOCAL-ARCHIVE CAVEAT (2026-07-31 verification pass). A citation here means
#: the manufacturer published such a document, not that a copy sits in this
#: repository. The re-verification pass checked every entry against the 270
#: PDFs in ``PDF/PROFILES/`` and found no copy on file for the Kodak/Eastman
#: cine entries (VISION3 5203/5207/5213/5219, DOUBLE-X 5222, PLUS-X 5231,
#: TRI-X Reversal 7266, EKTACHROME 100D 5285, EKTACHROME 7239, 5247), for
#: FUJI NEOPAN 1600 (the PDF on file is a pure scan with no text layer -- one of
#: five such files in the archive, the others documenting no profiled stock), for
#: FUJICOLOR ETERNA
#: Vivid 500, for FERRANIA P30 or for CINESTILL 800T. Those numbers therefore
#: could NOT be re-verified in this pass and were left untouched; see
#: NotFound.md for the full list and the specific parameters still missing.
_PROVENANCE_SOURCES: dict[str, tuple[str, ...]] = {
    # -- Kodak motion picture and still stocks ------------------------------
    "KODAK_VISION3_50D_5203": (
        "KODAK VISION3 50D 5203 Technical Data, Eastman Kodak Company",
    ),
    "KODAK_VISION3_250D_5207": (
        "KODAK VISION3 250D 5207 Technical Data, Eastman Kodak Company",
    ),
    "KODAK_VISION3_200T_5213": (
        "KODAK VISION3 200T 5213 Technical Data, Eastman Kodak Company",
    ),
    "KODAK_VISION3_500T_5219": (
        "KODAK VISION3 500T 5219 Technical Data, Eastman Kodak Company",
    ),
    "EASTMAN_DOUBLE_X_5222": (
        "EASTMAN DOUBLE-X 5222/7222 Technical Data, Eastman Kodak Company",
    ),
    "EASTMAN_PLUS_X_5231": (
        "EASTMAN PLUS-X 5231/7231 Technical Data, Eastman Kodak Company",
    ),
    "KODAK_TRI_X_REVERSAL_200": (
        "KODAK TRI-X Reversal Film 7266 Technical Data, Eastman Kodak Company",
    ),
    "KODACHROME_64": (
        "KODACHROME 25/64/200 Films, Kodak publication E-55, "
        "Eastman Kodak Company, 2009",
        # Second, independent Kodak publication, verified 2026-07-31:
        "KODACHROME 64 and 200 Films, Kodak publication E-88, "
        "Eastman Kodak Company, 2009",
    ),
    "EKTACHROME_64": (
        "KODAK EKTACHROME 64 Professional (EPR), Kodak publication E-8, "
        "Eastman Kodak Company",
    ),
    "EKTACHROME_160T": (
        "KODAK EKTACHROME 160T Professional (EPT), Kodak publication E-144, "
        "Eastman Kodak Company",
    ),
    "KODAK_EKTACHROME_100D_5285": (
        "KODAK EKTACHROME 100D Color Reversal Film 5285, "
        "Kodak publication H-1-5285, Eastman Kodak Company",
    ),
    "EASTMAN_EKTACHROME_7239": (
        "EASTMAN EKTACHROME Film (Daylight) 7239, "
        "Kodak publication H-1-5239, Eastman Kodak Company",
    ),
    "EASTMAN_5247_1974": (
        "EASTMAN Color Negative Film 5247, Kodak publication TI0835, "
        "Eastman Kodak Company",
    ),
    "KODAK_PORTRA_400": (
        "KODAK PROFESSIONAL PORTRA 400, Kodak publication E-4050, "
        "Eastman Kodak Company",
    ),
    # -- Ilford / HARMAN -----------------------------------------------------
    "ILFORD_HP5_PLUS_400": (
        "ILFORD HP5 PLUS technical datasheet, HARMAN technology Ltd",
    ),
    "ILFORD_DELTA_3200": (
        "ILFORD DELTA 3200 PROFESSIONAL technical datasheet, "
        "HARMAN technology Ltd, 2002",
    ),
    # -- Fujifilm ------------------------------------------------------------
    "FUJI_VELVIA_50": (
        "FUJICHROME Velvia 50 datasheet / Data Guide, FUJIFILM Corporation",
    ),
    "FUJI_PROVIA_400X": (
        "FUJICHROME PROVIA 400X Product Information Bulletin, "
        "FUJIFILM Corporation, 2007",
    ),
    "FUJI_SENSIA_100": (
        "FUJICHROME Sensia 100 [RA] datasheet AF3-091E, FUJIFILM Corporation",
    ),
    "FUJI_NEOPAN_1600": (
        "FUJI NEOPAN 1600 Super Presto datasheet, FUJIFILM Corporation",
    ),
    "FUJI_NEOPAN_ACROS_100": (
        "FUJI NEOPAN 100 ACROS datasheet (and ACROS II AF3-0258E), "
        "FUJIFILM Corporation",
    ),
    "FUJI_ETERNA_VIVID_500T_8547": (
        "FUJICOLOR ETERNA Vivid 500 datasheet, FUJIFILM Corporation, 2009",
    ),
    # -- Agfa ------------------------------------------------------------------
    "AGFA_APX_25": (
        "Agfa Professional Films Technical Data, Agfa-Gevaert AG",
        "AGFAPAN APX 25 PROFESSIONAL datasheet (agfapanapx25.pdf), "
        "Agfa-Gevaert AG",
    ),
    "AGFA_APX_100": (
        "Agfa Professional Films Technical Data, Agfa-Gevaert AG",
        "AGFAPAN APX 100 PROFESSIONAL datasheet (apx100.pdf), Agfa-Gevaert AG",
    ),
    "AGFA_APX_400": (
        "Agfa Professional Films Technical Data, Agfa-Gevaert AG",
        "AGFAPAN APX 400 PROFESSIONAL datasheet (apx400.pdf), Agfa-Gevaert AG",
    ),
    # Added 2026-07-31: OPTIMA II 100 IS documented -- it is specified across
    # the multi-film "Professional Films" brochure (granularity and resolving
    # power on p7, layer design on p5) rather than in a single-stock sheet,
    # which is why it was previously left on _NO_DATASHEET.
    "AGFA_OPTIMA_100": (
        "Agfa Professional Films -- AGFACOLOR OPTIMA II 100 pages "
        "(agfa_films.pdf p5, p7), Agfa-Gevaert AG",
    ),
    # -- Polaroid (added 2026-07-31) -------------------------------------------
    # All three were on _NO_DATASHEET although Polaroid Film Data Sheets are on
    # file. 664 and 667 carry full Technical Data pages; the SX-70 document is a
    # product page with no technical section at all, which is itself the reason
    # nearly every SX-70 number in this database stays an estimate.
    "POLAROID_664": (
        "POLAROID Polapan Pro 100 / Type 664 Film Data Sheet (664fds.pdf), "
        "Polaroid Corporation",
    ),
    "POLAROID_667": (
        "POLAROID Type 667 Film Data Sheet (667fds.pdf), Polaroid Corporation",
    ),
    "POLAROID_SX70": (
        "POLAROID Time-Zero SX-70 Integral Color Print Film product data "
        "sheet (timezfds.pdf), Polaroid Corporation -- product page only, no "
        "technical-data section",
    ),
    # -- Others ----------------------------------------------------------------
    "FOMAPAN_400_ACTION": (
        "FOMAPAN 400 Action technical datasheet, Foma Bohemia Ltd",
    ),
    "FERRANIA_P30": (
        "Ferrania P30 manufacturer product specification, "
        "Film Ferrania S.r.l., 2017",
    ),
    "CINESTILL_800T": (
        "CineStill 800T product documentation, CineStill Film, 2012 "
        "(base stock: KODAK VISION3 5219)",
    ),
}

#: Tier inference for profiles whose description carries no [T*] tag:
#: modern stocks with published datasheets are tier 1, partially documented
#: families tier 2, pre-datasheet reconstructions tier 3. Mirrors the
#: confidence-tier note in the database header.
_UNTAGGED_TIER: dict[str, int] = {
    "AGFACOLOR_NEU_1936": 3,
    "CINESTILL_800T": 2,
    "DUFAYCOLOR_1937": 3,
    "EASTMAN_5247_1974": 2,
    "EASTMAN_DOUBLE_X_5222": 1,
    "EASTMAN_EXR_500T_5296": 2,
    "EASTMAN_ORTHO_1930": 3,
    "EASTMAN_SUPER_XX_1938": 3,
    "FOMAPAN_400_ACTION": 1,
    "FUJICOLOR_SUPER_F500_8572": 2,
    "FUJI_ETERNA_VIVID_500T_8547": 2,
    "FUJI_VELVIA_50": 1,
    # Corrected 1 -> 2 on 2026-07-31. Tier 1 requires a published granularity
    # figure AND an MTF/resolving-power figure; Harman/ILFORD publish NEITHER for
    # any emulsion (checked across all 18 ILFORD datasheets on file, plus both
    # KENTMERE sheets). What IS documented for these two is ISO speed, the
    # characteristic-curve processing conditions, the reciprocity formula and the
    # development matrix -- real data, but not enough for tier 1 under this
    # database's own definition.
    # For accuracy: Harman DOES publish numeric average gradient (G-bar) for
    # some emulsions, e.g. "negatives of normal contrast (Gbar 0.62)" for
    # DELTA 400 and a full G-bar table for ORTHO PLUS -- so "Ilford publishes no
    # numbers at all" would be too strong a claim. It publishes no GRANULARITY
    # and no SHARPNESS numbers, which is what these two tiers turn on, and no
    # G-bar figure for either of these two stocks specifically.
    # Caveat on the mechanism: _FITTED_FROM maps tier 2 to "secondary_sources",
    # so both stocks now report that string even though their reciprocity
    # exponents are datasheet-derived. The tier is a whole-profile summary; the
    # per-field provenance lives in the comments at each field.
    "ILFORD_DELTA_3200": 2,
    "ILFORD_HP5_PLUS_400": 2,
    "KODACHROME_64": 1,
    "KODAK_EKTACHROME_100D_5285": 1,
    "KODAK_PORTRA_400": 1,
    "KODAK_TRI_X_REVERSAL_200": 1,
    "KODAK_VISION3_200T_5213": 1,
    "KODAK_VISION3_250D_5207": 1,
    "KODAK_VISION3_500T_5219": 1,
    "KODAK_VISION3_50D_5203": 1,
    "ORWOCOLOR_NC21": 3,
    "SOVIET_PANCHROM_1939": 3,
    "SVEMA_FN_64": 3,
    "TECHNICOLOR_THREE_STRIP": 3,
}

_FITTED_FROM = {1: "datasheet_curve", 2: "secondary_sources", 3: "analogy"}


def _provenance_for(p: FilmProfile) -> Provenance:
    """DM-19. Tier parsed from the [T*] tag; untagged stocks are inferred."""
    m = re.match(r"\[T([123])\]", p.description)
    tier = int(m.group(1)) if m else _UNTAGGED_TIER.get(p.name, 3)
    return Provenance(
        tier=tier,
        sources=_PROVENANCE_SOURCES.get(p.name, _NO_DATASHEET),
        fitted_from=_FITTED_FROM[tier],
        last_reviewed="2026-07-30",
    )


#: Colour negatives whose per-channel dmin values ladder upward (r << g << b)
#: because the dmin encodes the orange coupler mask directly (audit finding;
#: the rest keep near-neutral dmin and carry the mask in base_tint/dye data).
_DMIN_LADDER = {
    "AGFA_OPTIMA_100",
    "AGFA_VISTA_200",
    "FUJI_F125_8530",
    "FUJI_F125_8630",
    "GEVACOLOR_1952",
}

#: Datasheet-published resolving power, lp/mm at 1.6:1 / 1000:1 TOC (DM-11).
#: Deliberately sparse: numbers appear only where a datasheet states them.
#: Everything else stays 0.0 = "not published / not verified" -- do not
#: invent values here.
_RESOLVING_POWER: dict[str, tuple[float, float]] = {
    # 2026-07-31 datasheet additions (lowc 1.6:1 / highc 1000:1 lp/mm):
    "KONICA_IMPRESA_50": (63.0, 160.0),
    "KONICA_VX_100": (63.0, 125.0),
    "KONICA_CENTURIA_SUPER_400": (50.0, 100.0),
    "KONICA_CENTURIA_SUPER_1600": (50.0, 100.0),
    "KONICA_CHROME_CENTURIA_100": (60.0, 140.0),
    "KONICA_CHROME_R100": (50.0, 125.0),
    "ROLLEI_R3": (45.0, 100.0),        # 100 @ EI 400; sheet: up to 300 @ EI 25
    "ROLLEI_INFRARED_400": (55.0, 160.0),  # 160 high-contrast printed; low T3
    "ROLLEI_RETRO_400": (40.0, 110.0),     # sheet prints 110, contrast unstated
    # (low-contrast 1.6:1, high-contrast 1000:1). 0.0 means the manufacturer
    # does not publish that contrast, NOT that the film cannot resolve it. Every
    # entry below is a transcription; nothing here is interpolated between
    # contrasts or between films.
    #
    # !! UNIT CAVEAT. The field name says "lp_mm", but the sheets do not all use
    # the same unit: Agfa, Fuji and Foma print "lines/mm" while Polaroid prints
    # "line pairs/mm". No sheet states an equivalence between the two, so the
    # numbers below are stored AS PRINTED and are NOT normalised to a common
    # unit. A consumer comparing an Agfa figure with a Polaroid one is comparing
    # two different measurements. Renaming the field would break schema v2, so
    # the discrepancy is documented rather than papered over.
    #
    # -- Fujifilm (both contrasts published) --------------------------------
    "FUJI_NEOPAN_ACROS_100": (60.0, 200.0),   # NeopanAcros100.pdf p4
    "FUJI_VELVIA_50": (80.0, 160.0),          # velvia_50_datasheet.pdf p7
    "FUJI_PROVIA_400X": (55.0, 135.0),        # Provia_400X_PIB_1007.pdf p6
    "FUJI_SENSIA_100": (55.0, 135.0),         # sensia_100_datasheet.pdf p4
    # -- Agfa ---------------------------------------------------------------
    # The three APX sheets print only the 1000:1 column; the colour-negative
    # pages of the "Professional Films" brochure print both.
    "AGFA_APX_25": (0.0, 200.0),              # agfapanapx25.pdf p1
    "AGFA_APX_100": (0.0, 150.0),             # apx100.pdf p1
    "AGFA_APX_400": (0.0, 110.0),             # apx400.pdf p1
    "AGFA_OPTIMA_100": (50.0, 140.0),         # agfa_films.pdf p7 (OPTIMA II 100)
    # -- Foma ---------------------------------------------------------------
    # CAVEAT: Foma prints "Resolving power / 90 lines per mm" with NO
    # test-object contrast stated. It is recorded in the high-contrast slot
    # because an unqualified resolving-power figure is conventionally the
    # 1000:1 measurement, but that label is an interpretation of Foma's
    # omission, not something Foma printed. The number 90 itself is verbatim.
    "FOMAPAN_400_ACTION": (0.0, 90.0),        # fomapan-400.pdf p1
    # -- Polaroid -----------------------------------------------------------
    # Polaroid publishes a RANGE, e.g. "Resolution (1000:1): 20 - 25 line
    # pairs/mm". A single float cannot hold a range, so the LOWER bound is
    # recorded (the conservative end) rather than a midpoint, which would be an
    # invented number. Full published ranges: 664 = 20-25, 667 = 14-20.
    "POLAROID_664": (0.0, 20.0),              # 664fds.pdf p2
    "POLAROID_667": (0.0, 14.0),              # 667fds.pdf p2
}


#: Datasheet-documented reciprocity behaviour (DM-07). Consulted before the
#: era/type heuristics in ``_reciprocity_for``, so anything listed here is a
#: published figure rather than an estimate.
#:
#: HOW THE EXPONENT IS OBTAINED. Harman/ILFORD publish the correction as an
#: adjusted *time*: ``Ta = Tm ** k``, where Tm is the metered time and Ta the
#: time to actually give. This model instead writes the effective exposure as
#: ``E_eff = I * t ** p``. Correct exposure requires ``I * Ta**p == I * Tm``,
#: i.e. ``Ta == Tm ** (1/p)``, so ``p = 1 / k`` exactly. No curve fitting and
#: no free parameters -- the two formulations are algebraically identical.
#: ``onset_s`` is likewise taken from the sheets ("no adjustments are needed"
#: for exposures between 1/10 000 s and 1/2 s).
#:
#: !! NOTE FOR CONSUMERS OF THIS STRUCT. The bare form ``E_eff = I * t**p`` is
#: discontinuous at ``onset_s``: at t = 0.5 s with p = 0.7634 it returns
#: 0.5**0.7634 = 0.589, i.e. 0.24 stops of failure exactly where the datasheet
#: says there is none. The physically correct reading of these published figures
#: is the ONSET-NORMALISED form
#:     E_eff = I * t                          for t <= onset_s
#:     E_eff = I * t * (t / onset_s)**(p - 1) for t >  onset_s
#: which is continuous at the onset and reproduces the published adjusted-time
#: relation above it. Nothing in film_sim.py consumes ReciprocitySpec yet, so
#: this is latent rather than a live bug; any renderer that starts using these
#: fields must use the normalised form.
_RECIPROCITY_OVERRIDES: dict[str, ReciprocitySpec] = {
    # 2026-07-31, fitted from datasheet correction tables (t_a^p * onset^(1-p)
    # = t_m solved at the printed correction points):
    "KONICA_IMPRESA_50": ReciprocitySpec(0.87, 0.87, 0.87, onset_s=1.0),
    #   IMP50.pdf: +1/2 stop at 10 s, no CC -> p = ln10/ln14.1
    "KONICA_VX_100": ReciprocitySpec(0.77, 0.77, 0.77, onset_s=1.0),
    "KONICA_CENTURIA_SUPER_400": ReciprocitySpec(0.77, 0.77, 0.77, onset_s=1.0),
    "KONICA_CENTURIA_SUPER_1600": ReciprocitySpec(0.77, 0.77, 0.77, onset_s=1.0),
    #   VX/Centuria sheets: +1 stop at 10 s -> p = ln10/ln20 = 0.77
    "KONICA_CHROME_CENTURIA_100": ReciprocitySpec(0.82, 0.80, 0.80, onset_s=4.0),
    #   chrocen100.pdf: no correction to 4 s; +1 stop at 64 s with CC10C
    #   (cyan trim = red channel fails least -> p_r highest)
    "KONICA_CHROME_R100": ReciprocitySpec(0.88, 0.90, 0.91, onset_s=0.5),
    #   R100.pdf: +1/2 stop already at 1 s with CC5R (red trim = green/blue
    #   fail least); the old-generation reversal cliff
    "ROLLEI_R3": ReciprocitySpec(0.68, 0.68, 0.68, onset_s=1.0),
    #   TARoR3_e.pdf table: 15->60 s gives p=0.66, 60->350 s gives p=0.70;
    #   0.68 splits it. Severe -- the sheet is unusually honest about it.
    "ROLLEI_INFRARED_400": ReciprocitySpec(0.95, 0.95, 0.95, onset_s=0.5),
    #   Rollei_Infrared.pdf: "N/A" to 1/2 s, nothing documented beyond ->
    #   B&W default slope from a 0.5 s onset
    "KENTMERE_PAN_100": ReciprocitySpec(0.794, 0.794, 0.794, onset_s=1.0),
    #   Pan-100 sheet: Ta = Tm^1.26 -> p = 1/1.26
    "KENTMERE_PAN_400": ReciprocitySpec(0.769, 0.769, 0.769, onset_s=1.0),
    #   Pan-400 sheet: Ta = Tm^1.30 -> p = 1/1.30
    # ILFORD/HP5-Plus_201811.pdf p2: "The graph is based on the formulae
    # Ta = Tm1.31" (typeset exponent), no correction between 1/2 s and
    # 1/10 000 s. p = 1/1.31 = 0.7634.
    "ILFORD_HP5_PLUS_400": ReciprocitySpec(0.7634, 0.7634, 0.7634, onset_s=0.5),
    # ILFORD/Delta-3200_201811.pdf p2: "Ta = Tm1.33". p = 1/1.33 = 0.7519.
    # The 2018 sheet is internally inconsistent about the onset (it says no
    # correction is needed from 1/2 s, then refers to "exposures longer than
    # 1 second"); 0.5 s is used because the 2002 edition
    # (Delta_3200-200209.pdf p2) states 1/2 s unambiguously.
    "ILFORD_DELTA_3200": ReciprocitySpec(0.7519, 0.7519, 0.7519, onset_s=0.5),
}


def _grain_v2(p: FilmProfile) -> GrainSpec:
    """DM-09/DM-10 grain enrichment. Tier-2/3 estimates throughout."""
    g = p.grain
    kw: dict = {}
    is_colour_neg = (
        not p.is_monochrome
        and not p.is_reversal
        and p.reseau is None
        and p.name != "TECHNICOLOR_THREE_STRIP"  # three silver B&W records
    )
    if is_colour_neg:
        # Blue record noisiest (top, fastest layer), red slightly above
        # green: b ~1.3x, r ~1.1x of green = rms_granularity. Tier-2.
        kw.update(
            rms_r=round(1.1 * g.rms_granularity, 2),
            rms_g=g.rms_granularity,
            rms_b=round(1.3 * g.rms_granularity, 2),
        )
    # sigma(D) shape anchors at D = toe/1.0/dmax. Negatives are monotone;
    # reversal sigma turns over past mid-scale because the densest regions
    # of a slide received the least exposure. Tier-3 heuristic -- and ONLY
    # a heuristic: it fills the shape when the profile literal still holds
    # the dataclass defaults (0.4/1.0/1.2). A literal that sets its own
    # shape (measured stocks: the Soviet scan batches) is authoritative and
    # must never be silently overwritten here. This exact bug ate two
    # rounds of measured FN-64 shape adoptions before being caught.
    if (g.sigma_shape_toe, g.sigma_shape_mid, g.sigma_shape_dmax) == (0.0, 1.0, 0.0):
        # untouched dataclass defaults -> fill from the heuristic
        if p.is_reversal:
            kw.update(sigma_shape_toe=0.7, sigma_shape_mid=1.0,
                      sigma_shape_dmax=0.5)
        else:
            kw.update(sigma_shape_toe=0.4, sigma_shape_mid=1.0,
                      sigma_shape_dmax=1.2)
    # Grain-size dispersion: fast/pushed emulsions ~0.55, T-grain ~0.25,
    # conventional cubic keeps the 0.35 default. Tier-3.
    if p.exposure_index >= 800:
        kw["size_sigma_log"] = 0.55
    elif Feature.TABULAR_GRAIN in p.features:
        kw["size_sigma_log"] = 0.25
    # Dye clouds exist only in chromogenic images; B&W silver, the reseau
    # stock and Technicolor's silver camera records stay 0. Tier-2 practice:
    # ~1.5 um slow, ~2.0 um mid, ~2.5 um fast emulsions.
    if not p.is_monochrome and p.reseau is None and p.name != "TECHNICOLOR_THREE_STRIP":
        if p.exposure_index >= 400:
            kw["dye_cloud_um"] = 2.5
        elif p.exposure_index <= 64:
            kw["dye_cloud_um"] = 1.5
        else:
            kw["dye_cloud_um"] = 2.0
    return replace(g, **kw)


def _apply_schema_v2(p: FilmProfile) -> FilmProfile:
    """Fill every schema-v2 field from the rules and tables above."""
    y = _era_start(p.era)
    historic = y < 1960 and "present" not in p.era.lower()
    if p.is_monochrome:
        density_metric = "visual_iso"
        speed_criterion = "iso6"
        callier_q = 1.25 if p.is_reversal else 1.3
        mask_encoding = "none"
    elif p.is_reversal:
        density_metric = "status_a"
        speed_criterion = "iso2240"
        callier_q = 1.0
        mask_encoding = "none"
    else:
        density_metric = "status_m"
        speed_criterion = "iso5800"
        callier_q = 1.0
        mask_encoding = "dmin_ladder" if p.name in _DMIN_LADDER else "neutral_dmin"
    if historic:
        speed_criterion = "manufacturer_ei"

    mtf_kw: dict = {}
    if p.name in _RESOLVING_POWER:
        lo, hi = _RESOLVING_POWER[p.name]
        mtf_kw = dict(
            resolving_power_lp_mm_lowc=lo, resolving_power_lp_mm_highc=hi
        )

    # The HALATION flag is derived, never authoritative: the numeric gains in
    # p.halation decide, and the flag is forced to agree so that renderers
    # keying on either see the same truth.
    feats = p.features
    if p.halation.active:
        feats |= Feature.HALATION
    else:
        feats &= ~Feature.HALATION

    return replace(
        p,
        features=feats,
        grain=_grain_v2(p),
        mtf=replace(p.mtf, **mtf_kw) if mtf_kw else p.mtf,
        temporal=_TEMPORAL_OVERRIDES.get(
            p.name, _temporal_for_era(p.era, p.kind)
        ),
        reciprocity=_reciprocity_for(p),
        aging=AgingSpec(),  # every profile ships fresh; hooks only (DM-01)
        provenance=_provenance_for(p),
        trim=0.0,  # static trim; the per-render anchor solve refines it
        density_metric=density_metric,
        referred="print" if p.default_print != "SCAN_DI" else "scan",
        speed_point_x=0.0,
        speed_criterion=speed_criterion,
        mask_encoding=mask_encoding,
        callier_q=callier_q,
    )


FILM_PROFILES = tuple(_apply_schema_v2(_p) for _p in FILM_PROFILES)


PRINT_STOCKS: tuple[PrintStock, ...] = (
    PrintStock(
        name="SCAN_DI",
        description=(
            "Digital-intermediate style inversion. Modest print gamma so the "
            "system gamma lands near 1.0, as a graded scan would."
        ),
        curves=RGBCurves(
            r=ToneCurve(0.06, 1.72, -0.98, 0.26, 0.96, 0.34),
            g=ToneCurve(0.06, 1.75, -0.96, 0.25, 0.94, 0.33),
            b=ToneCurve(0.07, 1.78, -0.94, 0.25, 0.92, 0.33),
        ),
        mtf_f50=105.0,
        grain_rms=2.2,
        grain_clump_um=5.0,
    ),
    PrintStock(
        name="KODAK_2383_RELEASE",
        description=(
            "Theatrical release print emulation. High print gamma gives the "
            "contrasty, crushed-shadow projected look and the characteristic "
            "highlight shoulder."
        ),
        curves=RGBCurves(
            r=ToneCurve(0.08, 2.62, -0.72, 0.22, 0.74, 0.30),
            g=ToneCurve(0.08, 2.70, -0.70, 0.21, 0.72, 0.29),
            b=ToneCurve(0.09, 2.78, -0.68, 0.21, 0.70, 0.29),
        ),
        mtf_f50=80.0,
        grain_rms=3.4,
        grain_clump_um=6.0,
        dye_matrix=_dye(-0.09),
    ),
    PrintStock(
        name="DUPE_FINE_GRAIN",
        description=(
            "Fine-grain duplicating stock, gamma about 1.0. Used for the "
            "interpositive and dupe negative stages between the camera negative "
            "and the release print. Unity gamma is the whole design brief: a "
            "release print is three or four generations from the negative, and "
            "at any other gamma the contrast would compound catastrophically "
            "over that chain. What accumulates instead is grain and softness."
        ),
        curves=RGBCurves(
            r=ToneCurve(0.14, 1.00, -1.30, 0.30, 1.42, 0.44),
            g=ToneCurve(0.14, 1.01, -1.28, 0.30, 1.40, 0.44),
            b=ToneCurve(0.15, 1.02, -1.26, 0.30, 1.38, 0.44),
        ),
        mtf_f50=72.0,
        grain_rms=3.2,
        grain_clump_um=6.5,
    ),
    PrintStock(
        name="TECHNICOLOR_IB",
        description=(
            "Imbibition dye transfer print. Three gelatin matrices transfer "
            "pure cyan, magenta and yellow dyes onto a blank. Lower Dmax than a "
            "modern print but far cleaner dyes, which is why Technicolor looks "
            "saturated without looking harsh."
        ),
        curves=RGBCurves(
            r=ToneCurve(0.07, 2.16, -0.84, 0.24, 0.86, 0.32),
            g=ToneCurve(0.07, 2.22, -0.82, 0.23, 0.84, 0.31),
            b=ToneCurve(0.08, 2.28, -0.80, 0.23, 0.82, 0.31),
        ),
        mtf_f50=62.0,
        grain_rms=1.8,  # dye transfer adds almost no grain of its own
        grain_clump_um=7.0,
        dye_matrix=_dye(-0.16),
    ),
    PrintStock(
        name="TASMA_POSITIVE_28",
        description=(
            "[T3] Soviet B&W cine positive film, Tasma (Kazan), GOST 2.8 -- the "
            "release-print stock, sold in the yellow boxes. Note this is "
            "deliberately a PrintStock and not a FilmProfile: a positive film "
            "is not something you expose in a camera, it is what a negative is "
            "printed onto, which is exactly the role PrintStock fills in this "
            "pipeline. Pair it with a Soviet negative for a period Soviet "
            "release-print look: TASMA_FN_65 or SVEMA_FN_64 with "
            "--print TASMA_POSITIVE_28. High print gamma gives the contrasty, "
            "crushed-shadow projected image; grain is fine, as positive stock "
            "always is, so nearly all visible grain still comes from the "
            "negative. GOST 2.8 is roughly ISO 3 -- print stock is slow because "
            "it only ever sees a printer lamp."
        ),
        curves=RGBCurves(
            r=ToneCurve(0.09, 2.52, -0.74, 0.23, 0.78, 0.31),
            g=ToneCurve(0.09, 2.52, -0.74, 0.23, 0.78, 0.31),
            b=ToneCurve(0.09, 2.52, -0.74, 0.23, 0.78, 0.31),
        ),
        mtf_f50=62.0,
        grain_rms=4.2,
        grain_clump_um=7.5,
    ),
)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    """Normalise a lookup key: alphanumerics only, upper case."""
    return "".join(ch for ch in s if ch.isalnum()).upper()


_BY_NAME: dict[str, FilmProfile] = {p.name: p for p in FILM_PROFILES}
_PRINT_BY_NAME: dict[str, PrintStock] = {p.name: p for p in PRINT_STOCKS}

_INDEX: dict[str, FilmProfile] = {}
for _p in FILM_PROFILES:
    for _key in (_p.name, *_p.aliases):
        _n = _norm(_key)
        if _n in _INDEX and _INDEX[_n] is not _p:
            raise RuntimeError(f"duplicate profile alias {_key!r}")
        _INDEX[_n] = _p

_PRINT_INDEX: dict[str, PrintStock] = {_norm(p.name): p for p in PRINT_STOCKS}


def profile_names() -> list[str]:
    """Canonical names of all available film stocks."""
    return list(_BY_NAME)


def get_profile(name: str) -> FilmProfile:
    """Look up a stock by name, alias or catalogue number.

    Matching ignores case, spaces, hyphens and underscores, so all of
    ``"Kodak Vision3 500T (5219)"``, ``"vision3-500t"`` and ``"5219"`` resolve
    to the same profile.

    Raises:
        KeyError: If no stock matches, listing the valid names.
    """
    key = _norm(name)
    if key in _INDEX:
        return _INDEX[key]
    # Tolerate decorated input such as "Kodak Vision3 500T (5219)" by trying
    # each parenthesised or whitespace-separated token.
    for token in name.replace("(", " ").replace(")", " ").split():
        tk = _norm(token)
        if tk in _INDEX:
            return _INDEX[tk]
    raise KeyError(f"unknown film stock {name!r}; available: {profile_names()}")


def get_print_stock(name: str) -> PrintStock:
    """Look up a print stock by name, case and punctuation insensitive."""
    key = _norm(name)
    if key not in _PRINT_INDEX:
        raise KeyError(
            f"unknown print stock {name!r}; available: {list(_PRINT_BY_NAME)}"
        )
    return _PRINT_INDEX[key]


_DENSITY_METRICS = {"status_m", "status_a", "visual_iso"}
_MASK_ENCODINGS = {"dmin_ladder", "neutral_dmin", "none"}


def validate_all() -> None:
    """Validate every profile and print stock. Called by the renderer on start.

    Hard failures for data that cannot be right; a single aggregated
    ``UserWarning`` for flag/numeric disagreements, because the Feature flags
    are declared derived and non-authoritative (DM-20).
    """
    flag_notes: list[str] = []
    for p in FILM_PROFILES:
        p.validate()
        if not p.is_reversal:
            get_print_stock(p.default_print)
        # -- schema v2 checks -------------------------------------------------
        if min(p.grain.rms_r, p.grain.rms_g, p.grain.rms_b) < 0:
            raise ValueError(f"{p.name}: per-channel grain rms must be >= 0")
        if p.provenance.tier not in (1, 2, 3):
            raise ValueError(f"{p.name}: provenance tier must be 1..3")
        if p.density_metric not in _DENSITY_METRICS:
            raise ValueError(
                f"{p.name}: density_metric {p.density_metric!r} not in "
                f"{sorted(_DENSITY_METRICS)}"
            )
        if p.mask_encoding not in _MASK_ENCODINGS:
            raise ValueError(
                f"{p.name}: mask_encoding {p.mask_encoding!r} not in "
                f"{sorted(_MASK_ENCODINGS)}"
            )
        if p.mask_encoding == "dmin_ladder":
            dmins = [c.dmin for c in p.curves.as_tuple()]
            if max(dmins) - min(dmins) <= 0.3:
                raise ValueError(
                    f"{p.name}: mask_encoding 'dmin_ladder' but the dmin "
                    f"spread is only {max(dmins) - min(dmins):.2f} "
                    "(a mask ladder needs > 0.3)"
                )
        if p.callier_q <= 0:
            raise ValueError(f"{p.name}: callier_q must be > 0")
        # -- warn-only: flags are derived, so a mismatch is a documentation
        # smell rather than an error (DM-20).
        if (Feature.HALATION in p.features) != p.halation.active:
            flag_notes.append(
                f"{p.name}: HALATION flag "
                f"{'set but all gains are 0' if Feature.HALATION in p.features else 'clear but halation gains are nonzero'}"
            )
    if flag_notes:
        warnings.warn(
            "feature-flag / numeric-field disagreements (flags are derived "
            "and non-authoritative; renderers must key on the numeric "
            "fields): " + "; ".join(flag_notes),
            stacklevel=2,
        )
    for s in PRINT_STOCKS:
        s.validate()
        if s.density_metric not in _DENSITY_METRICS:
            raise ValueError(
                f"{s.name}: density_metric {s.density_metric!r} not in "
                f"{sorted(_DENSITY_METRICS)}"
            )


if __name__ == "__main__":
    validate_all()
    print(
        f"{len(FILM_PROFILES)} film stocks, {len(PRINT_STOCKS)} print stocks OK\n"
    )
    hdr = (
        f"{'stock':32s} {'kind':9s} {'EI':>5s} {'K':>5s} "
        f"{'gamma':>6s} {'Dmax':>5s} {'lat':>5s} {'RMS':>5s} "
        f"{'clump':>6s} {'f50':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p in FILM_PROFILES:
        c = p.curves.g
        if p.has_reseau:
            kind = "mosaic"
        elif p.is_reversal:
            kind = "reversal"
        elif p.is_monochrome:
            kind = "B&W neg"
        else:
            kind = "neg"
        print(
            f"{p.name:32s} {kind:9s} {p.exposure_index:>5d} "
            f"{p.balance_kelvin:>5d} {c.gamma:>6.2f} {c.dmax:>5.2f} "
            f"{c.latitude_stops:>5.1f} {p.grain.rms_granularity:>5.1f} "
            f"{p.grain.clump_um_g:>5.1f}u {p.mtf.f50_g:>5.0f}"
        )
