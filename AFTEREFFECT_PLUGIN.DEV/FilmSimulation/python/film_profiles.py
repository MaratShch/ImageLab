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

Requires Python 3.12+. Pure stdlib.
"""

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
    "FilmProfile",
    "PrintStock",
    "FILM_PROFILES",
    "PRINT_STOCKS",
    "FORMATS",
    "IDENTITY3",
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
    and guaranteed monotonicity as long as ``shoulder_k <= 2 * toe_k``.

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
    """

    rms_granularity: float
    clump_um_r: float
    clump_um_g: float
    clump_um_b: float
    clump_gain: float
    fog_grain: float = 0.18
    anisotropy: float = 1.0

    def clumps(self) -> tuple[float, float, float]:
        return (self.clump_um_r, self.clump_um_g, self.clump_um_b)


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
    """

    f50_r: float
    f50_g: float
    f50_b: float
    adjacency: float = 0.0
    adjacency_um: float = 25.0

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
        features: Optional behaviour flags.
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
    reseau: ReseauSpec | None = None
    default_flare: float = 0.0
    features: Feature = Feature.NONE

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

    def validate(self) -> None:
        self.curves.validate(self.name)


# ---------------------------------------------------------------------------
# Film gauges. Aperture widths per SMPTE / manufacturer specification.
# ---------------------------------------------------------------------------
FORMATS: dict[str, float] = {
    "super35": 24.89,     # Super 35 full aperture width
    "academy35": 21.95,   # Academy aperture
    "anamorphic35": 21.95,
    "techni35": 24.89,    # three-strip Technicolor used full aperture
    "super16": 12.52,
    "16mm": 10.26,
    "8mm": 4.80,
    "ff35": 36.00,        # 35 mm still full frame
    "medium645": 56.00,
    "large4x5": 127.00,
    "imax15": 70.41,
}
"""Image width on the negative, millimetres. Grain, halation, MTF and channel
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
        era="1991-2013",
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
        era="2008-2013",
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
        grain=GrainSpec(12.0, 14.0, 15.0, 18.0, clump_gain=1.35, fog_grain=0.30,
                        anisotropy=1.06),
        mtf=MTFSpec(26.0, 30.0, 34.0, adjacency=0.02),
        halation=HalationSpec(
            radii_um=(20.0, 100.0, 420.0),
            gain_r=0.22, gain_g=0.12, gain_b=0.08,
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
        grain=GrainSpec(4.0, 6.6, 7.2, 8.6, clump_gain=0.22, fog_grain=0.17),
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
        features=Feature.STRONG_DIR_COUPLERS | Feature.TABULAR_GRAIN,
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
        features=Feature.HALATION | Feature.NO_REMJET | Feature.STRONG_DIR_COUPLERS,
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
        grain=GrainSpec(2.2, 3.8, 4.0, 4.6, clump_gain=0.30, fog_grain=0.12),
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
        grain=GrainSpec(2.4, 3.6, 3.8, 4.4, clump_gain=0.18, fog_grain=0.12),
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
        features=Feature.STRONG_DIR_COUPLERS,
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
        exposure_index=400,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.12, 0.640, -1.62, 0.34, 2.30, 0.60)),
        grain=GrainSpec(9.0, 13.0, 13.0, 13.0, clump_gain=1.00, fog_grain=0.24),
        mtf=MTFSpec(52.0, 52.0, 52.0, adjacency=0.08),
        # Panchromatic, hotter in blue and red than video luma. This is why
        # B&W film darkens red lips and lightens a blue sky.
        spectral_weights=(0.34, 0.46, 0.20),
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
        curves=_mono(ToneCurve(0.14, 0.690, -1.44, 0.30, 1.98, 0.50)),
        grain=GrainSpec(11.5, 15.5, 15.5, 15.5, clump_gain=1.30, fog_grain=0.28,
                        anisotropy=1.03),
        mtf=MTFSpec(42.0, 42.0, 42.0, adjacency=0.05),
        spectral_weights=(0.30, 0.48, 0.22),
        misregistration_um=0.0,
        features=Feature.UNEVEN_EMULSION,
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
        balance_kelvin=3200,
        curves=_mono(ToneCurve(0.13, 0.620, -1.70, 0.32, 2.26, 0.58)),
        grain=GrainSpec(8.0, 12.0, 12.0, 12.0, clump_gain=1.05, fog_grain=0.22),
        mtf=MTFSpec(56.0, 56.0, 56.0, adjacency=0.09),
        spectral_weights=(0.32, 0.47, 0.21),
        misregistration_um=0.0,
        features=Feature.NONE,
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
        curves=_mono(ToneCurve(0.16, 0.860, -1.18, 0.24, 1.52, 0.34)),
        grain=GrainSpec(11.5, 15.0, 15.0, 15.0, clump_gain=1.55, fog_grain=0.32,
                        anisotropy=1.10),
        mtf=MTFSpec(34.0, 34.0, 34.0, adjacency=0.03),
        spectral_weights=(0.26, 0.50, 0.24),
        misregistration_um=0.0,
        features=Feature.UNEVEN_EMULSION | Feature.ORTHO_RESPONSE,
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
        exposure_index=3200,
        balance_kelvin=5500,
        curves=_mono(ToneCurve(0.18, 0.600, -1.56, 0.36, 2.10, 0.60)),
        grain=GrainSpec(16.0, 22.0, 22.0, 22.0, clump_gain=0.45, fog_grain=0.34),
        mtf=MTFSpec(30.0, 30.0, 30.0, adjacency=0.06),
        spectral_weights=(0.33, 0.46, 0.21),
        misregistration_um=0.0,
        features=Feature.TABULAR_GRAIN,
    ),
    # -----------------------------------------------------------------------
    # Black and white reversal
    # -----------------------------------------------------------------------
    FilmProfile(
        name="KODAK_TRI_X_REVERSAL_200",
        # The catalogue number here is worth a note: Tri-X Reversal ships as
        # 7266 in 16 mm. The requested "5266" designation does not correspond
        # to a Tri-X reversal product as far as I can establish, so this
        # profile is built as the 7266 emulsion. Correct the number if you have
        # a datasheet that says otherwise.
        aliases=("7266", "5266", "tri-x reversal", "tri x reversal", "trix"),
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
        balance_kelvin=3200,
        curves=_mono(ToneCurve(0.16, 1.50, -0.86, 0.22, 1.04, 0.34)),
        grain=GrainSpec(10.0, 14.0, 14.0, 14.0, clump_gain=1.20, fog_grain=0.26),
        mtf=MTFSpec(46.0, 46.0, 46.0, adjacency=0.07),
        spectral_weights=(0.32, 0.47, 0.21),
        misregistration_um=0.0,
        features=Feature.NONE,
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
        features=Feature.HALATION | Feature.THREE_STRIP | Feature.UNEVEN_EMULSION,
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
        balance_kelvin=3200,
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
        features=Feature.HALATION | Feature.UNEVEN_EMULSION | Feature.NITRATE_BASE,
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
)


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


def validate_all() -> None:
    """Validate every profile and print stock. Called by the renderer on start."""
    for p in FILM_PROFILES:
        p.validate()
        if not p.is_reversal:
            get_print_stock(p.default_print)
    for s in PRINT_STOCKS:
        s.validate()


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
