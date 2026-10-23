"""Control enumerations and numeric ranges for the Python reference.

GENERATED FILE -- DO NOT EDIT.

Emitted by gen_control_enums.py from AlgoControlEnums.hpp, which is the single
authoritative definition shared by the scalar build, the AVX2 build and
this reference. Edit the header and regenerate; a local edit here will be
overwritten and the build gate will fail before that happens.
"""

from __future__ import annotations

from enum import IntEnum


class FilmFormatCtrl(IntEnum):
    """Gate geometry. Values match FORMAT_GEOM keys through KEY below."""

    eFILM_FORMAT_8_MM = 0
    eFILM_FORMAT_SUPER_8 = 1
    eFILM_FORMAT_16_MM = 2
    eFILM_FORMAT_SUPER_16 = 3
    eFILM_FORMAT_ACADEMY_35 = 4
    eFILM_FORMAT_ANAMORPHIC_35 = 5
    eFILM_FORMAT_SUPER_35 = 6
    eFILM_FORMAT_TECHNI_35 = 7
    eFILM_FORMAT_FF_35 = 8
    eFILM_FORMAT_MEDIUM_645 = 9
    eFILM_FORMAT_IMAX_15 = 10
    eFILM_FORMAT_POLAROID_SX_70 = 11
    eFILM_FORMAT_POLAROID_PACK = 12
    eFILM_FORMAT_LARGE_4x5 = 13
    eFILM_FORMAT_TOTAL_FORMATS = 14

    @property
    def key(self) -> str:
        """The FORMAT_GEOM key, or "" for a value with no geometry."""
        return FILM_FORMAT_KEY.get(int(self), "")

    @property
    def label(self) -> str:
        return FILM_FORMAT_LABEL.get(int(self), "")


class PrintStockCtrl(IntEnum):
    """Positive stock. eSTOCKS_OWN is a sentinel, not a stock."""

    eSTOCKS_OWN = 0
    eSCAN_DI = 1
    eKODAK_2383_RELEASE = 2
    eDUPE_FINE_GRAIN = 3
    eTECHNICOLOR_IB = 4
    eKODAK_5302 = 5
    eTASMA_POSITIVE_28 = 6
    eTSP_1_POSITIVE = 7
    eTSP_3_POSITIVE = 8
    eTSP_6_POSITIVE = 9
    eEASTMANCOLOR_5382_1953 = 10
    eKODAK_VISION3_DI_2254 = 11
    ePRINT_STOCK_TOTAL = 12

    @property
    def key(self) -> str:
        """The PRINT_STOCKS name, or "" for the sentinel."""
        return PRINT_STOCK_KEY.get(int(self), "")

    @property
    def label(self) -> str:
        return PRINT_STOCK_LABEL.get(int(self), "")


class ProcessVariantCtrl(IntEnum):
    """A pushed or pulled STEP, the same on every stock (2026-10-11). eAS_SHIPPED is the absence of a
    selection and TOTAL_PROCESSES is a count; neither is selectable."""

    eAS_SHIPPED = 0
    ePULL_2 = 1
    ePULL_1 = 2
    ePUSH_1 = 3
    ePUSH_2 = 4
    ePUSH_3 = 5
    TOTAL_PROCESSES = 6

    @property
    def key(self) -> str:
        """The film::ProcessVariant.variant_id, or "" for the sentinel."""
        return PROCESS_VARIANT_KEY.get(int(self), "")

    @property
    def label(self) -> str:
        return PROCESS_VARIANT_LABEL.get(int(self), "As shipped")


class GrainTemporalModeCtrl(IntEnum):
    """How stage 11 grain moves between frames (still | motion | frozen)."""

    eGRAIN_TEMPORAL_STILL = 0
    eGRAIN_TEMPORAL_MOTION = 1
    eGRAIN_TEMPORAL_FROZEN = 2
    eGRAIN_TEMPORAL_TOTAL = 3

    @property
    def key(self) -> str:
        """The film_sim grain_temporal_mode string."""
        return GRAIN_TEMPORAL_MODE_KEY.get(int(self), "")

    @property
    def label(self) -> str:
        return GRAIN_TEMPORAL_MODE_LABEL.get(int(self), "")


class DeveloperCtrl(IntEnum):
    """The Developer list box (2026-10-16f): one developer, the same on every
    stock. eAS_SHIPPED = the stored profile; TOTAL_DEVELOPERS is a count."""

    eAS_SHIPPED = 0
    eATOMAL = 1
    eD76 = 2
    eFX39 = 3
    eMICROPHEN = 4
    eRODINAL = 5
    eSD2525 = 6
    eSHADOWMAX_19Z = 7
    eWEHNER_ALPHA = 8
    eXTOL = 9
    TOTAL_DEVELOPERS = 10

    @property
    def key(self) -> str:
        """The film DeveloperGrainPoint.developer name, or "" for eAS_SHIPPED."""
        return DEVELOPER_KEY.get(int(self), "")

    @property
    def label(self) -> str:
        return DEVELOPER_LABEL.get(int(self), "As shipped")


#: dupeStock draws on the same catalogue as printStock.
DupeStockCtrl = PrintStockCtrl

FILM_FORMAT_KEY: dict[int, str] = {
    0: '8mm',
    1: 'super8',
    2: '16mm',
    3: 'super16',
    4: 'academy35',
    5: 'anamorphic35',
    6: 'super35',
    7: 'techni35',
    8: 'ff35',
    9: 'medium645',
    10: 'imax15',
    11: 'polaroid_sx70',
    12: 'polaroid_pack',
    13: 'large4x5',
}

FILM_FORMAT_LABEL: dict[int, str] = {
    0: '8 mm',
    1: 'Super 8',
    2: '16 mm',
    3: 'Super 16',
    4: 'Academy 35',
    5: 'Anamorphic 35',
    6: 'Super 35',
    7: 'Techni 35',
    8: 'Full Frame 35',
    9: 'Medium 645',
    10: 'IMAX 15-perf',
    11: 'Polaroid SX-70',
    12: 'Polaroid pack',
    13: 'Large format 4x5',
}

PRINT_STOCK_KEY: dict[int, str] = {
    0: '',
    1: 'SCAN_DI',
    2: 'KODAK_2383_RELEASE',
    3: 'DUPE_FINE_GRAIN',
    4: 'TECHNICOLOR_IB',
    5: 'KODAK_5302',
    6: 'TASMA_POSITIVE_28',
    7: 'TSP_1_POSITIVE',
    8: 'TSP_3_POSITIVE',
    9: 'TSP_6_POSITIVE',
    10: 'EASTMANCOLOR_5382_1953',
    11: 'KODAK_VISION3_DI_2254',
}

PRINT_STOCK_LABEL: dict[int, str] = {
    0: "Stock's own",
    1: 'Scan DI',
    2: 'Kodak 2383 Release',
    3: 'Dupe Fine Grain',
    4: 'Technicolor IB',
    5: 'Kodak 5302',
    6: 'Tasma Positive 2.8',
    7: 'TSP-1 Positive',
    8: 'TSP-3 Positive',
    9: 'TSP-6 Positive',
    10: 'Eastman Color 5382 (1953)',
    11: 'Kodak Vision3 DI 2254',
}

PROCESS_VARIANT_KEY: dict[int, str] = {
    0: '',
    1: 'PULL_2',
    2: 'PULL_1',
    3: 'PUSH_1',
    4: 'PUSH_2',
    5: 'PUSH_3',
}

PROCESS_VARIANT_LABEL: dict[int, str] = {
    0: 'As shipped',
    1: 'Pull -2',
    2: 'Pull -1',
    3: 'Push +1',
    4: 'Push +2',
    5: 'Push +3',
}

#: A value stored before 2026-10-11 (index, 0 ... 48) -> the step it now
#: selects; see ProcessVariantCtrlMigrate in the header.
PROCESS_VARIANT_MIGRATE: tuple[int, ...] = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0,
    3, 4, 0, 0, 3, 0, 0, 0, 0, 3, 0, 3,
    4, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0, 3,
    4, 0, 0, 2, 3, 4, 5, 0, 0, 0, 0, 0,
    0,
)

DEVELOPER_KEY: dict[int, str] = {
    0: '',
    1: 'Atomal',
    2: 'D76',
    3: 'FX39',
    4: 'Microphen',
    5: 'Rodinal',
    6: 'SD2525',
    7: 'Shadowmax 19/z',
    8: 'Wehner/Alpha',
    9: 'XTOL',
}

DEVELOPER_LABEL: dict[int, str] = {
    0: 'As shipped',
    1: 'Atomal',
    2: 'D-76',
    3: 'FX-39',
    4: 'Microphen',
    5: 'Rodinal',
    6: 'SD 25/25',
    7: 'Shadowmax 19/z',
    8: 'Wehner/Alpha',
    9: 'XTOL',
}

GRAIN_TEMPORAL_MODE_KEY: dict[int, str] = {
    0: 'still',
    1: 'motion',
    2: 'frozen',
}

GRAIN_TEMPORAL_MODE_LABEL: dict[int, str] = {
    0: 'Still',
    1: 'Motion',
    2: 'Frozen',
}


# ---------------------------------------------------------------------------
# Numeric control metadata
# ---------------------------------------------------------------------------
# Transcribed by the header from the per-field documentation in
# AlgoControl.hpp. Most maxima are ADVISORY -- the engine does not clamp
# them -- so they describe where the model is meaningful, not where it is
# guarded. See the header for which bounds are enforced and at which stage.

ProcessVariantCtrlCount = 6
ProcessVariantCtrlMigrateCount = 49
BatchPositionMin = -1.0
BatchPositionMax = 1.0
BatchPositionDef = 0.0
BatchPositionStep = 0.05
ExposureStopsMin = -4.0
ExposureStopsMax = 4.0
ExposureStopsDef = 0.0
ExposureStopsStep = 0.1
ExposureTimeSOff = 0.0
ExposureTimeSMin = 1e-05
ExposureTimeSMax = 3600.0
ExposureTimeSDef = ExposureTimeSOff
DevelopmentMinutesSentinel = -1.0
DevelopmentMinutesMin = 1.9
DevelopmentMinutesMax = 36.0
DevelopmentMinutesDef = DevelopmentMinutesSentinel
DevelopmentMinutesStep = 0.1
DevelopmentCelsiusSentinel = -1.0
DevelopmentCelsiusMin = 18.0
DevelopmentCelsiusMax = 24.0
DevelopmentCelsiusDef = DevelopmentCelsiusSentinel
DevelopmentCelsiusStep = 0.5
DeveloperCtrlCount = 10
StorageYearsOff = 0.0
StorageYearsMin = 0.0
StorageYearsMax = 100.0
StorageYearsDef = StorageYearsOff
StorageYearsStep = 0.5
StorageCelsiusMin = -26.0
StorageCelsiusMax = 30.0
StorageCelsiusDef = 24.0
StorageCelsiusStep = 0.5
FilmAgeMonthsOff = 0.0
FilmAgeMonthsMin = 0.0
FilmAgeMonthsMax = 600.0
FilmAgeMonthsDef = FilmAgeMonthsOff
FilmAgeMonthsStep = 4.0
FilmAgeMin = 0.0
FilmAgeMax = 600.0
FilmAgeDef = FilmAgeMin
FilmAgeStep = 4.0
GenerationsMin = 0
GenerationsMax = 4
GenerationsDef = 0
GenerationsStep = 1
ScannerSpecularMin = 0.0
ScannerSpecularMax = 1.0
ScannerSpecularDef = 0.853
ScannerFixedPatternDef = 0.0
ScannerFixedPatternMin = 0.0
ScannerFixedPatternMax = 1.0
ScannerSpecularStep = 0.01
SceneKelvinMin = 2000.0
SceneKelvinMax = 12000.0
SceneKelvinDef = 5500.0
SceneKelvinStep = 50.0
WbStrengthMin = 0.0
WbStrengthMax = 1.0
WbStrengthDef = 0.0
WbStrengthStep = 0.01
GreyTargetMin = 0.02
GreyTargetMax = 0.6
GreyTargetDef = 0.18
GreyTargetStep = 0.005
BlackPointStretchMin = 0.0
BlackPointStretchMax = 1.0
BlackPointStretchDef = 1.0
BlackPointStretchStep = 0.01
GrainScaleMin = 0.0
GrainScaleMax = 4.0
GrainScaleDef = 1.0
GrainScaleStep = 0.01
HalationScaleMin = 0.0
HalationScaleMax = 4.0
HalationScaleDef = 1.0
HalationScaleStep = 0.01
CouplerScaleMin = 0.0
CouplerScaleMax = 3.0
CouplerScaleDef = 1.0
CouplerScaleStep = 0.01
MisregScaleMin = 0.0
MisregScaleMax = 4.0
MisregScaleDef = 1.0
MisregScaleStep = 0.01
CoatingScaleMin = 0.0
CoatingScaleMax = 3.0
CoatingScaleDef = 1.0
CoatingScaleStep = 0.01
FlareSentinel = -1.0
FlareMin = 0.0
FlareMax = 0.5
FlareDef = FlareSentinel
FlareStep = 0.005
VignetteSentinel = -1.0
VignetteMin = 0.0
VignetteMax = 4.0
VignetteDef = VignetteSentinel
VignetteStep = 0.05
FrameRateMin = 1.0
FrameRateMax = 240.0
FrameRateDef = 24.0
FrameRateStep = 0.001
FrameIndexDef = 0
FrameIndexStep = 1
SeedDef = 12345
SeedStep = 1
PrintGrainDef = True
ReseauDef = True
FilmDamageEnabledDef = False
SpectralTakingDef = False
SpectralUpsamplingDef = False
AimExposureDef = False
GrainTemporalIntegrationS = 0.2
GrainTemporalMaxFrames = 8.0
DamageStrengthMin = 0.0
DamageStrengthMax = 4.0
DamageStrengthDef = 1.0
DamageStrengthStep = 0.01
DamageSeedDef = 20250803
DamageSeedStep = 1
DustLevelMin = 0.0
DustLevelMax = 4.0
DustLevelDef = 1.0
DustLevelStep = 0.01
DebrisLevelMin = 0.0
DebrisLevelMax = 4.0
DebrisLevelDef = 1.0
DebrisLevelStep = 0.01
FibreLevelMin = 0.0
FibreLevelMax = 4.0
FibreLevelDef = 1.0
FibreLevelStep = 0.01
DirtClumpingMin = 0.0
DirtClumpingMax = 2.0
DirtClumpingDef = 1.0
DirtClumpingStep = 0.01
ScratchTransportMin = 0.0
ScratchTransportMax = 120.0
ScratchTransportDef = 4.8
ScratchTransportStep = 0.1
ScratchTransportFloor = 4.8
ScratchHandlingMin = 0.0
ScratchHandlingMax = 4.0
ScratchHandlingDef = 0.3
ScratchHandlingStep = 0.01
GateDirtMin = 0.0
GateDirtMax = 2.0
GateDirtDef = 0.6
GateDirtStep = 0.01
WeaveAmountMin = 0.0
WeaveAmountMax = 2.0
WeaveAmountDef = 0.5
WeaveAmountStep = 0.01
DamageEventsMin = 0.0
DamageEventsMax = 2.0
DamageEventsDef = 0.2
DamageEventsStep = 0.01
ProcessingQualityMin = 0.0
ProcessingQualityMax = 1.0
ProcessingQualityDef = 0.3
ProcessingQualityStep = 0.01
DryingMarksMin = 0.0
DryingMarksMax = 2.0
DryingMarksDef = 0.25
DryingMarksStep = 0.01
StorageSeverityMin = 0.0
StorageSeverityMax = 1.0
StorageSeverityDef = 0.2
StorageSeverityStep = 0.01
ColourVeilMin = 0.0
ColourVeilMax = 2.0
ColourVeilDef = 0.15
ColourVeilStep = 0.01
FlickerStopsMin = 0.0
FlickerStopsMax = 0.5
FlickerStopsDef = 0.15
FlickerStopsStep = 0.01
FlickerStopsStock = -1.0
ScannerArtifactsMin = 0.0
ScannerArtifactsMax = 2.0
ScannerArtifactsDef = 0.2
ScannerArtifactsStep = 0.01
DefectOpticsDef = False
ScratchLifetimeMin = 0.0
ScratchLifetimeMax = 60.0
ScratchLifetimeDef = 0.0
ScratchLifetimeStep = 0.1
DustLifetimeMin = 0.5
DustLifetimeMax = 600.0
DustLifetimeDef = 42.0
DustLifetimeStep = 0.5
DefectVariabilityMin = 0.0
DefectVariabilityMax = 1.0
DefectVariabilityDef = 0.5
DefectVariabilityStep = 0.01
DefectVisibilityMin = 0.0
DefectVisibilityMax = 3.0
DefectVisibilityDef = 1.0
DefectVisibilityStep = 0.01
SpliceAndTearsEventstMin = 0.0
SpliceAndTearsEventstMax = 2.0
SpliceAndTearsEventstDef = 0.2
SpliceAndTearsEventsttep = 0.01


def film_format_key(value) -> str:
    """Resolve a control value to a FORMAT_GEOM key.

    Accepts the enumerator, its integer value, or a bare key string, so
    existing callers that pass a string keep working. An unrecognised
    value yields "", which every caller already treats as "use the
    stock's own default" -- the same degradation the engine applies.
    """
    if isinstance(value, str):
        return value
    try:
        return FILM_FORMAT_KEY.get(int(value), "")
    except (TypeError, ValueError):
        return ""


def process_variant_key(value) -> str:
    """Resolve a control value to a film::ProcessVariant.variant_id.

    Accepts the enumerator, its integer value, or a bare key string.
    An unrecognised value yields "", which every caller treats as "as
    shipped" -- the same degradation both engines apply, and
    deliberately not a clamp into range.
    """
    if isinstance(value, str):
        return value
    try:
        return PROCESS_VARIANT_KEY.get(int(value), "")
    except (TypeError, ValueError):
        return ""


def process_variant_from_legacy(old) -> int:
    """A value stored before 2026-10-11 as the step it now selects;
    out-of-range input answers 0 (as shipped), as in the header."""
    try:
        o = int(old)
    except (TypeError, ValueError):
        return 0
    return PROCESS_VARIANT_MIGRATE[o] if 0 <= o < len(PROCESS_VARIANT_MIGRATE) else 0


def grain_temporal_mode_key(value) -> str:
    """Resolve a control value to film_sim's grain_temporal_mode string.

    Accepts the enumerator, its integer value, or the string itself. An
    unrecognised value yields "still", the default both engines clamp to.
    """
    if isinstance(value, str):
        return value
    try:
        return GRAIN_TEMPORAL_MODE_KEY.get(int(value), "still")
    except (TypeError, ValueError):
        return "still"


def print_stock_key(value) -> str:
    """Resolve a control value to a PRINT_STOCKS name. See above."""
    if isinstance(value, str):
        return value
    try:
        return PRINT_STOCK_KEY.get(int(value), "")
    except (TypeError, ValueError):
        return ""
