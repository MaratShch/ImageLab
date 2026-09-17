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
    """A DEVELOPMENT, globally. eAS_SHIPPED is the absence of a
    selection and TOTAL_PROCESSES is a count; neither is selectable."""

    eAS_SHIPPED = -1
    eAGFA_REFINAL = 0
    eAGFA_RODINAL_1_25 = 1
    eAGFA_RODINAL_1_50 = 2
    eAGFA_RODINAL_SPECIAL = 3
    eAGFA_STUDIONAL_LIQUID = 4
    eCINESTILL_C41_AS_SHIPPED = 5
    eCINESTILL_CS2_TWO_BATH = 6
    eCINESTILL_ECN2_NATIVE = 7
    eGEVACHROME_23DIN_160ASA = 8
    eGEVACHROME_26DIN_320ASA = 9
    ePORTRA800_EI800 = 10
    ePORTRA800_EI1600_PUSH1 = 11
    ePORTRA800_EI3200_PUSH2 = 12
    eULTRA400UC_EI400_E4035 = 13
    eULTRA400UC_EI400_E190 = 14
    eULTRA400UC_EI800_E4035 = 15
    eULTRA400UC_EI800_E190 = 16
    eANSCOCHROME_A_14MIN_EI80 = 17
    eANSCOCHROME_B_16MIN_EI100 = 18
    eANSCOCHROME_C_19MIN_EI150 = 19
    eANSCOCHROME_D_22MIN_EI200 = 20
    TOTAL_PROCESSES = 21

    @property
    def key(self) -> str:
        """The film::ProcessVariant.variant_id, or "" for the sentinel."""
        return PROCESS_VARIANT_KEY.get(int(self), "")

    @property
    def label(self) -> str:
        return PROCESS_VARIANT_LABEL.get(int(self), "As shipped")


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
    0: 'AGFA_REFINAL',
    1: 'AGFA_RODINAL_1_25',
    2: 'AGFA_RODINAL_1_50',
    3: 'AGFA_RODINAL_SPECIAL',
    4: 'AGFA_STUDIONAL_LIQUID',
    5: 'CINESTILL_C41_AS_SHIPPED',
    6: 'CINESTILL_CS2_TWO_BATH',
    7: 'CINESTILL_ECN2_NATIVE',
    8: 'GEVACHROME_23DIN_160ASA',
    9: 'GEVACHROME_26DIN_320ASA',
    10: 'PORTRA800_EI800',
    11: 'PORTRA800_EI1600_PUSH1',
    12: 'PORTRA800_EI3200_PUSH2',
    13: 'ULTRA400UC_EI400_E4035',
    14: 'ULTRA400UC_EI400_E190',
    15: 'ULTRA400UC_EI800_E4035',
    16: 'ULTRA400UC_EI800_E190',
    17: 'ANSCOCHROME_A_14MIN_EI80',
    18: 'ANSCOCHROME_B_16MIN_EI100',
    19: 'ANSCOCHROME_C_19MIN_EI150',
    20: 'ANSCOCHROME_D_22MIN_EI200',
}

PROCESS_VARIANT_LABEL: dict[int, str] = {
    0: 'REFINAL',
    1: 'RODINAL 1+25',
    2: 'RODINAL 1+50',
    3: 'RODINAL SPECIAL',
    4: 'STUDIONAL LIQUID',
    5: 'C-41 cross-process, as shipped',
    6: 'Cs2 two-bath kit',
    7: "ECN-2, the base stock's native process",
    8: '23 DIN / 160 ASA (box speed)',
    9: '26 DIN / 320 ASA (push 1)',
    10: 'EI 800 (box speed)',
    11: 'EI 1600 (Push 1)',
    12: 'EI 3200 (Push 2)',
    13: 'EI 400 (box speed) -- E-4035',
    14: 'EI 400 (box speed) -- E-190 (2003)',
    15: 'EI 800 (Push 1) -- E-4035',
    16: 'EI 800 (Push 1) -- E-190 (2003)',
    17: 'A -- 14 min first developer, EI 80',
    18: 'B -- 16 min first developer, EI 100',
    19: 'C -- 19 min first developer, EI 150',
    20: 'D -- 22 min first developer, EI 200',
}


# ---------------------------------------------------------------------------
# Numeric control metadata
# ---------------------------------------------------------------------------
# Transcribed by the header from the per-field documentation in
# AlgoControl.hpp. Most maxima are ADVISORY -- the engine does not clamp
# them -- so they describe where the model is meaningful, not where it is
# guarded. See the header for which bounds are enforced and at which stage.

ProcessVariantCtrlCount = 21
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
StorageYearsOff = 0.0
StorageYearsMin = 0.0
StorageYearsMax = 100.0
StorageYearsDef = StorageYearsOff
StorageYearsStep = 0.5
GenerationsMin = 0
GenerationsMax = 4
GenerationsDef = 0
GenerationsStep = 1
ScannerSpecularMin = 0.0
ScannerSpecularMax = 1.0
ScannerSpecularDef = 0.853
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
ScratchTransportMax = 4.0
ScratchTransportDef = 0.4
ScratchTransportStep = 0.01
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
ScannerArtifactsMin = 0.0
ScannerArtifactsMax = 2.0
ScannerArtifactsDef = 0.2
ScannerArtifactsStep = 0.01


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


def print_stock_key(value) -> str:
    """Resolve a control value to a PRINT_STOCKS name. See above."""
    if isinstance(value, str):
        return value
    try:
        return PRINT_STOCK_KEY.get(int(value), "")
    except (TypeError, ValueError):
        return ""
