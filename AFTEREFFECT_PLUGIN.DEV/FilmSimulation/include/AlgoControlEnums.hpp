#ifndef __IMAGE_LAB2_ALGO_CONTROL_ENUMERATORS__
#define __IMAGE_LAB2_ALGO_CONTROL_ENUMERATORS__

#include <cstdint>

/**
 * @file AlgoControlEnums.hpp
 * @brief Selectable control values and numeric control metadata.
 *
 * This header is the single authoritative definition of every enumerated
 * control value and every numeric control range in the film simulation. It is
 * included by AlgoControl.hpp, so the scalar build, the AVX2 build and any
 * host plug-in compile against the same definitions rather than agreeing by
 * convention. Nothing here is duplicated in either implementation.
 *
 * Two rules govern the contents.
 *
 * The enumerations mirror the film database exactly. A film format offered
 * here must exist in FORMAT_GEOM; a print stock offered here must exist in
 * PRINT_STOCKS. A user interface cannot present a choice the engine will
 * silently discard, so the ordering below is fixed and the string tables are
 * index aligned with the enumerators.
 *
 * The numeric constants are transcribed from the per-field documentation in
 * AlgoControl.hpp, not invented here. Where that documentation distinguishes
 * an enforced bound from an advisory one the distinction is preserved, because
 * it tells the host which limits it must apply itself.
 */


// ---------------------------------------------------------------------------
//  Film format
// ---------------------------------------------------------------------------
//  Selects the gate geometry: image width and height on the negative, the
//  anamorphic squeeze, and the perforation pitch that sets how far the web
//  advances between frames. Grain size, halation radius, modulation transfer
//  and channel registration are all expressed in micrometres on the negative
//  and converted to pixels through this geometry, which is what makes a
//  profile resolution independent.
//
//  The fourteen entries correspond one for one with the keys of FORMAT_GEOM in
//  the generated database. Four of them (super 8, super 16, anamorphic 35 and
//  IMAX 15) are not the default format of any stock currently in the database,
//  but they carry full published geometry and remain selectable against any
//  stock. That is deliberate: the format describes how the film was exposed,
//  not what the emulsion is.
enum class FilmFormatCtrl : int32_t
{
    eFILM_FORMAT_8_MM = 0,
    eFILM_FORMAT_SUPER_8,
    eFILM_FORMAT_16_MM,
    eFILM_FORMAT_SUPER_16,
    eFILM_FORMAT_ACADEMY_35,
    eFILM_FORMAT_ANAMORPHIC_35,
    eFILM_FORMAT_SUPER_35,
    eFILM_FORMAT_TECHNI_35,
    eFILM_FORMAT_FF_35,
    eFILM_FORMAT_MEDIUM_645,
    eFILM_FORMAT_IMAX_15,
    eFILM_FORMAT_POLAROID_SX_70,
    eFILM_FORMAT_POLAROID_PACK,
    eFILM_FORMAT_LARGE_4x5,
    eFILM_FORMAT_TOTAL_FORMATS
};

// Display names, pipe separated, index aligned with FilmFormatCtrl.
constexpr char FilmFormatCtrlStr[] =
    "8 mm|"
    "Super 8|"
    "16 mm|"
    "Super 16|"
    "Academy 35|"
    "Anamorphic 35|"
    "Super 35|"
    "Techni 35|"
    "Full Frame 35|"
    "Medium 645|"
    "IMAX 15-perf|"
    "Polaroid SX-70|"
    "Polaroid pack|"
    "Large format 4x5";

// Database keys, index aligned with FilmFormatCtrl. These are the exact
// FORMAT_GEOM keys. The engine resolves a format through this table rather
// than through a display name, so the two can change independently.
constexpr const char* const FilmFormatCtrlKey[] =
{
    "8mm", "super8", "16mm", "super16", "academy35", "anamorphic35",
    "super35", "techni35", "ff35", "medium645", "imax15",
    "polaroid_sx70", "polaroid_pack", "large4x5"
};

// Sentinel meaning "use the selected stock's own default_format".
constexpr FilmFormatCtrl FilmFormatCtrlUseStockDefault =
    static_cast<FilmFormatCtrl>(-1);

constexpr FilmFormatCtrl FilmFormatCtrlDef =
    FilmFormatCtrl::eFILM_FORMAT_SUPER_35;


// ---------------------------------------------------------------------------
//  Print stock
// ---------------------------------------------------------------------------
//  Selects the positive stock the developed negative is printed onto, which
//  supplies the print characteristic curve, the print grain and the print
//  modulation transfer. Reversal originals bypass the print chain entirely.
//
//  eSTOCKS_OWN is a sentinel, not a stock: it means "use the selected film's
//  own default_print". The remaining eleven are the entire contents of
//  PRINT_STOCKS in the generated database, in database order.
//
//  The Soviet positive stocks are spelled TSP to match the database and the
//  source literature. An earlier draft of this header wrote them CP; the two
//  name the same three stocks and the database spelling is authoritative.
enum class PrintStockCtrl : int32_t
{
    eSTOCKS_OWN = 0,
    eSCAN_DI,
    eKODAK_2383_RELEASE,
    eDUPE_FINE_GRAIN,
    eTECHNICOLOR_IB,
    eKODAK_5302,
    eTASMA_POSITIVE_28,
    eTSP_1_POSITIVE,
    eTSP_3_POSITIVE,
    eTSP_6_POSITIVE,
    eEASTMANCOLOR_5382_1953,
    eKODAK_VISION3_DI_2254,
    ePRINT_STOCK_TOTAL
};

// Display names, pipe separated, index aligned with PrintStockCtrl.
constexpr char PrintStockCtrlStr[] =
    "Stock's own|"
    "Scan DI|"
    "Kodak 2383 Release|"
    "Dupe Fine Grain|"
    "Technicolor IB|"
    "Kodak 5302|"
    "Tasma Positive 2.8|"
    "TSP-1 Positive|"
    "TSP-3 Positive|"
    "TSP-6 Positive|"
    "Eastman Color 5382 (1953)|"
    "Kodak Vision3 DI 2254";

// Database keys, index aligned with PrintStockCtrl. Index 0 is the sentinel
// and resolves to the stock's own default_print, so it carries an empty key.
constexpr const char* const PrintStockCtrlKey[] =
{
    "",
    "SCAN_DI", "KODAK_2383_RELEASE", "DUPE_FINE_GRAIN", "TECHNICOLOR_IB",
    "KODAK_5302", "TASMA_POSITIVE_28", "TSP_1_POSITIVE", "TSP_3_POSITIVE",
    "TSP_6_POSITIVE", "EASTMANCOLOR_5382_1953", "KODAK_VISION3_DI_2254"
};

constexpr PrintStockCtrl PrintStockCtrlDef = PrintStockCtrl::eSTOCKS_OWN;


// ---------------------------------------------------------------------------
//  Enumerator to database key
// ---------------------------------------------------------------------------
//  These two are the only conversion the engine performs on a selection, and
//  they exist so that no call site has to know the table layout.
//
//  AN OUT-OF-RANGE VALUE RETURNS THE EMPTY KEY RATHER THAN CLAMPING, and that
//  is a deliberate preservation of the behaviour these controls had when they
//  were character arrays: an empty key means "use the stock's own default",
//  so a stale preset from an older database degrades to the stock's intended
//  choice instead of silently selecting whichever entry happens to sit at the
//  clamped index. Degrading to the stock default is recoverable and visible;
//  selecting the wrong stock is neither.

constexpr const char* AlgoFilmFormatKey(const FilmFormatCtrl v) noexcept
{
    return (v >= FilmFormatCtrl::eFILM_FORMAT_8_MM
            && v < FilmFormatCtrl::eFILM_FORMAT_TOTAL_FORMATS)
        ? FilmFormatCtrlKey[static_cast<int32_t>(v)]
        : "";
}

constexpr const char* AlgoPrintStockKey(const PrintStockCtrl v) noexcept
{
    return (v >= PrintStockCtrl::eSTOCKS_OWN
            && v < PrintStockCtrl::ePRINT_STOCK_TOTAL)
        ? PrintStockCtrlKey[static_cast<int32_t>(v)]
        : "";
}


// ---------------------------------------------------------------------------
//  Duplication stock
// ---------------------------------------------------------------------------
//  The intermediate stock used for each duplication generation. It draws on
//  the same catalogue as the print stock, so the type is shared rather than
//  duplicated. Its sentinel behaves differently: where printStock's sentinel
//  means "the stock's own default print", dupeStock's means "whatever the
//  print stock resolved to".
using DupeStockCtrl = PrintStockCtrl;
constexpr const char* const DupeStockCtrlStr = PrintStockCtrlStr;

constexpr DupeStockCtrl DupeStockCtrlDef = DupeStockCtrl::eDUPE_FINE_GRAIN;


// ---------------------------------------------------------------------------
//  Process variant -- the strongly typed selection
// ---------------------------------------------------------------------------
//  Selects an alternative development for the chosen emulsion: a push, a pull,
//  a cross-process or an alternate kit, as the manufacturer plotted it.
//
//  \warning THIS WAS AN int32_t INDEX UNTIL 2026-09-17 AND THE CHANGE IS NOT
//  COSMETIC. The host used to select a development by its POSITION in the
//  chosen stock's own process_variants vector. Position is not an identity.
//  Eight stocks offer between two and five developments each, so the stored
//  value 2 meant "RODINAL 1+50" on an AGFAPAN, "ECN-2, the base stock's native
//  process" on CINESTILL 800T and "EI 3200 (Push 2)" on PORTRA 800 - one saved
//  project file, three films, three meanings, every one of them in range and
//  therefore undetectable. Inserting a variant, or re-ordering two, silently
//  changed what every saved project selected.
//
//  The control is now this enumeration. It is GLOBAL: each development has one
//  value that means the same thing whatever stock is loaded. Which stock
//  OFFERS which remains per stock and is answered by searching that stock's
//  process_variants for the id - see AlgoProcessVariant.hpp.
//
//  \warning AND AN UNRECOGNISED VALUE RENDERS AS SHIPPED RATHER THAN BEING
//  CLAMPED. A project saved against a database offering a variant this one
//  does not takes the same inert path an unselected control takes. Clamping
//  would render a DIFFERENT development and call it the one that was asked
//  for, which is worse than rendering none.
//
//  THE ORDER IS FIXED AND THE TABLES BELOW ARE INDEX ALIGNED WITH IT. Every
//  entry corresponds one for one with an entry of _PROCESS_VARIANT_IDS in the
//  generated database, in the same order, and verify.py refuses the build if
//  the two disagree in either direction.
enum class ProcessVariantCtrl : int32_t
{
    //  The absence of an ALTERNATE selection: the development the stored
    //  curves already represent. It IS an item in the list box, and it is the
    //  first one.
    //
    //  \warning IT WAS -1 UNTIL 2026-09-18e AND THE REBASE IS AN OWNER
    //  REQUIREMENT, not a tidy-up: every enumerated value in the AlgoControl
    //  structure starts from zero. The old comment argued that a negative
    //  value "can never collide with an enumerator" and sits "outside the
    //  [0, TOTAL_PROCESSES) range every validity test uses" -- both true, and
    //  both bought with a value that a plain unsigned control, a list-box
    //  index or a serialiser writing a size_t cannot represent at all.
    //
    //  NOTHING BEHAVIOURAL CHANGES, and the mechanism is the KEY table rather
    //  than the range test. `ProcessVariantCtrlKey[eAS_SHIPPED]` is the empty
    //  string, `AlgoResolveProcessVariant` looks that key up in the stock's
    //  own process_variants and finds nothing, and returns the shipped
    //  profile by reference -- which is exactly what the sentinel did when it
    //  failed the validity test instead. The static_assert below pins the
    //  empty key so the two can never come apart.
    //
    //  \warning A PROJECT SAVED BEFORE THE REBASE STORES DIFFERENT NUMBERS.
    //  -1 is no longer a legal value and is rejected by the range test, which
    //  takes the same inert as-shipped path; every other old value is one
    //  LESS than its new one. A host that migrates saved projects adds one.
    eAS_SHIPPED = 0,

    eAGFA_REFINAL = 1,
    eAGFA_RODINAL_1_25,
    eAGFA_RODINAL_1_50,
    eAGFA_RODINAL_SPECIAL,
    eAGFA_STUDIONAL_LIQUID,
    eCINESTILL_C41_AS_SHIPPED,
    eCINESTILL_CS2_TWO_BATH,
    eCINESTILL_ECN2_NATIVE,
    eGEVACHROME_23DIN_160ASA,
    eGEVACHROME_26DIN_320ASA,
    ePORTRA800_EI800,
    ePORTRA800_EI1600_PUSH1,
    ePORTRA800_EI3200_PUSH2,
    eULTRA400UC_EI400_E4035,
    eULTRA400UC_EI400_E190,
    eULTRA400UC_EI800_E4035,
    eULTRA400UC_EI800_E190,
    eANSCOCHROME_A_14MIN_EI80,
    eANSCOCHROME_B_16MIN_EI100,
    eANSCOCHROME_C_19MIN_EI150,
    eANSCOCHROME_D_22MIN_EI200,

    //  \warning TOTAL_PROCESSES IS A COUNT, NOT A PROCESS. It must stay last,
    //  it must never be offered in the list box, and it must never be stored
    //  as a selection. It exists so that the number of developments, the size
    //  of the display tables and the bound of every range check are all one
    //  fact written once - the compiler computes it from the enumerators, so
    //  adding a variant cannot leave a table behind.
    TOTAL_PROCESSES
};

//  The count as a plain integer, for array bounds and loop limits.
constexpr int32_t ProcessVariantCtrlCount =
    static_cast<int32_t>(ProcessVariantCtrl::TOTAL_PROCESSES);

constexpr ProcessVariantCtrl ProcessVariantCtrlDef =
    ProcessVariantCtrl::eAS_SHIPPED;

//  Display strings for the list box, pipe separated in enumerator order, in
//  the same form as every other control's string in this header.
//  \warning TOTAL_PROCESSES IS ABSENT BY DESIGN -- it is a count, not a
//  process. eAS_SHIPPED IS PRESENT AND IS THE FIRST ENTRY, as of the
//  2026-09-18e rebase: it is a legal selection meaning "the development the
//  stored curves already represent", so the list box offers it like any other
//  and the string, name and key tables are index aligned with the enumeration
//  from zero. ProcessVariantCtrlNoneStr below is kept, and is now the same
//  text as entry 0.
constexpr char ProcessVariantCtrlStr[] =
    "As shipped|"
    "REFINAL|"
    "RODINAL 1+25|"
    "RODINAL 1+50|"
    "RODINAL SPECIAL|"
    "STUDIONAL LIQUID|"
    "C-41 cross-process, as shipped|"
    "Cs2 two-bath kit|"
    "ECN-2, the base stock's native process|"
    "23 DIN / 160 ASA (box speed)|"
    "26 DIN / 320 ASA (push 1)|"
    "EI 800 (box speed)|"
    "EI 1600 (Push 1)|"
    "EI 3200 (Push 2)|"
    "EI 400 (box speed) -- E-4035|"
    "EI 400 (box speed) -- E-190 (2003)|"
    "EI 800 (Push 1) -- E-4035|"
    "EI 800 (Push 1) -- E-190 (2003)|"
    "A -- 14 min first developer, EI 80|"
    "B -- 16 min first developer, EI 100|"
    "C -- 19 min first developer, EI 150|"
    "D -- 22 min first developer, EI 200";

constexpr const char* const ProcessVariantCtrlNoneStr = "As shipped";

//  The same strings indexed by enumerator, for code that needs one name rather
//  than the whole list.
constexpr const char* const ProcessVariantCtrlName[] =
{
    "As shipped",
    "REFINAL",
    "RODINAL 1+25",
    "RODINAL 1+50",
    "RODINAL SPECIAL",
    "STUDIONAL LIQUID",
    "C-41 cross-process, as shipped",
    "Cs2 two-bath kit",
    "ECN-2, the base stock's native process",
    "23 DIN / 160 ASA (box speed)",
    "26 DIN / 320 ASA (push 1)",
    "EI 800 (box speed)",
    "EI 1600 (Push 1)",
    "EI 3200 (Push 2)",
    "EI 400 (box speed) -- E-4035",
    "EI 400 (box speed) -- E-190 (2003)",
    "EI 800 (Push 1) -- E-4035",
    "EI 800 (Push 1) -- E-190 (2003)",
    "A -- 14 min first developer, EI 80",
    "B -- 16 min first developer, EI 100",
    "C -- 19 min first developer, EI 150",
    "D -- 22 min first developer, EI 200",
};

//  Database keys, index aligned with ProcessVariantCtrl. These are the exact
//  film::ProcessVariant::variant_id values, so the engine resolves a variant
//  through this table rather than through a display name and the two can
//  change independently.
constexpr const char* const ProcessVariantCtrlKey[] =
{
    //  \warning ENTRY 0 IS EMPTY AND MUST STAY EMPTY. eAS_SHIPPED names no
    //  film::ProcessVariant, so the resolver's search finds nothing and
    //  returns the shipped profile. A key here would select a development.
    "",
    "AGFA_REFINAL",
    "AGFA_RODINAL_1_25",
    "AGFA_RODINAL_1_50",
    "AGFA_RODINAL_SPECIAL",
    "AGFA_STUDIONAL_LIQUID",
    "CINESTILL_C41_AS_SHIPPED",
    "CINESTILL_CS2_TWO_BATH",
    "CINESTILL_ECN2_NATIVE",
    "GEVACHROME_23DIN_160ASA",
    "GEVACHROME_26DIN_320ASA",
    "PORTRA800_EI800",
    "PORTRA800_EI1600_PUSH1",
    "PORTRA800_EI3200_PUSH2",
    "ULTRA400UC_EI400_E4035",
    "ULTRA400UC_EI400_E190",
    "ULTRA400UC_EI800_E4035",
    "ULTRA400UC_EI800_E190",
    "ANSCOCHROME_A_14MIN_EI80",
    "ANSCOCHROME_B_16MIN_EI100",
    "ANSCOCHROME_C_19MIN_EI150",
    "ANSCOCHROME_D_22MIN_EI200",
};

static_assert(
    static_cast<int32_t>(sizeof(ProcessVariantCtrlName)
                         / sizeof(ProcessVariantCtrlName[0]))
    == ProcessVariantCtrlCount,
    "ProcessVariantCtrlName must have exactly TOTAL_PROCESSES entries");

static_assert(
    static_cast<int32_t>(sizeof(ProcessVariantCtrlKey)
                         / sizeof(ProcessVariantCtrlKey[0]))
    == ProcessVariantCtrlCount,
    "ProcessVariantCtrlKey must have exactly TOTAL_PROCESSES entries");

static_assert(
    static_cast<int32_t>(ProcessVariantCtrl::eAS_SHIPPED) == 0,
    "every enumerated control value starts at zero (owner requirement, "
    "2026-09-18e); eAS_SHIPPED is the first list-box entry");

static_assert(
    ProcessVariantCtrlKey[static_cast<int32_t>(ProcessVariantCtrl::eAS_SHIPPED)]
        [0] == '\0',
    "eAS_SHIPPED must carry an EMPTY database key -- that empty key is what "
    "makes the resolver fall through to the shipped profile");

//  True for a value the tables can be indexed by. Since the 2026-09-18e
//  rebase that INCLUDES eAS_SHIPPED, which is a legal selection at index 0;
//  TOTAL_PROCESSES and any out-of-range value answer false. "Names an actual
//  development" is answered by the KEY being non-empty, not by this test.
constexpr bool ProcessVariantCtrlValid (const ProcessVariantCtrl v) noexcept
{
    return (static_cast<int32_t>(v) >= 0)
        && (static_cast<int32_t>(v) < ProcessVariantCtrlCount);
}

//  Display name for one value, or the sentinel's label. Never returns nullptr.
constexpr const char* ProcessVariantCtrlLabel (const ProcessVariantCtrl v) noexcept
{
    return ProcessVariantCtrlValid(v)
        ? ProcessVariantCtrlName[static_cast<int32_t>(v)]
        : ProcessVariantCtrlNoneStr;
}

//  Database key for one value, or an empty string for the sentinel. This is
//  what AlgoProcessVariant.hpp matches against film::ProcessVariant::variant_id.
constexpr const char* ProcessVariantCtrlKeyOf (const ProcessVariantCtrl v) noexcept
{
    return ProcessVariantCtrlValid(v)
        ? ProcessVariantCtrlKey[static_cast<int32_t>(v)]
        : "";
}



// ---------------------------------------------------------------------------
//  Numeric control metadata
// ---------------------------------------------------------------------------
//  MIN, MAX, DEFAULT and STEP for every numeric control, transcribed from the
//  per-field documentation in AlgoControl.hpp.
//
//  READ THE ENFORCEMENT NOTES BEFORE RELYING ON A BOUND. Most maxima in this
//  structure are ADVISORY: the engine does not clamp them, so they describe
//  the range over which the model is meaningful, and the host is what keeps a
//  value inside it. Where a bound is enforced in the algorithm the comment
//  says so and names the stage. Three cases deserve particular care, because
//  exceeding them does not merely look wrong:
//
//    flare              above 1.0 the direct weight goes negative and the
//                       image inverts. The host must enforce this one.
//    blackPointStretch  above 1.0 the black point passes Dmax; at exactly
//                       tMax / tMin the stage divides by zero.
//    sceneKelvin        the proxy white-balance path has no positivity guard,
//                       so a zero or negative value divides by zero there.
//
//  Types follow the field each constant describes: double for the continuous
//  controls, int32_t for counts, indices and seeds. They are deliberately not
//  unified, because a seed and a reflectance are not the same kind of number.

// -- exposure and development ----------------------------------------------

// Exposure offset in stops. Advisory both ends; no guard of any kind exists.
constexpr double ExposureStopsMin  = -4.0;
constexpr double ExposureStopsMax  =  4.0;
constexpr double ExposureStopsDef  =  0.0;
constexpr double ExposureStopsStep =  0.1;   // 1/3-stop detents at 0.333

// Exposure duration in seconds, for reciprocity failure. Zero is the OFF
// sentinel rather than a minimum; the active range begins at 1e-5 s. The span
// is eight decades, so the control needs a logarithmic slider or a typed
// value. A linear step cannot serve both ends.
constexpr double ExposureTimeSOff = 0.0;
constexpr double ExposureTimeSMin = 1.0e-5;
constexpr double ExposureTimeSMax = 3600.0;
constexpr double ExposureTimeSDef = ExposureTimeSOff;

// Development time in minutes. -1 is the sentinel meaning "the development the
// stored curves represent".
// THE EFFECTIVE RANGE IS PER STOCK, NOT THE FIGURES BELOW, which are only the
// envelope across every traced family: the shortest traced point runs from
// 1.9 min (AGFA APX 25) to 10.0 min (KODAK VERICHROME 1952), the longest from
// 7.0 min (PANATOMIC-X) to 36.0 min (VERICHROME). Outside a stock's own traced
// range the value is treated as the sentinel rather than clamped, because
// extrapolating a development family is not a measurement.
constexpr double DevelopmentMinutesSentinel = -1.0;
constexpr double DevelopmentMinutesMin  = 1.9;
constexpr double DevelopmentMinutesMax  = 36.0;
constexpr double DevelopmentMinutesDef  = DevelopmentMinutesSentinel;
constexpr double DevelopmentMinutesStep = 0.1;

// Development temperature in Celsius, with the same sentinel.
// The traced span is the monochrome families' 18-24 C. A colour process would
// need a 0.05 C step to respect C-41's stated +/- 0.15 C, which is one reason
// this control does not apply to colour stocks.
constexpr double DevelopmentCelsiusSentinel = -1.0;
constexpr double DevelopmentCelsiusMin  = 18.0;
constexpr double DevelopmentCelsiusMax  = 24.0;
constexpr double DevelopmentCelsiusDef  = DevelopmentCelsiusSentinel;
constexpr double DevelopmentCelsiusStep = 0.5;

// Years of DARK STORAGE since processing, for image-dye fade. Zero is the OFF
// sentinel and is fresh film, not a minimum.
// ⚠ THE CONTROL IS INERT ON 188 OF 191 STOCKS and must hide or disable itself
// there, by the same rule processVariant and developmentMinutes already
// follow: only a stock carrying a PUBLISHED dark-fade rate in
// film::DyeStabilitySpec can respond, and today that is KODAK EKTAR 125 (8
// years to a 10 % yellow loss) and KODAK VERICOLOR III 160 (23 years).
// ⚠ THE UPPER BOUND IS NOT A PHYSICAL LIMIT. It is the span over which a
// first-order fade from a published 10 %-loss time stays a restatement of that
// figure rather than an extrapolation of it: at 100 years EKTAR 125's yellow
// record is down 65 %, which is already past anything the source measured.
constexpr double StorageYearsOff  =   0.0;
constexpr double StorageYearsMin  =   0.0;
constexpr double StorageYearsMax  = 100.0;
constexpr double StorageYearsDef  = StorageYearsOff;
constexpr double StorageYearsStep =   0.5;

// -- duplication -----------------------------------------------------------

// Duplication generations. THE ONLY CONTROL IN THE STRUCTURE WITH AN ENFORCED
// UPPER BOUND, clamped at stage 13 against ALGO_DUPE_MAX_GENERATIONS so a
// mistyped value cannot turn one frame into an unbounded render.
constexpr int32_t GenerationsMin  = 0;
constexpr int32_t GenerationsMax  = 4;
constexpr int32_t GenerationsDef  = 0;
constexpr int32_t GenerationsStep = 1;

// -- scanning and white balance --------------------------------------------

// Specular fraction of the scanner illuminant, for the Callier correction.
constexpr double ScannerSpecularMin  = 0.0;
constexpr double ScannerSpecularMax  = 1.0;
constexpr double ScannerSpecularDef  = 0.853;  // 1 - Streiffert's E = 0.1471
constexpr double ScannerSpecularStep = 0.01;

// Scene illuminant colour temperature in kelvin. Advisory bounds.
// The proxy path has no positivity guard; see the section note above.
constexpr double SceneKelvinMin  = 2000.0;
constexpr double SceneKelvinMax  = 12000.0;
constexpr double SceneKelvinDef  = 5500.0;     // nominal daylight
constexpr double SceneKelvinStep = 50.0;

// White-balance correction strength. Values at or below zero disable the stage
// by short circuit rather than being clamped. The default of zero is
// deliberate: an uncorrected tungsten stock shot in daylight stays blue, which
// is what the film did.
constexpr double WbStrengthMin  = 0.0;
constexpr double WbStrengthMax  = 1.0;
constexpr double WbStrengthDef  = 0.0;
constexpr double WbStrengthStep = 0.01;

// -- tone ------------------------------------------------------------------

// Mid-grey anchor as a scene reflectance. Advisory; zero or negative flows
// into a division and then into the bisection solve.
constexpr double GreyTargetMin  = 0.02;
constexpr double GreyTargetMax  = 0.60;
constexpr double GreyTargetDef  = 0.18;
constexpr double GreyTargetStep = 0.005;

// Black-point stretch. Above 1.0 the black point passes Dmax and the divisor
// shrinks toward zero. The host should hold the slider at 1.0.
constexpr double BlackPointStretchMin  = 0.0;
constexpr double BlackPointStretchMax  = 1.0;
constexpr double BlackPointStretchDef  = 1.0;
constexpr double BlackPointStretchStep = 0.01;

// -- film character scalars ------------------------------------------------

// Grain amplitude scale. Floor enforced at all three reading stages.
constexpr double GrainScaleMin  = 0.0;
constexpr double GrainScaleMax  = 4.0;
constexpr double GrainScaleDef  = 1.0;
constexpr double GrainScaleStep = 0.01;

// Halation amplitude scale. Floor enforced at stage 05.
// This scales AMPLITUDE only. Halo WIDTH is derived from the support thickness
// and refractive index at schema v34 and is not user adjustable, because it is
// geometry rather than a preference.
constexpr double HalationScaleMin  = 0.0;
constexpr double HalationScaleMax  = 4.0;
constexpr double HalationScaleDef  = 1.0;
constexpr double HalationScaleStep = 0.01;

// DIR coupler strength scale. Floored in two different shapes: the field
// itself at stage 09, and the product strength * couplerScale in the tone path
// at stages 08, 08b and 13.
constexpr double CouplerScaleMin  = 0.0;
constexpr double CouplerScaleMax  = 3.0;
constexpr double CouplerScaleDef  = 1.0;
constexpr double CouplerScaleStep = 0.01;

// Layer misregistration scale. Floor enforced at stage 10.
constexpr double MisregScaleMin  = 0.0;
constexpr double MisregScaleMax  = 4.0;
constexpr double MisregScaleDef  = 1.0;
constexpr double MisregScaleStep = 0.01;

// Coating non-uniformity scale. Floor enforced at all three reading stages.
// One product cap exists at stage 06b, where buckle_mtf_loss * coatingScale is
// capped at ALGO_DEFOCUS_MAX_LOSS; stages 04 and 10b have no ceiling.
constexpr double CoatingScaleMin  = 0.0;
constexpr double CoatingScaleMax  = 3.0;
constexpr double CoatingScaleDef  = 1.0;
constexpr double CoatingScaleStep = 0.01;

// Veiling flare fraction. -1 is the sentinel meaning "the stock's own
// era-appropriate value". Above 1.0 the direct weight goes negative and the
// image inverts; the host must enforce the maximum.
constexpr double FlareSentinel = -1.0;
constexpr double FlareMin  = 0.0;
constexpr double FlareMax  = 0.5;
constexpr double FlareDef  = FlareSentinel;
constexpr double FlareStep = 0.005;

// Corner falloff. -1 is the sentinel meaning "the stock's era default"; any
// resolved value at or below zero disables the vignette half of stage 04.
constexpr double VignetteSentinel = -1.0;
constexpr double VignetteMin  = 0.0;
constexpr double VignetteMax  = 4.0;
constexpr double VignetteDef  = VignetteSentinel;
constexpr double VignetteStep = 0.05;

// -- transport and determinism ---------------------------------------------

// Frames per second. Floor enforced, but only at the two sites that use it
// (stages 15 and 16), not at the driver. The step must express 23.976 and
// 29.97 recognisably, which a coarser one cannot.
constexpr double FrameRateMin  = 1.0;
constexpr double FrameRateMax  = 240.0;      // advisory; none documented
constexpr double FrameRateDef  = 24.0;
constexpr double FrameRateStep = 0.001;

// Frame ordinal, host supplied. One unguarded signed multiplication exists at
// stage 13 (frameIndex * 9), so extreme values can overflow.
constexpr int32_t FrameIndexDef  = 0;
constexpr int32_t FrameIndexStep = 1;

// Deterministic render seed. The whole int32_t range is valid: negatives wrap
// into the upper half of uint32_t, which the mixer treats alike. The panel
// should offer a randomise button rather than a drag.
constexpr int32_t SeedDef  = 12345;
constexpr int32_t SeedStep = 1;

// -- boolean controls ------------------------------------------------------

constexpr bool PrintGrainDef        = true;
constexpr bool ReseauDef            = true;
constexpr bool FilmDamageEnabledDef = false;   // clean film is the default

// -- film damage sub-structure ---------------------------------------------
//  SIX OF THESE SEVENTEEN CONTROLS HAVE NO READER IN THE CURRENT ENGINE and
//  are marked individually below. They are retained because each names a real
//  degradation the model does not yet implement, and deleting the field would
//  lose the requirement along with it. A host should present them as inactive
//  rather than as controls that appear to do something.

// Master damage scale. Floor enforced at stages 09b, 15 and 16; stages 09b and
// 16 return immediately at or below zero.
constexpr double DamageStrengthMin  = 0.0;
constexpr double DamageStrengthMax  = 4.0;
constexpr double DamageStrengthDef  = 1.0;
constexpr double DamageStrengthStep = 0.01;

// Damage-layer seed, independent of the render seed.
constexpr int32_t DamageSeedDef  = 20250803;
constexpr int32_t DamageSeedStep = 1;

// Particulate populations. Floors enforced at stage 09b.
// Only dust saturates downstream, at ALGO_DUST_DENSITY_MAX; debris and fibre
// scale their Poisson intensity without limit.
constexpr double DustLevelMin  = 0.0;
constexpr double DustLevelMax  = 4.0;
constexpr double DustLevelDef  = 1.0;
constexpr double DustLevelStep = 0.01;

constexpr double DebrisLevelMin  = 0.0;
constexpr double DebrisLevelMax  = 4.0;
constexpr double DebrisLevelDef  = 1.0;
constexpr double DebrisLevelStep = 0.01;

constexpr double FibreLevelMin  = 0.0;
constexpr double FibreLevelMax  = 4.0;
constexpr double FibreLevelDef  = 1.0;
constexpr double FibreLevelStep = 0.01;

// Spatial clustering of the particulate populations. At exactly zero the Cox
// field is bypassed and the process degenerates to uniform Poisson.
constexpr double DirtClumpingMin  = 0.0;
constexpr double DirtClumpingMax  = 2.0;
constexpr double DirtClumpingDef  = 1.0;
constexpr double DirtClumpingStep = 0.01;

// Abrasion populations. Floors enforced at stage 09b.
constexpr double ScratchTransportMin  = 0.0;
constexpr double ScratchTransportMax  = 4.0;
constexpr double ScratchTransportDef  = 0.40;
constexpr double ScratchTransportStep = 0.01;

constexpr double ScratchHandlingMin  = 0.0;
constexpr double ScratchHandlingMax  = 4.0;
constexpr double ScratchHandlingDef  = 0.30;
constexpr double ScratchHandlingStep = 0.01;

// Gate contamination. Floor enforced at stage 16. Two indirect caps exist
// downstream: the persistent-mark window and the per-frame sparkle cap.
constexpr double GateDirtMin  = 0.0;
constexpr double GateDirtMax  = 2.0;
constexpr double GateDirtDef  = 0.60;
constexpr double GateDirtStep = 0.01;

// Gate weave amplitude. Floor enforced at stage 15.
constexpr double WeaveAmountMin  = 0.0;
constexpr double WeaveAmountMax  = 2.0;
constexpr double WeaveAmountDef  = 0.50;
constexpr double WeaveAmountStep = 0.01;

// Gate weave amplitude. Floor enforced at stage 15.
constexpr double SpliceAndTearsEventstMin = 0.0;
constexpr double SpliceAndTearsEventstMax = 2.0;
constexpr double SpliceAndTearsEventstDef = 0.20;
constexpr double SpliceAndTearsEventsttep = 0.01;
// Discrete damage event rate. Floor enforced at stage 16. An implicit
// switch-off exists far above the advisory range: once the derived interval
// falls below one frame the generator stops entirely.
constexpr double DamageEventsMin  = 0.0;
constexpr double DamageEventsMax  = 2.0;
constexpr double DamageEventsDef  = 0.20;
constexpr double DamageEventsStep = 0.01;

// NO READER IN THE CURRENT ENGINE - processing quality axis.
constexpr double ProcessingQualityMin  = 0.0;
constexpr double ProcessingQualityMax  = 1.0;
constexpr double ProcessingQualityDef  = 0.30;
constexpr double ProcessingQualityStep = 0.01;

// NO READER IN THE CURRENT ENGINE - drying mark severity.
constexpr double DryingMarksMin  = 0.0;
constexpr double DryingMarksMax  = 2.0;
constexpr double DryingMarksDef  = 0.25;
constexpr double DryingMarksStep = 0.01;

// NO READER IN THE CURRENT ENGINE - storage degradation severity.
constexpr double StorageSeverityMin  = 0.0;
constexpr double StorageSeverityMax  = 1.0;
constexpr double StorageSeverityDef  = 0.20;
constexpr double StorageSeverityStep = 0.01;

// NO READER IN THE CURRENT ENGINE - overall colour veil from ageing.
constexpr double ColourVeilMin  = 0.0;
constexpr double ColourVeilMax  = 2.0;
constexpr double ColourVeilDef  = 0.15;
constexpr double ColourVeilStep = 0.01;

// NO READER IN THE CURRENT ENGINE - frame-to-frame exposure flicker.
constexpr double FlickerStopsMin  = 0.0;
constexpr double FlickerStopsMax  = 0.5;
constexpr double FlickerStopsDef  = 0.15;
constexpr double FlickerStopsStep = 0.01;

// NO READER IN THE CURRENT ENGINE - scanner-introduced artefacts.
constexpr double ScannerArtifactsMin  = 0.0;
constexpr double ScannerArtifactsMax  = 2.0;
constexpr double ScannerArtifactsDef  = 0.20;
constexpr double ScannerArtifactsStep = 0.01;


#endif // __IMAGE_LAB2_ALGO_CONTROL_ENUMERATORS__
