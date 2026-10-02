#pragma once

//============================================================================
//  ColorTransformMatrix.hpp
//
//  Color-space conversion matrices, observers, and illuminant white points.
//
//  Verified / regenerated: 2026-07-05 12:09:45
//  - All matrices and white points regenerated to 7 significant digits.
//  - White points and the sRGB matrix are CONSISTENT: both derived from the
//    same CIE chromaticities (D65 = 0.31270, 0.32900). Original ASTM E308 /
//    Lindbloom values are preserved in comments where they differed.
//  - RGB<->YCbCr matrices are FULL/DATA range (Y in [0,1], Cb/Cr in [-0.5,0.5]).
//    Studio/legal-range scaling (e.g. 16-235) must be applied separately.
//============================================================================

#include "Common.hpp"

typedef enum
{
    BT601 = 0,
    BT709,
    BT2020,
    SMPTE240M
}eCOLOR_SPACE;

typedef enum
{
    observer_CIE_1931 = 0, /*  2 deg (CIE 1931) - standard for video/grading   */
    observer_CIE_1964,     /* 10 deg (CIE 1964) - large-field / textiles       */
    observer_TOTAL_OBSERVERS
}eCOLOR_OBSERVER;

// NOTE: enum order matches the cCOLOR_ILLUMINANT[] table row order below.
typedef enum
{
    color_ILLUMINANT_A = 0,
    color_ILLUMINANT_B = 1,
    color_ILLUMINANT_C = 2,
    color_ILLUMINANT_D50 = 3,
    color_ILLUMINANT_D55 = 4,
    color_ILLUMINANT_D60 = 5,
    color_ILLUMINANT_D65 = 6,
    color_ILLUMINANT_D75 = 7,
    color_ILLUMINANT_D93 = 8,
    color_ILLUMINANT_E = 9,
    color_ILLUMINANT_P3_D65 = 10,
    color_ILLUMINANT_F1 = 11,
    color_ILLUMINANT_F2 = 12,
    color_ILLUMINANT_F3 = 13,
    color_ILLUMINANT_F4 = 14,
    color_ILLUMINANT_F5 = 15,
    color_ILLUMINANT_F6 = 16,
    color_ILLUMINANT_F7 = 17,
    color_ILLUMINANT_F8 = 18,
    color_ILLUMINANT_F9 = 19,
    color_ILLUMINANT_F10 = 20,
    color_ILLUMINANT_F11 = 21,
    color_ILLUMINANT_F12 = 22,
    color_TOTAL_ILLUMINANTS
}eCOLOR_ILLUMINANT;

constexpr eCOLOR_OBSERVER   CieLabDefaultObserver  { observer_CIE_1931 };
constexpr eCOLOR_ILLUMINANT CieLabDefaultIlluminant{ color_ILLUMINANT_D65 };

// ==========================================================================
//  RGB -> YCbCr  (full range). Rows: Y, Cb, Cr.
//  Derived from ITU luma coefficients (Kr,Kb) via
//    Cb = 0.5*(B-Y)/(1-Kb),  Cr = 0.5*(R-Y)/(1-Kr).
// ==========================================================================
CACHE_ALIGN float constexpr RGB2YUV[][9] =
{
    // BT.601  (Kr=0.2990, Kb=0.1140)
    {
         0.2990000f,  0.5870000f,  0.1140000f,
        -0.1687359f, -0.3312641f,  0.5000000f,
         0.5000000f, -0.4186876f, -0.0813124f,
    },
    // BT.709  (Kr=0.2126, Kb=0.0722)
    {
         0.2126000f,  0.7152000f,  0.0722000f,
        -0.1145721f, -0.3854279f,  0.5000000f,
         0.5000000f, -0.4541529f, -0.0458471f,
    },
    // BT.2020 (Kr=0.2627, Kb=0.0593)
    {
         0.2627000f,  0.6780000f,  0.0593000f,
        -0.1396301f, -0.3603699f,  0.5000000f,
         0.5000000f, -0.4597857f, -0.0402143f,
    },
    // SMPTE 240M (Kr=0.2120, Kb=0.0870) - OBSOLETE early-HDTV; kept for legacy
    {
         0.2120000f,  0.7010000f,  0.0870000f,
        -0.1161008f, -0.3838992f,  0.5000000f,
         0.5000000f, -0.4447970f, -0.0552030f,
    }
};

// ==========================================================================
//  YCbCr -> RGB  (full range). Exact analytic inverses of the above
//  (near-zero terms cleaned to true 0.0).
// ==========================================================================
CACHE_ALIGN float constexpr YUV2RGB[][9] =
{
    // BT.601  (Kr=0.2990, Kb=0.1140)
    {
        1.0000000f,  0.0000000f,  1.4020000f,
        1.0000000f, -0.3441363f, -0.7141363f,
        1.0000000f,  1.7720000f, -0.0000000f,
    },
    // BT.709  (Kr=0.2126, Kb=0.0722)
    {
        1.0000000f, -0.0000000f,  1.5748000f,
        1.0000000f, -0.1873243f, -0.4681243f,
        1.0000000f,  1.8556000f,  0.0000000f,
    },
    // BT.2020 (Kr=0.2627, Kb=0.0593)
    {
        1.0000000f, -0.0000000f,  1.4746000f,
        1.0000000f, -0.1645531f, -0.5713531f,
        1.0000000f,  1.8814000f,  0.0000000f,
    },
    // SMPTE 240M (Kr=0.2120, Kb=0.0870) - OBSOLETE early-HDTV; kept for legacy
    {
        1.0000000f, -0.0000000f,  1.5760000f,
        1.0000000f, -0.2266220f, -0.4766220f,
        1.0000000f,  1.8260000f, -0.0000000f,
    }
};

// ==========================================================================
//  Linear sRGB <-> XYZ (D65). Regenerated from sRGB primaries + D65
//  (0.31270, 0.32900) to 7 digits; consistent with color_ILLUMINANT_D65 below.
//
//  Previous (Bruce Lindbloom / ASTM-D65) values, kept for reference:
//    sRGBtoXYZ = { 0.4124564, 0.3575761, 0.1804375,
//                  0.2126729, 0.7151522, 0.0721750,
//                  0.0193339, 0.1191920, 0.9503041 }
//    XYZtosRGB = { 3.240455, -1.537139, -0.498532,
//                 -0.969266,  1.876011,  0.041556,
//                  0.055643, -0.204026,  1.057225 }
// ==========================================================================
CACHE_ALIGN constexpr float sRGBtoXYZ[9] =
{
    0.4123908f, 0.3575843f, 0.1804808f,
    0.2126390f, 0.7151687f, 0.0721923f,
    0.0193308f, 0.1191948f, 0.9505322f,
};

CACHE_ALIGN constexpr float XYZtosRGB[9] =
{
     3.2409699f, -1.5373832f, -0.4986108f,
    -0.9692436f,  1.8759675f,  0.0415551f,
     0.0556301f, -0.2039770f,  1.0569715f,
};

// ==========================================================================
//  YCoCg  (RGB <-> YCoCg).  NOT a broadcast standard.
//  Used in image/video COMPRESSION and screen-content coding (e.g. H.264/
//  HEVC RExt lossless). Excellent decorrelation, integer-friendly, and
//  near-lossless. Full range. Rows: Y, Co, Cg.
//    Y  =  0.25 R + 0.5 G + 0.25 B
//    Co =  0.5  R        - 0.5  B
//    Cg = -0.25 R + 0.5 G - 0.25 B
//  Inverse is exact and multiply-free: R=Y+Co-Cg, G=Y+Cg, B=Y-Co-Cg.
//  (A reversible lifting variant, YCoCg-R, exists for true lossless but is
//   an integer lifting scheme, not a matrix.)
// ==========================================================================
CACHE_ALIGN constexpr float RGB2YCoCg[9] =
{
     0.2500000f, 0.5000000f,  0.2500000f,
     0.5000000f, 0.0000000f, -0.5000000f,
    -0.2500000f, 0.5000000f, -0.2500000f,
};

CACHE_ALIGN constexpr float YCoCg2RGB[9] =
{
    1.0000000f,  1.0000000f, -1.0000000f,
    1.0000000f,  0.0000000f,  1.0000000f,
    1.0000000f, -1.0000000f, -1.0000000f,
};

// ==========================================================================
//  ICtCp  (BT.2100).  Modern HDR/WCG luma-chroma encoding, successor to
//  Y'CbCr; used in Dolby Vision. Far better hue linearity & perceptual
//  uniformity than Y'CbCr for HDR. This is a PIPELINE, not a single matrix:
//    1) RGB (BT.2020 linear) -> LMS         : use RGB2LMS_BT2100
//    2) apply PQ (ST 2084) OETF to L,M,S -> L' M' S'   (or HLG variant)
//    3) L'M'S' -> ICtCp                     : use LMS2ICtCp_BT2100_PQ
//  Decode reverses: ICtCp2LMS -> inverse-PQ -> LMS2RGB.
//  NOTE: step 2 is nonlinear - the matrices alone do NOT produce ICtCp
//  without the PQ/HLG transfer stage in between.
// ==========================================================================
CACHE_ALIGN constexpr float RGB2LMS_BT2100[9] =
{
    0.4121094f, 0.5239258f, 0.0639648f,
    0.1667480f, 0.7204590f, 0.1127930f,
    0.0241699f, 0.0754395f, 0.9003906f,
};

CACHE_ALIGN constexpr float LMS2RGB_BT2100[9] =
{
     3.4366067f, -2.5064521f,  0.0698454f,
    -0.7913296f,  1.9836005f, -0.1922709f,
    -0.0259499f, -0.0989137f,  1.1248636f,
};

// L'M'S' (after PQ) -> ICtCp. Rows: I, Ct, Cp.
CACHE_ALIGN constexpr float LMS2ICtCp_BT2100_PQ[9] =
{
    0.5000000f,  0.5000000f,  0.0000000f,
    1.6137695f, -3.3234863f,  1.7097168f,
    4.3781738f, -4.2456055f, -0.1325684f,
};

CACHE_ALIGN constexpr float ICtCp2LMS_BT2100_PQ[9] =
{
    1.0000000f,  0.0086090f,  0.1110296f,
    1.0000000f, -0.0086090f, -0.1110296f,
    1.0000000f,  0.5600313f, -0.3206272f,
};

// ==========================================================================
//  BT.2020 Constant Luminance (CL) - Yc'CbcCrc.
//  Alternative to the ordinary (non-constant-luminance) BT.2020 matrix in
//  RGB2YUV[BT2020]. CL derives luma in LINEAR light for better luma/chroma
//  separation. Rarely deployed, but standardized. This is NONLINEAR and
//  cannot be a 3x3 matrix; procedure:
//    Yc' = OETF( 0.2627*Rlin + 0.6780*Glin + 0.0593*Blin )   // luma in linear
//    Cb  = (B' - Yc')/Nb  if (B'-Yc')<=0  else  (B' - Yc')/Pb
//    Cr  = (R' - Yc')/Nr  if (R'-Yc')<=0  else  (R' - Yc')/Pr
//  where R',B' are the OETF-encoded channels and the normalization consts:
CACHE_ALIGN constexpr float BT2020_CL_Luma[3] = { 0.2627000f, 0.6780000f, 0.0593000f }; // Kr,Kg,Kb
CACHE_ALIGN constexpr float BT2020_CL_Nb = 1.9404000f;
CACHE_ALIGN constexpr float BT2020_CL_Pb = 1.5816000f;
CACHE_ALIGN constexpr float BT2020_CL_Nr = 1.7184000f;
CACHE_ALIGN constexpr float BT2020_CL_Pr = 0.9936000f;

// ==========================================================================
//  !!! NOT VALID - DO NOT USE !!!
//  A YCbCr<->XYZ conversion requires a full 3x3 (9 coefficients). The two
//  3-element vectors below cannot represent any such transform; their origin
//  is unknown (likely leftover scratch constants). Retained per request but
//  marked invalid - do not use them for color conversion.
// ==========================================================================
CACHE_ALIGN constexpr float yuv2xyz[3] = { 0.114653800f, 0.083911980f, 0.082220770f }; // NOT VALID
CACHE_ALIGN constexpr float xyz2yuv[3] = { 0.083911980f, 0.283096500f, 0.466178900f }; // NOT VALID

// ==========================================================================
//  Reference white points as tristimulus XYZ, normalized to Y = 100.
//  Derived from CIE chromaticities (7 significant digits). Indexed
//  [observer][illuminant][X,Y,Z]; illuminant order matches the enum above.
// ==========================================================================
CACHE_ALIGN constexpr float cCOLOR_ILLUMINANT[observer_TOTAL_OBSERVERS][color_TOTAL_ILLUMINANTS][3] =
{
    /* 2 deg (CIE 1931) */
    {
        // A: Incandescent/tungsten ~2856K. Reference for warm studio light; legacy WB anchor.
        { 109.84906f, 100.00000f, 35.57983f },
        // B: Direct sunlight ~4874K. Obsolete (superseded by D-series); kept for legacy.
        { 99.09274f, 100.00000f, 85.31327f },
        // C: Average daylight ~6774K. Obsolete (no UV); legacy NTSC/print reference.
        { 98.07060f, 100.00000f, 118.22495f },
        // D50: Horizon daylight 5003K. ICC/print & graphic-arts viewing standard.
        { 96.42957f, 100.00000f, 82.51046f },
        // D55: Mid-morning daylight 5503K. Photographic/film reference.
        { 95.67983f, 100.00000f, 92.13965f },
        // D60: Daylight ~6003K. ACES white point; common in VFX/DI mastering.
        { 95.25999f, 100.00000f, 100.93106f },
        // D65: Noon daylight 6504K. sRGB/Rec.709/Rec.2020/Display-P3 white; primary video reference.
        { 95.04559f, 100.00000f, 108.90578f },
        // D75: North-sky daylight 7504K. Cooler daylight reference.
        { 94.96634f, 100.00000f, 122.61496f },
        // D93: Daylight ~9305K. High-CCT display/broadcast white (e.g. some monitors, JEITA).
        { 95.32850f, 100.00000f, 141.42730f },
        // E: Equal-energy 5454K. Theoretical flat SPD; sanity/reference only.
        { 100.00000f, 100.00000f, 100.00000f },
        // P3_D65: DCI-P3 D65 variant white (== D65). Display-P3 mastering; distinct from DCI theatrical white.
        { 95.04559f, 100.00000f, 108.90578f },
        // F1: Daylight fluorescent. Fluorescent series F1.
        { 92.88045f, 100.00000f, 103.76743f },
        // F2: Cool white fluorescent (very common office). Fluorescent series F2.
        { 99.20021f, 100.00000f, 67.39536f },
        // F3: White fluorescent. Fluorescent series F3.
        { 103.80614f, 100.00000f, 49.93656f },
        // F4: Warm white fluorescent. Fluorescent series F4.
        { 109.20367f, 100.00000f, 38.87373f },
        // F5: Daylight fluorescent. Fluorescent series F5.
        { 90.90382f, 100.00000f, 98.78331f },
        // F6: Light white fluorescent. Fluorescent series F6.
        { 97.34673f, 100.00000f, 60.25245f },
        // F7: D65 broadband fluorescent (high CRI). Fluorescent series F7.
        { 95.04860f, 100.00000f, 108.71810f },
        // F8: D50 broadband fluorescent (high CRI). Fluorescent series F8.
        { 96.43056f, 100.00000f, 82.43168f },
        // F9: Broadband fluorescent. Fluorescent series F9.
        { 100.37564f, 100.00000f, 67.93668f },
        // F10: Tri-band fluorescent ~5000K (common modern tube). Fluorescent series F10.
        { 96.37681f, 100.00000f, 82.32999f },
        // F11: Tri-band fluorescent ~4000K. Fluorescent series F11.
        { 100.95516f, 100.00000f, 64.36721f },
        // F12: Tri-band fluorescent ~3000K. Fluorescent series F12.
        { 108.11479f, 100.00000f, 39.28748f }
    },
    /* 10 deg (CIE 1964) */
    {
        // A: Incandescent/tungsten ~2856K. Reference for warm studio light; legacy WB anchor.
        { 111.14204f, 100.00000f, 35.19978f },
        // B: Direct sunlight ~4874K. Obsolete (superseded by D-series); kept for legacy.
        { 99.17777f, 100.00000f, 84.34931f },
        // C: Average daylight ~6774K. Obsolete (no UV); legacy NTSC/print reference.
        { 97.28569f, 100.00000f, 116.14480f },
        // D50: Horizon daylight 5003K. ICC/print & graphic-arts viewing standard.
        { 96.72063f, 100.00000f, 81.42802f },
        // D55: Mid-morning daylight 5503K. Photographic/film reference.
        { 95.79952f, 100.00000f, 90.92238f },
        // D60: Daylight ~6003K. ACES white point; common in VFX/DI mastering.
        { 95.19895f, 100.00000f, 99.54657f },
        // D65: Noon daylight 6504K. sRGB/Rec.709/Rec.2020/Display-P3 white; primary video reference.
        { 94.80967f, 100.00000f, 107.30514f },
        // D75: North-sky daylight 7504K. Cooler daylight reference.
        { 94.41714f, 100.00000f, 120.64272f },
        // D93: Daylight ~9305K. High-CCT display/broadcast white (e.g. some monitors, JEITA).
        { 94.29270f, 100.00000f, 138.63660f },
        // E: Equal-energy 5454K. Theoretical flat SPD; sanity/reference only.
        { 100.00000f, 100.00000f, 100.00000f },
        // P3_D65: DCI-P3 D65 variant white (== D65). Display-P3 mastering; distinct from DCI theatrical white.
        { 94.80967f, 100.00000f, 107.30514f },
        // F1: Daylight fluorescent. Fluorescent series F1.
        { 94.79126f, 100.00000f, 103.19139f },
        // F2: Cool white fluorescent (very common office). Fluorescent series F2.
        { 103.24504f, 100.00000f, 68.98974f },
        // F3: White fluorescent. Fluorescent series F3.
        { 108.96827f, 100.00000f, 51.96483f },
        // F4: Warm white fluorescent. Fluorescent series F4.
        { 114.96136f, 100.00000f, 40.96330f },
        // F5: Daylight fluorescent. Fluorescent series F5.
        { 93.36857f, 100.00000f, 98.63634f },
        // F6: Light white fluorescent. Fluorescent series F6.
        { 102.14812f, 100.00000f, 62.07361f },
        // F7: D65 broadband fluorescent (high CRI). Fluorescent series F7.
        { 95.77973f, 100.00000f, 107.61833f },
        // F8: D50 broadband fluorescent (high CRI). Fluorescent series F8.
        { 97.11456f, 100.00000f, 81.13470f },
        // F9: Broadband fluorescent. Fluorescent series F9.
        { 102.11634f, 100.00000f, 67.82562f },
        // F10: Tri-band fluorescent ~5000K (common modern tube). Fluorescent series F10.
        { 99.00124f, 100.00000f, 83.13396f },
        // F11: Tri-band fluorescent ~4000K. Fluorescent series F11.
        { 103.81973f, 100.00000f, 65.55505f },
        // F12: Tri-band fluorescent ~3000K. Fluorescent series F12.
        { 111.42836f, 100.00000f, 40.35300f }
    }
};

