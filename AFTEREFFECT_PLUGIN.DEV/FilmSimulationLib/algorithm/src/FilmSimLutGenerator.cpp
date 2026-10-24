// ===========================================================================
//  FilmSimLutGenerator.cpp -- 3D LUT export of the film simulation.
//  See FilmSimLutGenerator.hpp for the contract. Shared by the scalar and the
//  AVX2 builds: it only fills the source planes, calls Algorithm_Main and reads
//  the destination planes, so each build exports through its own engine.
//  Every conversion here is computed in double precision.
// ===========================================================================

#include "FilmSimLutGenerator.hpp"

#include "AlgorithmMain.hpp"
#include "AlgoControlEnums.hpp"
#include "film_profiles.hpp"
#include "film_schema_version.h"

#include <cmath>
#include <cstddef>
#include <ctime>
#include <new>

namespace
{
    // -----------------------------------------------------------------------
    //  CONSTANTS -- full double precision. Same values as lut_export.py.
    // -----------------------------------------------------------------------

    // BT.709 / BT.2020 OETF, continuous in value and slope: alpha, beta solved
    // exactly (BT.2020 prints 1.09929682680944 / 0.018053968510807; BT.709 the
    // four-digit rounding 1.099 / 0.018).
    constexpr double kBt709Alpha    = 1.099296826809443;
    constexpr double kBt709Beta     = 0.01805396851080781;
    constexpr double kBt709Gamma    = 0.45;
    constexpr double kBt709Slope    = 4.5;
    constexpr double kBt709VBreak   = 0.08124285829863515;   // 4.5 * beta

    // IEC 61966-2-1 (sRGB): the standard's defining values.
    constexpr double kSrgbA         = 0.055;
    constexpr double kSrgbGamma     = 2.4;
    constexpr double kSrgbSlope     = 12.92;
    constexpr double kSrgbVBreak    = 0.04045;
    constexpr double kSrgbLBreak    = 0.0031308;

    // BT.1886 reference display.
    constexpr double kBt1886Gamma   = 2.4;

    // ACEScct, Academy S-2016-001.
    constexpr double kAcescctYBrk   = 0.155251141552511;
    constexpr double kAcescctA      = 10.5402377416545;
    constexpr double kAcescctB      = 0.0729055341958355;
    constexpr double kAcescctHalf   = 65504.0;
    constexpr double kAcescctVMax   = 1.4679963120447153;    // (log2(65504) + 9.72) / 17.52

    // ACES AP1 linear -> Rec.709 linear, Bradford D60 -> D65, derived in double
    // from the S-2014-004 and BT.709-6 chromaticities (lut_export.AP1_TO_REC709).
    constexpr double kAp1To709[3][3] =
    {
        {  1.7050509926579829,   -0.6217921206570044,   -0.0832588720009786   },
        { -0.13025641750704353,   1.1408047365754013,   -0.010548319068358064 },
        { -0.024003356804618032, -0.12896897606497051,   1.1529723328695884   }
    };

    // -----------------------------------------------------------------------
    //  Transfer functions (double).
    // -----------------------------------------------------------------------
    inline double Rec709InverseOetf (const double v) noexcept
    {
        return (v < kBt709VBreak) ? (v / kBt709Slope)
                                  : std::pow((v + (kBt709Alpha - 1.0)) / kBt709Alpha, 1.0 / kBt709Gamma);
    }

    inline double SrgbDecode (const double v) noexcept
    {
        return (v <= kSrgbVBreak) ? (v / kSrgbSlope)
                                  : std::pow((v + kSrgbA) / (1.0 + kSrgbA), kSrgbGamma);
    }

    inline double SrgbEncode (const double l) noexcept
    {
        return (l <= kSrgbLBreak) ? (l * kSrgbSlope)
                                  : ((1.0 + kSrgbA) * std::pow(l, 1.0 / kSrgbGamma) - kSrgbA);
    }

    inline double AcescctToLin (const double v) noexcept
    {
        if (v >= kAcescctVMax)
            return kAcescctHalf;
        return (v <= kAcescctYBrk) ? ((v - kAcescctB) / kAcescctA)
                                   : std::pow(2.0, v * 17.52 - 9.72);
    }

    inline double Clamp01 (const double v) noexcept
    {
        return (v < 0.0) ? 0.0 : ((v > 1.0) ? 1.0 : v);
    }

    //: code values -> linear Rec.709 (0.18 = mid grey), negatives clipped to 0
    void DecodeInput (const LutInputSpaceCtrl sp, const double code[3], double lin[3]) noexcept
    {
        switch (sp)
        {
        case LutInputSpaceCtrl::eLUT_IN_SRGB:
            for (int k = 0; k < 3; k++) lin[k] = SrgbDecode(code[k]);
            break;
        case LutInputSpaceCtrl::eLUT_IN_ACESCCT:
        {
            const double a[3] = { AcescctToLin(code[0]), AcescctToLin(code[1]), AcescctToLin(code[2]) };
            for (int k = 0; k < 3; k++)
                lin[k] = kAp1To709[k][0] * a[0] + kAp1To709[k][1] * a[1] + kAp1To709[k][2] * a[2];
            break;
        }
        case LutInputSpaceCtrl::eLUT_IN_REC709:
        default:
            for (int k = 0; k < 3; k++) lin[k] = Rec709InverseOetf(code[k]);
            break;
        }
        for (int k = 0; k < 3; k++)
            if (lin[k] < 0.0) lin[k] = 0.0;
    }

    //: display-linear 0..1 -> output code value
    inline double EncodeOutput (const LutOutputSpaceCtrl sp, const double displayLin) noexcept
    {
        const double l = Clamp01(displayLin);
        return (LutOutputSpaceCtrl::eLUT_OUT_SRGB == sp) ? SrgbEncode(l)
                                                         : std::pow(l, 1.0 / kBt1886Gamma);
    }

    bool ValidControls (const LutGeneratorControls& c) noexcept
    {
        return (c.size >= LutSizeCtrl::eLUT_SIZE_SMALL && c.size < LutSizeCtrl::eLUT_SIZE_TOTAL)
            && (c.inputSpace >= LutInputSpaceCtrl::eLUT_IN_REC709 && c.inputSpace < LutInputSpaceCtrl::eLUT_IN_TOTAL)
            && (c.outputSpace >= LutOutputSpaceCtrl::eLUT_OUT_REC709 && c.outputSpace < LutOutputSpaceCtrl::eLUT_OUT_TOTAL)
            && (c.interpolation >= LutInterpolationCtrl::eLUT_INTERP_TRILINEAR
                && c.interpolation < LutInterpolationCtrl::eLUT_INTERP_TOTAL);
    }

    //: The controls a LUT cannot carry, forced off (lut_export.LUT_FORCED).
    AlgoControls LutControls (const AlgoControls& in) noexcept
    {
        AlgoControls c = in;
        c.grainScale          = 0.0;     // grain (random)
        c.printGrain          = false;   // print grain (random)
        c.halationScale       = 0.0;     // halation (spatial)
        c.misregScale         = 0.0;     // scan misregistration (spatial)
        c.coatingScale        = 0.0;     // coating field, gate buckle, edge fog (position)
        c.vignette            = 0.0;     // lens vignette (position)
        c.flare               = 0.0;     // veiling flare (whole-frame mean)
        c.reseau              = false;   // additive mosaic (spatial)
        c.scannerFixedPattern = 0.0;     // scanner fixed pattern (position)
        c.filmDamageEnabled   = false;   // damage, gate weave, flicker (spatial / temporal)
        return c;
    }

    // -----------------------------------------------------------------------
    //  Locale-independent number formatting for the .cube text.
    // -----------------------------------------------------------------------
    void AppendUInt (std::string& s, unsigned long long v)
    {
        char buf[24];
        int n = 0;
        do { buf[n++] = static_cast<char>('0' + static_cast<int>(v % 10ULL)); v /= 10ULL; } while (v != 0ULL);
        while (n > 0) s.push_back(buf[--n]);
    }

    //: fixed-point with `decimals` digits, correctly rounded, '.' always
    void AppendFixed (std::string& s, double v, const int decimals)
    {
        if (!(v == v)) { s += "nan"; return; }
        if (v < 0.0) { s.push_back('-'); v = -v; }
        double scale = 1.0;
        for (int i = 0; i < decimals; i++) scale *= 10.0;
        const double scaled = std::floor(v * scale + 0.5);
        const unsigned long long whole = static_cast<unsigned long long>(scaled / scale);
        unsigned long long frac = static_cast<unsigned long long>(scaled - static_cast<double>(whole) * scale + 0.5);
        unsigned long long lim = static_cast<unsigned long long>(scale);
        unsigned long long w = whole;
        if (frac >= lim) { frac -= lim; w += 1ULL; }
        AppendUInt(s, w);
        if (decimals > 0)
        {
            s.push_back('.');
            std::string f;
            AppendUInt(f, frac);
            for (int i = static_cast<int>(f.size()); i < decimals; i++) s.push_back('0');
            s += f;
        }
    }

    //: shortest-ish general form for header values (up to 6 decimals, trailing zeros dropped)
    void AppendNumber (std::string& s, const double v)
    {
        std::string t;
        AppendFixed(t, v, 6);
        if (t.find('.') != std::string::npos)
        {
            while (!t.empty() && t[t.size() - 1] == '0') t.erase(t.size() - 1);
            if (!t.empty() && t[t.size() - 1] == '.') t.erase(t.size() - 1);
        }
        s += t;
    }

    void TodayIso (char out[11]) noexcept
    {
        const std::time_t t = std::time(nullptr);
        std::tm tmv;
#if defined(_MSC_VER)
        if (0 != localtime_s(&tmv, &t)) { out[0] = '\0'; return; }
#else
        if (nullptr == localtime_r(&t, &tmv)) { out[0] = '\0'; return; }
#endif
        const int y = tmv.tm_year + 1900, m = tmv.tm_mon + 1, d = tmv.tm_mday;
        out[0] = static_cast<char>('0' + (y / 1000) % 10);
        out[1] = static_cast<char>('0' + (y / 100) % 10);
        out[2] = static_cast<char>('0' + (y / 10) % 10);
        out[3] = static_cast<char>('0' + y % 10);
        out[4] = '-';
        out[5] = static_cast<char>('0' + m / 10);
        out[6] = static_cast<char>('0' + m % 10);
        out[7] = '-';
        out[8] = static_cast<char>('0' + d / 10);
        out[9] = static_cast<char>('0' + d % 10);
        out[10] = '\0';
    }

    const char* InputName (const LutInputSpaceCtrl v) noexcept
    {
        return (LutInputSpaceCtrl::eLUT_IN_SRGB == v) ? "srgb"
             : (LutInputSpaceCtrl::eLUT_IN_ACESCCT == v) ? "acescct" : "rec709";
    }

    const char* OutputName (const LutOutputSpaceCtrl v) noexcept
    {
        return (LutOutputSpaceCtrl::eLUT_OUT_SRGB == v) ? "srgb" : "rec709";
    }

    // "name=value" for the LUT-relevant controls that differ from the defaults
    void AppendNonDefault (std::string& s, const AlgoControls& c)
    {
        const AlgoControls d = getAlgoControlsDefault();
        bool any = false;
        auto sep = [&]() { s += any ? ", " : ""; any = true; };
        auto num = [&](const char* name, const double v, const double dv)
        {
            if (v != dv) { sep(); s += name; s.push_back('='); AppendNumber(s, v); }
        };
        if (c.printStock != d.printStock) { sep(); s += "printStock="; s += AlgoPrintStockKey(c.printStock); }
        if (c.dupeStock != d.dupeStock)   { sep(); s += "dupeStock=";  s += AlgoPrintStockKey(c.dupeStock); }
        num("generations",        static_cast<double>(c.generations), static_cast<double>(d.generations));
        num("exposureStops",      c.exposureStops,      d.exposureStops);
        num("exposureTimeS",      c.exposureTimeS,      d.exposureTimeS);
        num("processVariant",     static_cast<double>(static_cast<int32_t>(c.processVariant)),
                                  static_cast<double>(static_cast<int32_t>(d.processVariant)));
        num("batchPosition",      c.batchPosition,      d.batchPosition);
        num("developmentMinutes", c.developmentMinutes, d.developmentMinutes);
        num("developmentCelsius", c.developmentCelsius, d.developmentCelsius);
        num("developerIndex",     static_cast<double>(static_cast<int32_t>(c.developerIndex)),
                                  static_cast<double>(static_cast<int32_t>(d.developerIndex)));
        num("storageYears",       c.storageYears,       d.storageYears);
        num("storageCelsius",     c.storageCelsius,     d.storageCelsius);
        num("filmAgeMonths",      c.filmAgeMonths,      d.filmAgeMonths);
        num("scannerSpecular",    c.scannerSpecular,    d.scannerSpecular);
        num("sceneKelvin",        c.sceneKelvin,        d.sceneKelvin);
        num("wbStrength",         c.wbStrength,         d.wbStrength);
        num("greyTarget",         c.greyTarget,         d.greyTarget);
        num("blackPointStretch",  c.blackPointStretch,  d.blackPointStretch);
        num("couplerScale",       c.couplerScale,       d.couplerScale);
        num("spectralTaking",     c.spectralTaking ? 1.0 : 0.0,     d.spectralTaking ? 1.0 : 0.0);
        num("spectralUpsampling", c.spectralUpsampling ? 1.0 : 0.0, d.spectralUpsampling ? 1.0 : 0.0);
        num("aimExposure",        c.aimExposure ? 1.0 : 0.0,        d.aimExposure ? 1.0 : 0.0);
        if (!any) s += "none";
    }
}


LutGeneratorControls getLutGeneratorControlsDefault () noexcept
{
    LutGeneratorControls c;
    c.size          = LutSizeCtrl::eLUT_SIZE_MEDIUM;
    c.inputSpace    = LutInputSpaceCtrl::eLUT_IN_REC709;
    c.outputSpace   = LutOutputSpaceCtrl::eLUT_OUT_REC709;
    c.interpolation = LutInterpolationCtrl::eLUT_INTERP_TRILINEAR;
    return c;
}


int32_t FilmSimLutEdge (const LutSizeCtrl size) noexcept
{
    switch (size)
    {
    case LutSizeCtrl::eLUT_SIZE_SMALL:  return 17;
    case LutSizeCtrl::eLUT_SIZE_MEDIUM: return 33;
    case LutSizeCtrl::eLUT_SIZE_BIG:    return 65;
    case LutSizeCtrl::eLUT_SIZE_HUGE:   return 96;
    default:                            return 0;
    }
}


std::vector<LutRGB> FilmSimLutGenerator (const MemHandler&           memHandler,
                                         const AlgoControls&         algoCtrl,
                                         const LutGeneratorControls& lutCtrl)
{
    std::vector<LutRGB> lut;
    const int32_t n = FilmSimLutEdge(lutCtrl.size);
    const int32_t sizeX = memHandler.activeW;
    const int32_t sizeY = memHandler.activeH;
    const int32_t idx = static_cast<int32_t>(algoCtrl.filmProfile);
    if ((n <= 1) || (false == ValidControls(lutCtrl))
        || (nullptr == memHandler.SuperBufferHead)
        || (sizeX <= 0) || (sizeY <= 0)
        || (nullptr == memHandler.pProfileDb) || (idx < 0) || (idx >= memHandler.profileCount))
    {
        return lut;
    }

    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n)
                            * static_cast<std::size_t>(n);
    try
    {
        lut.resize(total);
    }
    catch (const std::bad_alloc&)
    {
        return std::vector<LutRGB>();
    }

    // A private copy of each: the caller's handler and controls are untouched.
    MemHandler mem = memHandler;
    mem.lutMode = 1;
    mem.Freq.dcOnly = 1;
    const AlgoControls ctrl = LutControls(algoCtrl);

    const std::ptrdiff_t pitch = memHandler.padW;
    const std::size_t capacity = static_cast<std::size_t>(sizeX) * static_cast<std::size_t>(sizeY);
    const double step = 1.0 / static_cast<double>(n - 1);

    for (std::size_t first = 0; first < total; first += capacity)
    {
        const std::size_t count = (total - first < capacity) ? (total - first) : capacity;

        // Fill the frame: lattice colours first, mid grey in the unused rest
        // (every pixel is independent in LUT mode, so the filler is inert).
        for (std::size_t p = 0; p < capacity; p++)
        {
            double lin[3] = { 0.18, 0.18, 0.18 };
            if (p < count)
            {
                const std::size_t i = first + p;
                const std::size_t ir = i % static_cast<std::size_t>(n);
                const std::size_t ig = (i / static_cast<std::size_t>(n)) % static_cast<std::size_t>(n);
                const std::size_t ib = i / (static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
                const double code[3] = { static_cast<double>(ir) * step,
                                         static_cast<double>(ig) * step,
                                         static_cast<double>(ib) * step };
                DecodeInput(lutCtrl.inputSpace, code, lin);
            }
            const std::ptrdiff_t y = static_cast<std::ptrdiff_t>(p / static_cast<std::size_t>(sizeX));
            const std::ptrdiff_t x = static_cast<std::ptrdiff_t>(p % static_cast<std::size_t>(sizeX));
            const std::ptrdiff_t o = y * pitch + x;
            mem.Src_R[o] = static_cast<ImgType>(lin[0]);
            mem.Src_G[o] = static_cast<ImgType>(lin[1]);
            mem.Src_B[o] = static_cast<ImgType>(lin[2]);
        }

        Algorithm_Main(mem, sizeX, sizeY, ctrl);

        for (std::size_t p = 0; p < count; p++)
        {
            const std::ptrdiff_t y = static_cast<std::ptrdiff_t>(p / static_cast<std::size_t>(sizeX));
            const std::ptrdiff_t x = static_cast<std::ptrdiff_t>(p % static_cast<std::size_t>(sizeX));
            const std::ptrdiff_t o = y * pitch + x;
            LutRGB& e = lut[first + p];
            e.r = EncodeOutput(lutCtrl.outputSpace, static_cast<double>(mem.Dst_R[o]));
            e.g = EncodeOutput(lutCtrl.outputSpace, static_cast<double>(mem.Dst_G[o]));
            e.b = EncodeOutput(lutCtrl.outputSpace, static_cast<double>(mem.Dst_B[o]));
        }
    }
    return lut;
}


std::string FilmSimLutToCubeText (const std::vector<LutRGB>&  lut,
                                  const MemHandler&           memHandler,
                                  const AlgoControls&         algoCtrl,
                                  const LutGeneratorControls& lutCtrl,
                                  const char*                 dateIso)
{
    std::string s;
    const int32_t n = FilmSimLutEdge(lutCtrl.size);
    const int32_t idx = static_cast<int32_t>(algoCtrl.filmProfile);
    if ((n <= 1) || (false == ValidControls(lutCtrl))
        || (lut.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(n))
        || (nullptr == memHandler.pProfileDb) || (idx < 0) || (idx >= memHandler.profileCount))
    {
        return s;
    }

    try
    {
        const std::string& stock = memHandler.pProfileDb[idx].name;
        char today[11];
        const char* date = dateIso;
        if ((nullptr == date) || ('\0' == date[0]))
        {
            TodayIso(today);
            date = today;
        }
        int year = 2026;
        if ((nullptr != date) && date[0] >= '0' && date[0] <= '9' && date[1] && date[2] && date[3])
            year = (date[0] - '0') * 1000 + (date[1] - '0') * 100 + (date[2] - '0') * 10 + (date[3] - '0');

        s.reserve(static_cast<std::size_t>(n) * n * n * 40 + 2048);

        s += "TITLE \""; s += stock; s.push_back(' ');
        s += InputName(lutCtrl.inputSpace); s += "->"; s += OutputName(lutCtrl.outputSpace);
        s.push_back(' '); AppendUInt(s, static_cast<unsigned long long>(n)); s += "\"\n";

        s += "# Generated on "; s += date;
        s += " by Film Simulator, developed by Marat Shchuchinsky.\n";
        s += "# Copyright (c) 2026";
        if (year > 2026) { s.push_back('-'); AppendUInt(s, static_cast<unsigned long long>(year)); }
        s += " Marat Shchuchinsky. All rights reserved.\n";
        s += "# No part of this LUT may be redistributed or modified without the prior "
             "written permission of Marat Shchuchinsky.\n";
        s += "# generated by FilmSimLutGenerator (C++ engine), schema v";
        AppendUInt(s, static_cast<unsigned long long>(FilmDatabaseSchemaVersion()));
        s += "\n# stock: "; s += stock;
        s += "\n# input: "; s += InputName(lutCtrl.inputSpace);
        s += "   output: "; s += OutputName(lutCtrl.outputSpace);
        s += "   size: "; AppendUInt(s, static_cast<unsigned long long>(n));
        s += "\n# interpolation: ";
        s += (LutInterpolationCtrl::eLUT_INTERP_TETRAHEDRAL == lutCtrl.interpolation) ? "tetrahedral" : "trilinear";
        s += " intended (set it in the host; the table holds exact samples)\n";
        s += "# settings (non-default): "; AppendNonDefault(s, algoCtrl); s.push_back('\n');
        s += "# excluded (not representable in a LUT): grain, print grain, halation, MTF / blur, "
             "scan misregistration, veiling flare, vignette, coating field / gate buckle / edge fog, "
             "reseau mosaic, damage / gate weave / flicker\n";
        s += "LUT_3D_SIZE "; AppendUInt(s, static_cast<unsigned long long>(n)); s.push_back('\n');
        s += "DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n";
        for (std::size_t i = 0; i < lut.size(); i++)
        {
            AppendFixed(s, lut[i].r, 10); s.push_back(' ');
            AppendFixed(s, lut[i].g, 10); s.push_back(' ');
            AppendFixed(s, lut[i].b, 10); s.push_back('\n');
        }
    }
    catch (...)
    {
        return std::string();
    }
    return s;
}
