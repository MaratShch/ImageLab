#pragma once

#include "CommonPixFormat.hpp"
#include "ClassRestrictions.hpp"
#include "MosaicMemHandler.hpp"
#include <stack>
#include <queue>
#include <memory>

void MosaicAlgorithmMain(const MemHandler& memHndl, A_long width, A_long height, A_long K = 1000);

namespace ArtMosaic
{

    class Color final
    {
    public:
        float r, g, b;

        constexpr Color() noexcept : r(0), g(0), b(0) {};
        constexpr Color(const float r0, const float g0, const float b0) noexcept : r(r0), g(g0), b(b0) {};

        ~Color() = default;
    };


    void fillProcBuf(Color* pBuf, const A_long pixNumber, const float val) noexcept;
    void fillProcBuf(std::unique_ptr<Color[]>& pBuf, const A_long pixNumber, const float val) noexcept;
    void fillProcBuf(A_long* pBuf, const A_long pixNumber, const A_long val) noexcept;
    void fillProcBuf(std::unique_ptr<A_long[]>& pBuf, const A_long pixNumber, const A_long val) noexcept;
    void fillProcBuf(float* pBuf, const A_long pixNumber, const float val) noexcept;
    void fillProcBuf(std::unique_ptr<float[]>& pBuf, const A_long pixNumber, const float val) noexcept;


}; // ArtMosaic/


void rgb2planar
(
    const PF_Pixel_BGRA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void planar2rgb
(
    const PF_Pixel_BGRA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_BGRA_8u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch
);

void rgb2planar
(
    const PF_Pixel_ARGB_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void planar2rgb
(
    const PF_Pixel_ARGB_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_ARGB_8u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch
);

void rgb2planar
(
    const PF_Pixel_BGRA_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void planar2rgb
(
    const PF_Pixel_BGRA_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_BGRA_16u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch
);

void rgb2planar
(
    const PF_Pixel_ARGB_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void planar2rgb
(
    const PF_Pixel_ARGB_16u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_ARGB_16u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch
);

void rgb2planar
(
    const PF_Pixel_BGRA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void planar2rgb
(
    const PF_Pixel_BGRA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_BGRA_32f* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch
);

void rgb2planar
(
    const PF_Pixel_ARGB_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long linePitch
);

void planar2rgb
(
    const PF_Pixel_ARGB_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_ARGB_32f* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch
);


void vuya2planar
(
    const PF_Pixel_VUYA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    bool is709
);

void planar2vuya
(
    const PF_Pixel_VUYA_8u* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_VUYA_8u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    bool is709
);

void vuya2planar
(
    const PF_Pixel_VUYA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    bool is709
);

void planar2vuya
(
    const PF_Pixel_VUYA_32f* RESTRICT pSrc,
    const MemHandler& memHndl,
    PF_Pixel_VUYA_32f* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    bool is709
);
