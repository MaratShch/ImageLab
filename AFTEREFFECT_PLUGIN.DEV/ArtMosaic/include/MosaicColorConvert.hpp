#pragma once

#include "ImageMosaicUtils.hpp"
#include "CommonPixFormat.hpp"


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


void rgb2planar
(
    const PF_Pixel_RGB_10u* RESTRICT pSrc,
    const MemHandler& memHndl,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch
);

void planar2rgb
(
    const MemHandler& memHndl,
    PF_Pixel_RGB_10u* RESTRICT pDst,
    A_long sizeX,
    A_long sizeY,
    A_long dstPitch
);


