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

