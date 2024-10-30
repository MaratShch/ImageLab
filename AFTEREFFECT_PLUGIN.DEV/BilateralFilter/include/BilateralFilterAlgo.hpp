#ifndef __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_ALGO__
#define __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_ALGO__

#include "Common.hpp"
#include "Param_Utils.h"
#include "CompileTimeUtils.hpp"
#include "CommonPixFormat.hpp"
#include "BilateralFilterEnum.hpp"
#include "GaussMesh.hpp"



template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline T ClampPixelValue
(
    const T& input,
    const T& black,
    const T& white
) noexcept
{
    T output;
    output.R = CLAMP_VALUE(input.R, black.R, white.R);
    output.G = CLAMP_VALUE(input.G, black.G, white.G);
    output.B = CLAMP_VALUE(input.B, black.B, white.B);
    output.A = input.A; // not touch Alpha Channel 
    return output;
}


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
void BilateralFilterAlgorithm
(
    const T* __restrict pSrc,
          T* __restrict pDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    A_long fRadius,
    const T& blackPix,
    const T& whitePix
) noexcept
{
    A_long i, j;
    const GaussMesh* pMeshObj = getMeshHandler();
    const MeshT* __restrict pMesh = pMeshObj->geCenterMesh();

    for (j = 0; j < sizeY; j++)
    {
        for (i = 0; i < sizeX; i++)
        {

        }
    }
    return;
}


#endif // __IMAGE_LAB_BILATERAL_FILTER_STANDALONE_ALGO__