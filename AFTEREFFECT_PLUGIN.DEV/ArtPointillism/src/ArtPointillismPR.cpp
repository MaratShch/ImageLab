#include "ArtPointillism.hpp"
#include "ArtPointillismAlgo.hpp"
#include "ArtPointillismControl.hpp"
#include "ArtPointillismEnums.hpp"
#include "ImageLabMemInterface.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
)
{
	PF_Err err = PF_Err_NONE;
	PF_Err errFormat = PF_Err_INVALID_INDEX;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_INPUT)]->u.ld);
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;

    // Allocate memory storage for store temporary results
    constexpr A_long doubleBuf = 2 * FastCompute::Max(static_cast<A_long>(sizeof(fRGB)), static_cast<A_long>(sizeof(fCIELabPix)));
    const A_long singleTmpFrameSize = sizeX * sizeY;
    const A_long totalProcMem  = CreateAlignment(singleTmpFrameSize * doubleBuf, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fRGB* pTmpBuf1 = static_cast<fRGB*>(pMemoryBlock);
        fRGB* pTmpBuf2 = pTmpBuf1 + singleTmpFrameSize;
        fCIELabPix* pCieLabBuf = static_cast<fCIELabPix*>(pMemoryBlock); // Aliased pointer!!!

        // This plugin called frop PR - check video fomat
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            const PontillismControls algoControls = GetControlParametersStruct (params);

            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                    const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
                    constexpr float fCoeff{ static_cast<float>(u8_value_white) };

                    // convert to CieLAB color space
                    ConvertToCIELab_BGRA_8u (localSrc, pCieLabBuf, sizeX, sizeY, linePitch, sizeX);

                    // back convert to native buffer format after processing complete
                    ConvertFromCIELab (localSrc, pCieLabBuf, localDst, sizeX, sizeY, linePitch, sizeX, linePitch);
                }
                break;

                case PrPixelFormat_BGRA_4444_16u:
                {
                    const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
                    constexpr float fCoeff{ static_cast<float>(u16_value_white) };

                    // convert to CieLAB color space
                    ConvertToCIELab (localSrc, pCieLabBuf, sizeX, sizeY, linePitch, sizeX);

                    // back convert to native buffer format after processing complete
                    ConvertFromCIELab (localSrc, pCieLabBuf, localDst, sizeX, sizeY, linePitch, sizeX, linePitch);
                }
                break;

                case PrPixelFormat_BGRA_4444_32f:
                {
                    const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
                    constexpr float fCoeff{ 1.f };

                    // convert to CieLAB color space
                    ConvertToCIELab_BGRA_32f (localSrc, pCieLabBuf, sizeX, sizeY, linePitch, sizeX);

                    // back convert to native buffer format after processing complete
                    ConvertFromCIELab (localSrc, pCieLabBuf, localDst, sizeX, sizeY, linePitch, sizeX, linePitch);
                }
                break;

                case PrPixelFormat_VUYA_4444_8u_709:
                case PrPixelFormat_VUYA_4444_8u:
                {
                    const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                    constexpr float fCoeff{ static_cast<float>(u8_value_white) };

                    // convert to CieLAB color space
                    ConvertToCIELab (localSrc, pCieLabBuf, sizeX, sizeY, linePitch, sizeX);
 
                    // back convert to native buffer format after processing complete
                    ConvertFromCIELab (localSrc, pCieLabBuf, localDst, sizeX, sizeY, linePitch, sizeX, linePitch);
                }
                break;

                case PrPixelFormat_VUYA_4444_32f_709:
                case PrPixelFormat_VUYA_4444_32f:
                {
                    const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                    constexpr float fCoeff{ static_cast<float>(1.f) };

                    // convert to CieLAB color space
                    ConvertToCIELab (localSrc, pCieLabBuf, sizeX, sizeY, linePitch, sizeX);
                    
                    // back convert to native buffer format after processing complete
                    ConvertFromCIELab (localSrc, pCieLabBuf, localDst, sizeX, sizeY, linePitch, sizeX, linePitch);
                }
                break;

                case PrPixelFormat_RGB_444_10u:
                {
                    const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
                          PF_Pixel_RGB_10u* __restrict localDst = reinterpret_cast<      PF_Pixel_RGB_10u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);
                    constexpr float fCoeff{ static_cast<float>(u10_value_white) };

                    // convert to CieLAB color space
                    ConvertToCIELab (localSrc, pCieLabBuf, sizeX, sizeY, linePitch, sizeX);

                    // back convert to native buffer format after processing complete
                    ConvertFromCIELab (pCieLabBuf, localDst, sizeX, sizeY, sizeX, linePitch);
                }
                break;

                default:
                    err = PF_Err_INTERNAL_STRUCT_DAMAGED;
                break;
            } // switch (destinationPixelFormat)

        } // if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        else
        {
            // error in determine pixel format
            err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
        }

        pMemoryBlock = nullptr;
        pTmpBuf1 = pTmpBuf2 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    } // if (nullptr != pMemoryBlock && blockId >= 0)
    else
        err = PF_Err_OUT_OF_MEMORY;

	return err;
}
