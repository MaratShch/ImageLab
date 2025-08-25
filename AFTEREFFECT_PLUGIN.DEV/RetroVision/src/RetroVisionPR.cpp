#include "CompileTimeUtils.hpp"
#include "RetroVision.hpp"
#include "RetroVisionEnum.hpp"
#include "PrSDKAESupport.h"
#include "ImageLabMemInterface.hpp"
#include "CommonAuxPixFormat.hpp"

PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) 
{
    // check if effect is enabled by CheckBox
    const A_long isEnabled = params[UnderlyingType(RetroVision::eRETRO_VISION_ENABLE)]->u.bd.value;
    if (0 == isEnabled)
        return PF_COPY(&params[UnderlyingType(RetroVision::eRETRO_VISION_INPUT)]->u.ld, output, NULL, NULL);

    PF_Err err{ PF_Err_NONE };
    PF_Err errFormat{ PF_Err_INVALID_INDEX };
    PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[UnderlyingType(RetroVision::eRETRO_VISION_INPUT)]->u.ld);
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;

    // Allocate memory storage for store temporary results
    const A_long singleTmpFrameSize = sizeX * sizeY;
    constexpr A_long doubleBuf = 2 * static_cast<A_long>(sizeof(fRGB));
    const A_long totalProcMem = CreateAlignment (singleTmpFrameSize * doubleBuf, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fRGB* __restrict pTmpBuf1 = static_cast<fRGB* __restrict>(pMemoryBlock);
        fRGB* __restrict pTmpBuf2 = pTmpBuf1 + singleTmpFrameSize;

        // This plugin called frop PR - check video fomat
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                    const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

                }
                break;

                case PrPixelFormat_BGRA_4444_16u:
                {
                    const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

                }
                break;

                case PrPixelFormat_BGRA_4444_32f:
                {
                    const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

                }
                break;

                case PrPixelFormat_VUYA_4444_8u_709:
                case PrPixelFormat_VUYA_4444_8u:
                {
                    const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                    const bool isBT709 = (PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat);

                }
                break;

                case PrPixelFormat_VUYA_4444_32f_709:
                case PrPixelFormat_VUYA_4444_32f:
                {
                    const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                    const bool isBT709 = (PrPixelFormat_VUYA_4444_32f_709 == destinationPixelFormat);

                }
                break;

                case PrPixelFormat_RGB_444_10u:
                {
                    const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
                          PF_Pixel_RGB_10u* __restrict localDst = reinterpret_cast<      PF_Pixel_RGB_10u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

                }
                break;

                default:
                    err = PF_Err_INTERNAL_STRUCT_DAMAGED;
                break;
            } /* switch (destinationPixelFormat) */

        } /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
        else
        {
            /* error in determine pixel format */
            err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
        }

        pMemoryBlock = nullptr;
        pTmpBuf1 = pTmpBuf2 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    }
    else
        err = PF_Err_OUT_OF_MEMORY;

	return err;
}
