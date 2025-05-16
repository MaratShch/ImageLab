#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureAlgo.hpp"
#include "CompileTimeUtils.hpp"
#include "CommonAuxPixFormat.hpp"
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
	PF_Err err{ PF_Err_NONE };
    PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

    // This plugin called frop PR - check video fomat
    auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

    if (PF_Err_NONE == (err = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
    {
        const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);

        // Allocate memory storage for store temporary results
        const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
        const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;

        const A_long totalProcMem = CreateAlignment(sizeX * sizeY * static_cast<A_long>(sizeof(PixComponentsStr32)), CACHE_LINE);

        void* pMemoryBlock = nullptr;
        A_long blockId = ::GetMemoryBlock (totalProcMem, 0, &pMemoryBlock);

        if (nullptr != pMemoryBlock && blockId >= 0)
        {
            PixComponentsStr32* __restrict pTmpBuffer = static_cast<PixComponentsStr32* __restrict>(pMemoryBlock);

            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                    const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
                    constexpr float coeff = 1.f / static_cast<float>(u8_value_white);
//                    Convert2Linear_sRGB (localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff);
                }
                break;

                case PrPixelFormat_BGRA_4444_16u:
                {
                    const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
                    constexpr float coeff = 1.f / static_cast<float>(u16_value_white);
//                    Convert2Linear_sRGB (localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff);
                }
                break;

                case PrPixelFormat_BGRA_4444_32f:
                {
                    const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
                    constexpr float coeff = 1.f;
//                    Convert2Linear_sRGB (localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff);
                }
                break;

                case PrPixelFormat_VUYA_4444_8u_709:
                case PrPixelFormat_VUYA_4444_8u:
                {
                    const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                    constexpr float coeff = 1.f / static_cast<float>(u8_value_white);
//                    Convert2Linear_sRGB (localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff, destinationPixelFormat == PrPixelFormat_VUYA_4444_8u_709);
                }
                break;

                case PrPixelFormat_VUYA_4444_32f_709:
                case PrPixelFormat_VUYA_4444_32f:
                {
                    const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                    constexpr float coeff = 1.f;
//                    Convert2Linear_sRGB (localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff, destinationPixelFormat == PrPixelFormat_VUYA_4444_32f_709);
                }
                break;

                case PrPixelFormat_RGB_444_10u:
                {
                    const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
                          PF_Pixel_RGB_10u* __restrict localDst = reinterpret_cast<      PF_Pixel_RGB_10u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);
                    constexpr float coeff = 1.f / static_cast<float>(u10_value_white);
//                    Convert2Linear_sRGB (localSrc, pTmpBuffer, sizeX, sizeY, linePitch, sizeX, coeff);
                }
                break;

                default:
                    err = PF_Err_INTERNAL_STRUCT_DAMAGED;
                break;
            } // switch (destinationPixelFormat)

            ::FreeMemoryBlock (blockId);
            blockId = -1;
            pMemoryBlock = nullptr;

        } // if (nullptr != pMemoryBlock && blockId >= 0)

    } // if (PF_Err_NONE == (err = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))

	return err;
}
