#include "ArtMosaic.hpp"
#include "ArtMosaicEnum.hpp"
#include "MosaicMemHandler.hpp"
#include "MosaicColorConvert.hpp"
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
	PF_Err errFormat{ PF_Err_INVALID_INDEX };
	PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_INPUT)]->u.ld);
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
    const A_long lineRawPitch = pfLayer->rowbytes;

    const A_long cellsNumber = 1000; // value should be read from slider

    MemHandler algoMemHandler = alloc_memory_buffers (sizeX, sizeY, cellsNumber);
    if (true == mem_handler_valid(algoMemHandler))
    {
        /* This plugin called frop PR - check video fomat */
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                    const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
                    const A_long linePitch = lineRawPitch / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

                    rgb2planar (localSrc, algoMemHandler, sizeX, sizeY, linePitch);     // convewrt interleaved to planar format (range 0.f ... 225.f)
                    MosaicAlgorithmMain (algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
                    planar2rgb (localSrc, algoMemHandler, localDst, sizeX, sizeY, linePitch, linePitch); // back convert from planar to interleaved format
                }
                break;

                case PrPixelFormat_BGRA_4444_16u:
                break;

                case PrPixelFormat_BGRA_4444_32f:
                break;

                case PrPixelFormat_VUYA_4444_8u_709:
                case PrPixelFormat_VUYA_4444_8u:
                break;

                case PrPixelFormat_VUYA_4444_32f_709:
                case PrPixelFormat_VUYA_4444_32f:
                break;

                case PrPixelFormat_RGB_444_10u:
                break;

                default:
                break;
            } // switch (destinationPixelFormat)/

            free_memory_buffers(algoMemHandler);

        } // if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))/
        else
        {
            // error in determine pixel format/
            err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
        }
    } // if (true == mem_handler_valid(algoMemHandler))
    else
        err = PF_Err_OUT_OF_MEMORY;

	return err;
}
