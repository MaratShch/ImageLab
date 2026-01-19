#include "ArtPointillism.hpp"
#include "ArtPointillismAlgo.hpp"
#include "ArtPointillismControl.hpp"
#include "ArtPointillismEnums.hpp"
#include "ImageLabMemInterface.hpp"
#include "Avx2ColorConverts.hpp"
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

    MemHandler algoMemHandler = alloc_memory_buffers (sizeX, sizeY);
    if (true == mem_handler_valid(algoMemHandler))
    {
        float* __restrict srcL  = algoMemHandler.L;
        float* __restrict srcAB = algoMemHandler.ab;
        float* __restrict dstL  = algoMemHandler.dst_L;
        float* __restrict dstAB = algoMemHandler.dst_ab;

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

                    // convert to semi-planar CieLAB color space
                    AVX2_ConvertRgbToCIELab_SemiPlanar (localSrc, srcL, srcAB, sizeX, sizeY, linePitch, sizeX);

                    // execute algorithm
                    ArtPointillismAlgorithmExec (algoMemHandler, algoControls, sizeX, sizeY);

                    // back convert to native buffer format after processing complete
                    AVX2_ConvertCIELab_SemiPlanar_ToRgb (localSrc, dstL, dstAB, localDst, sizeX, sizeY, linePitch, linePitch);
                }
                break;

                case PrPixelFormat_BGRA_4444_16u:
                {
                    const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

                    // convert to semi-planar CieLAB color space
                    AVX2_ConvertRgbToCIELab_SemiPlanar (localSrc, srcL, srcAB, sizeX, sizeY, linePitch, sizeX);

                    // execute algorithm
                    ArtPointillismAlgorithmExec (algoMemHandler, algoControls, sizeX, sizeY);

                    // back convert to native buffer format after processing complete
                    AVX2_ConvertCIELab_SemiPlanar_ToRgb (localSrc, dstL, dstAB, localDst, sizeX, sizeY, linePitch, linePitch);
                }
                break;

                case PrPixelFormat_BGRA_4444_32f:
                {
                    const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

                    // convert to semi-planar CieLAB color space
                    AVX2_ConvertRgbToCIELab_SemiPlanar(localSrc, srcL, srcAB, sizeX, sizeY, linePitch, sizeX);

                    // execute algorithm
                    ArtPointillismAlgorithmExec(algoMemHandler, algoControls, sizeX, sizeY);

                    // back convert to native buffer format after processing complete
                    AVX2_ConvertCIELab_SemiPlanar_ToRgb(localSrc, dstL, dstAB, localDst, sizeX, sizeY, linePitch, linePitch);
                }
                break;

                case PrPixelFormat_VUYA_4444_8u_709:
                case PrPixelFormat_VUYA_4444_8u:
                {
                    const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
                    const bool isBT709 = (PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat);

                    // convert to semi-planar CieLAB color space
                    AVX2_ConvertVuyaToCIELab_SemiPlanar (localSrc, srcL, srcAB, sizeX, sizeY, linePitch, sizeX, isBT709);

                    // execute algorithm
                    ArtPointillismAlgorithmExec (algoMemHandler, algoControls, sizeX, sizeY);

                    // back convert to native buffer format after processing complete
                    AVX2_ConvertCIELab_SemiPlanar_To_YUV (localSrc, dstL, dstAB, localDst, sizeX, sizeY, linePitch, linePitch, isBT709);
                }
                break;

                case PrPixelFormat_VUYA_4444_32f_709:
                case PrPixelFormat_VUYA_4444_32f:
                {
                    const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                          PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
                    const bool isBT709 = (PrPixelFormat_VUYA_4444_32f_709 == destinationPixelFormat);

                    // convert to semi-planar CieLAB color space
                    AVX2_ConvertVuyaToCIELab_SemiPlanar (localSrc, srcL, srcAB, sizeX, sizeY, linePitch, sizeX, isBT709);

                    // execute algorithm
                    ArtPointillismAlgorithmExec (algoMemHandler, algoControls, sizeX, sizeY);

                    // back convert to native buffer format after processing complete
                    AVX2_ConvertCIELab_SemiPlanar_To_YUV (localSrc, dstL, dstAB, localDst, sizeX, sizeY, linePitch, linePitch, isBT709);
                }
                break;

                case PrPixelFormat_RGB_444_10u:
                {
                    const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
                          PF_Pixel_RGB_10u* __restrict localDst = reinterpret_cast<      PF_Pixel_RGB_10u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

                    // convert to drmi-planat CieLAB color space
                    AVX2_ConvertRgbToCIELab_SemiPlanar(localSrc, srcL, srcAB, sizeX, sizeY, linePitch, sizeX);

                    // execute algorithm
                    ArtPointillismAlgorithmExec(algoMemHandler, algoControls, sizeX, sizeY);

                    // back convert to native buffer format after processing complete
                    AVX2_ConvertCIELab_SemiPlanar_To_Rgb (dstL, dstAB, localDst, sizeX, sizeY, linePitch);
                }
                break;

                case PrPixelFormat_ARGB_4444_8u:
                {
                    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(pfLayer->data);
                          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                    // convert to semi-planar CieLAB color space
                    AVX2_ConvertRgbToCIELab_SemiPlanar(localSrc, srcL, srcAB, sizeX, sizeY, linePitch, sizeX);

                    // execute algorithm
                    ArtPointillismAlgorithmExec(algoMemHandler, algoControls, sizeX, sizeY);

                    // back convert to native buffer format after processing complete
                    AVX2_ConvertCIELab_SemiPlanar_ToRgb (localSrc, dstL, dstAB, localDst, sizeX, sizeY, linePitch, linePitch);
                }
                break;

                case PrPixelFormat_ARGB_4444_16u:
                {
                    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(pfLayer->data);
                          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                    // convert to semi-planar CieLAB color space
                    AVX2_ConvertRgbToCIELab_SemiPlanar(localSrc, srcL, srcAB, sizeX, sizeY, linePitch, sizeX);

                    // execute algorithm
                    ArtPointillismAlgorithmExec(algoMemHandler, algoControls, sizeX, sizeY);

                    // back convert to native buffer format after processing complete
                    AVX2_ConvertCIELab_SemiPlanar_ToRgb(localSrc, dstL, dstAB, localDst, sizeX, sizeY, linePitch, linePitch);
                }
                break;

                case PrPixelFormat_ARGB_4444_32f:
                {
                    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(pfLayer->data);
                          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

                    // convert to semi-planar CieLAB color space
                    AVX2_ConvertRgbToCIELab_SemiPlanar(localSrc, srcL, srcAB, sizeX, sizeY, linePitch, sizeX);

                    // execute algorithm
                    ArtPointillismAlgorithmExec(algoMemHandler, algoControls, sizeX, sizeY);

                    // back convert to native buffer format after processing complete
                    AVX2_ConvertCIELab_SemiPlanar_ToRgb(localSrc, dstL, dstAB, localDst, sizeX, sizeY, linePitch, linePitch);
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

        free_memory_buffers (algoMemHandler);

    } // if (true == mem_handler_valid (algoMemHandler))
    else
        err = PF_Err_OUT_OF_MEMORY;

	return err;
}
