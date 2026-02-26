#include "ImageLabDenoise.hpp"
#include "ImageLabDenoiseEnum.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgoControls.hpp"
#include "AlgorithmMain.hpp"
#include "AVX2_AlgoColorConvert.hpp"
#include "ColorConvert.hpp"
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

    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_INPUT)]->u.ld);
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long sizeX = pfLayer->extent_hint.right - pfLayer->extent_hint.left;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY);
    if (true == mem_handler_valid(algoMemHandler))
    {
        // This plugin called frop PR - check video fomat
        auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
        {
            const AlgoControls algoControls = GetControlParametersStruct (params);

            switch (destinationPixelFormat)
            {
                case PrPixelFormat_BGRA_4444_8u:
                {
                    const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
                          PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

                    // convert BGRA_8u interleaved buffer to YUV (Orthonormal) planar format
                    AVX2_Convert_BGRA_8u_YUV (localSrc, algoMemHandler.Y_planar, algoMemHandler.U_planar, algoMemHandler.V_planar, sizeX, sizeY, linePitch);

                    // call algorithm flow
                    Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

                    // convert denoised image to BGRA_8u interleaved output buffer
                    AVX2_Convert_YUV_to_BGRA_8u (algoMemHandler.Accum_Y, algoMemHandler.Accum_U, algoMemHandler.Accum_V, localSrc, localDst, sizeX, sizeY, linePitch, linePitch);
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

                    // convert BGRA_8u interleaved buffer to YUV (Orthonormal) planar format
                    AVX2_Convert_VUYA_8u_YUV (localSrc, algoMemHandler.Y_planar, algoMemHandler.U_planar, algoMemHandler.V_planar, sizeX, sizeY, linePitch, isBT709);

                    // call algorithm flow
                    Algorithm_Main(algoMemHandler, sizeX, sizeY, algoControls);

                    // convert denoised image to BGRA_8u interleaved output buffer
                    AVX2_Convert_YUV_to_VUYA_8u (algoMemHandler.Accum_Y, algoMemHandler.Accum_U, algoMemHandler.Accum_V, localSrc, localDst, sizeX, sizeY, linePitch, linePitch, isBT709);
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

                case PrPixelFormat_ARGB_4444_8u:
                {
                    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(pfLayer->data);
                    PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                }
                break;

                case PrPixelFormat_ARGB_4444_16u:
                {
                    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(pfLayer->data);
                    PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                }
                break;

                case PrPixelFormat_ARGB_4444_32f:
                {
                    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(pfLayer->data);
                    PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);
                    const A_long linePitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

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

        free_memory_buffers(algoMemHandler);

    } // if (true == mem_handler_valid (algoMemHandler))
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}
