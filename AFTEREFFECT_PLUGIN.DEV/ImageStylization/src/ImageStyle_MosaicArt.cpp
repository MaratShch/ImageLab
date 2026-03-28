#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "SegmentationUtils.hpp"
#include "ImageAuxPixFormat.hpp"
#include "ImageMosaicUtils.hpp"


PF_Err PR_ImageStyle_MosaicArt
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

    const PF_LayerDef* pfLayer = reinterpret_cast<const PF_LayerDef*>(&params[IMAGE_STYLE_INPUT]->u.ld);
    const A_long cellsNumber = 1000;

    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long sizeX = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
    const A_long lineRawPitch = pfLayer->rowbytes;


    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, cellsNumber);
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

                rgb2planar(localSrc, algoMemHandler, sizeX, sizeY, linePitch);     // convert interleaved to planar format (range 0.f ... 225.f)
                MosaicAlgorithmMain(algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
                planar2rgb(localSrc, algoMemHandler, localDst, sizeX, sizeY, linePitch, linePitch); // back convert from planar to interleaved format
            }
            break;

            case PrPixelFormat_BGRA_4444_16u:
            {
                const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
                PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
                const A_long linePitch = lineRawPitch / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

                rgb2planar(localSrc, algoMemHandler, sizeX, sizeY, linePitch);     // convert interleaved to planar format (range 0.f ... 225.f)
                MosaicAlgorithmMain(algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
                planar2rgb(localSrc, algoMemHandler, localDst, sizeX, sizeY, linePitch, linePitch); // back convert from planar to interleaved format
            }
            break;

            case PrPixelFormat_BGRA_4444_32f:
            {
                const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
                PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
                const A_long linePitch = lineRawPitch / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

                rgb2planar(localSrc, algoMemHandler, sizeX, sizeY, linePitch);     // convert interleaved to planar format (range 0.f ... 225.f)
                MosaicAlgorithmMain(algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
                planar2rgb(localSrc, algoMemHandler, localDst, sizeX, sizeY, linePitch, linePitch); // back convert from planar to interleaved format
            }
            break;

            case PrPixelFormat_VUYA_4444_8u_709:
            case PrPixelFormat_VUYA_4444_8u:
            {
                const bool is709 = (PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat);
                const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
                PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
                const A_long linePitch = lineRawPitch / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

                vuya2planar(localSrc, algoMemHandler, sizeX, sizeY, linePitch, is709);     // convert interleaved to planar format (range 0.f ... 225.f)
                MosaicAlgorithmMain(algoMemHandler, sizeX, sizeY, cellsNumber);            // perform SLIC algorithm
                planar2vuya(localSrc, algoMemHandler, localDst, sizeX, sizeY, linePitch, linePitch, is709); // back convert from planar to interleaved format
            }
            break;

            case PrPixelFormat_VUYA_4444_32f_709:
            case PrPixelFormat_VUYA_4444_32f:
            {
                const bool is709 = (PrPixelFormat_VUYA_4444_32f_709 == destinationPixelFormat);
                const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
                PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
                const A_long linePitch = lineRawPitch / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

                vuya2planar(localSrc, algoMemHandler, sizeX, sizeY, linePitch, is709);     // convert interleaved to planar format (range 0.f ... 225.f)
                MosaicAlgorithmMain(algoMemHandler, sizeX, sizeY, cellsNumber);            // perform SLIC algorithm
                planar2vuya(localSrc, algoMemHandler, localDst, sizeX, sizeY, linePitch, linePitch, is709); // back convert from planar to interleaved format
            }
            break;

            case PrPixelFormat_ARGB_4444_8u:
            {
                const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(pfLayer->data);
                PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);
                const A_long linePitch = lineRawPitch / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                rgb2planar(localSrc, algoMemHandler, sizeX, sizeY, linePitch);     // convert interleaved to planar format (range 0.f ... 225.f)
                MosaicAlgorithmMain(algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
                planar2rgb(localSrc, algoMemHandler, localDst, sizeX, sizeY, linePitch, linePitch); // back convert from planar to interleaved format
            }
            break;

            case PrPixelFormat_ARGB_4444_16u:
            {
                const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(pfLayer->data);
                PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);
                const A_long linePitch = lineRawPitch / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                rgb2planar(localSrc, algoMemHandler, sizeX, sizeY, linePitch);     // convert interleaved to planar format (range 0.f ... 225.f)
                MosaicAlgorithmMain(algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
                planar2rgb(localSrc, algoMemHandler, localDst, sizeX, sizeY, linePitch, linePitch); // back convert from planar to interleaved format
            }
            break;

            case PrPixelFormat_ARGB_4444_32f:
            {
                const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(pfLayer->data);
                PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);
                const A_long linePitch = lineRawPitch / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                rgb2planar(localSrc, algoMemHandler, sizeX, sizeY, linePitch);     // convert interleaved to planar format (range 0.f ... 225.f)
                MosaicAlgorithmMain(algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
                planar2rgb(localSrc, algoMemHandler, localDst, sizeX, sizeY, linePitch, linePitch); // back convert from planar to interleaved format
            }
            break;

            default:
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
                break;
            } // switch (destinationPixelFormat)/

            free_memory_buffers(algoMemHandler);

        } // if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))/
        else
        {
            // error in determine pixel format
            err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
        }
    } // if (true == mem_handler_valid(algoMemHandler))
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}


PF_Err AE_ImageStyle_MosaicArt_ARGB_8u
(
    PF_InData*   __restrict in_data,
    PF_OutData*  __restrict out_data,
    PF_ParamDef* __restrict params[],
    PF_LayerDef* __restrict output
)
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    constexpr A_long cellsNumber = 1000;

    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, cellsNumber);
    if (true == mem_handler_valid(algoMemHandler))
    {
        rgb2planar(localSrc, algoMemHandler, sizeX, sizeY, src_pitch);     // convert interleaved to planar format (range 0.f ... 225.f)
        MosaicAlgorithmMain(algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
        planar2rgb(localSrc, algoMemHandler, localDst, sizeX, sizeY, src_pitch, dst_pitch); // back convert from planar to interleaved format

        free_memory_buffers(algoMemHandler);
    }
    else
    {
        err = PF_Err_OUT_OF_MEMORY;
    }

    return err;
}


PF_Err AE_ImageStyle_MosaicArt_ARGB_16u
(
    PF_InData*   __restrict in_data,
    PF_OutData*  __restrict out_data,
    PF_ParamDef* __restrict params[],
    PF_LayerDef* __restrict output
)
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    constexpr A_long cellsNumber = 1000;

    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, cellsNumber);
    if (true == mem_handler_valid(algoMemHandler))
    {
        rgb2planar (localSrc, algoMemHandler, sizeX, sizeY, src_pitch);     // convert interleaved to planar format (range 0.f ... 225.f)
        MosaicAlgorithmMain(algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
        planar2rgb(localSrc, algoMemHandler, localDst, sizeX, sizeY, src_pitch, dst_pitch); // back convert from planar to interleaved format

        free_memory_buffers(algoMemHandler);
    }
    else
    {
        err = PF_Err_OUT_OF_MEMORY;
    }

    return err;
}


PF_Err AE_ImageStyle_MosaicArt_ARGB_32f
(
    PF_InData*   __restrict in_data,
    PF_OutData*  __restrict out_data,
    PF_ParamDef* __restrict params[],
    PF_LayerDef* __restrict output
)
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    constexpr A_long cellsNumber = 1000;

    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, cellsNumber);
    if (true == mem_handler_valid(algoMemHandler))
    {
        rgb2planar(localSrc, algoMemHandler, sizeX, sizeY, src_pitch);     // convert interleaved to planar format (range 0.f ... 225.f)
        MosaicAlgorithmMain(algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
        planar2rgb(localSrc, algoMemHandler, localDst, sizeX, sizeY, src_pitch, dst_pitch); // back convert from planar to interleaved format

        free_memory_buffers(algoMemHandler);
    }
    else
    {
        err = PF_Err_OUT_OF_MEMORY;
    }

    return err;
}

