#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "CompileTimeUtils.hpp"
#include "CommonAuxPixFormat.hpp"
#include "AlgoControl.hpp"
#include "AlgoMemHandler.hpp"
#include "ColorLocus.hpp"
#include "AlgoSuperPixel.hpp"
#include "LinearLut/LinearLut.hpp"
#include "AlgoPrFormatIngest.hpp"
#include "AlgoPrFormatEgress.hpp"


PF_Err ColorTemperature_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
    static const auto& lut8  = LinLut_srgb_8bit_double ::LINEARIZE_LUT_SRGB_8BIT_F64;
    static const auto& lut16 = LinLut_srgb_16bit_double::LINEARIZE_LUT_SRGB_16BIT_F64;
    static const auto& lut10 = LinLut_srgb_10bit_double::LINEARIZE_LUT_SRGB_10BIT_F64;

    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);

    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
    PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY);
    if (true == mem_handler_valid(algoMemHandler))
    {
        const AlgoControls algoCtrl = getAlgoControlsDefault();
        const auto& locusGate = getLocusGate(obs_CIE_1931_2deg == algoCtrl.observer);

        SuperPixel<double> super{};     // computed in double, max accuracy
        CctDuv<double> cct_duv{};       // computed in double, max accuracy

        AlgoPrIngest::ingest_and_superpixel
        (
            localSrc,
            sizeX,
            sizeY,
            src_pitch,
            AlgoPrIngest::fmt_ARGB_4444_8u,
            lut8, lut16, lut10,
            locusGate,
            algoMemHandler.input_f32_interleaved,
            super,
            algoCtrl.confidenceMap
        );

        Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

        AlgoPrIngest::egress_from_linear_f32
        (
            (0 == algoCtrl.confidenceMap ? algoMemHandler.output_f32_interleaved : algoMemHandler.input_f32_interleaved),
            sizeX,
            sizeY,
            localDst,
            dst_pitch,
            AlgoPrIngest::fmt_ARGB_4444_8u,
            lut8, lut16, lut10,
            localSrc,
            sizeX,
            src_pitch
        );

        free_memory_buffers(algoMemHandler);
    }
    else
    {
        err = PF_Err_OUT_OF_MEMORY;
    }

    return err;
}


PF_Err ColorTemperature_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
    static const auto& lut8  = LinLut_srgb_8bit_double ::LINEARIZE_LUT_SRGB_8BIT_F64;
    static const auto& lut16 = LinLut_srgb_16bit_double::LINEARIZE_LUT_SRGB_16BIT_F64;
    static const auto& lut10 = LinLut_srgb_10bit_double::LINEARIZE_LUT_SRGB_10BIT_F64;

    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);

    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY);
    if (true == mem_handler_valid(algoMemHandler))
    {
        const AlgoControls algoCtrl = getAlgoControlsDefault();
        const auto& locusGate = getLocusGate(obs_CIE_1931_2deg == algoCtrl.observer);

        SuperPixel<double> super{};     // computed in double, max accuracy
        CctDuv<double> cct_duv{};       // computed in double, max accuracy

        AlgoPrIngest::ingest_and_superpixel
        (
            localSrc,
            sizeX,
            sizeY,
            src_pitch,
            AlgoPrIngest::fmt_ARGB_4444_16u,
            lut8, lut16, lut10,
            locusGate,
            algoMemHandler.input_f32_interleaved,
            super,
            algoCtrl.confidenceMap
        );

        Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

        AlgoPrIngest::egress_from_linear_f32
        (
            (0 == algoCtrl.confidenceMap ? algoMemHandler.output_f32_interleaved : algoMemHandler.input_f32_interleaved),
            sizeX,
            sizeY,
            localDst,
            dst_pitch,
            AlgoPrIngest::fmt_ARGB_4444_16u,
            lut8, lut16, lut10,
            localSrc,
            sizeX,
            src_pitch
        );

        free_memory_buffers(algoMemHandler);
    }
    else
    {
        err = PF_Err_OUT_OF_MEMORY;
    }

    return err;
}


PF_Err ColorTemperature_InAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    static const auto& lut8  = LinLut_srgb_8bit_double ::LINEARIZE_LUT_SRGB_8BIT_F64;
    static const auto& lut16 = LinLut_srgb_16bit_double::LINEARIZE_LUT_SRGB_16BIT_F64;
    static const auto& lut10 = LinLut_srgb_10bit_double::LINEARIZE_LUT_SRGB_10BIT_F64;

    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);

    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY);
    if (true == mem_handler_valid(algoMemHandler))
    {
        const AlgoControls algoCtrl = getAlgoControlsDefault();
        const auto& locusGate = getLocusGate(obs_CIE_1931_2deg == algoCtrl.observer);

        SuperPixel<double> super{};     // computed in double, max accuracy
        CctDuv<double> cct_duv{};       // computed in double, max accuracy

        AlgoPrIngest::ingest_and_superpixel
        (
            localSrc,
            sizeX,
            sizeY,
            src_pitch,
            AlgoPrIngest::fmt_ARGB_4444_32f,
            lut8, lut16, lut10,
            locusGate,
            algoMemHandler.input_f32_interleaved,
            super,
            algoCtrl.confidenceMap
        );

        Algorithm_Main(getCctHndl(), super, algoMemHandler, sizeX, sizeY, algoCtrl, cct_duv);

        AlgoPrIngest::egress_from_linear_f32
        (
            (0 == algoCtrl.confidenceMap ? algoMemHandler.output_f32_interleaved : algoMemHandler.input_f32_interleaved),
            sizeX,
            sizeY,
            localDst,
            dst_pitch,
            AlgoPrIngest::fmt_ARGB_4444_32f,
            lut8, lut16, lut10,
            localSrc,
            sizeX,
            src_pitch
        );

        free_memory_buffers(algoMemHandler);
    }
    else
    {
        err = PF_Err_OUT_OF_MEMORY;
    }

    return err;
}


inline PF_Err ColorTemperature_InAE_DeepWord
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    PF_Err	err = PF_Err_NONE;
    PF_PixelFormat format = PF_PixelFormat_INVALID;
    AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld), &format))
        err = (format == PF_PixelFormat_ARGB128 ?
            ColorTemperature_InAE_32bits(in_data, out_data, params, output) : ColorTemperature_InAE_16bits(in_data, out_data, params, output));
    else
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;

    return err;
}


PF_Err
ProcessImgInAE
(
	PF_InData*   in_data,
	PF_OutData*	 out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
	return (PF_WORLD_IS_DEEP(output) ?
      ColorTemperature_InAE_DeepWord(in_data, out_data, params, output) :
  	  ColorTemperature_InAE_8bits (in_data, out_data, params, output));
}