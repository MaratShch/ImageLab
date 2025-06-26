#include "ColorCorrectionCIELab.hpp"
#include "ColorCorrectionCIELabEnums.hpp"
#include "ColorTransform.hpp"
#include "PrSDKAESupport.h"


PF_Err ColorCorrectionCieLABInAe_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	/* get sliders values */
	auto const L_coarse = params[eCIELAB_SLIDER_L_COARSE]->u.sd.value;
	auto const L_fine   = params[eCIELAB_SLIDER_L_FINE  ]->u.fs_d.value;
	auto const A_coarse = params[eCIELAB_SLIDER_A_COARSE]->u.sd.value;
	auto const A_fine   = params[eCIELAB_SLIDER_A_FINE  ]->u.fs_d.value;
	auto const B_coarse = params[eCIELAB_SLIDER_B_COARSE]->u.sd.value;
	auto const B_fine   = params[eCIELAB_SLIDER_B_FINE  ]->u.fs_d.value;

	const float L_level = static_cast<float>(static_cast<double>(L_coarse) + L_fine);
	const float A_level = static_cast<float>(static_cast<double>(A_coarse) + A_fine);
	const float B_level = static_cast<float>(static_cast<double>(B_coarse) + B_fine);

	PF_Err err = PF_Err_NONE;

	if ((0.f == L_level) && (0.f == A_level) && (0.f == B_level))
	{
		PF_EffectWorld* input = reinterpret_cast<PF_EffectWorld*>(&params[eCIELAB_INPUT]->u.ld);
		auto const& worldTransformSuite = AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data);
		err = worldTransformSuite->copy(in_data->effect_ref, input, output, NULL, NULL);
	}
	else
	{
		/* in case of processing enable - let's check illuminant and observer using for compute color transofrm */
		const eCOLOR_OBSERVER  iObserver   = static_cast<const eCOLOR_OBSERVER >(params[eCIELAB_POPUP_OBSERVER  ]->u.pd.value - 1);
		const eCOLOR_ILLUMINANT iIlluminant = static_cast<const eCOLOR_ILLUMINANT>(params[eCIELAB_POPUP_ILLUMINANT]->u.pd.value - 1);

		const PF_EffectWorld*    __restrict input   = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[eCIELAB_INPUT]->u.ld);
		const PF_Pixel_ARGB_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
		      PF_Pixel_ARGB_8u*  __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

		auto const src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
		auto const dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

		auto const sizeY = output->height;
		auto const sizeX = output->width;
		constexpr float reciproc = 1.f / static_cast<float>(u8_value_white);

		const float* __restrict fReferences = cCOLOR_ILLUMINANT[iObserver][iIlluminant];

		for (A_long j = 0; j < sizeY; j++)
		{
			const PF_Pixel_ARGB_8u* __restrict pSrcLine = localSrc + j * src_line_pitch;
			      PF_Pixel_ARGB_8u* __restrict pDstLine = localDst + j * dst_line_pitch;

			for (A_long i = 0; i < sizeX; i++)
			{
				/* convert RGB to CIELab */
				fRGB pixRGB;
				pixRGB.R = static_cast<float>(pSrcLine[i].R) * reciproc;
				pixRGB.G = static_cast<float>(pSrcLine[i].G) * reciproc;
				pixRGB.B = static_cast<float>(pSrcLine[i].B) * reciproc;

				fCIELabPix pixCIELab = RGB2CIELab(pixRGB, fReferences);

				/* add values from sliders */
				pixCIELab.L += L_level;
				pixCIELab.a += A_level;
				pixCIELab.b += B_level;

				pixCIELab.L = CLAMP_VALUE(pixCIELab.L, static_cast<float>(L_coarse_min_level), static_cast<float>(L_coarse_max_level));
				pixCIELab.a = CLAMP_VALUE(pixCIELab.a, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));
				pixCIELab.b = CLAMP_VALUE(pixCIELab.b, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));

				/* back convert to RGB */
				fRGB pixRGBOut = CIELab2RGB(pixCIELab, fReferences);
				pDstLine[i].B = static_cast<A_u_char>(CLAMP_VALUE(pixRGBOut.B * 255.f, static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
				pDstLine[i].G = static_cast<A_u_char>(CLAMP_VALUE(pixRGBOut.G * 255.f, static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
				pDstLine[i].R = static_cast<A_u_char>(CLAMP_VALUE(pixRGBOut.R * 255.f, static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
				pDstLine[i].A = pSrcLine[i].A;

			} /* for (A_long i = 0; i < sizeX; i++) */
		} /* for (A_long j = 0; j < sizeY; j++) */
	}
	return err;
}


PF_Err ColorCorrectionCieLABInAe_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	/* get sliders values */
	auto const L_coarse = params[eCIELAB_SLIDER_L_COARSE]->u.sd.value;
	auto const L_fine   = params[eCIELAB_SLIDER_L_FINE  ]->u.fs_d.value;
	auto const A_coarse = params[eCIELAB_SLIDER_A_COARSE]->u.sd.value;
	auto const A_fine   = params[eCIELAB_SLIDER_A_FINE  ]->u.fs_d.value;
	auto const B_coarse = params[eCIELAB_SLIDER_B_COARSE]->u.sd.value;
	auto const B_fine   = params[eCIELAB_SLIDER_B_FINE  ]->u.fs_d.value;

	const float L_level = static_cast<float>(static_cast<double>(L_coarse) + L_fine);
	const float A_level = static_cast<float>(static_cast<double>(A_coarse) + A_fine);
	const float B_level = static_cast<float>(static_cast<double>(B_coarse) + B_fine);

	PF_Err err = PF_Err_NONE;

	if ((0.f == L_level) && (0.f == A_level) && (0.f == B_level))
	{
		PF_EffectWorld* input = reinterpret_cast<PF_EffectWorld*>(&params[eCIELAB_INPUT]->u.ld);
		auto const& worldTransformSuite = AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data);
		err = worldTransformSuite->copy_hq (in_data->effect_ref, input, output, NULL, NULL);
	}
	else
	{
		/* in case of processing enable - let's check illuminant and observer using for compute color transofrm */
		const eCOLOR_OBSERVER  iObserver   = static_cast<const eCOLOR_OBSERVER >(params[eCIELAB_POPUP_OBSERVER  ]->u.pd.value - 1);
		const eCOLOR_ILLUMINANT iIlluminant = static_cast<const eCOLOR_ILLUMINANT>(params[eCIELAB_POPUP_ILLUMINANT]->u.pd.value - 1);

		const PF_EffectWorld*     __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[eCIELAB_INPUT]->u.ld);
		const PF_Pixel_ARGB_16u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
		      PF_Pixel_ARGB_16u*  __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

		auto const src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
		auto const dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

		auto const sizeY = output->height;
		auto const sizeX = output->width;
		constexpr float reciproc = 1.f / static_cast<float>(u16_value_white);

		const float* __restrict fReferences = cCOLOR_ILLUMINANT[iObserver][iIlluminant];

		for (A_long j = 0; j < sizeY; j++)
		{
			const PF_Pixel_ARGB_16u* __restrict pSrcLine = localSrc + j * src_line_pitch;
			      PF_Pixel_ARGB_16u* __restrict pDstLine = localDst + j * dst_line_pitch;

			for (A_long i = 0; i < sizeX; i++)
			{
				/* convert RGB to CIELab */
				fRGB pixRGB;
				pixRGB.R = static_cast<float>(pSrcLine[i].R) * reciproc;
				pixRGB.G = static_cast<float>(pSrcLine[i].G) * reciproc;
				pixRGB.B = static_cast<float>(pSrcLine[i].B) * reciproc;

				fCIELabPix pixCIELab = RGB2CIELab(pixRGB, fReferences);

				/* add values from sliders */
				pixCIELab.L += L_level;
				pixCIELab.a += A_level;
				pixCIELab.b += B_level;

				pixCIELab.L = CLAMP_VALUE(pixCIELab.L, static_cast<float>(L_coarse_min_level), static_cast<float>(L_coarse_max_level));
				pixCIELab.a = CLAMP_VALUE(pixCIELab.a, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));
				pixCIELab.b = CLAMP_VALUE(pixCIELab.b, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));

				/* back convert to RGB */
				fRGB pixRGBOut = CIELab2RGB(pixCIELab, fReferences);
				pDstLine[i].B = static_cast<A_u_short>(CLAMP_VALUE(pixRGBOut.B * static_cast<float>(u16_value_white), static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
				pDstLine[i].G = static_cast<A_u_short>(CLAMP_VALUE(pixRGBOut.G * static_cast<float>(u16_value_white), static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
				pDstLine[i].R = static_cast<A_u_short>(CLAMP_VALUE(pixRGBOut.R * static_cast<float>(u16_value_white), static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
				pDstLine[i].A = pSrcLine[i].A;

			} /* for (A_long i = 0; i < sizeX; i++) */
		} /* for (A_long j = 0; j < sizeY; j++) */
	}
	return err;
}


PF_Err ColorCorrectionCieLABInAe_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    /* get sliders values */
    auto const L_coarse = params[eCIELAB_SLIDER_L_COARSE]->u.sd.value;
    auto const L_fine   = params[eCIELAB_SLIDER_L_FINE]->u.fs_d.value;
    auto const A_coarse = params[eCIELAB_SLIDER_A_COARSE]->u.sd.value;
    auto const A_fine   = params[eCIELAB_SLIDER_A_FINE]->u.fs_d.value;
    auto const B_coarse = params[eCIELAB_SLIDER_B_COARSE]->u.sd.value;
    auto const B_fine   = params[eCIELAB_SLIDER_B_FINE]->u.fs_d.value;

    const float L_level = static_cast<float>(static_cast<double>(L_coarse) + L_fine);
    const float A_level = static_cast<float>(static_cast<double>(A_coarse) + A_fine);
    const float B_level = static_cast<float>(static_cast<double>(B_coarse) + B_fine);

    PF_Err err = PF_Err_NONE;

    if ((0.f == L_level) && (0.f == A_level) && (0.f == B_level))
    {
        PF_EffectWorld* input = reinterpret_cast<PF_EffectWorld*>(&params[eCIELAB_INPUT]->u.ld);
        auto const& worldTransformSuite = AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data);
        err = worldTransformSuite->copy_hq(in_data->effect_ref, input, output, NULL, NULL);
    }
    else
    {
        /* in case of processing enable - let's check illuminant and observer using for compute color transofrm */
        const eCOLOR_OBSERVER  iObserver = static_cast<const eCOLOR_OBSERVER >(params[eCIELAB_POPUP_OBSERVER]->u.pd.value - 1);
        const eCOLOR_ILLUMINANT iIlluminant = static_cast<const eCOLOR_ILLUMINANT>(params[eCIELAB_POPUP_ILLUMINANT]->u.pd.value - 1);

        const PF_EffectWorld*     __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[eCIELAB_INPUT]->u.ld);
        const PF_Pixel_ARGB_32f*  __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
              PF_Pixel_ARGB_32f*  __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

        auto const src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
        auto const dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

        auto const sizeY = output->height;
        auto const sizeX = output->width;

        const float* __restrict fReferences = cCOLOR_ILLUMINANT[iObserver][iIlluminant];

        for (A_long j = 0; j < sizeY; j++)
        {
            const PF_Pixel_ARGB_32f* __restrict pSrcLine = localSrc + j * src_line_pitch;
                  PF_Pixel_ARGB_32f* __restrict pDstLine = localDst + j * dst_line_pitch;

            for (A_long i = 0; i < sizeX; i++)
            {
                /* convert RGB to CIELab */
                fRGB pixRGB;
                pixRGB.R = pSrcLine[i].R;
                pixRGB.G = pSrcLine[i].G;
                pixRGB.B = pSrcLine[i].B;

                fCIELabPix pixCIELab = RGB2CIELab(pixRGB, fReferences);

                /* add values from sliders */
                pixCIELab.L += L_level;
                pixCIELab.a += A_level;
                pixCIELab.b += B_level;

                pixCIELab.L = CLAMP_VALUE (pixCIELab.L, static_cast<float>(L_coarse_min_level ), static_cast<float>(L_coarse_max_level));
                pixCIELab.a = CLAMP_VALUE (pixCIELab.a, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));
                pixCIELab.b = CLAMP_VALUE (pixCIELab.b, static_cast<float>(AB_coarse_min_level), static_cast<float>(AB_coarse_max_level));

                /* back convert to RGB */
                const fRGB pixRGBOut = CIELab2RGB(pixCIELab, fReferences);
                pDstLine[i].B = CLAMP_VALUE (pixRGBOut.B, f32_value_black, f32_value_white);
                pDstLine[i].G = CLAMP_VALUE (pixRGBOut.G, f32_value_black, f32_value_white);
                pDstLine[i].R = CLAMP_VALUE (pixRGBOut.R, f32_value_black, f32_value_white);
                pDstLine[i].A = pSrcLine[i].A;
            } /* for (A_long i = 0; i < sizeX; i++) */
        } /* for (A_long j = 0; j < sizeY; j++) */
    }
    return err;
}


inline PF_Err ColorCorrectionCieLABInAe_DeepWorld
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[eCIELAB_INPUT]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            ColorCorrectionCieLABInAe_32bits(in_data, out_data, params, output) : ColorCorrectionCieLABInAe_16bits(in_data, out_data, params, output));
    }
    else
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;

    return err;
}


PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept
{
	return (PF_WORLD_IS_DEEP(output) ?
        ColorCorrectionCieLABInAe_DeepWorld (in_data, out_data, params, output) :
		ColorCorrectionCieLABInAe_8bits (in_data, out_data, params, output));
}