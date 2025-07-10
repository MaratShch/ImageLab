#include "AuthomaticWhiteBalance.hpp"
#include "AlgCommonFunctions.hpp"
#include "AlgCorrectionMatrix.hpp"

bool ProcessImgInAE_8bits
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	CACHE_ALIGN float U_avg[gMaxCnt]{};
	CACHE_ALIGN float V_avg[gMaxCnt]{};
	PF_Pixel_ARGB_8u* __restrict pMem[2]{};

	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[AWB_INPUT]->u.ld);
	PF_Pixel_ARGB_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

	PF_Pixel_ARGB_8u* __restrict srcInput = nullptr;
	PF_Pixel_ARGB_8u* __restrict dstOutput = nullptr;

	const A_long height = output->height;
	const A_long width  = output->width;
	const A_long src_line_pitch = input->rowbytes  / PF_Pixel_ARGB_8u_size;
	const A_long dst_line_pitch = output->rowbytes / PF_Pixel_ARGB_8u_size;

	int32_t srcIdx = 0;
	int32_t dstIdx = 1;
	A_long srcPitch = 0;
	A_long dstPitch = 0;
	int32_t memBlockId = -1;

	/* acquire parameters */
	const eCOLOR_SPACE colorSpace = static_cast<eCOLOR_SPACE>(params[AWB_COLOR_SPACE_POPUP]->u.pd.value - 1);
	const eChromaticAdaptation chromatic = static_cast<eChromaticAdaptation>(params[AWB_CHROMATIC_POPUP]->u.pd.value - 1);
	const eILLUMINATE  illuminate = static_cast<eILLUMINATE> (params[AWB_ILLUMINATE_POPUP]->u.pd.value - 1);
	const int32_t sliderIterCnt = params[AWB_ITERATIONS_SLIDER]->u.sd.value;
	const int32_t sliderThreshold = params[AWB_THRESHOLD_SLIDER]->u.sd.value;
	const int32_t iterCnt = FastCompute::Min(sliderIterCnt, gMaxCnt);

	constexpr float reciproc100 = 1.0f / 100.f;
	float T = static_cast<float>(sliderThreshold) * reciproc100;
	float uAvg, vAvg;

	/* Get memory block */
	void* pMemoryBlock = nullptr;
	const int32_t blocksNumber = FastCompute::Min(2, (iterCnt - 1));

	if (blocksNumber)
	{
		const size_t  frameSize = width * height;
		const size_t  requiredMemSize = blocksNumber * PF_Pixel_ARGB_8u_size * frameSize;
		memBlockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemoryBlock);
        if (nullptr == pMemoryBlock || memBlockId < 0)
            return PF_Err_OUT_OF_MEMORY; // not enough memory for allocate temporary buffer for ALGO
        
        pMem[0] = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(pMemoryBlock);
		pMem[1] = (2 == blocksNumber) ? pMem[0] + frameSize : nullptr;
	}

	/* pass iterations in corresponding to slider value */
	for (A_long k = 0; k < iterCnt; k++)
	{
		if (0 == k)
		{
			srcInput = localSrc;
			dstIdx++;
			dstIdx &= 0x1;
			dstOutput = (1 == iterCnt) ? localDst : pMem[dstIdx];
			srcPitch = src_line_pitch;
			dstPitch = (1 == iterCnt) ? dst_line_pitch : width;
		}
		else if ((iterCnt - 1) == k)
		{
			srcInput = pMem[dstIdx];
			dstOutput = localDst;
			srcPitch = width;
			dstPitch = dst_line_pitch;
		} /* if (k > 0) */
		else
		{
			srcIdx = dstIdx;
			dstIdx++;
			dstIdx &= 0x1;
			srcInput = pMem[srcIdx];
			dstOutput = pMem[dstIdx];
			srcPitch = dstPitch = width;
		}

		uAvg = vAvg = 0.f;
		/* collect statistics about image and compute averages values for U and for V components */
		collect_rgb_statistics (srcInput, width, height, srcPitch, T, colorSpace, &uAvg, &vAvg);
		U_avg[k] = uAvg;
		V_avg[k] = vAvg;

		if (k > 0)
		{
			const float U_diff = U_avg[k] - U_avg[k - 1];
			const float V_diff = V_avg[k] - V_avg[k - 1];

			const float normVal = FastCompute::Sqrt(U_diff * U_diff + V_diff * V_diff);

			if (normVal < algAWBepsilon)
			{
				// U and V no longer improving, so just copy source to destination and break the loop
				simple_image_copy (srcInput, localDst, width, height, srcPitch, dst_line_pitch);

				if (-1 != memBlockId)
					::FreeMemoryBlock(memBlockId);
                memBlockId = -1;

				/* release temporary memory buffers on exit from function */
				return true; // U and V no longer improving
			}
		} /* if (k > 0) */

		  /* compute correction matrix */
		float correctionMatrix[3]{};
		compute_correction_matrix(uAvg, vAvg, colorSpace, illuminate, chromatic, correctionMatrix);

		/* in second: perform image color correction */
		image_rgb_correction(srcInput, dstOutput, width, height, srcPitch, dstPitch, correctionMatrix);

	} /* for (k = 0; k < iterCnt; k++) */

	  /* release temporary memory buffers on exit from function */
	if (-1 != memBlockId)
		::FreeMemoryBlock(memBlockId);
	memBlockId = -1;

	return true;
}


bool ProcessImgInAE_16bits
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	CACHE_ALIGN float U_avg[gMaxCnt]{};
	CACHE_ALIGN float V_avg[gMaxCnt]{};
	PF_Pixel_ARGB_16u* __restrict pMem[2]{};

	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[AWB_INPUT]->u.ld);
	PF_Pixel_ARGB_16u*  __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(input->data);
	PF_Pixel_ARGB_16u*  __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output->data);

	PF_Pixel_ARGB_16u* __restrict srcInput = nullptr;
	PF_Pixel_ARGB_16u* __restrict dstOutput = nullptr;

	const A_long height = output->height;
	const A_long width  = output->width;
	const A_long src_line_pitch = input->rowbytes  / PF_Pixel_ARGB_16u_size;
	const A_long dst_line_pitch = output->rowbytes / PF_Pixel_ARGB_16u_size;

	int32_t srcIdx = 0;
	int32_t dstIdx = 1;

	A_long srcPitch = 0;
	A_long dstPitch = 0;

	int32_t memBlockId = -1;

	/* acquire parameters */
	const eCOLOR_SPACE colorSpace = static_cast<eCOLOR_SPACE>(params[AWB_COLOR_SPACE_POPUP]->u.pd.value - 1);
	const eChromaticAdaptation chromatic = static_cast<eChromaticAdaptation>(params[AWB_CHROMATIC_POPUP]->u.pd.value - 1);
	const eILLUMINATE  illuminate = static_cast<eILLUMINATE> (params[AWB_ILLUMINATE_POPUP]->u.pd.value - 1);
	const int32_t sliderIterCnt = params[AWB_ITERATIONS_SLIDER]->u.sd.value;
	const int32_t sliderThreshold = params[AWB_THRESHOLD_SLIDER]->u.sd.value;
	const int32_t iterCnt = FastCompute::Min(sliderIterCnt, gMaxCnt);

	constexpr float reciproc100 = 1.0f / 100.f;
	float T = static_cast<float>(sliderThreshold) * reciproc100;
	float uAvg, vAvg;

	/* Get memory block */
	void* pMemoryBlock = nullptr;
	const int32_t blocksNumber = FastCompute::Min(2, (iterCnt - 1));

	if (blocksNumber)
	{
		const size_t  frameSize = width * height;
		const size_t  requiredMemSize = blocksNumber * frameSize * PF_Pixel_ARGB_16u_size;
		memBlockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemoryBlock);
        if (nullptr == pMemoryBlock || memBlockId < 0)
            return PF_Err_OUT_OF_MEMORY; // not enough memory for allocate temporary buffer for ALGO
        
        pMem[0] = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(pMemoryBlock);
		pMem[1] = (2 == blocksNumber) ? pMem[0] + frameSize : nullptr;
	}

	/* pass iterations in corresponding to slider position */
	for (A_long k = 0; k < iterCnt; k++)
	{
		if (0 == k)
		{
			srcInput = localSrc;
			dstIdx++;
			dstIdx &= 0x1;
			dstOutput = (1 == iterCnt) ? localDst : pMem[dstIdx];
			srcPitch = src_line_pitch;
			dstPitch = (1 == iterCnt) ? dst_line_pitch : width;
		}
		else if ((iterCnt - 1) == k)
		{
			srcInput = pMem[dstIdx];
			dstOutput = localDst;
			srcPitch = width;
			dstPitch = dst_line_pitch;
		} /* if (k > 0) */
		else
		{
			srcIdx = dstIdx;
			dstIdx++;
			dstIdx &= 0x1;
			srcInput = pMem[srcIdx];
			dstOutput = pMem[dstIdx];
			srcPitch = dstPitch = width;
		}

		uAvg = vAvg = 0.f;
		/* collect statistics about image and compute averages values for U and for V components */
		collect_rgb_statistics(srcInput, width, height, srcPitch, T, colorSpace, &uAvg, &vAvg);
		U_avg[k] = uAvg;
		V_avg[k] = vAvg;

		if (k > 0)
		{
			const float U_diff = U_avg[k] - U_avg[k - 1];
			const float V_diff = V_avg[k] - V_avg[k - 1];

			const float normVal = FastCompute::Sqrt(U_diff * U_diff + V_diff * V_diff);

			if (normVal < algAWBepsilon)
			{
				// U and V no longer improving, so just copy source to destination and break the loop
				simple_image_copy (srcInput, localDst, height, width, src_line_pitch, dst_line_pitch);

				if (-1 != memBlockId)
					::FreeMemoryBlock(memBlockId);
                memBlockId = -1;

				/* release temporary memory buffers on exit from function */
				return true; // U and V no longer improving
			}
		} /* if (k > 0) */

		  /* compute correction matrix */
		float correctionMatrix[3]{};
		compute_correction_matrix(uAvg, vAvg, colorSpace, illuminate, chromatic, correctionMatrix);

		/* in second: perform image color correction */
		image_rgb_correction(srcInput, dstOutput, width, height, srcPitch, dstPitch, correctionMatrix);

	} /* for (k = 0; k < iterCnt; k++) */

	  /* release temporary memory buffers on exit from function */
	if (-1 != memBlockId)
		::FreeMemoryBlock(memBlockId);
	memBlockId = -1;

	return true;
}


bool ProcessImgInAE_32bits
(
    PF_InData*    in_data,
    PF_OutData*   out_data,
    PF_ParamDef*  params[],
    PF_LayerDef*  output
) noexcept
{
    CACHE_ALIGN float U_avg[gMaxCnt]{};
    CACHE_ALIGN float V_avg[gMaxCnt]{};
    PF_Pixel_ARGB_32f* __restrict pMem[2]{};

    const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[AWB_INPUT]->u.ld);
    PF_Pixel_ARGB_32f*  __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(input->data);
    PF_Pixel_ARGB_32f*  __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(output->data);

    PF_Pixel_ARGB_32f* __restrict srcInput = nullptr;
    PF_Pixel_ARGB_32f* __restrict dstOutput = nullptr;

    const A_long height = output->height;
    const A_long width = output->width;
    const A_long src_line_pitch = input->rowbytes  / PF_Pixel_ARGB_32f_size;
    const A_long dst_line_pitch = output->rowbytes / PF_Pixel_ARGB_32f_size;

    int32_t srcIdx = 0;
    int32_t dstIdx = 1;

    A_long srcPitch = 0;
    A_long dstPitch = 0;

    int32_t memBlockId = -1;

    /* acquire parameters */
    const eCOLOR_SPACE colorSpace = static_cast<eCOLOR_SPACE>(params[AWB_COLOR_SPACE_POPUP]->u.pd.value - 1);
    const eChromaticAdaptation chromatic = static_cast<eChromaticAdaptation>(params[AWB_CHROMATIC_POPUP]->u.pd.value - 1);
    const eILLUMINATE  illuminate = static_cast<eILLUMINATE> (params[AWB_ILLUMINATE_POPUP]->u.pd.value - 1);
    const int32_t sliderIterCnt = params[AWB_ITERATIONS_SLIDER]->u.sd.value;
    const int32_t sliderThreshold = params[AWB_THRESHOLD_SLIDER]->u.sd.value;
    const int32_t iterCnt = FastCompute::Min(sliderIterCnt, gMaxCnt);

    constexpr float reciproc100 = 1.0f / 100.f;
    float T = static_cast<float>(sliderThreshold) * reciproc100;
    float uAvg, vAvg;

    /* Get memory block */
    void* pMemoryBlock = nullptr;
    const int32_t blocksNumber = FastCompute::Min(2, (iterCnt - 1));

    if (blocksNumber)
    {
        const size_t  frameSize = width * height;
        const size_t  requiredMemSize = blocksNumber * frameSize * PF_Pixel_ARGB_16u_size;
        memBlockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemoryBlock);
        if (nullptr == pMemoryBlock || memBlockId < 0)
            return PF_Err_OUT_OF_MEMORY; // not enough memory for allocate temporary buffer for ALGO

        pMem[0] = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(pMemoryBlock);
        pMem[1] = (2 == blocksNumber) ? pMem[0] + frameSize : nullptr;
    }

    /* pass iterations in corresponding to slider position */
    for (A_long k = 0; k < iterCnt; k++)
    {
        if (0 == k)
        {
            srcInput = localSrc;
            dstIdx++;
            dstIdx &= 0x1;
            dstOutput = (1 == iterCnt) ? localDst : pMem[dstIdx];
            srcPitch = src_line_pitch;
            dstPitch = (1 == iterCnt) ? dst_line_pitch : width;
        }
        else if ((iterCnt - 1) == k)
        {
            srcInput = pMem[dstIdx];
            dstOutput = localDst;
            srcPitch = width;
            dstPitch = dst_line_pitch;
        } /* if (k > 0) */
        else
        {
            srcIdx = dstIdx;
            dstIdx++;
            dstIdx &= 0x1;
            srcInput = pMem[srcIdx];
            dstOutput = pMem[dstIdx];
            srcPitch = dstPitch = width;
        }

        uAvg = vAvg = 0.f;
        /* collect statistics about image and compute averages values for U and for V components */
        collect_rgb_statistics(srcInput, width, height, srcPitch, T, colorSpace, &uAvg, &vAvg);
        U_avg[k] = uAvg;
        V_avg[k] = vAvg;

        if (k > 0)
        {
            const float U_diff = U_avg[k] - U_avg[k - 1];
            const float V_diff = V_avg[k] - V_avg[k - 1];

            const float normVal = FastCompute::Sqrt(U_diff * U_diff + V_diff * V_diff);

            if (normVal < algAWBepsilon)
            {
                // U and V no longer improving, so just copy source to destination and break the loop
                simple_image_copy(srcInput, localDst, height, width, src_line_pitch, dst_line_pitch);

                if (-1 != memBlockId)
                    ::FreeMemoryBlock(memBlockId);
                memBlockId = -1;

                /* release temporary memory buffers on exit from function */
                return true; // U and V no longer improving
            }
        } /* if (k > 0) */

          /* compute correction matrix */
        float correctionMatrix[3]{};
        compute_correction_matrix(uAvg, vAvg, colorSpace, illuminate, chromatic, correctionMatrix);

        /* in second: perform image color correction */
        image_rgb_correction(srcInput, dstOutput, width, height, srcPitch, dstPitch, correctionMatrix);

    } /* for (k = 0; k < iterCnt; k++) */

      /* release temporary memory buffers on exit from function */
    if (-1 != memBlockId)
        ::FreeMemoryBlock(memBlockId);
    memBlockId = -1;

    return true;
}


inline PF_Err ProcessImgInAE_DeepWord
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[AWB_INPUT]->u.ld), &format))
        err = (format == PF_PixelFormat_ARGB128 ?
            ProcessImgInAE_32bits(in_data, out_data, params, output) : ProcessImgInAE_16bits(in_data, out_data, params, output));
    else
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;

    return err;
}


PF_Err ProcessImgInAE
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return (true == (PF_WORLD_IS_DEEP(output) ?
        ProcessImgInAE_DeepWord (in_data, out_data, params, output) :
		ProcessImgInAE_8bits (in_data, out_data, params, output) ) ? PF_Err_NONE : PF_Err_INVALID_INDEX);
}