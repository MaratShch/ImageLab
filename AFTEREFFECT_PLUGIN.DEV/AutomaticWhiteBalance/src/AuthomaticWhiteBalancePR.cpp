#include "AuthomaticWhiteBalance.hpp"
#include "AlgCommonFunctions.hpp"
#include "AlgCorrectionMatrix.hpp"

static bool ProcessPrImage_BGRA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	CACHE_ALIGN float U_avg[gMaxCnt]{};
	CACHE_ALIGN float V_avg[gMaxCnt]{};
	PF_Pixel_BGRA_8u* __restrict pMem[2]{};

	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[AWB_INPUT]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	PF_Pixel_BGRA_8u* __restrict srcInput  = nullptr;
	PF_Pixel_BGRA_8u* __restrict dstOutput = nullptr;

	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	int32_t srcIdx = 0;
	int32_t dstIdx = 1;

	A_long srcPitch = 0;
	A_long dstPitch = 0;

	int32_t memBlockId = -1;

	/* acquire parameters */
	const eCOLOR_SPACE colorSpace = static_cast<eCOLOR_SPACE>(params[AWB_COLOR_SPACE_POPUP]->u.pd.value - 1);
	const eChromaticAdaptation chromatic = static_cast<eChromaticAdaptation>(params[AWB_CHROMATIC_POPUP]->u.pd.value - 1);
	const eILLUMINATE  illuminate = static_cast<eILLUMINATE> (params[AWB_ILLUMINATE_POPUP]->u.pd.value - 1);
	const int32_t sliderIterCnt   = params[AWB_ITERATIONS_SLIDER]->u.sd.value;
	const int32_t sliderThreshold = params[AWB_THRESHOLD_SLIDER]->u.sd.value;
	const int32_t iterCnt = FastCompute::Min (sliderIterCnt, gMaxCnt);

	constexpr float reciproc100 = 1.0f / 100.f;
	float T = static_cast<float>(sliderThreshold) * reciproc100;
	float uAvg, vAvg;

	/* Get memory block */
	void* pMemoryBlock = nullptr;
	const int32_t blocksNumber = FastCompute::Min (2, (iterCnt - 1));

	if (blocksNumber > 0)
	{
		const size_t  frameSize = width * height;
		const size_t  requiredMemSize = blocksNumber * frameSize * PF_Pixel_BGRA_8u_size;
		memBlockId = ::GetMemoryBlock (requiredMemSize, 0, &pMemoryBlock);
		pMem[0] = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pMemoryBlock);
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
			srcPitch  = line_pitch;
			dstPitch  = (1 == iterCnt) ? line_pitch : width;
		} else if ((iterCnt-1) == k)
		{
			srcIdx = dstIdx;
			srcInput  = pMem[srcIdx];
			dstOutput = localDst;
			srcPitch  = width;
			dstPitch  = line_pitch;
		} /* if (k > 0) */
		else
		{
			srcIdx = dstIdx;
			dstIdx++;
			dstIdx &= 0x1;
			srcInput  = pMem[srcIdx];
			dstOutput = pMem[dstIdx];
			srcPitch  = dstPitch = width;
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
				simple_image_copy (srcInput, localDst, height, width, srcPitch, line_pitch);

				if (-1 != memBlockId)
					::FreeMemoryBlock (memBlockId);

				/* release temporary memory buffers on exit from function */
				return true; // U and V no longer improving
			}
		} /* if (k > 0) */

		/* compute correction matrix */
		float correctionMatrix[3]{};
		compute_correction_matrix (uAvg, vAvg, colorSpace, illuminate, chromatic, correctionMatrix);

		/* in second: perform image color correction */
		image_rgb_correction(srcInput, dstOutput, width, height, srcPitch, dstPitch, correctionMatrix);

	} /* for (k = 0; k < iterCnt; k++) */

	  /* release temporary memory buffers on exit from function */
	if (-1 != memBlockId)
		::FreeMemoryBlock(memBlockId);
	memBlockId = -1;

	return true;
}


static bool ProcessPrImage_BGRA_4444_16u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	CACHE_ALIGN float U_avg[gMaxCnt]{};
	CACHE_ALIGN float V_avg[gMaxCnt]{};
	PF_Pixel_BGRA_16u* __restrict pMem[2]{};

	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[AWB_INPUT]->u.ld);
	PF_Pixel_BGRA_16u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	PF_Pixel_BGRA_16u* __restrict srcInput = nullptr;
	PF_Pixel_BGRA_16u* __restrict dstOutput = nullptr;

	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	int32_t srcIdx = 0;
	int32_t dstIdx = 1;

	A_long srcPitch = 0;
	A_long dstPitch = 0;

	int32_t memBlockId = -1;

	/* acquire parameters */
	const eCOLOR_SPACE colorSpace = static_cast<eCOLOR_SPACE>(params[AWB_COLOR_SPACE_POPUP]->u.pd.value - 1);
	const eChromaticAdaptation chromatic = static_cast<eChromaticAdaptation>(params[AWB_CHROMATIC_POPUP]->u.pd.value - 1);
	const eILLUMINATE  illuminate = static_cast<eILLUMINATE> (params[AWB_ILLUMINATE_POPUP]->u.pd.value - 1);
	const int32_t sliderIterCnt   = params[AWB_ITERATIONS_SLIDER]->u.sd.value;
	const int32_t sliderThreshold = params[AWB_THRESHOLD_SLIDER]->u.sd.value;
	const int32_t iterCnt = FastCompute::Min(sliderIterCnt, gMaxCnt);

	constexpr float reciproc100 = 1.0f / 100.f;
	float T = static_cast<float>(sliderThreshold) * reciproc100;
	float uAvg, vAvg;

	/* Get memory block */
	void* pMemoryBlock = nullptr;
	const int32_t blocksNumber = FastCompute::Min (2, (iterCnt - 1));

	if (blocksNumber > 0)
	{
		const size_t  frameSize = width * height;
		const size_t  requiredMemSize = blocksNumber * frameSize * PF_Pixel_BGRA_16u_size;
		memBlockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemoryBlock);
		pMem[0] = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(pMemoryBlock);
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
			srcPitch  = line_pitch;
			dstPitch  = (1 == iterCnt) ? line_pitch : width;
		}
		else if ((iterCnt - 1) == k)
		{
			srcIdx    = dstIdx;
			srcInput  = pMem[srcIdx];
			dstOutput = localDst;
			srcPitch  = width;
			dstPitch  = line_pitch;
		} /* if (k > 0) */
		else
		{
			srcIdx = dstIdx;
			dstIdx++;
			dstIdx &= 0x1;
			srcInput  = pMem[srcIdx];
			dstOutput = pMem[dstIdx];
			srcPitch  = dstPitch = width;
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

			const float normVal = FastCompute::Sqrt (U_diff * U_diff + V_diff * V_diff);

			if (normVal < algAWBepsilon)
			{
				// U and V no longer improving, so just copy source to destination and break the loop
				simple_image_copy (srcInput, localDst, height, width, srcPitch, line_pitch);

				if (-1 != memBlockId)
					::FreeMemoryBlock(memBlockId);

				/* release temporary memory buffers on exit from function */
				return true; // U and V no longer improving
			}
		} /* if (k > 0) */

		  /* compute correction matrix */
		float correctionMatrix[3]{};
		compute_correction_matrix (uAvg, vAvg, colorSpace, illuminate, chromatic, correctionMatrix);

		/* in second: perform image color correction */
		image_rgb_correction (srcInput, dstOutput, width, height, srcPitch, dstPitch, correctionMatrix);

	} /* for (k = 0; k < iterCnt; k++) */

	  /* release temporary memory buffers on exit from function */
	if (-1 != memBlockId)
		::FreeMemoryBlock(memBlockId);
	memBlockId = -1;

	return true;

}


 bool ProcessPrImage_BGRA_4444_32f
(
    PF_InData*    in_data,
    PF_OutData*   out_data,
    PF_ParamDef*  params[],
    PF_LayerDef*  output
) noexcept
{
    CACHE_ALIGN float U_avg[gMaxCnt]{};
    CACHE_ALIGN float V_avg[gMaxCnt]{};
    PF_Pixel_BGRA_32f* __restrict pMem[2]{};

    const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[AWB_INPUT]->u.ld);
    PF_Pixel_BGRA_32f*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
    PF_Pixel_BGRA_32f*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

    PF_Pixel_BGRA_32f* __restrict srcInput = nullptr;
    PF_Pixel_BGRA_32f* __restrict dstOutput = nullptr;

    const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
    const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

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

    if (blocksNumber > 0)
    {
        const size_t  frameSize = width * height;
        const size_t  requiredMemSize = blocksNumber * frameSize * PF_Pixel_BGRA_32f_size;
        memBlockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemoryBlock);
        pMem[0] = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(pMemoryBlock);
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
            srcPitch = line_pitch;
            dstPitch = (1 == iterCnt) ? line_pitch : width;
        }
        else if ((iterCnt - 1) == k)
        {
            srcIdx = dstIdx;
            srcInput = pMem[srcIdx];
            dstOutput = localDst;
            srcPitch = width;
            dstPitch = line_pitch;
        } /* if (k > 0) */
        else
        {
            srcIdx = dstIdx;
            dstIdx++;
            dstIdx &= 0x1;
            srcInput  = pMem[srcIdx];
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
                simple_image_copy(srcInput, localDst, height, width, srcPitch, line_pitch);

                if (-1 != memBlockId)
                    ::FreeMemoryBlock(memBlockId);

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



static bool ProcessPrImage_VUYA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output,
	const bool isBT709
) noexcept
{
	CACHE_ALIGN float U_avg[gMaxCnt]{};
	CACHE_ALIGN float V_avg[gMaxCnt]{};
	PF_Pixel_VUYA_8u* __restrict pMem[2]{};

	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<PF_LayerDef* __restrict>(&params[AWB_INPUT]->u.ld);
	PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	PF_Pixel_VUYA_8u* __restrict srcInput = nullptr;
	PF_Pixel_VUYA_8u* __restrict dstOutput = nullptr;

	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

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

	if (blocksNumber > 0)
	{
		const size_t  frameSize = width * height;
		const size_t  requiredMemSize = blocksNumber * frameSize * PF_Pixel_VUYA_8u_size;
		memBlockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemoryBlock);
		pMem[0] = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(pMemoryBlock);
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
			srcPitch = line_pitch;
			dstPitch = (1 == iterCnt) ? line_pitch : width;
		}
		else if ((iterCnt - 1) == k)
		{
			srcIdx = dstIdx;
			srcInput = pMem[srcIdx];
			dstOutput = localDst;
			srcPitch = width;
			dstPitch = line_pitch;
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
		collect_yuv_statistics (srcInput, width, height, srcPitch, T, (isBT709 ? BT709 : BT601), &uAvg, &vAvg);
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
				simple_image_copy(srcInput, localDst, height, width, srcPitch, line_pitch);

				/* release temporary memory buffers on exit from function */
				if (-1 != memBlockId)
					::FreeMemoryBlock(memBlockId);

				memBlockId = -1;
				return true; // U and V no longer improving
			}
		} /* if (k > 0) */

		  /* compute correction matrix */
		float correctionMatrix[3]{};
		compute_correction_matrix (uAvg, vAvg, colorSpace, illuminate, chromatic, correctionMatrix);

		/* in second: perform image color correction */
		image_yuv_correction (srcInput, dstOutput, width, height, srcPitch, dstPitch, correctionMatrix, isBT709);

	} /* for (k = 0; k < iterCnt; k++) */

	  /* release temporary memory buffers on exit from function */
	if (-1 != memBlockId)
		::FreeMemoryBlock(memBlockId);
	memBlockId = -1;

	return true;
}


static bool ProcessPrImage_VUYA_4444_32f
(
    PF_InData*    in_data,
    PF_OutData*   out_data,
    PF_ParamDef*  params[],
    PF_LayerDef*  output,
    const bool isBT709
) noexcept
{
    CACHE_ALIGN float U_avg[gMaxCnt]{};
    CACHE_ALIGN float V_avg[gMaxCnt]{};
    PF_Pixel_VUYA_32f* __restrict pMem[2]{};

    const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[AWB_INPUT]->u.ld);
    PF_Pixel_VUYA_32f*  __restrict localSrc = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
    PF_Pixel_VUYA_32f*  __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

    PF_Pixel_VUYA_32f* __restrict srcInput = nullptr;
    PF_Pixel_VUYA_32f* __restrict dstOutput = nullptr;

    const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
    const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

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

    if (blocksNumber > 0)
    {
        const size_t  frameSize = width * height;
        const size_t  requiredMemSize = blocksNumber * frameSize * PF_Pixel_VUYA_32f_size;
        memBlockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemoryBlock);
        pMem[0] = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(pMemoryBlock);
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
            srcPitch = line_pitch;
            dstPitch = (1 == iterCnt) ? line_pitch : width;
        }
        else if ((iterCnt - 1) == k)
        {
            srcIdx = dstIdx;
            srcInput = pMem[srcIdx];
            dstOutput = localDst;
            srcPitch = width;
            dstPitch = line_pitch;
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
        collect_yuv_statistics (srcInput, width, height, srcPitch, T, (isBT709 ? BT709 : BT601), &uAvg, &vAvg);
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
                simple_image_copy(srcInput, localDst, height, width, srcPitch, line_pitch);

                /* release temporary memory buffers on exit from function */
                if (-1 != memBlockId)
                    ::FreeMemoryBlock(memBlockId);

                memBlockId = -1;
                return true; // U and V no longer improving
            }
        } /* if (k > 0) */

          /* compute correction matrix */
        float correctionMatrix[3]{};
        compute_correction_matrix(uAvg, vAvg, colorSpace, illuminate, chromatic, correctionMatrix);

        /* in second: perform image color correction */
        image_yuv_correction(srcInput, dstOutput, width, height, srcPitch, dstPitch, correctionMatrix, isBT709);

    } /* for (k = 0; k < iterCnt; k++) */

      /* release temporary memory buffers on exit from function */
    if (-1 != memBlockId)
        ::FreeMemoryBlock(memBlockId);
    memBlockId = -1;

    return true;
}



PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output,
	const PrPixelFormat&    pixelFormat
) noexcept
{
	bool bValue = true;

	/* acquire controls parameters */

	switch (pixelFormat)
	{
		case PrPixelFormat_BGRA_4444_8u:
			bValue = ProcessPrImage_BGRA_4444_8u (in_data, out_data, params, output);
		break;

		case PrPixelFormat_BGRA_4444_16u:
			bValue = ProcessPrImage_BGRA_4444_16u (in_data, out_data, params, output);
		break;

        case PrPixelFormat_BGRA_4444_32f:
           bValue = ProcessPrImage_BGRA_4444_32f (in_data, out_data, params, output);
        break;

        case PrPixelFormat_VUYA_4444_8u:
		case PrPixelFormat_VUYA_4444_8u_709:
			bValue = ProcessPrImage_VUYA_4444_8u (in_data, out_data, params, output, PrPixelFormat_VUYA_4444_8u_709 == pixelFormat);
		break;

        case PrPixelFormat_VUYA_4444_32f:
        case PrPixelFormat_VUYA_4444_32f_709:
            bValue = ProcessPrImage_VUYA_4444_32f (in_data, out_data, params, output, PrPixelFormat_VUYA_4444_8u_709 == pixelFormat);
        break;

        default:
			bValue = false;
		break;
	}

	return (true == bValue ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED);
}