#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "SegmentationUtils.hpp"
#include "ImageAuxPixFormat.hpp"

#include <mutex>
#include <math.h> 

constexpr float qH = 6.0f;
constexpr float qS = 5.0f;
constexpr float qI = 5.0f;


inline void sRgb2NewHsi (float R, float G, float B, float& H, float& S, float& I) noexcept
{
	constexpr float Sqrt2 = 1.414213562373f; /* sqrt isn't defined as constexpr */
	constexpr float OneRadian = 180.f / FastCompute::PI;
	constexpr float div3  = 1.f / 3.f;
	constexpr float denom = 1.f / 1e7f;

	I = (R + G + B) * div3;
	const float RminusI = R - I;
	const float GminusI = G - I;
	const float BminusI = B - I;

	H = 0.f;

//	S = FastCompute::Sqrt (RminusI * RminusI + GminusI * GminusI + BminusI * BminusI);
	S = std::sqrt(RminusI * RminusI + GminusI * GminusI + BminusI * BminusI);

	if (S == 0.f)
		return;

//	if (std::fabs(S) > denom)
//	{
		float cosH = (G - B) / (Sqrt2 * S);
		const float cosAbsH = std::abs(cosH);

		if (cosAbsH > 1.f) 
			cosH /= cosAbsH;

//		float h = (FastCompute::Acos(cosH)) * OneRadian;
		float h = std::acos(cosH) * OneRadian;
		float proj2 = -2.f * RminusI + GminusI + BminusI;
		
		if (proj2 < 0.f)
			h = -h;
		
		if (h < 0)
			h += 360.f;

		if (h == 360.f)
			h = 0.f;

		H = h;
//	}
//	else
//	{
//		H = S = 0.f;
//	}

	return;
}


static PF_Err PR_ImageStyle_CartoonEffect_BGRA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN int32_t histH[hist_size_H]{};
	CACHE_ALIGN int32_t histI[hist_size_I]{};

	ImageStyleTmpStorage*   __restrict pTmpStorageHdnl = nullptr;
	PF_Pixel_HSI_32f*       __restrict pTmpStorage = nullptr;
	const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	PF_Err errCode = PF_Err_NONE;
	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
	const A_long imgSize = height * width;

	const size_t requiredMemSize = imgSize * sizeof(float) * 3;
	int j, i, nhighsat;
	int hH, sS, iI;
	float H, S, I;

	int nBinsH = static_cast<int>(static_cast<float>(hist_size_H) / qH);
	int nBinsS = static_cast<int>(static_cast<float>(hist_size_S) / qS);
	int nBinsI = static_cast<int>(static_cast<float>(hist_size_I) / qI);

	if (nBinsH * qH < hist_size_H) nBinsH++;
	if (nBinsS * qS < hist_size_S) nBinsS++;
	if (nBinsI * qI < hist_size_I) nBinsI++;

	const float sMin = static_cast<float>(nBinsH) / FastCompute::PIx2; // compute minimum saturation value that prevents quantization problems 
	const bool  bForceGray = false; // reads this value from check box on control pannel
	bool bMemSizeTest = false;

	j = i = nhighsat = 0;
	hH = sS = iI = 0;
	H = S = I = 0.f;

	auto tmpFloatData = std::make_unique<fDataRGB []>(imgSize);

	if (tmpFloatData)
	{
		bufHandle* pGlobal = static_cast<bufHandle*>(GET_OBJ_FROM_HNDL(in_data->global_data));
		if (nullptr != pGlobal)
		{
			pTmpStorageHdnl = static_cast<ImageStyleTmpStorage* __restrict>(pGlobal->pBufHndl);
			bMemSizeTest = test_temporary_buffers(pTmpStorageHdnl, requiredMemSize);
		}

		if (true == bMemSizeTest)
		{
			const std::lock_guard<std::mutex> lock(pTmpStorageHdnl->guard_buffer);
			pTmpStorage = reinterpret_cast<PF_Pixel_HSI_32f* __restrict>(pTmpStorageHdnl->pStorage1);

			auto tmpFloatPtr = tmpFloatData.get();
			utils_prepare_data (localSrc, tmpFloatPtr, width, height, line_pitch, -1.f);

			memset(reinterpret_cast<void*>(pTmpStorage), 0, requiredMemSize);

			/* first path - build the statistics about frame [build histogram] */
			for (j = 0; j < height; j++)
			{
				const A_long line_idx = j * width;
				for (i = 0; i < width; i++)
				{
					const A_long pix_idx = line_idx + i;

					/* convert sRGB to HSI color space */
					sRgb2NewHsi (tmpFloatPtr[pix_idx].R, tmpFloatPtr[pix_idx].G, tmpFloatPtr[pix_idx].B, H, S, I);

					pTmpStorage[pix_idx].H = H;
					pTmpStorage[pix_idx].S = S;
					pTmpStorage[pix_idx].I = I;

					if (S > sMin)
					{
						hH = static_cast<int>(H / qH);
						histH[hH]++;
						nhighsat++;
					}
					else
					{
						iI = static_cast<int>(I / qI);
						histI[iI]++;
					}
				} /* for (i = 0; i < width; i++) */
			} /* for (j = 0; j < height; j++) */

			/* second path - segment histogram and build palette */
			auto const isGray{ 0 == nhighsat };
			constexpr float epsilon{ 1.0f };

			std::vector<int32_t> ftcSegI;
			std::vector<int32_t> ftcSegH;
			std::vector<Hsegment>hSegments;
			std::vector<Isegment>iSegments;

			if (true == isGray || true == bForceGray)
			{
				ftcSegI = ftc_utils_segmentation (histI, nBinsI, epsilon, isGray);
				iSegments = compute_gray_palette (pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsI, qI, ftcSegI);
			}
			else
			{
				ftcSegH = ftc_utils_segmentation  (histH, nBinsH, epsilon, isGray);
				hSegments = compute_color_palette (pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsH, nBinsS, nBinsI, qH, qS, qI, ftcSegH, epsilon);
			}

			std::vector<dataRGB> meanRGB_I, meanRGB_H, meanRGB_HS, meanRGB_HSI;
			std::vector<int32_t> icolorsH;
			std::vector<int32_t> icolorsS;

			/* get list of gray levels and colors */
	//		get_list_grays_colors (iSegments, hSegments, meanRGB_I, meanRGB_H, meanRGB_HS, meanRGB_HSI, icolorsH, icolorsS);

			/* create segmented image */


		} /* if (true == bMemSizeTest) */
		else
		{
			errCode = PF_Err_OUT_OF_MEMORY;
		}
	}
	else
	{
		errCode = PF_Err_OUT_OF_MEMORY;
	}

	return errCode;
}



static PF_Err PR_ImageStyle_CartoonEffect_VUYA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_CartoonEffect_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_CartoonEffect_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_CartoonEffect_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}




PF_Err PR_ImageStyle_CartoonEffect
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PF_Err errFormat = PF_Err_INVALID_INDEX;

	/* This plugin called frop PR - check video fomat */
	AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
		AEFX_SuiteScoper<PF_PixelFormatSuite1>(
			in_data,
			kPFPixelFormatSuite,
			kPFPixelFormatSuiteVersion1,
			out_data);

	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;
	if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = PR_ImageStyle_CartoonEffect_BGRA_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageStyle_CartoonEffect_VUYA_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageStyle_CartoonEffect_VUYA_32f(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageStyle_CartoonEffect_BGRA_16u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageStyle_CartoonEffect_BGRA_32f(in_data, out_data, params, output);
			break;

			default:
				err = PF_Err_INVALID_INDEX;
			break;
		}
	}
	else
	{
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}


PF_Err AE_ImageStyle_CartoonEffect_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err AE_ImageStyle_CartoonEffect_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}