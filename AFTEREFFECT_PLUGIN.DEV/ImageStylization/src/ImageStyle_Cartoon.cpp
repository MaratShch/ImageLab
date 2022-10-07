#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "SegmentationUtils.hpp"
#include "ImageAuxPixFormat.hpp"

#include <mutex>
#include <math.h> 

constexpr float qH = 6.f;
constexpr float qS = 5.f;
constexpr float qI = 5.f;
constexpr float divQh = 1.f / qH;
constexpr float divQs = 1.f / qS;
constexpr float divQi = 1.f / qI;

constexpr int nbinsH { static_cast<int>(hist_size_H / qH) };
constexpr int nbinsS { static_cast<int>(hist_size_S / qS) };
constexpr int nbinsI { static_cast<int>(hist_size_I / qI) };


inline void sRgb2NewHsi (const float& R, const float& G, const float& B, float& H, float& S, float& I) noexcept
{
	constexpr float Sqrt2 = 1.414213562373f; /* sqrt isn't defined as constexpr */
	constexpr float OneRadian = 180.f / FastCompute::PI;
	constexpr float div3 = 1.f / 3.f;
	constexpr float denom = 1.f / 1e7f;

	I = div3 * (R + G + B);
	const float RminusI = R - I;
	const float GminusI = G - I;
	const float BminusI = B - I;

	S = FastCompute::Sqrt (RminusI * RminusI + GminusI * GminusI + BminusI * BminusI);

	if (fabs(S) > denom)
	{
		float cosH = (G - B) / (Sqrt2 * S);
		const float FabsH = fabs(cosH);

		if (FabsH > 1.f)
			cosH /= FabsH;

		float h = (FastCompute::Acos(cosH)) * OneRadian;
		float proj2 = -2.f * RminusI + GminusI + BminusI;
		
		if (proj2 < 0.f)
			h = -h;
		
		H = h + ((h < 0.f) ? 360.f : 0.f);
		
		if (H == 360.f)
			H = 0.f;
	}
	else
	{
		H = 0.f;
	}

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

	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
	const A_long tmp_pitch = width;

	constexpr float sMin = static_cast<float>(nbinsH) / FastCompute::PIx2; // compute minimum saturation value that prevents quantization problems 
	const size_t requiredMemSize = tmp_pitch * height * sizeof(float) * 3;
	int j, i, nhighsat;
	int hH, sS, iI;
	float H, S, I;

	const int nBinsH = static_cast<int>(static_cast<float>(hist_size_H) * divQh);
	const int nBinsS = static_cast<int>(static_cast<float>(hist_size_S) * divQs);
	const int nBinsI = static_cast<int>(static_cast<float>(hist_size_I) * divQi);

	bool bMemSizeTest = false;

	j = i = nhighsat = 0;
	hH = sS = iI = 0;
	H = S = I = 0.f;

	bufHandle* pGlobal = static_cast<bufHandle*>(GET_OBJ_FROM_HNDL(in_data->global_data));
	if (nullptr != pGlobal)
	{
		pTmpStorageHdnl = static_cast<ImageStyleTmpStorage* __restrict>(pGlobal->pBufHndl);
		bMemSizeTest = test_temporary_buffers(pTmpStorageHdnl, requiredMemSize);
	}

	if (true == bMemSizeTest)
	{
		const std::lock_guard<std::mutex> lock (pTmpStorageHdnl->guard_buffer);
		pTmpStorage = reinterpret_cast<PF_Pixel_HSI_32f* __restrict>(pTmpStorageHdnl->pStorage1);

		/* first path - build the statistics about frame [build histogram] */
		for (j = 0; j < height; j++)
		{
			const A_long line_idx = j * line_pitch;
			const A_long tmpBufLineidx = j * width;

			__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				const A_long tmpBufpixIdx = tmpBufLineidx + i;
				const A_long pix_idx = line_idx + i;

				/* convert RGB to sRGB */
				const float B = static_cast<float>(localSrc[pix_idx].B);
				const float G = static_cast<float>(localSrc[pix_idx].G);
				const float R = static_cast<float>(localSrc[pix_idx].R);

				/* convert sRGB to HSI color space */
				sRgb2NewHsi (R, G, B, H, S, I);

				pTmpStorage[tmpBufpixIdx].H = H;
				pTmpStorage[tmpBufpixIdx].S = S;
				pTmpStorage[tmpBufpixIdx].I = I;

				if (S > sMin)
				{
					hH = static_cast<int>(H * divQh);
					histH[hH]++;
					nhighsat++;
				}
				else
				{
					iI = static_cast<int>(I * divQi);
					histI[iI]++;
				}
			} /* for (i = 0; i < width; i++) */
		} /* for (j = 0; j < height; j++) */

		/* second path - segment histogram and build palette */
		auto const isGray{ 0 == nhighsat };
		constexpr float epsilon{ 1.0f };
		
		std::vector<int32_t> ftcSeg;
		std::vector<Hsegment>hSegments;
		std::vector<Isegment>iSegments;

		if (true == isGray)
		{
			ftcSeg = ftc_utils_segmentation(histI, nBinsI, epsilon, isGray);
		}
		else
		{
			ftcSeg = ftc_utils_segmentation(histH, nBinsH, epsilon, isGray);
			hSegments = compute_color_palette (pTmpStorage, localSrc, width, height, line_pitch, tmp_pitch, sMin, nbinsH, nbinsS, nbinsI, qH, qS, qI, ftcSeg, epsilon);
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
		return PF_Err_OUT_OF_MEMORY;
	}

	return PF_Err_NONE;
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