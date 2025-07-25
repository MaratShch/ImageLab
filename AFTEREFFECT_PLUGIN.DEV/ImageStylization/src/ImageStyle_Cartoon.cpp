#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "SegmentationUtils.hpp"
#include "ImageAuxPixFormat.hpp"
#include "ImageLabMemInterface.hpp"
#include <mutex>
#include <math.h> 

constexpr float qH = 6.0f;
constexpr float qS = 5.0f;
constexpr float qI = 5.0f;


inline void sRgb2NewHsi (const float& R, const float& G, const float& B, float& H, float& S, float& I) noexcept
{
	constexpr float Sqrt2 = 1.414213562373f; /* sqrt isn't defined as constexpr */
	constexpr float OneRadian = 180.f / FastCompute::PI;
	constexpr float div3  = 1.f / 3.f;
	constexpr float denom = 1.f / 1e7f;

	I = (R + G + B) * div3;

	const float RminusI = R - I;
	const float GminusI = G - I;
	const float BminusI = B - I;

	S = FastCompute::Sqrt (RminusI * RminusI + GminusI * GminusI + BminusI * BminusI);

	if (FastCompute::Abs(S) > denom)
	{
		float cosH = (G - B) / (Sqrt2 * S);
		const float cosAbsH = FastCompute::Abs(cosH);

		if (cosAbsH > 1.f) 
			cosH /= cosAbsH;

		float h = (FastCompute::Acos(cosH)) * OneRadian;
		float proj2 = -2.f * RminusI + GminusI + BminusI;
		
		if (proj2 < 0.f)
			h = -h;
		
		if (h < 0)
			h += 360.f;

		if (h == 360.f)
			h = 0.f;

		H = h;
	}
	else
	{
		H = S = 0.f;
	}

	return;
}


A_long convert2HSI
(
	const fDataRGB* __restrict   pRGB,
	PF_Pixel_HSI_32f* __restrict pHSI,
	int32_t* __restrict histH,
	int32_t* __restrict histI,
	const A_long& sizeX,
	const A_long& sizeY,
	const float& qH,
	const float& qI,
	const float& sMin
) noexcept
{
	A_long nSaturated = 0;
	float H = 0.f, S = 0.f, I = 0.f;

	memset(histH, 0, hist_size_H * sizeof(histH[0]));
	memset(histI, 0, hist_size_I * sizeof(histI[0]));

	for (A_long j = 0; j < sizeY; j++)
	{
		const A_long lineIdx = j * sizeX;
		__VECTOR_ALIGNED__
		for (A_long i = 0; i < sizeX; i++)
		{
			sRgb2NewHsi (pRGB[lineIdx + i].R, pRGB[lineIdx + i].G, pRGB[lineIdx + i].B, H, S, I);

			pHSI[lineIdx + i].H = H;
			pHSI[lineIdx + i].S = S;
			pHSI[lineIdx + i].I = I;

			if (S > sMin)
			{
				const int32_t hH = static_cast<int32_t>(H / qH);
				histH[hH]++;
				nSaturated++;
			}
			else
			{
				const int32_t iI = static_cast<int>(I / qI);
				histI[iI]++;
			}
		}
	}
	return nSaturated;
}



static PF_Err PR_ImageStyle_CartoonEffect_BGRA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN int32_t histH[hist_size_H];
	CACHE_ALIGN int32_t histI[hist_size_I];

	const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	PF_Err errCode = PF_Err_NONE;
	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
	const A_long imgSize = height * width;

	int nBinsH = static_cast<int>(static_cast<float>(hist_size_H) / qH);
	int nBinsS = static_cast<int>(static_cast<float>(hist_size_S) / qS);
	int nBinsI = static_cast<int>(static_cast<float>(hist_size_I) / qI);

	if (nBinsH * qH < hist_size_H) nBinsH++;
	if (nBinsS * qS < hist_size_S) nBinsS++;
	if (nBinsI * qI < hist_size_I) nBinsI++;

	const float sMin = static_cast<float>(nBinsH) / FastCompute::PIx2; // compute minimum saturation value that prevents quantization problems 
	const bool  bForceGray = false; // reads this value from check box on control pannel

    constexpr A_long bufferElementsSize = static_cast<A_long>(sizeof(fDataRGB) + sizeof(PF_Pixel_HSI_32f));
    const A_long totalProcMem = CreateAlignment(imgSize * bufferElementsSize, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fDataRGB* __restrict tmpFloatPtr = static_cast<fDataRGB* __restrict>(pMemoryBlock);
        PF_Pixel_HSI_32f* __restrict pTmpStorage  = reinterpret_cast<PF_Pixel_HSI_32f* __restrict>(tmpFloatPtr + imgSize);

        utils_prepare_data (localSrc, tmpFloatPtr, width, height, line_pitch, -1.f);

        // first path - convert image to HSI and build H and I histogramm
        const A_long nhighsat = convert2HSI (tmpFloatPtr, pTmpStorage, histH, histI, width, height, qH, qI, sMin);

        // second path - segment histogram and build palette */
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

        // create segmented image (lets re-use temporary buffer for assemble image)
        float* __restrict fR = reinterpret_cast<float* __restrict>(tmpFloatPtr);
        float* __restrict fG = fR + imgSize;
        float* __restrict fB = fG + imgSize;
        assemble_segmented_image (iSegments, hSegments, fR, fG, fB, width, height);
        
        // store segmented image
        store_segmented_image (localSrc, localDst, fR, fG, fB, width, height, line_pitch, line_pitch);

        ::FreeMemoryBlock(blockId);
        blockId = -1;
        pMemoryBlock = nullptr;
        tmpFloatPtr = nullptr;
        pTmpStorage = nullptr;
    } // if (true == bMemSizeTest)
    else
        errCode = PF_Err_OUT_OF_MEMORY;

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
	CACHE_ALIGN int32_t histH[hist_size_H];
	CACHE_ALIGN int32_t histI[hist_size_I];

	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*        __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	PF_Err errCode = PF_Err_NONE;
	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
	const A_long imgSize = height * width;

	int nBinsH = static_cast<int>(static_cast<float>(hist_size_H) / qH);
	int nBinsS = static_cast<int>(static_cast<float>(hist_size_S) / qS);
	int nBinsI = static_cast<int>(static_cast<float>(hist_size_I) / qI);

	if (nBinsH * qH < hist_size_H) nBinsH++;
	if (nBinsS * qS < hist_size_S) nBinsS++;
	if (nBinsI * qI < hist_size_I) nBinsI++;

	const float sMin = static_cast<float>(nBinsH) / FastCompute::PIx2; // compute minimum saturation value that prevents quantization problems 
	const bool  bForceGray = false; // reads this value from check box on control pannel

    constexpr A_long bufferElementsSize = static_cast<A_long>(sizeof(fDataRGB) + sizeof(PF_Pixel_HSI_32f));
    const A_long totalProcMem = CreateAlignment(imgSize * bufferElementsSize, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fDataRGB* __restrict tmpFloatPtr = static_cast<fDataRGB* __restrict>(pMemoryBlock);
        PF_Pixel_HSI_32f* __restrict pTmpStorage = reinterpret_cast<PF_Pixel_HSI_32f* __restrict>(tmpFloatPtr + imgSize);

        utils_prepare_data(localSrc, tmpFloatPtr, width, height, line_pitch, -1.f);

		// first path - convert image to HSI and build H and I histogramm/
		const A_long nhighsat = convert2HSI(tmpFloatPtr, pTmpStorage, histH, histI, width, height, qH, qI, sMin);

		// second path - segment histogram and build palette
		auto const isGray{ 0 == nhighsat };
		constexpr float epsilon{ 1.0f };

		std::vector<int32_t> ftcSegI;
		std::vector<int32_t> ftcSegH;
		std::vector<Hsegment>hSegments;
		std::vector<Isegment>iSegments;

		if (true == isGray || true == bForceGray)
		{
			ftcSegI = ftc_utils_segmentation(histI, nBinsI, epsilon, isGray);
			iSegments = compute_gray_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsI, qI, ftcSegI);
		}
		else
		{
			ftcSegH = ftc_utils_segmentation(histH, nBinsH, epsilon, isGray);
			hSegments = compute_color_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsH, nBinsS, nBinsI, qH, qS, qI, ftcSegH, epsilon);
		}

		// create segmented image (lets re-use temporary buffer for assemble image) */
		float* __restrict fR = reinterpret_cast<float*>(tmpFloatPtr);
		float* __restrict fG = fR + imgSize;
		float* __restrict fB = fG + imgSize;
		assemble_segmented_image(iSegments, hSegments, fR, fG, fB, width, height);

        // store segmented image */
		store_segmented_image(localSrc, localDst, fR, fG, fB, width, height, line_pitch, line_pitch);

        ::FreeMemoryBlock(blockId);
        blockId = -1;
        pMemoryBlock = nullptr;
        tmpFloatPtr = nullptr;
        pTmpStorage = nullptr;

	} // if (nullptr != pMemoryBlock && blockId >= 0)
	else
		errCode = PF_Err_OUT_OF_MEMORY;

	return errCode;
}


static PF_Err PR_ImageStyle_CartoonEffect_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN int32_t histH[hist_size_H];
	CACHE_ALIGN int32_t histI[hist_size_I];

	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

	PF_Err errCode = PF_Err_NONE;
	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
	const A_long imgSize = height * width;

	int nBinsH = static_cast<int>(static_cast<float>(hist_size_H) / qH);
	int nBinsS = static_cast<int>(static_cast<float>(hist_size_S) / qS);
	int nBinsI = static_cast<int>(static_cast<float>(hist_size_I) / qI);

	if (nBinsH * qH < hist_size_H) nBinsH++;
	if (nBinsS * qS < hist_size_S) nBinsS++;
	if (nBinsI * qI < hist_size_I) nBinsI++;

	const float sMin = static_cast<float>(nBinsH) / FastCompute::PIx2; // compute minimum saturation value that prevents quantization problems 
	const bool  bForceGray = false; // reads this value from check box on control pannel

    constexpr A_long bufferElementsSize = static_cast<A_long>(sizeof(fDataRGB) + sizeof(PF_Pixel_HSI_32f));
    const A_long totalProcMem = CreateAlignment(imgSize * bufferElementsSize, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fDataRGB* __restrict tmpFloatPtr = static_cast<fDataRGB* __restrict>(pMemoryBlock);
        PF_Pixel_HSI_32f* __restrict pTmpStorage = reinterpret_cast<PF_Pixel_HSI_32f* __restrict>(tmpFloatPtr + imgSize);

        utils_prepare_data(localSrc, tmpFloatPtr, width, height, line_pitch, 255.f);

		// first path - convert image to HSI and build H and I histogramm
		const A_long nhighsat = convert2HSI(tmpFloatPtr, pTmpStorage, histH, histI, width, height, qH, qI, sMin);

		// second path - segment histogram and build palette
		auto const isGray{ 0 == nhighsat };
		constexpr float epsilon{ 1.0f };

		std::vector<int32_t> ftcSegI;
		std::vector<int32_t> ftcSegH;
		std::vector<Hsegment>hSegments;
		std::vector<Isegment>iSegments;

		if (true == isGray || true == bForceGray)
		{
			ftcSegI = ftc_utils_segmentation(histI, nBinsI, epsilon, isGray);
			iSegments = compute_gray_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsI, qI, ftcSegI);
		}
		else
		{
			ftcSegH = ftc_utils_segmentation(histH, nBinsH, epsilon, isGray);
			hSegments = compute_color_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsH, nBinsS, nBinsI, qH, qS, qI, ftcSegH, epsilon);
		}

		// create segmented image (lets re-use temporary buffer for assemble image) */
		float* __restrict fR = reinterpret_cast<float*>(tmpFloatPtr);
		float* __restrict fG = fR + imgSize;
		float* __restrict fB = fG + imgSize;
		assemble_segmented_image(iSegments, hSegments, fR, fG, fB, width, height);
		// store segmented image */
		store_segmented_image(localSrc, localDst, fR, fG, fB, width, height, line_pitch, line_pitch);

        ::FreeMemoryBlock(blockId);
        blockId = -1;
        pMemoryBlock = nullptr;
        tmpFloatPtr = nullptr;
        pTmpStorage = nullptr;

    } // if (nullptr != pMemoryBlock && blockId >= 0)
	else
		errCode = PF_Err_OUT_OF_MEMORY;

	return errCode;
}


static PF_Err PR_ImageStyle_CartoonEffect_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN int32_t histH[hist_size_H];
	CACHE_ALIGN int32_t histI[hist_size_I];

	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	PF_Err errCode = PF_Err_NONE;
	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
	const A_long imgSize = height * width;

	int nBinsH = static_cast<int>(static_cast<float>(hist_size_H) / qH);
	int nBinsS = static_cast<int>(static_cast<float>(hist_size_S) / qS);
	int nBinsI = static_cast<int>(static_cast<float>(hist_size_I) / qI);

	if (nBinsH * qH < hist_size_H) nBinsH++;
	if (nBinsS * qS < hist_size_S) nBinsS++;
	if (nBinsI * qI < hist_size_I) nBinsI++;

	const float sMin = static_cast<float>(nBinsH) / FastCompute::PIx2; // compute minimum saturation value that prevents quantization problems 
	const bool  bForceGray = false; // reads this value from check box on control pannel
    constexpr float reciprocVal = 256.f / 32768.f;

    constexpr A_long bufferElementsSize = static_cast<A_long>(sizeof(fDataRGB) + sizeof(PF_Pixel_HSI_32f));
    const A_long totalProcMem = CreateAlignment(imgSize * bufferElementsSize, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fDataRGB* __restrict tmpFloatPtr = static_cast<fDataRGB* __restrict>(pMemoryBlock);
        PF_Pixel_HSI_32f* __restrict pTmpStorage = reinterpret_cast<PF_Pixel_HSI_32f* __restrict>(tmpFloatPtr + imgSize);

        utils_prepare_data(localSrc, tmpFloatPtr, width, height, line_pitch, reciprocVal);

		// first path - convert image to HSI and build H and I histogramm
		const A_long nhighsat = convert2HSI(tmpFloatPtr, pTmpStorage, histH, histI, width, height, qH, qI, sMin);

		// second path - segment histogram and build palette
		auto const isGray{ 0 == nhighsat };
		constexpr float epsilon{ 1.0f };

		std::vector<int32_t> ftcSegI;
		std::vector<int32_t> ftcSegH;
		std::vector<Hsegment>hSegments;
		std::vector<Isegment>iSegments;

		if (true == isGray || true == bForceGray)
		{
			ftcSegI = ftc_utils_segmentation(histI, nBinsI, epsilon, isGray);
			iSegments = compute_gray_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsI, qI, ftcSegI);
		}
		else
		{
			ftcSegH = ftc_utils_segmentation(histH, nBinsH, epsilon, isGray);
			hSegments = compute_color_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsH, nBinsS, nBinsI, qH, qS, qI, ftcSegH, epsilon);
		}

		// create segmented image (lets re-use temporary buffer for assemble image)
		float* __restrict fR = reinterpret_cast<float*>(tmpFloatPtr);
		float* __restrict fG = fR + imgSize;
		float* __restrict fB = fG + imgSize;
		assemble_segmented_image (iSegments, hSegments, fR, fG, fB, width, height);
		// store segmented image
		store_segmented_image (localSrc, localDst, fR, fG, fB, width, height, line_pitch, line_pitch);

        ::FreeMemoryBlock(blockId);
        blockId = -1;
        pMemoryBlock = nullptr;
        tmpFloatPtr = nullptr;
        pTmpStorage = nullptr;

    } // if (nullptr != pMemoryBlock && blockId >= 0)
    else
        errCode = PF_Err_OUT_OF_MEMORY;

	return errCode;
}


static PF_Err PR_ImageStyle_CartoonEffect_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN int32_t histH[hist_size_H];
	CACHE_ALIGN int32_t histI[hist_size_I];

	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	PF_Err errCode = PF_Err_NONE;
	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
	const A_long imgSize = height * width;

	const size_t requiredMemSize = imgSize * sizeof(float) * 3;
	int j = 0, i = 0;

	int nBinsH = static_cast<int>(static_cast<float>(hist_size_H) / qH);
	int nBinsS = static_cast<int>(static_cast<float>(hist_size_S) / qS);
	int nBinsI = static_cast<int>(static_cast<float>(hist_size_I) / qI);

	if (nBinsH * qH < hist_size_H) nBinsH++;
	if (nBinsS * qS < hist_size_S) nBinsS++;
	if (nBinsI * qI < hist_size_I) nBinsI++;

	const float sMin = static_cast<float>(nBinsH) / FastCompute::PIx2; // compute minimum saturation value that prevents quantization problems 
	const bool  bForceGray = false; // reads this value from check box on control pannel
    constexpr float mulVal = 255.f;

    constexpr A_long bufferElementsSize = static_cast<A_long>(sizeof(fDataRGB) + sizeof(PF_Pixel_HSI_32f));
    const A_long totalProcMem = CreateAlignment(imgSize * bufferElementsSize, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fDataRGB* __restrict tmpFloatPtr = static_cast<fDataRGB* __restrict>(pMemoryBlock);
        PF_Pixel_HSI_32f* __restrict pTmpStorage = reinterpret_cast<PF_Pixel_HSI_32f* __restrict>(tmpFloatPtr + imgSize);

        utils_prepare_data(localSrc, tmpFloatPtr, width, height, line_pitch, mulVal);

		// first path - convert image to HSI and build H and I histogramm 
		const A_long nhighsat = convert2HSI(tmpFloatPtr, pTmpStorage, histH, histI, width, height, qH, qI, sMin);

		// second path - segment histogram and build palette 
		auto const isGray{ 0 == nhighsat };
		constexpr float epsilon{ 1.0f };

		std::vector<int32_t> ftcSegI;
		std::vector<int32_t> ftcSegH;
		std::vector<Hsegment>hSegments;
		std::vector<Isegment>iSegments;

		if (true == isGray || true == bForceGray)
		{
			ftcSegI = ftc_utils_segmentation(histI, nBinsI, epsilon, isGray);
			iSegments = compute_gray_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsI, qI, ftcSegI);
		}
		else
		{
			ftcSegH = ftc_utils_segmentation(histH, nBinsH, epsilon, isGray);
			hSegments = compute_color_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsH, nBinsS, nBinsI, qH, qS, qI, ftcSegH, epsilon);
		}

		// create segmented image (lets re-use temporary buffer for assemble image)
		float* __restrict fR = reinterpret_cast<float*>(tmpFloatPtr);
		float* __restrict fG = fR + imgSize;
		float* __restrict fB = fG + imgSize;
		assemble_segmented_image(iSegments, hSegments, fR, fG, fB, width, height);

		// store segmented image
		store_segmented_image(localSrc, localDst, fR, fG, fB, width, height, line_pitch, line_pitch);

        ::FreeMemoryBlock(blockId);
        blockId = -1;
        pMemoryBlock = nullptr;
        tmpFloatPtr = nullptr;
        pTmpStorage = nullptr;

    } // if (nullptr != pMemoryBlock && blockId >= 0)
	else
		errCode = PF_Err_OUT_OF_MEMORY;

	return errCode;
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
	CACHE_ALIGN int32_t histH[hist_size_H];
	CACHE_ALIGN int32_t histI[hist_size_I];

	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_8u*     __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*     __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

	PF_Err errCode = PF_Err_NONE;

	const A_long height = output->height;
	const A_long width  = output->width;
	const A_long src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	const A_long dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	const A_long imgSize = height * width;

	int nBinsH = static_cast<int>(static_cast<float>(hist_size_H) / qH);
	int nBinsS = static_cast<int>(static_cast<float>(hist_size_S) / qS);
	int nBinsI = static_cast<int>(static_cast<float>(hist_size_I) / qI);

	if (nBinsH * qH < hist_size_H) nBinsH++;
	if (nBinsS * qS < hist_size_S) nBinsS++;
	if (nBinsI * qI < hist_size_I) nBinsI++;

	const float sMin = static_cast<float>(nBinsH) / FastCompute::PIx2; // compute minimum saturation value that prevents quantization problems 
	const bool  bForceGray = false; // reads this value from check box on control pannel

    constexpr A_long bufferElementsSize = static_cast<A_long>(sizeof(fDataRGB) + sizeof(PF_Pixel_HSI_32f));
    const A_long totalProcMem = CreateAlignment(imgSize * bufferElementsSize, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fDataRGB* __restrict tmpFloatPtr = static_cast<fDataRGB* __restrict>(pMemoryBlock);
        PF_Pixel_HSI_32f* __restrict pTmpStorage = reinterpret_cast<PF_Pixel_HSI_32f* __restrict>(tmpFloatPtr + imgSize);

        utils_prepare_data (localSrc, tmpFloatPtr, width, height, src_line_pitch, -1.f);

		// first path - convert image to HSI and build H and I histogramm
		const A_long nhighsat = convert2HSI (tmpFloatPtr, pTmpStorage, histH, histI, width, height, qH, qI, sMin);

		// second path - segment histogram and build palette
		auto const isGray{ 0 == nhighsat };
		constexpr float epsilon{ 1.0f };

		std::vector<int32_t> ftcSegI;
		std::vector<int32_t> ftcSegH;
		std::vector<Hsegment>hSegments;
		std::vector<Isegment>iSegments;

		if (true == isGray || true == bForceGray)
		{
			ftcSegI = ftc_utils_segmentation(histI, nBinsI, epsilon, isGray);
			iSegments = compute_gray_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsI, qI, ftcSegI);
		}
		else
		{
			ftcSegH = ftc_utils_segmentation(histH, nBinsH, epsilon, isGray);
			hSegments = compute_color_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsH, nBinsS, nBinsI, qH, qS, qI, ftcSegH, epsilon);
		}

		// create segmented image (lets re-use temporary buffer for assemble image)
		float* __restrict fR = reinterpret_cast<float*>(tmpFloatPtr);
		float* __restrict fG = fR + imgSize;
		float* __restrict fB = fG + imgSize;
		assemble_segmented_image(iSegments, hSegments, fR, fG, fB, width, height);

		// store segmented image
		store_segmented_image(localSrc, localDst, fR, fG, fB, width, height, src_line_pitch, dst_line_pitch);

        ::FreeMemoryBlock(blockId);
        blockId = -1;
        pMemoryBlock = nullptr;
        tmpFloatPtr = nullptr;
        pTmpStorage = nullptr;

    } // if (true == bMemSizeTest)
		errCode = PF_Err_OUT_OF_MEMORY;

	return errCode;
}


PF_Err AE_ImageStyle_CartoonEffect_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN int32_t histH[hist_size_H];
	CACHE_ALIGN int32_t histI[hist_size_I];

	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_16u*    __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(input->data);
	PF_Pixel_ARGB_16u*    __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output->data);

	PF_Err errCode = PF_Err_NONE;

	const A_long height = output->height;
	const A_long width = output->width;
	const A_long src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	const A_long dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

	const A_long imgSize = height * width;

	int nBinsH = static_cast<int>(static_cast<float>(hist_size_H) / qH);
	int nBinsS = static_cast<int>(static_cast<float>(hist_size_S) / qS);
	int nBinsI = static_cast<int>(static_cast<float>(hist_size_I) / qI);

	if (nBinsH * qH < hist_size_H) nBinsH++;
	if (nBinsS * qS < hist_size_S) nBinsS++;
	if (nBinsI * qI < hist_size_I) nBinsI++;

	const float sMin = static_cast<float>(nBinsH) / FastCompute::PIx2; // compute minimum saturation value that prevents quantization problems 
	const bool  bForceGray = false; // reads this value from check box on control pannel
    constexpr float reciprocVal = 256.0f / 32768.0f;

    constexpr A_long bufferElementsSize = static_cast<A_long>(sizeof(fDataRGB) + sizeof(PF_Pixel_HSI_32f));
    const A_long totalProcMem = CreateAlignment(imgSize * bufferElementsSize, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fDataRGB* __restrict tmpFloatPtr = static_cast<fDataRGB* __restrict>(pMemoryBlock);
        PF_Pixel_HSI_32f* __restrict pTmpStorage = reinterpret_cast<PF_Pixel_HSI_32f* __restrict>(tmpFloatPtr + imgSize);

        utils_prepare_data(localSrc, tmpFloatPtr, width, height, src_line_pitch, reciprocVal);

		// first path - convert image to HSI and build H and I histogramm
		const A_long nhighsat = convert2HSI(tmpFloatPtr, pTmpStorage, histH, histI, width, height, qH, qI, sMin);

		// second path - segment histogram and build palette
		auto const isGray{ 0 == nhighsat };
		constexpr float epsilon{ 1.0f };

		std::vector<int32_t> ftcSegI;
		std::vector<int32_t> ftcSegH;
		std::vector<Hsegment>hSegments;
		std::vector<Isegment>iSegments;

		if (true == isGray || true == bForceGray)
		{
			ftcSegI = ftc_utils_segmentation(histI, nBinsI, epsilon, isGray);
			iSegments = compute_gray_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsI, qI, ftcSegI);
		}
		else
		{
			ftcSegH = ftc_utils_segmentation(histH, nBinsH, epsilon, isGray);
			hSegments = compute_color_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsH, nBinsS, nBinsI, qH, qS, qI, ftcSegH, epsilon);
		}

		// create segmented image (lets re-use temporary buffer for assemble image)
		float* __restrict fR = reinterpret_cast<float*>(tmpFloatPtr);
		float* __restrict fG = fR + imgSize;
		float* __restrict fB = fG + imgSize;
		assemble_segmented_image (iSegments, hSegments, fR, fG, fB, width, height);

		// store segmented image
		store_segmented_image (localSrc, localDst, fR, fG, fB, width, height, src_line_pitch, dst_line_pitch);

        ::FreeMemoryBlock(blockId);
        blockId = -1;
        pMemoryBlock = nullptr;
        tmpFloatPtr = nullptr;
        pTmpStorage = nullptr;

    } // if (true == bMemSizeTest)
	else
		errCode = PF_Err_OUT_OF_MEMORY;

	return errCode;
}


PF_Err AE_ImageStyle_CartoonEffect_ARGB_32f
(
    PF_InData*   __restrict in_data,
    PF_OutData*  __restrict out_data,
    PF_ParamDef* __restrict params[],
    PF_LayerDef* __restrict output
) noexcept
{
    CACHE_ALIGN int32_t histH[hist_size_H];
    CACHE_ALIGN int32_t histI[hist_size_I];

    const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    PF_Pixel_ARGB_32f*    __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(input->data);
    PF_Pixel_ARGB_32f*    __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(output->data);

    PF_Err errCode = PF_Err_NONE;

    const A_long height = output->height;
    const A_long width = output->width;
    const A_long src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

    const A_long imgSize = height * width;

    const size_t requiredMemSize = imgSize * sizeof(float) * 3;
    int j = 0, i = 0;

    int nBinsH = static_cast<int>(static_cast<float>(hist_size_H) / qH);
    int nBinsS = static_cast<int>(static_cast<float>(hist_size_S) / qS);
    int nBinsI = static_cast<int>(static_cast<float>(hist_size_I) / qI);

    if (nBinsH * qH < hist_size_H) nBinsH++;
    if (nBinsS * qS < hist_size_S) nBinsS++;
    if (nBinsI * qI < hist_size_I) nBinsI++;

    const float sMin = static_cast<float>(nBinsH) / FastCompute::PIx2; // compute minimum saturation value that prevents quantization problems 
    const bool  bForceGray = false; // reads this value from check box on control pannel
    constexpr float mulVal = 255.f;

    constexpr A_long bufferElementsSize = static_cast<A_long>(sizeof(fDataRGB) + sizeof(PF_Pixel_HSI_32f));
    const A_long totalProcMem = CreateAlignment(imgSize * bufferElementsSize, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fDataRGB* __restrict tmpFloatPtr = static_cast<fDataRGB* __restrict>(pMemoryBlock);
        PF_Pixel_HSI_32f* __restrict pTmpStorage = reinterpret_cast<PF_Pixel_HSI_32f* __restrict>(tmpFloatPtr + imgSize);

        utils_prepare_data(localSrc, tmpFloatPtr, width, height, src_line_pitch, mulVal);

        // first path - convert image to HSI and build H and I histogramm */
        const A_long nhighsat = convert2HSI(tmpFloatPtr, pTmpStorage, histH, histI, width, height, qH, qI, sMin);

        // second path - segment histogram and build palette
        auto const isGray{ 0 == nhighsat };
        constexpr float epsilon{ 1.0f };

        std::vector<int32_t> ftcSegI;
        std::vector<int32_t> ftcSegH;
        std::vector<Hsegment>hSegments;
        std::vector<Isegment>iSegments;

        if (true == isGray || true == bForceGray)
        {
            ftcSegI = ftc_utils_segmentation(histI, nBinsI, epsilon, isGray);
            iSegments = compute_gray_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsI, qI, ftcSegI);
        }
        else
        {
            ftcSegH = ftc_utils_segmentation(histH, nBinsH, epsilon, isGray);
            hSegments = compute_color_palette(pTmpStorage, tmpFloatPtr, width, height, sMin, nBinsH, nBinsS, nBinsI, qH, qS, qI, ftcSegH, epsilon);
        }

        // create segmented image (lets re-use temporary buffer for assemble image)
        float* __restrict fR = reinterpret_cast<float*>(tmpFloatPtr);
        float* __restrict fG = fR + imgSize;
        float* __restrict fB = fG + imgSize;
        assemble_segmented_image (iSegments, hSegments, fR, fG, fB, width, height);

        // store segmented image
        store_segmented_image (localSrc, localDst, fR, fG, fB, width, height, src_line_pitch, dst_line_pitch);

        ::FreeMemoryBlock(blockId);
        blockId = -1;
        pMemoryBlock = nullptr;
        tmpFloatPtr = nullptr;
        pTmpStorage = nullptr;

    } // if (true == bMemSizeTest)
    else
        errCode = PF_Err_OUT_OF_MEMORY;

    return errCode;
}