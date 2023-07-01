#include "FastAriphmetics.hpp"
#include "ImageEqualization.hpp"
#include "ColorTransform.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ImageEqualizationColorTransform.hpp"
#include <memory>


PF_Err PR_ImageEq_Advanced_BGRA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	A_long i, j;
	const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	      PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);

	auto const sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	constexpr size_t components_size = fCIELabPix_size /* CIELab buffer */ + fXYZPix_size /* XYZ buffer */ + sizeof(float)/* cs_out buffer */;
	const size_t frameSize = sizeX * sizeY;
	const size_t memSizeForTmpBuffers = frameSize * components_size;
	const size_t requiredMemSize = CreateAlignment(memSizeForTmpBuffers, static_cast<size_t>(CACHE_LINE));

	/* Get memory block */
	void* pMemoryBlock = nullptr;
	const int32_t blockId = ::GetMemoryBlock (requiredMemSize, 0, &pMemoryBlock);

	if (-1 != blockId && nullptr != pMemoryBlock)
	{
		constexpr float scale = static_cast<float>(u8_value_white);
		constexpr float scale2sRGB = 1.f / scale;
		const float* __restrict fReferences = cCOLOR_ILLUMINANT[observer_CIE_1931][color_ILLUMINANT_D65];
		
		fCIELabPix* srcLabBuffer = reinterpret_cast<fCIELabPix*>(pMemoryBlock);
		float*      cs_out       = reinterpret_cast<float*     >(srcLabBuffer + frameSize);
		fXYZPix*    xyz_buffer   = reinterpret_cast<fXYZPix*   >(cs_out + frameSize);

		constexpr float lightness_enhanecemnt = 1.25f;

#ifdef _DEBUG
		memset(pMemoryBlock, 0, requiredMemSize);
#endif // _DEBUG

		/* convert from RGB to CIEL*a*b color space */
		const float cs_out_max = ConvertToLabColorSpace (localSrc, srcLabBuffer, cs_out, sizeX, sizeY, line_pitch, sizeX, scale2sRGB);

		/* improve */
		fCIELabHueImprove (srcLabBuffer, xyz_buffer, cs_out, sizeX, sizeY, 85.f, lightness_enhanecemnt, cs_out_max);

		/* convert improved XYZ to RGB */
		ImprovedImageRestore (localSrc, localDst, xyz_buffer, sizeX, sizeY, line_pitch, sizeX, line_pitch, scale);

		/* release memory block */
		::FreeMemoryBlock(blockId);
	}

	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Advanced_BGRA_4444_16u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	A_long i, j;
	const PF_LayerDef*      __restrict pfLayer   = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	      PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);

	auto const sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	constexpr size_t components_size = fCIELabPix_size /* CIELab buffer */ + fXYZPix_size /* XYZ buffer */ + sizeof(float)/* cs_out buffer */;
	const size_t frameSize = sizeX * sizeY;
	const size_t memSizeForTmpBuffers = frameSize * components_size;
	const size_t requiredMemSize = CreateAlignment(memSizeForTmpBuffers, static_cast<size_t>(CACHE_LINE));

	/* Get memory block */
	void* pMemoryBlock = nullptr;
	const int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemoryBlock);

	if (-1 != blockId && nullptr != pMemoryBlock)
	{
		constexpr float scale = static_cast<float>(u16_value_white);
		constexpr float scale2sRGB = 1.f / scale;
		const float* __restrict fReferences = cCOLOR_ILLUMINANT[observer_CIE_1931][color_ILLUMINANT_D65];

		fCIELabPix* srcLabBuffer = reinterpret_cast<fCIELabPix*>(pMemoryBlock);
		float*      cs_out = reinterpret_cast<float*     >(srcLabBuffer + frameSize);
		fXYZPix*    xyz_buffer = reinterpret_cast<fXYZPix*   >(cs_out + frameSize);

		constexpr float lightness_enhanecemnt = 1.25f;

#ifdef _DEBUG
		memset(pMemoryBlock, 0, requiredMemSize);
#endif // _DEBUG

		/* convert from RGB to CIEL*a*b color space */
		const float cs_out_max = ConvertToLabColorSpace(localSrc, srcLabBuffer, cs_out, sizeX, sizeY, line_pitch, sizeX, scale2sRGB);

		/* improve */
		fCIELabHueImprove(srcLabBuffer, xyz_buffer, cs_out, sizeX, sizeY, 85.f, lightness_enhanecemnt, cs_out_max);

		/* convert improved XYZ to RGB */
		ImprovedImageRestore(localSrc, localDst, xyz_buffer, sizeX, sizeY, line_pitch, sizeX, line_pitch, scale);

		/* release memory block */
		::FreeMemoryBlock(blockId);
	}

	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Advanced_BGRA_4444_32f
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err AE_ImageEq_Advanced_ARGB_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err AE_ImageEq_Advanced_ARGB_4444_16u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Advanced_VUYA_4444_8u_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Advanced_VUYA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PR_ImageEq_Advanced_VUYA_4444_8u_709(in_data, out_data, params, output);
}


PF_Err PR_ImageEq_Advanced_VUYA_4444_32f_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Advanced_VUYA_4444_32f
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PR_ImageEq_Advanced_VUYA_4444_32f_709(in_data, out_data, params, output);
}
