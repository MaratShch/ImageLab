#include "NoiseClean.hpp"


PF_Err NoiseCleanPr_VUYA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const bool isBT709
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[NOISE_CLEAN_TOTAL_PARAMS]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

	PF_Err err = PF_Err_NONE;
	return err;
}
