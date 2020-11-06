#include "Convolution.hpp"
#include "ColorTransformMatrix.hpp"
#include "Kernels.hpp"


static bool
ProcessPrImage_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosedKernel
) noexcept
{
	IAbsrtactKernel<int32_t>* iKernel = nullptr;
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<PF_LayerDef* __restrict>(&params[CONVOLUTION_INPUT]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	A_long i = 0, j = 0;
	A_long l = 0, m = 0, k = 0;
	A_long accumR, accumG, accumB, k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;

	bool bResult = true;

	if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel)))
	{
		if (true == (bResult = iKernel->LoadKernel()))
		{
			const uint32_t kernSize = iKernel->GetSize();
			const int32_t* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
			const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

			for (j = 0; j < height; j++)
			{
				const A_long line_idx = j * line_pitch;

				for (i = 0; i < width; i++)
				{
					k_idx = elements - 1;
					accumR = accumG = accumB = 0;

					for (l = -halfKernelSize; l <= halfKernelSize; l++)
					{
						j_idx = j + l;
						if (j_idx < 0 || j_idx >= height)
						{
							k_idx--;
							continue;
						}

						for (m = -halfKernelSize; m <= halfKernelSize; m++)
						{
							i_idx = i + m;
							if (i_idx < 0 || i_idx >= width)
							{
								k_idx--;
								continue;
							}

							src_idx = j_idx * line_pitch + i_idx;

							accumR += (static_cast<int32_t>(localSrc[src_idx].R) * kernArray[k_idx]);
							accumG += (static_cast<int32_t>(localSrc[src_idx].G) * kernArray[k_idx]);
							accumB += (static_cast<int32_t>(localSrc[src_idx].B) * kernArray[k_idx]);

							k_idx--;

						} /* for (m = -halfKernelSize; m < halfKernelSize; m++) */

					} /* for (l = -halfKernelSize; l < halfKernelSize; l++) */

					const float outR = static_cast<float>(accumR) / kernSum;
					const float outG = static_cast<float>(accumG) / kernSum;
					const float outB = static_cast<float>(accumB) / kernSum;

					dst_idx = line_idx + i;
					localDst[dst_idx].R = static_cast<A_u_char>(CLAMP_VALUE(outR, 0.f, 255.f));
					localDst[dst_idx].G = static_cast<A_u_char>(CLAMP_VALUE(outG, 0.f, 255.f));
					localDst[dst_idx].B = static_cast<A_u_char>(CLAMP_VALUE(outB, 0.f, 255.f));
					localDst[dst_idx].A = localSrc[dst_idx].A;

				} /* for (i = 0; i < in_data->width; i++) */

			} /* for (j = 0; j < height; j++) */

		} /* if (true == iKernel->LoadKernel()) */
		else
			bResult = false;
	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel))) */
	else
		bResult = false;

	iKernel = nullptr; /* for DBG purpose */

	return bResult;
}


static bool
ProcessPrImage_BGRA_4444_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosedKernel
) noexcept
{
	IAbsrtactKernel<int32_t>* iKernel = nullptr;
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[CONVOLUTION_INPUT]->u.ld);
	PF_Pixel_BGRA_16u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	A_long i = 0, j = 0;
	A_long l = 0, m = 0, k = 0;
	A_long accumR, accumG, accumB, k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;

	bool bResult = true;

	if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const int32_t* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
			const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

			for (j = 0; j < height; j++)
			{
				const A_long line_idx = j * line_pitch;

				for (i = 0; i < width; i++)
				{
					k_idx = elements - 1;
					accumR = accumG = accumB = 0;

					for (l = -halfKernelSize; l <= halfKernelSize; l++)
					{
						j_idx = j + l;
						if (j_idx < 0 || j_idx >= height)
						{
							k_idx--;
							continue;
						}

						for (m = -halfKernelSize; m <= halfKernelSize; m++)
						{
							i_idx = i + m;
							if (i_idx < 0 || i_idx >= width)
							{
								k_idx--;
								continue;
							}

							src_idx = j_idx * line_pitch + i_idx;

							accumR += (static_cast<int32_t>(localSrc[src_idx].R) * kernArray[k_idx]);
							accumG += (static_cast<int32_t>(localSrc[src_idx].G) * kernArray[k_idx]);
							accumB += (static_cast<int32_t>(localSrc[src_idx].B) * kernArray[k_idx]);

							k_idx--;

						} /* for (m = -halfKernelSize; m < halfKernelSize; m++) */

					} /* for (l = -halfKernelSize; l < halfKernelSize; l++) */

					const float outR = static_cast<float>(accumR) / kernSum;
					const float outG = static_cast<float>(accumG) / kernSum;
					const float outB = static_cast<float>(accumB) / kernSum;

					dst_idx = line_idx + i;
					localDst[dst_idx].R = static_cast<A_u_short>(CLAMP_VALUE(outR, 0.f, 32767.0f));
					localDst[dst_idx].G = static_cast<A_u_short>(CLAMP_VALUE(outG, 0.f, 32767.0f));
					localDst[dst_idx].B = static_cast<A_u_short>(CLAMP_VALUE(outB, 0.f, 32767.0f));
					localDst[dst_idx].A = localSrc[dst_idx].A;

				} /* for (i = 0; i < in_data->width; i++) */

			} /* for (j = 0; j < height; j++) */

		} /* if (true == iKernel->LoadKernel()) */
		else
			bResult = false;
	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel))) */
	else
		bResult = false;

	iKernel = nullptr; /* for DBG purpose */

	return bResult;
}


static bool
ProcessPrImage_BGRA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosedKernel
) noexcept
{
	IAbsrtactKernel<float>* iKernel = nullptr;
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[CONVOLUTION_INPUT]->u.ld);
	PF_Pixel_BGRA_32f*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	A_long i = 0, j = 0;
	A_long l = 0, m = 0, k = 0;
	A_long k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;
	float accumR, accumG, accumB;

	bool bResult = true;

	if (nullptr != (iKernel = GetKernel<float>(choosedKernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const float* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
			const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

			for (j = 0; j < height; j++)
			{
				const A_long line_idx = j * line_pitch;

				for (i = 0; i < width; i++)
				{
					k_idx = elements - 1;
					accumR = accumG = accumB = 0.f;

					for (l = -halfKernelSize; l <= halfKernelSize; l++)
					{
						j_idx = j + l;
						if (j_idx < 0 || j_idx >= height)
						{
							k_idx--;
							continue;
						}

						for (m = -halfKernelSize; m <= halfKernelSize; m++)
						{
							i_idx = i + m;
							if (i_idx < 0 || i_idx >= width)
							{
								k_idx--;
								continue;
							}

							src_idx = j_idx * line_pitch + i_idx;

							accumR += localSrc[src_idx].R * kernArray[k_idx];
							accumG += localSrc[src_idx].G * kernArray[k_idx];
							accumB += localSrc[src_idx].B * kernArray[k_idx];

							k_idx--;

						} /* for (m = -halfKernelSize; m < halfKernelSize; m++) */

					} /* for (l = -halfKernelSize; l < halfKernelSize; l++) */

					const float outR = static_cast<float>(accumR) / kernSum;
					const float outG = static_cast<float>(accumG) / kernSum;
					const float outB = static_cast<float>(accumB) / kernSum;

					dst_idx = line_idx + i;
					localDst[dst_idx].R = CLAMP_VALUE(outR, f32_value_black, f32_value_white);
					localDst[dst_idx].G = CLAMP_VALUE(outG, f32_value_black, f32_value_white);
					localDst[dst_idx].B = CLAMP_VALUE(outB, f32_value_black, f32_value_white);
					localDst[dst_idx].A = localSrc[dst_idx].A;

				} /* for (i = 0; i < in_data->width; i++) */

			} /* for (j = 0; j < height; j++) */

		} /* if (true == iKernel->LoadKernel()) */
		else
			bResult = false;

	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel))) */
	else
		bResult = false;

	iKernel = nullptr; /* for DBG purpose */

	return bResult;
}


static bool
ProcessPrImage_VUYA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosedKernel,
	const bool& isBT709
) noexcept
{
	IAbsrtactKernel<float>* iKernel = nullptr;
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[CONVOLUTION_INPUT]->u.ld);
	PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	const float* __restrict yuv2rgb = (true == isBT709 ? YUV2RGB[BT709] : YUV2RGB[BT601]);
	const float* __restrict rgb2yuv = (true == isBT709 ? RGB2YUV[BT709] : RGB2YUV[BT601]);

	A_long i = 0, j = 0;
	A_long l = 0, m = 0, k = 0;
	A_long k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;
	float R, G, B, accumR, accumG, accumB, newY, newU, newV;

	bool bResult = true;

	if (nullptr != (iKernel = GetKernel<float>(choosedKernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const float* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
			const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

			for (j = 0; j < height; j++)
			{
				const A_long line_idx = j * line_pitch;

				for (i = 0; i < width; i++)
				{
					k_idx = elements - 1;
					accumR = accumG = accumB = 0.f;

					for (l = -halfKernelSize; l <= halfKernelSize; l++)
					{
						j_idx = j + l;
						if (j_idx < 0 || j_idx >= height)
						{
							k_idx--;
							continue;
						}

						for (m = -halfKernelSize; m <= halfKernelSize; m++)
						{
							i_idx = i + m;
							if (i_idx < 0 || i_idx >= width)
							{
								k_idx--;
								continue;
							}

							src_idx = j_idx * line_pitch + i_idx;

							R = static_cast<float>(localSrc[src_idx].Y) * yuv2rgb[0] +
								static_cast<float>(localSrc[src_idx].U - 128) * yuv2rgb[1] +
								static_cast<float>(localSrc[src_idx].V - 128) * yuv2rgb[2];

							G = static_cast<float>(localSrc[src_idx].Y) * yuv2rgb[3] +
								static_cast<float>(localSrc[src_idx].U - 128) * yuv2rgb[4] +
								static_cast<float>(localSrc[src_idx].V - 128) * yuv2rgb[5];

							B = static_cast<float>(localSrc[src_idx].Y) * yuv2rgb[6] +
								static_cast<float>(localSrc[src_idx].U - 128) * yuv2rgb[7] +
								static_cast<float>(localSrc[src_idx].V - 128) * yuv2rgb[8];

							accumR += R * kernArray[k_idx];
							accumG += G * kernArray[k_idx];
							accumB += B * kernArray[k_idx];

							k_idx--;

						} /* for (m = -halfKernelSize; m < halfKernelSize; m++) */

					} /* for (l = -halfKernelSize; l < halfKernelSize; l++) */

					const float outR = CLAMP_VALUE(accumR / kernSum, 0.f, 255.f);
					const float outG = CLAMP_VALUE(accumG / kernSum, 0.f, 255.f);
					const float outB = CLAMP_VALUE(accumB / kernSum, 0.f, 255.f);

					newY = outR * rgb2yuv[0] + outG * rgb2yuv[1] + outB * rgb2yuv[2];
					newU = outR * rgb2yuv[3] + outG * rgb2yuv[4] + outB * rgb2yuv[5] + 128.f;
					newV = outR * rgb2yuv[6] + outG * rgb2yuv[7] + outB * rgb2yuv[8] + 128.f;

					dst_idx = line_idx + i;

					localDst[dst_idx].Y = static_cast<A_u_char>(newY);
					localDst[dst_idx].U = static_cast<A_u_char>(newU);
					localDst[dst_idx].V = static_cast<A_u_char>(newV);
					localDst[dst_idx].A = localSrc[dst_idx].A;

				} /* for (i = 0; i < in_data->width; i++) */

			} /* for (j = 0; j < height; j++) */

		} /* if (true == iKernel->LoadKernel()) */
		else
			bResult = false;
	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel))) */
	else
		bResult = false;

	iKernel = nullptr; /* for DBG purpose */

	return bResult;
}

static bool
ProcessPrImage_VUYA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosedKernel,
	const bool& isBT709
) noexcept
{
	IAbsrtactKernel<float>* iKernel = nullptr;
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[CONVOLUTION_INPUT]->u.ld);
	PF_Pixel_VUYA_32f*  __restrict localSrc = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_32f*  __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

	const float* __restrict yuv2rgb = (true == isBT709 ? YUV2RGB[BT709] : YUV2RGB[BT601]);
	const float* __restrict rgb2yuv = (true == isBT709 ? RGB2YUV[BT709] : RGB2YUV[BT601]);

	A_long i = 0, j = 0;
	A_long l = 0, m = 0, k = 0;
	A_long k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;
	float R, G, B, accumR, accumG, accumB, newY, newU, newV;

	bool bResult = true;

	if (nullptr != (iKernel = GetKernel<float>(choosedKernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const float* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
			const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

			for (j = 0; j < height; j++)
			{
				const A_long line_idx = j * line_pitch;

				for (i = 0; i < width; i++)
				{
					k_idx = elements - 1;
					accumR = accumG = accumB = 0.f;

					for (l = -halfKernelSize; l <= halfKernelSize; l++)
					{
						j_idx = j + l;
						if (j_idx < 0 || j_idx >= height)
						{
							k_idx--;
							continue;
						}

						for (m = -halfKernelSize; m <= halfKernelSize; m++)
						{
							i_idx = i + m;
							if (i_idx < 0 || i_idx >= width)
							{
								k_idx--;
								continue;
							}

							src_idx = j_idx * line_pitch + i_idx;

							R = localSrc[src_idx].Y * yuv2rgb[0] + localSrc[src_idx].U * yuv2rgb[1] + localSrc[src_idx].V * yuv2rgb[2];
							G = localSrc[src_idx].Y * yuv2rgb[3] + localSrc[src_idx].U * yuv2rgb[4] + localSrc[src_idx].V * yuv2rgb[5];
							B = localSrc[src_idx].Y * yuv2rgb[6] + localSrc[src_idx].U * yuv2rgb[7] + localSrc[src_idx].V * yuv2rgb[8];

							accumR += R * kernArray[k_idx];
							accumG += G * kernArray[k_idx];
							accumB += B * kernArray[k_idx];

							k_idx--;

						} /* for (m = -halfKernelSize; m < halfKernelSize; m++) */

					} /* for (l = -halfKernelSize; l < halfKernelSize; l++) */

					const float outR = CLAMP_VALUE(accumR / kernSum, f32_value_black, f32_value_white);
					const float outG = CLAMP_VALUE(accumG / kernSum, f32_value_black, f32_value_white);
					const float outB = CLAMP_VALUE(accumB / kernSum, f32_value_black, f32_value_white);

					newY = outR * rgb2yuv[0] + outG * rgb2yuv[1] + outB * rgb2yuv[2];
					newU = outR * rgb2yuv[3] + outG * rgb2yuv[4] + outB * rgb2yuv[5];
					newV = outR * rgb2yuv[6] + outG * rgb2yuv[7] + outB * rgb2yuv[8];

					dst_idx = line_idx + i;

					localDst[dst_idx].Y = newY;
					localDst[dst_idx].U = newU;
					localDst[dst_idx].V = newV;
					localDst[dst_idx].A = localSrc[dst_idx].A;

				} /* for (i = 0; i < in_data->width; i++) */

			} /* for (j = 0; j < height; j++) */

		} /* if (true == iKernel->LoadKernel()) */
		else
			bResult = false;
	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel))) */
	else
		bResult = false;

	iKernel = nullptr; /* for DBG purpose */

	return bResult;
}


static bool
ProcessPrImage_RGB_444_10u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const uint32_t& choosedKernel
) noexcept
{
	IAbsrtactKernel<int32_t>* iKernel = nullptr;
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<PF_LayerDef* __restrict>(&params[CONVOLUTION_INPUT]->u.ld);
	PF_Pixel_RGB_10u*  __restrict localSrc = reinterpret_cast<PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
	PF_Pixel_RGB_10u*  __restrict localDst = reinterpret_cast<PF_Pixel_RGB_10u* __restrict>(output->data);

	A_long i = 0, j = 0;
	A_long l = 0, m = 0, k = 0;
	A_long accumR, accumG, accumB, k_idx = 0, j_idx = 0, i_idx = 0, src_idx = 0, dst_idx = 0;

	bool bResult = true;

	if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel)))
	{
		if (true == iKernel->LoadKernel())
		{
			const uint32_t kernSize = iKernel->GetSize();
			const int32_t* __restrict kernArray = iKernel->GetArray();
			const int32_t elements = static_cast<int32_t>(kernSize * kernSize);
			const int32_t halfKernelSize = static_cast<int32_t>(kernSize) >> 1;

			const float kernSum = iKernel->Normalizing();

			const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
			const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
			const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);

			for (j = 0; j < height; j++)
			{
				const A_long line_idx = j * line_pitch;

				for (i = 0; i < width; i++)
				{
					k_idx = elements - 1;
					accumR = accumG = accumB = 0;

					for (l = -halfKernelSize; l <= halfKernelSize; l++)
					{
						j_idx = j + l;
						if (j_idx < 0 || j_idx >= height)
						{
							k_idx--;
							continue;
						}

						for (m = -halfKernelSize; m <= halfKernelSize; m++)
						{
							i_idx = i + m;
							if (i_idx < 0 || i_idx >= width)
							{
								k_idx--;
								continue;
							}

							src_idx = j_idx * line_pitch + i_idx;

							accumR += (static_cast<int32_t>(localSrc[src_idx].R) * kernArray[k_idx]);
							accumG += (static_cast<int32_t>(localSrc[src_idx].G) * kernArray[k_idx]);
							accumB += (static_cast<int32_t>(localSrc[src_idx].B) * kernArray[k_idx]);

							k_idx--;

						} /* for (m = -halfKernelSize; m < halfKernelSize; m++) */

					} /* for (l = -halfKernelSize; l < halfKernelSize; l++) */

					const float outR = static_cast<float>(accumR) / kernSum;
					const float outG = static_cast<float>(accumG) / kernSum;
					const float outB = static_cast<float>(accumB) / kernSum;

					dst_idx = line_idx + i;
					localDst[dst_idx].R = static_cast<A_u_long>(CLAMP_VALUE(outR, 0.f, 1023.f));
					localDst[dst_idx].G = static_cast<A_u_long>(CLAMP_VALUE(outG, 0.f, 1023.f));
					localDst[dst_idx].B = static_cast<A_u_long>(CLAMP_VALUE(outB, 0.f, 1023.f));

				} /* for (i = 0; i < in_data->width; i++) */

			} /* for (j = 0; j < height; j++) */

		} /* if (true == iKernel->LoadKernel()) */
		else
			bResult = false;
	} /* if (nullptr != (iKernel = GetKernel<int32_t>(choosedKernel))) */
	else
		bResult = false;

	iKernel = nullptr; /* for DBG purpose */

	return bResult;
}


bool ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const PrPixelFormat& destinationPixelFormat,
	const uint32_t& choosedKernel
) noexcept
{
	bool bValue = true;

		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				bValue = ProcessPrImage_BGRA_4444_8u (in_data, out_data, params, output, choosedKernel);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				bValue = ProcessPrImage_BGRA_4444_16u (in_data, out_data, params, output, choosedKernel);
			break;

			case PrPixelFormat_VUYA_4444_8u:
			case PrPixelFormat_VUYA_4444_8u_709:
				bValue = ProcessPrImage_VUYA_4444_8u (in_data, out_data, params, output, choosedKernel, PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				bValue = ProcessPrImage_BGRA_4444_32f (in_data, out_data, params, output, choosedKernel);
			break;

			case PrPixelFormat_VUYA_4444_32f:
			case PrPixelFormat_VUYA_4444_32f_709:
				bValue = ProcessPrImage_VUYA_4444_32f (in_data, out_data, params, output, choosedKernel, PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat);
			break;

			case PrPixelFormat_RGB_444_10u:
				bValue = ProcessPrImage_RGB_444_10u (in_data, out_data, params, output, choosedKernel);
			break;

			default:
				bValue = false;
			break;
		} /* switch (destinationPixelFormat) */

		if (false == bValue)
		{
			/* something going wrong - let's make simple copy from input to output */
			PF_COPY(&params[0]->u.ld, output, nullptr, nullptr);
		}

	return bValue;
}
